"""Regenerate the train/val curves of an ACT run from its train.log.

The in-training plots are drawn at every 100 epochs by `plot_history()` in
`third_party/act/imitate_episodes.py`. This script reproduces the same figures
from the log, so a finished run can be re-plotted without retraining.

    python plot_history.py runs/train25_e2000
    python plot_history.py runs/train25_e2000 --smooth 25 --dpi 200
"""

import argparse
import os
import re

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

KEYS = ('loss', 'l1', 'kl')
COLORS = {'train': '#1f77b4', 'validation': '#d62728'}


def rolling_mean(values, window):
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or values.size == 0:
        return values
    window = min(window, values.size)
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode='edge')
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode='valid')[:values.size]


def rolling_std(values, window):
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or values.size == 0:
        return np.zeros_like(values)
    window = min(window, values.size)
    pad = window // 2
    kernel = np.ones(window) / window
    mean = rolling_mean(values, window)
    sq = np.convolve(np.pad(values ** 2, (pad, pad), mode='edge'), kernel, mode='valid')[:values.size]
    return np.sqrt(np.clip(sq - mean ** 2, 0, None))


def parse_log(path):
    """Return {'epoch': [...], 'val': {key: [...]}, 'train': {key: [...]}}.

    Each epoch block looks like:
        Epoch 12
        Val loss:   0.12345
        l1: 0.233 kl: 0.031 loss: 0.123
        Train loss: 0.54321
        l1: 0.255 kl: 0.230 loss: 0.544
    So a block is parsed by collecting its lines and taking the two `l1:`/`kl:`/
    `loss:` summaries in order (first = validation, second = training).
    """
    history = {'epoch': [], 'val': {k: [] for k in KEYS}, 'train': {k: [] for k in KEYS}}

    def flush(block):
        val_loss = train_loss = None
        summaries = []
        for line in block['lines']:
            if line.startswith('Val loss:'):
                val_loss = float(line.split(':')[1])
            elif line.startswith('Train loss:'):
                train_loss = float(line.split(':')[1])
            elif line.startswith('l1:'):
                summaries.append({k: float(v) for k, v in re.findall(r'(\w+):\s*(-?[\d.eE+-]+)', line)})
        if val_loss is None or train_loss is None or len(summaries) < 2:
            return  # partial block, e.g. the run was killed mid-epoch
        history['epoch'].append(block['epoch'])
        for key in ('l1', 'kl'):
            history['val'][key].append(summaries[0].get(key, np.nan))
            history['train'][key].append(summaries[1].get(key, np.nan))
        history['val']['loss'].append(val_loss)      # 5 decimals, unlike the summary line
        history['train']['loss'].append(train_loss)

    block = None
    for line in open(path, 'r', errors='replace'):
        if line.startswith('Epoch '):
            if block is not None:
                flush(block)
            block = {'epoch': int(line.split()[1]), 'lines': []}
        elif block is not None:
            block['lines'].append(line.strip())
    if block is not None:
        flush(block)
    return history


def plot_key(history, key, run_dir, seed, smooth, dpi, skip_start_ratio=0.05):
    epochs = np.asarray(history['epoch'], dtype=np.float64)
    train_values = np.asarray(history['train'][key], dtype=np.float64)
    val_values = np.asarray(history['val'][key], dtype=np.float64)

    x_train = epochs[:train_values.size]
    x_val = epochs[:val_values.size]
    train_smooth = rolling_mean(train_values, smooth)
    val_smooth = rolling_mean(val_values, smooth)
    val_std = rolling_std(val_values, smooth)

    fig, ax = plt.subplots(figsize=(11, 5), dpi=dpi)
    # training: the raw curve is already fairly smooth, keep it as context
    ax.plot(x_train, train_values, color=COLORS['train'], lw=0.5, alpha=0.15)
    ax.plot(x_train, train_smooth, color=COLORS['train'], lw=2.0, label='train')
    # validation: only 5 episodes, so a single epoch swings wildly; show the
    # rolling mean with a +/-1 sigma band instead of the per-epoch spikes
    if val_values.size:
        ax.fill_between(x_val, np.clip(val_smooth - val_std, 0, None), val_smooth + val_std,
                        color=COLORS['validation'], alpha=0.18, lw=0, label='validation ±1σ')
        ax.plot(x_val, val_smooth, color=COLORS['validation'], lw=2.0, label='validation')

    # y range from the smoothed curves, so a single noisy epoch cannot decide the
    # scale. Loss/kl start ~2 orders of magnitude above their plateau: only then
    # is the head of the run dropped from the range.
    smoothed = np.concatenate([s for s in (train_smooth, val_smooth) if s.size])
    positive = smoothed[smoothed > 0]
    median = float(np.median(positive)) if positive.size else 0.0
    ratio = skip_start_ratio if (median > 0 and smoothed.max() > 5 * median) else 0.0
    tail = np.concatenate([s[int(s.size * ratio):] for s in (train_smooth, val_smooth) if s.size])
    if tail.size:
        ax.set_ylim(0, max(float(tail.max()) * 1.08, 1e-6))
    n_above = int((np.concatenate([train_values, val_values]) > ax.get_ylim()[1]).sum())
    if n_above:
        ax.text(0.985, 0.80, f'{n_above} startup points above the y range',
                ha='right', va='top', transform=ax.transAxes, fontsize=9, color='0.45',
                bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.5))

    if val_values.size:
        best_idx = int(np.argmin(val_values))
        best_epoch = int(epochs[best_idx])
        best_value = float(val_values[best_idx])
        ax.axvline(best_epoch, color='0.5', ls=':', lw=1.2)
        ax.annotate(f'best val {best_value:.4f} @ {best_epoch}',
                    xy=(best_epoch, min(best_value, ax.get_ylim()[1] * 0.9)),
                    xytext=(-12, 34), textcoords='offset points',
                    ha='right', fontsize=9, color=COLORS['validation'],
                    arrowprops=dict(arrowstyle='->', color='0.5', lw=1, shrinkA=0, shrinkB=2))

    ax.set_xlabel('epoch', fontsize=11)
    ax.set_ylabel(key, fontsize=11)
    ax.set_title(f'{key}   (rolling mean w={smooth}; shaded = validation ±1σ)', fontsize=12)
    ax.grid(alpha=0.3, ls=':')
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10, loc='upper right')
    ax.margins(x=0.01)
    fig.tight_layout()

    out = os.path.join(run_dir, f'train_val_{key}_seed_{seed}.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'wrote {out}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', help='run directory containing train.log')
    parser.add_argument('--log', default=None, help='log path (default: <run_dir>/train.log)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--smooth', type=int, default=0, help='rolling window (default: epochs/100)')
    parser.add_argument('--dpi', type=int, default=150)
    args = parser.parse_args()

    log_path = args.log or os.path.join(args.run_dir, 'train.log')
    history = parse_log(log_path)
    n = len(history['epoch'])
    if n == 0:
        raise SystemExit(f'no epochs parsed from {log_path}')

    smooth = args.smooth or max(1, n // 100)
    print(f'{n} epochs from {log_path}, rolling window = {smooth}')
    for key in KEYS:
        plot_key(history, key, args.run_dir, args.seed, smooth, args.dpi)


if __name__ == '__main__':
    main()
