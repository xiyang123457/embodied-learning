"""Draw the evaluation summary of an ACT run from result_policy_best.txt.

    python plot_eval.py runs/train25_e2000
    python plot_eval.py runs/train25_e2000 --baseline 26 --baseline-label "200 epochs"

Left panel: how far the 50 rollouts got (reward buckets).
Right panel: success rate against an optional baseline run.
"""

import argparse
import os
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

BUCKET_HINTS = {
    0: 'any contact',
    1: 'grasp',
    2: 'lifted',
    3: 'reach other arm',
    4: 'handover',
}


def parse_result(path):
    text = open(path, 'r', errors='replace').read()
    success = re.search(r'Success rate:\s*([\d.]+)', text)
    ret = re.search(r'Average return:\s*([\d.]+)', text)
    buckets = []
    for level, count, total, pct in re.findall(r'Reward >= (\d+): (\d+)/(\d+) = ([\d.]+)%', text):
        buckets.append({'level': int(level), 'count': int(count), 'total': int(total), 'pct': float(pct)})
    return {
        'success': float(success.group(1)) if success else None,
        'return': float(ret.group(1)) if ret else None,
        'buckets': buckets,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir')
    parser.add_argument('--result', default=None, help='default: <run_dir>/result_policy_best.txt')
    parser.add_argument('--out', default=None, help='default: <run_dir>/eval_summary.png')
    parser.add_argument('--baseline', type=float, default=None, help='baseline success rate in percent')
    parser.add_argument('--baseline-label', default='baseline')
    parser.add_argument('--title', default=None)
    args = parser.parse_args()

    result_path = args.result or os.path.join(args.run_dir, 'result_policy_best.txt')
    out_path = args.out or os.path.join(args.run_dir, 'eval_summary.png')
    data = parse_result(result_path)
    if not data['buckets']:
        raise SystemExit(f'no "Reward >= N" lines in {result_path}')

    total = data['buckets'][0]['total']
    labels = [f">={b['level']}\n{BUCKET_HINTS.get(b['level'], '')}" for b in data['buckets']]
    counts = [b['count'] for b in data['buckets']]
    colors = ['#8c8c8c'] + ['#4c72b0'] * (len(counts) - 2) + ['#2e7d32'] if len(counts) > 2 else ['#2e7d32']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)

    ax = axes[0]
    bars = ax.bar(labels, counts, color=colors, width=0.62)
    for bar, count, bucket in zip(bars, counts, data['buckets']):
        ax.text(bar.get_x() + bar.get_width() / 2, count + total * 0.025,
                f'{count}/{total}\n{bucket["pct"]:.0f}%', ha='center', va='bottom', fontsize=10)
    ax.set_ylim(0, total * 1.16)
    ax.set_ylabel(f'rollouts (of {total})', fontsize=11)
    ax.set_title('reward bucket distribution', fontsize=12)
    ax.grid(axis='y', alpha=0.3, ls=':')
    ax.set_axisbelow(True)

    ax = axes[1]
    names, values, notes, cols = [], [], [], []
    if args.baseline is not None:
        names.append(args.baseline_label)
        values.append(args.baseline)
        notes.append('')
        cols.append('#bbbbbb')
    names.append('this run')
    values.append(data['success'] * 100 if data['success'] is not None else 0)
    notes.append(f'{(data["success"] or 0) * total:.0f}/{total}')
    cols.append('#2e7d32')
    bars = ax.bar(names, values, color=cols, width=0.5)
    for bar, value, note in zip(bars, values, notes):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5,
                f'{value:.0f}%' + (f'  ({note})' if note else ''), ha='center', va='bottom', fontsize=11)
    ax.set_ylim(0, max(values + [1]) * 1.3)
    ax.set_ylabel('success rate (%)', fontsize=11)
    ax.set_title('success rate', fontsize=12)
    ax.grid(axis='y', alpha=0.3, ls=':')
    ax.set_axisbelow(True)

    fig.suptitle(args.title or os.path.basename(os.path.normpath(args.run_dir)), fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
