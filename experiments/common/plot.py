"""阶段实验的绘图工具：从 run 目录生成图表。

数据来源只有两个，都不依赖 tensorboard：
  - `<run>/stdout.log`：官方 baseline 打印的 `Epoch/global_step`、`SPS`、
    `eval_*_mean` 三类记录
  - `<run>/eval/per_episode.csv`：独立评估的逐回合结果（含分解指标）

产出（默认写到 runs/plots/）：
  - `<run>_eval.png`      评估曲线：success 与 return 随训练步数变化
  - `<run>_sps.png`       吞吐曲线：每个 epoch 的 steps/s
  - `<run>_breakdown.png` 独立评估的指标分解柱状图（有 per_episode.csv 时）
  - `compare_eval.png`    多个 run 的 success 曲线对比

用法（仓库根目录）：
    ./.venv-maniskill/bin/python experiments/common/plot.py \
        experiments/04_ppo_pickcube/runs/train_20261103_095601_seed1 \
        experiments/04_ppo_pickcube/runs/train6m_20261103_095904_seed1
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EPOCH_RE = re.compile(r"Epoch: (\d+), global_step=(\d+)")
EVAL_RE = re.compile(r"eval_([a-z_]+)_mean=([-\d.eE+]+)")
SPS_RE = re.compile(r"SPS: ([-\d.]+)")


def setup_font() -> None:
    """系统里有 Noto Sans CJK，显式指定，否则中文全是方框。"""
    matplotlib.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def short_name(run_name: str) -> str:
    """把 train6m_20261103_095904_seed1 压成 train6m seed1，图例才放得下。"""
    m = re.match(r"(.+?)_\d{8}_\d{6}_seed(\d+)", run_name)
    return f"{m.group(1)} seed{m.group(2)}" if m else run_name


def parse_stdout(path: Path) -> dict:
    """从 stdout.log 抽出 epoch 序列、吞吐序列和评估序列。"""
    epochs: list[tuple[int, int]] = []          # (epoch, global_step)
    sps: list[tuple[int, float]] = []           # (epoch, steps/s)
    evals: list[dict] = []                      # {epoch, step, metrics...}

    cur_epoch, cur_step = 0, 0
    pending: dict = {}

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = EPOCH_RE.search(line)
        if m:
            if pending:  # 上一个 epoch 还留着未提交的评估
                evals.append({"epoch": cur_epoch, "step": cur_step, **pending})
                pending = {}
            cur_epoch, cur_step = int(m.group(1)), int(m.group(2))
            epochs.append((cur_epoch, cur_step))
            continue

        m = EVAL_RE.search(line)
        if m:
            pending[m.group(1)] = float(m.group(2))
            continue

        m = SPS_RE.search(line)
        if m:
            sps.append((cur_epoch, float(m.group(1))))
            if pending:
                evals.append({"epoch": cur_epoch, "step": cur_step, **pending})
                pending = {}

    if pending:
        evals.append({"epoch": cur_epoch, "step": cur_step, **pending})

    return {"epochs": epochs, "sps": sps, "evals": evals}


def parse_per_episode(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_eval(parsed: dict, name: str, out_png: Path) -> bool:
    name = short_name(name)
    evals = parsed["evals"]
    if not evals:
        return False

    steps = [e["step"] for e in evals]
    succ_key = next((k for k in evals[0] if k.endswith("success_once")), None)
    ret_key = next((k for k in evals[0] if k == "return"), None)

    fig, ax1 = plt.subplots(figsize=(9, 4.6), dpi=130)
    if ret_key:
        ax1.plot(steps, [e[ret_key] for e in evals], "o-",
                 color="#08519c", lw=1.8, ms=5, label="评估回报 (return)")
        ax1.set_xlabel("训练步数 (global_step)")
        ax1.set_ylabel("评估回报", color="#08519c")
        ax1.tick_params(axis="y", labelcolor="#08519c")
        ax1.grid(alpha=0.3)

    if succ_key:
        ax2 = ax1.twinx()
        ax2.plot(steps, [e[succ_key] for e in evals], "s--",
                 color="#cb181d", lw=1.8, ms=6, label="成功率 (success_once)")
        ax2.set_ylabel("成功率", color="#cb181d")
        ax2.set_ylim(-0.05, 1.05)
        ax2.tick_params(axis="y", labelcolor="#cb181d")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = (ax2.get_legend_handles_labels() if succ_key else ([], []))
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    ax1.set_title(f"{name}\n评估曲线（每次评估约 8 个回合，注意样本量小）", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return True


def plot_sps(parsed: dict, name: str, out_png: Path) -> bool:
    name = short_name(name)
    sps = parsed["sps"]
    if not sps:
        return False
    xs = [e for e, _ in sps]
    ys = [v for _, v in sps]
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=130)
    ax.plot(xs, ys, color="#238b45", lw=1.6)
    ax.set_xlabel("迭代 (epoch)")
    ax.set_ylabel("吞吐 (steps/s)")
    ax.set_title(f"{name}   吞吐曲线（含评估与梯度更新开销）", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return True


def plot_breakdown(rows: list[dict], name: str, out_png: Path) -> bool:
    """把独立评估里的 *_ever 指标画成横向柱状图。"""
    name = short_name(name)
    if not rows:
        return False
    keys = [k for k in rows[0] if k.endswith("_ever")]
    if not keys:
        return False

    n = len(rows)
    labels, rates = [], []
    for k in keys:
        labels.append(k.replace("_ever", ""))
        rates.append(sum(int(r[k]) for r in rows) / n)

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(labels) + 1.8), dpi=130)
    colors = ["#cb181d" if lab == "success" else "#6baed6" for lab in labels]
    ax.barh(labels, rates, color=colors, height=0.55)
    for i, r in enumerate(rates):
        ax.text(min(r + 0.02, 0.98), i, f"{r:.0%}", va="center", fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(f"比例（{n} 个评估回合）")
    ax.set_title(f"{name}\n独立评估指标分解", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return True


def plot_compare(parsed_list: list[tuple[str, dict]], out_png: Path) -> bool:
    series = [(short_name(n), p) for n, p in parsed_list
              if any(k.endswith("success_once") for k in (p["evals"][0] if p["evals"] else {}))]
    if len(series) < 2:
        return False

    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=130)
    cmap = plt.get_cmap("tab10")
    for i, (name, parsed) in enumerate(series):
        evals = parsed["evals"]
        key = next(k for k in evals[0] if k.endswith("success_once"))
        ax.plot([e["step"] for e in evals], [e[key] for e in evals],
                "o-", color=cmap(i), lw=1.8, ms=5, label=name)
    ax.set_xlabel("训练步数 (global_step)")
    ax.set_ylabel("成功率 (success_once)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("不同训练预算的评估成功率对比（每次评估约 8 个回合）", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="从 run 目录生成图表")
    ap.add_argument("runs", nargs="+", help="run 目录（含 stdout.log）")
    ap.add_argument("--out", default=None,
                    help="输出目录，默认是第一个 run 的上一级目录下 plots/")
    args = ap.parse_args()

    setup_font()

    run_dirs = [Path(r).resolve() for r in args.runs]
    out_dir = Path(args.out).resolve() if args.out else run_dirs[0].parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed_all: list[tuple[str, dict]] = []
    for run_dir in run_dirs:
        log = run_dir / "stdout.log"
        if not log.exists():
            print(f"[跳过] 没有 stdout.log: {run_dir}")
            continue

        parsed = parse_stdout(log)
        parsed_all.append((run_dir.name, parsed))
        print(f"{run_dir.name}: {len(parsed['epochs'])} epochs, "
              f"{len(parsed['evals'])} 次评估, {len(parsed['sps'])} 个吞吐点")

        if plot_eval(parsed, run_dir.name, out_dir / f"{run_dir.name}_eval.png"):
            print(f"  -> {run_dir.name}_eval.png")
        if plot_sps(parsed, run_dir.name, out_dir / f"{run_dir.name}_sps.png"):
            print(f"  -> {run_dir.name}_sps.png")

        ep = run_dir / "eval" / "per_episode.csv"
        if ep.exists():
            rows = parse_per_episode(ep)
            if plot_breakdown(rows, run_dir.name, out_dir / f"{run_dir.name}_breakdown.png"):
                print(f"  -> {run_dir.name}_breakdown.png")

    if plot_compare(parsed_all, out_dir / "compare_eval.png"):
        print("  -> compare_eval.png")

    print(f"\n输出目录: {out_dir}")


if __name__ == "__main__":
    main()
