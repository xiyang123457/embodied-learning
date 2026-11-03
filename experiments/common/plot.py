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
    # newline="" 交给 csv 模块处理 CRLF，否则字段会带上 "\r" 导致 float() 失败
    with path.open(encoding="utf-8", newline="") as f:
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


def plot_goal_height(rows: list[dict], name: str, out_png: Path,
                     threshold: float = 0.15) -> bool:
    """目标高度与成败的关系：左侧散点，右侧分层成功率。"""
    if not rows:
        return False
    gz = [float(r["goal_z"]) for r in rows]
    succ = [int(r["success_ever"]) for r in rows]

    rng = np.random.default_rng(0)
    jitter = rng.normal(0, 0.02, len(gz))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=130)

    ax = axes[0]
    for g, s, j in zip(gz, succ, jitter):
        ax.scatter(g, s + j, s=55, alpha=0.85,
                   color="#238b45" if s else "#cb181d")
    ax.axvline(threshold, color="#636363", ls=":", lw=1.2)
    ax.set_xlabel("目标高度 goal_z (m)")
    ax.set_ylabel("是否成功")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["失败", "成功"])
    ax.set_ylim(-0.35, 1.35)
    ax.grid(alpha=0.3)
    ax.set_title(f"{short_name(name)}  每个种子的目标高度与结果", fontsize=10)

    ax = axes[1]
    lo = [s for g, s in zip(gz, succ) if g < threshold]
    hi = [s for g, s in zip(gz, succ) if g >= threshold]
    labels, rates, colors = [], [], []
    for lab, group, col in ((f"goal_z < {threshold}\n(n={len(lo)})", lo, "#238b45"),
                            (f"goal_z >= {threshold}\n(n={len(hi)})", hi, "#cb181d")):
        labels.append(lab)
        rates.append(sum(group) / len(group) if group else 0.0)
        colors.append(col)
    ax.bar(labels, rates, color=colors, width=0.5)
    for i, r in enumerate(rates):
        ax.text(i, r + 0.02, f"{r:.0%}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("成功率")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("按目标高度分层成功率", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return True


def plot_trajectory(traj_path: Path, seed: int, name: str, out_png: Path,
                    goal_thresh: float = 0.025, static_thresh: float = 0.2) -> bool:
    """单回合轨迹：obj→goal 距离与 max|qvel| 随步数变化，并标出两条阈值线。

    这张图是用来看「到位」与「静止」是否错开的：绿色底色是方块在目标圈内的区间，
    若红线（速度）在底色结束后才降到虚线以下，就是「减速时把方块推出了目标」。
    """
    with traj_path.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r["seed"]) == seed]
    if not rows:
        return False

    steps = [int(r["step"]) for r in rows]
    dist = [float(r["obj_to_goal_dist"]) for r in rows]
    qv = [float(r["qvel_max"]) for r in rows]
    placed = [int(r["is_obj_placed"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(9.5, 4.6), dpi=130)

    ax1.fill_between(steps, 0, 1, where=[p > 0 for p in placed],
                     transform=ax1.get_xaxis_transform(),
                     color="#a1d99b", alpha=0.35, step="mid")
    ax1.plot(steps, dist, color="#08519c", lw=2.0, label="obj→goal 距离")
    ax1.axhline(goal_thresh, color="#08519c", ls="--", lw=1.1,
                label=f"成功距离阈值 {goal_thresh * 100:.1f} cm")
    ax1.set_xlabel("步数 (step)")
    ax1.set_ylabel("obj→goal 距离 (m)", color="#08519c")
    ax1.tick_params(axis="y", labelcolor="#08519c")
    ax1.set_ylim(0, max(max(dist) * 1.15, goal_thresh * 4))

    ax2 = ax1.twinx()
    ax2.plot(steps, qv, color="#cb181d", lw=2.0, label="max|qvel|")
    ax2.axhline(static_thresh, color="#cb181d", ls="--", lw=1.1,
                label=f"静止阈值 {static_thresh}")
    ax2.set_ylabel("max|qvel| (rad/s)", color="#cb181d")
    ax2.tick_params(axis="y", labelcolor="#cb181d")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.5)
    ax1.set_title(f"{short_name(name)}  seed={seed}   绿色底色 = 方块在目标圈内", fontsize=10)
    ax1.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return True


def plot_compare(parsed_list: list[tuple[str, dict]], out_png: Path) -> bool:
    """多 run 的 success 对比。

    只画一条绝对步数曲线时，短预算的线会全部挤在 x 轴最左端、并且贴在 y=0 上
    互相遮挡，看起来像只有一条。所以拆成两栏：
      左：绝对步数（看谁能训到多高）
      右：训练进度归一化（看同样进度下谁学得更快，三条线形状可直接对比）
    同时用不同线型与标记，即使数值重合也能分辨。
    """
    series = [(short_name(n), p) for n, p in parsed_list
              if p["evals"] and any(k.endswith("success_once") for k in p["evals"][0])]
    if len(series) < 2:
        return False
    # 按总步数从短到长排序，短预算先画，长预算压在上面
    series.sort(key=lambda x: max(e["step"] for e in x[1]["evals"]))

    cmap = plt.get_cmap("tab10")
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=130)
    for i, (name, parsed) in enumerate(series):
        evals = parsed["evals"]
        key = next(k for k in evals[0] if k.endswith("success_once"))
        xs = [e["step"] for e in evals]
        ys = [e[key] for e in evals]
        total = xs[-1] if xs and xs[-1] > 0 else 1

        axes[0].plot([x / 1e6 for x in xs], ys, linestyle=linestyles[i % 4],
                     marker=markers[i % 4], color=cmap(i), lw=1.8, ms=5,
                     label=name, alpha=0.9)
        axes[1].plot([100 * x / total for x in xs], ys, linestyle=linestyles[i % 4],
                     marker=markers[i % 4], color=cmap(i), lw=1.8, ms=5,
                     label=name, alpha=0.9)

    axes[0].set_xlabel("训练步数（百万步）")
    axes[0].set_title("按绝对步数", fontsize=10)
    axes[1].set_xlabel("训练进度 (%)")
    axes[1].set_title("按训练进度归一化（对比学习速度）", fontsize=10)

    for ax in axes:
        ax.set_ylabel("成功率 (success_once)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5, loc="upper left")

    fig.suptitle("不同训练预算的评估成功率对比（每次评估约 8 个回合）", fontsize=10)
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

        # 诊断产物：任何含 episode_summary.csv 的子目录（diagnose_* / eval50* 等）
        for ep_sum in sorted(run_dir.glob("*/episode_summary.csv")):
            diag = ep_sum.parent
            traj = diag / "trajectory.csv"
            eps = parse_per_episode(ep_sum)
            tag = f"{run_dir.name}_{diag.name}"
            if plot_goal_height(eps, run_dir.name, out_dir / f"{tag}_goal_height.png"):
                print(f"  -> {tag}_goal_height.png")
            if not traj.exists():
                continue
            ok = next((e for e in eps if int(e["success_ever"]) == 1), None)
            bad = next((e for e in eps if int(e["success_ever"]) == 0
                        and int(e["first_placed"]) > 0), None)
            for lab, ref in (("success", ok), ("failure", bad)):
                if ref is None:
                    continue
                png = out_dir / f"{tag}_traj_{lab}_seed{ref['seed']}.png"
                if plot_trajectory(traj, int(ref["seed"]), run_dir.name, png):
                    print(f"  -> {png.name}")

    if plot_compare(parsed_all, out_dir / "compare_eval.png"):
        print("  -> compare_eval.png")

    print(f"\n输出目录: {out_dir}")


if __name__ == "__main__":
    main()
