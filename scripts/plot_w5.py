"""W5 · D3：两张图。

    outputs/w5/path_xy.png       一条专家轨迹的末端 x-y 路径（claim：直线靠近，不绕路）
    outputs/w5/error_curve.png   误差随步数变化，k=5/50/100 与零动作（claim：k=50 最快且稳，
                                 k=100 越过 0.05·r·k<2 后开始抖）

用法（仓库根目录）：
    ./.venv/bin/python scripts/plot_w5.py
    ./.venv/bin/python scripts/plot_w5.py --seed 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 不依赖窗口
import matplotlib.pyplot as plt
import numpy as np

from w5_core import make_env, make_policy_feedback, policy_zero, run_episode

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "w5"
THR = 0.05


def plot_path_xy(seed: int) -> Path:
    env = make_env()
    try:
        obs, _ = env.reset(seed=seed)
        goal = obs["desired_goal"].copy()
        start = obs["achieved_goal"].copy()
        pts = [start]
        policy = make_policy_feedback(50)
        for _ in range(50):
            obs, _r, term, trunc, _i = env.step(policy(obs, {}))
            pts.append(obs["achieved_goal"].copy())
            if term or trunc:
                break
    finally:
        env.close()

    pts = np.asarray(pts)
    path_len = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    straight = float(np.linalg.norm(goal - start))

    fig, ax = plt.subplots(figsize=(4.0, 5.8))
    ax.plot([start[0], goal[0]], [start[1], goal[1]], ls="--", lw=1.0,
            color="#999999", label="straight line start→goal")
    ax.plot(pts[:, 0], pts[:, 1], "-o", ms=3.0, lw=1.5, color="#1f77b4",
            label="end-effector path")
    ax.scatter(*pts[0, :2], s=90, marker="s", color="#2ca02c", zorder=5, label="start")
    ax.scatter(*pts[-1, :2], s=90, marker="D", color="#1f77b4", zorder=5, label="end")
    ax.scatter(*goal[:2], s=170, marker="*", color="#d62728", zorder=5, label="goal")
    ax.text(0.03, 0.03,
            f"end−goal = {np.linalg.norm(pts[-1] - goal) * 1000:.1f} mm "
            f"(tolerance {THR * 1000:.0f} mm)",
            transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Straight approach, no detour\n"
                 f"length {path_len:.3f} m vs straight {straight:.3f} m "
                 f"({path_len / straight:.2f}×)", fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.10)
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", fontsize=7.5, framealpha=0.95)

    path = OUT / "path_xy.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"写入 {path.relative_to(ROOT)}  （路径长度 {path_len:.3f} m，"
          f"直线距离 {straight:.3f} m，比值 {path_len / straight:.2f}）")
    return path


def plot_error_curve(seed: int) -> Path:
    env = make_env()
    curves = {}
    try:
        for label, policy in [
            ("k=5 feedback", make_policy_feedback(5)),
            ("k=50 feedback", make_policy_feedback(50)),
            ("k=100 feedback", make_policy_feedback(100)),
            ("zero action", policy_zero),
        ]:
            rec = run_episode(env, policy, seed, return_trace=True)
            curves[label] = rec["dist_trace"]
    finally:
        env.close()

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    colors = {"k=5 feedback": "#ff7f0e", "k=50 feedback": "#1f77b4",
              "k=100 feedback": "#9467bd", "zero action": "#7f7f7f"}
    for label, trace in curves.items():
        ax.semilogy(np.arange(len(trace)), np.maximum(trace, 1e-5), lw=1.6,
                    color=colors[label], label=label)
    ax.axhline(THR, ls="--", lw=1.2, color="#d62728")
    ax.text(0.5, THR * 1.15, "tolerance 0.05 m", color="#d62728", fontsize=9)
    ax.set_xlabel("control step")
    ax.set_ylabel("‖desired − achieved‖ (m)")
    ax.set_title(f"Position error vs step (same seed={seed}, 50-step limit)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9)

    path = OUT / "error_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    for label, trace in curves.items():
        under = np.where(trace < THR)[0]
        first = int(under[0]) if under.size else None
        print(f"  {label:<14s} first_in_tol={first}  final={trace[-1]:.4f} m  "
              f"min={trace.min():.4f} m")
    print(f"写入 {path.relative_to(ROOT)}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    plot_path_xy(args.seed)
    print()
    plot_error_curve(args.seed)


if __name__ == "__main__":
    main()
