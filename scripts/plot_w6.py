"""W6 · D4：两张图。

    outputs/w6/train_curves.png     3 个种子的 train/val loss（n20 与 n40 两栏）
    outputs/w6/closed_loop.png      左：同一测试回合的误差曲线；右：容差扫描

用法（仓库根目录）：
    ./.venv/bin/python scripts/plot_w6.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_closed_loop import fit_linear_regression, load_mlp, policy_linear  # noqa: E402
from w5_core import make_env, make_policy_feedback, make_policy_random, run_episode  # noqa: E402

OUT = ROOT / "outputs" / "w6"
THR = 0.05
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _smooth(v: NDArray[np.float64], w: int = 5) -> NDArray[np.float64]:
    """滑动平均：一个 epoch 只有 3–7 个 batch，原始 train loss 抖动太大看不出趋势。

    两端用边缘值补齐，否则卷积的补零会把首尾拉出假尖峰。
    """
    if len(v) < w:
        return v
    pad = w // 2
    vp = np.pad(v, pad, mode="edge")
    return np.convolve(vp, np.ones(w) / w, mode="valid")


def _best_points() -> dict:
    """每个 (tag, seed) 真正存进 ckpt 的那个 epoch。

    训练时每个 epoch 都比一次 val_loss，但日志只每 10 个 epoch 记一条。
    所以曲线上的 argmin != 实际 checkpoint，必须从 train_summary.csv 读。
    """
    path = OUT / "train_summary.csv"
    out = {}
    if not path.exists():
        return out
    for r in csv.DictReader(path.open(encoding="utf-8")):
        out[(r["tag"], r["seed"])] = (int(r["best_epoch"]), float(r["best_val"]))
    return out


def fig_train_curves() -> Path:
    best = _best_points()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    for ax, tag in zip(axes[:2], ("n20", "n40")):
        rows = list(csv.DictReader((OUT / f"train_curves_{tag}.csv").open(encoding="utf-8")))
        for i, seed in enumerate(("0", "1", "2")):
            sub = [r for r in rows if r["seed"] == seed]
            ep = np.array([int(r["epoch"]) for r in sub])
            tr = _smooth(np.array([float(r["train_loss"]) for r in sub]))
            va = np.array([float(r["val_loss"]) for r in sub])
            c = COLORS[i % len(COLORS)]
            ax.semilogy(ep, tr, ls="--", lw=1.1, color=c, alpha=0.8)
            ax.semilogy(ep, va, ls="-", lw=1.5, color=c, label=f"seed{seed} val")
            be, bv = best.get((tag, seed), (ep[int(np.argmin(va))], va.min()))
            ax.scatter(be, bv, marker="*", s=130, color=c, zorder=5,
                       edgecolor="black", linewidth=0.4)
        ax.plot([], [], "--", color="0.35", lw=1.1, label="train (dashed)")
        ax.plot([], [], "*", color="0.35", ls="none", ms=9, label="saved ckpt epoch")
        ax.set_xlabel("epoch")
        ax.set_ylabel("weighted MSE loss")
        ax.set_title(f"{tag}: {15 if tag == 'n20' else 35} train / 5 val episodes", fontsize=9)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  frameon=False)
        ax.set_ylim(5e-6, 2.0)

    ax = axes[2]
    rows = list(csv.DictReader((OUT / "train_curves_n20.csv").open(encoding="utf-8")))
    for i, seed in enumerate(("0", "1", "2")):
        sub = [r for r in rows if r["seed"] == seed]
        ep = np.array([int(r["epoch"]) for r in sub])
        far = _smooth(np.array([float(r["val_mse_far"]) for r in sub]), 7)
        ax.plot(ep, far, lw=1.4, color=COLORS[i % len(COLORS)], label=f"seed{seed}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val MSE on dist≥5 cm subset")
    ax.set_title("n20: error on far validation samples drifts up", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Train loss falls 3 orders below validation (overfitting by construction); "
                 "doubling the data halves validation loss", fontsize=10)
    path = OUT / "train_curves.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_closed_loop(seed: int = 100) -> Path:
    coef, intercept, _ = fit_linear_regression()
    mlp, _meta = load_mlp(OUT / "ckpt" / "bc_n40_seed0.pt")
    policies = {
        "feedback k=50": make_policy_feedback(50),
        "MLP-BC (n40, seed0)": mlp,
        "linear regression": policy_linear(coef, intercept),
        "random": make_policy_random(seed),
    }

    env = make_env()
    traces = {}
    try:
        for name, pol in policies.items():
            traces[name] = run_episode(env, pol, seed, return_trace=True)["dist_trace"]
    finally:
        env.close()

    tol = list(csv.DictReader((OUT / "results_tolerance.csv").open(encoding="utf-8")))
    thrs = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for i, (name, tr) in enumerate(traces.items()):
        ax1.semilogy(np.arange(len(tr)), np.maximum(tr, 1e-5), lw=1.6,
                     color=COLORS[i % len(COLORS)], label=name)
    ax1.axhline(THR, ls="--", lw=1.2, color="#d62728")
    ax1.text(0.5, THR * 1.2, "tolerance 0.05 m", color="#d62728", fontsize=8)
    ax1.set_xlabel("control step")
    ax1.set_ylabel("‖desired − achieved‖ (m)")
    ax1.set_title(f"Closed loop on test seed {seed}", fontsize=10)
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(fontsize=8, loc="lower left")

    # feedback 与 mlp_n40_s0 在 6 个阈值上数值完全相同（20,20,20,20,19,19），
    # 同色同宽会互相遮住。把 experts 画成粗淡的"带子"垫底，MLP 用虚线压在上面。
    show = [
        ("feedback", "feedback k=50 (band)", dict(color=COLORS[0], lw=9, alpha=0.30, marker="o", ms=8)),
        ("mlp_n40_s0", "MLP-BC n40 s0 (dashed)", dict(color=COLORS[1], lw=1.6, ls="--", marker="o", ms=4)),
        ("linear", "linear regression", dict(color=COLORS[2], lw=1.7, ls="-", marker="o", ms=4)),
        ("random", "random", dict(color=COLORS[3], lw=1.7, ls="-", marker="o", ms=4)),
    ]
    sub = {r["policy"]: r for r in tol if r["condition"] == "std"}
    for key, label, style in show:
        if key not in sub:
            continue
        ok = [int(sub[key][f"ok_{t}"]) for t in thrs]
        ax2.plot(thrs, ok, label=label, **style)
    ax2.text(0.03, 0.10, "feedback and MLP overlap at all 6 thresholds",
             transform=ax2.transAxes, fontsize=7.5, color="0.35")
    ax2.set_xscale("log")
    ax2.set_xlabel("success threshold (m)")
    ax2.set_ylabel("successful episodes (out of 20)")
    ax2.set_ylim(-1, 21)
    ax2.set_yticks(range(0, 21, 5))
    ax2.set_title("Standard tolerance saturates; 2 mm separates methods", fontsize=10)
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8)

    path = OUT / "closed_loop.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p1 = fig_train_curves()
    print(f"写入 {p1.relative_to(ROOT)}")
    p2 = fig_closed_loop()
    print(f"写入 {p2.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
