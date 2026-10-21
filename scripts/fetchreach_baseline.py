"""W5 · D2 + D3：接口断言、单轴小动作、位置误差基线、扫增益、三策略 × 20 回合。

用法（仓库根目录）：
    ./.venv/bin/python scripts/fetchreach_baseline.py                 # 全跑
    ./.venv/bin/python scripts/fetchreach_baseline.py --section axis  # 只跑单轴
    ./.venv/bin/python scripts/fetchreach_baseline.py --section sweep
    ./.venv/bin/python scripts/fetchreach_baseline.py --section policies --k 10

产出：
    outputs/w5/baseline_results.csv    三策略 × 20 回合（60 行）
    outputs/w5/k_sweep.csv             k 扫描逐回合（7 × 20 行）
    outputs/w5/k_sweep_summary.csv      k 扫描按 k 汇总（7 行）
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from w5_core import (
    N_EPISODES,
    SEEDS,
    make_env,
    make_policy_feedback,
    make_policy_random,
    policy_zero,
    run_episode,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "w5"
K_GRID = [1, 2, 5, 10, 20, 50, 100]
CSV_FIELDS = [
    "policy", "k", "seed", "steps", "first_success_step", "frac_in_tol",
    "final_dist", "min_dist", "success_end", "success_ever", "terminated", "truncated",
]


# ------------------------------------------------------------------ B1
def check_interface(env) -> dict:
    obs, _info = env.reset(seed=0)
    required = {"observation", "achieved_goal", "desired_goal"}
    assert required.issubset(obs), "Missing observation keys"
    assert obs["achieved_goal"].shape == (3,)
    assert obs["desired_goal"].shape == (3,)
    assert env.action_space.shape == (4,)
    assert np.isfinite(obs["observation"]).all()
    assert env.action_space.contains(np.zeros(4, dtype=np.float32))
    print("[B1] 接口断言全部通过：键 / 形状 / 有限性 / 零动作合法")
    print(f"     obs keys = {sorted(obs.keys())}")
    return obs


# ------------------------------------------------------------------ B2
def single_axis(env) -> None:
    print("\n[B2] 单轴小动作：a=0.5 连走 5 步，打印相对起点的位移 (m)")
    for axis, sign in [(0, +1), (0, -1), (1, +1), (2, +1)]:
        obs, _ = env.reset(seed=0)
        p0 = obs["achieved_goal"].copy()
        print(f"--- axis={axis} sign={sign:+d} ---")
        for t in range(5):
            a = np.zeros(4, dtype=np.float32)
            a[axis] = sign * 0.5
            obs, *_ = env.step(a)
            print(f"    t={t}  Δachieved = {np.round(obs['achieved_goal'] - p0, 4)}")
        total = float(np.linalg.norm(obs["achieved_goal"] - p0))
        print(f"    5 步累计位移 = {total:.4f} m，命令量 = {5 * 0.05 * 0.5:.4f} m，"
              f"比例 = {total / (5 * 0.05 * 0.5):.3f}")

    print("\n[B2-补] a=1.0 连走 20 步：每步位移是不是常数？（饱和行为）")
    obs, _ = env.reset(seed=0)
    prev = obs["achieved_goal"].copy()
    steps = []
    for _ in range(20):
        a = np.zeros(4, dtype=np.float32)
        a[0] = 1.0
        obs, _r, term, trunc, _i = env.step(a)
        steps.append(float(np.linalg.norm(obs["achieved_goal"] - prev)))
        prev = obs["achieved_goal"].copy()
        if term or trunc:
            break
    print("    per-step |Δ| =", np.round(steps, 4).tolist())
    print(f"    前 5 步中位数 = {np.median(steps[:5]):.4f} m，"
          f"全 20 步中位数 = {np.median(steps):.4f} m，std = {np.std(steps):.4f}")


# ------------------------------------------------------------------ B4
def sweep_k(env) -> tuple[list[dict], list[dict]]:
    print("\n[B4] 扫增益 k = 1/2/5/10/20/50/100，每组 20 回合")
    rows, summary_rows = [], []
    for k in K_GRID:
        policy = make_policy_feedback(k)
        recs = [run_episode(env, policy, s) for s in SEEDS]
        for r in recs:
            rows.append({"policy": "feedback", "k": k, **{f: r[f] for f in CSV_FIELDS if f in r}})
        s = summarize(recs)
        summary_rows.append({"k": k, **s})
        print(f"    k={k:>3d}  success_end {s['success_end']:>2d}/20  "
              f"success_ever {s['success_ever']:>2d}/20  "
              f"first_success {s['first_success_mean']:6.2f}  "
              f"final {s['final_dist_mean']:.4f}  min {s['min_dist_mean']:.4f}")
    return rows, summary_rows


# ------------------------------------------------------------------ C
def run_policies(env, k: float) -> list[dict]:
    print(f"\n[C] 三策略 × 20 回合（反馈策略 k={k:g}，同一组 seed 0–19）")
    configs = [
        ("zero", 0, policy_zero),
        ("random", 0, None),
        ("feedback", k, make_policy_feedback(k)),
    ]
    rows, summaries = [], []
    for name, kk, policy in configs:
        recs = []
        for s in SEEDS:
            pol = make_policy_random(s) if name == "random" else policy
            r = run_episode(env, pol, s)
            recs.append(r)
            rows.append({"policy": name, "k": kk, **{f: r[f] for f in CSV_FIELDS if f in r}})
        s = summarize(recs)
        summaries.append((name, kk, s))
        print(f"    {name:<8s} k={kk:<3g} success_end {s['success_end']:>2d}/20  "
              f"success_ever {s['success_ever']:>2d}/20  "
              f"first_success {s['first_success_mean']:6.2f}  "
              f"final {s['final_dist_mean']:.4f}  min {s['min_dist_mean']:.4f}")

    # 现象归类：只对反馈策略做失败分析
    print("\n[C2] 反馈策略失败回合归类（按 final_dist / min_dist / 轨迹形态）")
    failures = [r for r in rows if r["policy"] == "feedback" and not r["success_ever"]]
    print(f"    失败回合数 = {len(failures)}")
    for r in failures:
        print(f"    seed={r['seed']:>2d} final={r['final_dist']:.4f} min={r['min_dist']:.4f}")
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"    写入 {path.relative_to(ROOT)}（{len(rows)} 行）")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", default="all",
                        choices=["all", "assert", "axis", "sweep", "policies"])
    parser.add_argument("--k", type=float, default=None,
                        help="反馈策略增益；缺省则先用扫增益结果自动挑一个")
    args = parser.parse_args()

    env = make_env()
    try:
        check_interface(env)

        summary_rows = None
        if args.section in ("all", "axis"):
            single_axis(env)

        if args.section in ("all", "sweep"):
            rows, summary_rows = sweep_k(env)
            write_csv(OUT_DIR / "k_sweep.csv", rows, CSV_FIELDS)
            write_csv(OUT_DIR / "k_sweep_summary.csv", summary_rows,
                      ["k", "n", "success_end", "success_ever", "first_success_mean",
                       "first_success_median", "final_dist_mean", "min_dist_mean"])

        if args.section in ("all", "policies"):
            k = args.k
            if k is None:
                # 自动选择：success_ever 最高者里取 first_success_mean 最小（并列取较小 k）
                if summary_rows is None:
                    summary_rows = sweep_k(env)[1]
                ok = [r for r in summary_rows if r["success_ever"] == N_EPISODES]
                pool = ok if ok else summary_rows
                k = min(pool, key=lambda r: (r["first_success_mean"], r["k"]))["k"]
                print(f"\n[C] 依据扫增益结果选定 k = {k:g}")
            rows = run_policies(env, k)
            write_csv(OUT_DIR / "baseline_results.csv", rows, CSV_FIELDS)
    finally:
        env.close()


if __name__ == "__main__":
    main()
