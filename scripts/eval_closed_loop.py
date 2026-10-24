"""W6 · D3：闭环测试 20 回合（seed 100–119），四类模型同一起点。

模型：
    feedback     位置反馈专家 a = clip(k·(goal−pos))，k=50（W5 最优）
    linear       同样的 8 维输入做最小二乘（指南 7.2 要求）
    mlp_n20/n40  BC 的 3 个训练种子
    random       均匀随机，下界

产出：
    outputs/w6/results.csv        每回合一行
    outputs/w6/results_summary.csv 每个模型一行（成功率、末距、步数、延迟、失败分类）
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.bc_dataset import (  # noqa: E402
    EXTRA_SEEDS,
    TEST_SEEDS,
    TRAIN_SEEDS,
    episodes_to_arrays,
    load_episodes,
)
from src.bc_model import BCPolicy, TorchPolicy  # noqa: E402
from w5_core import make_env, make_policy_feedback, make_policy_random, run_episode  # noqa: E402

OUT = ROOT / "outputs" / "w6"
CKPT = OUT / "ckpt"
THR = 0.05


# ------------------------------------------------------------------ baselines
def run_episode_warmup(env, policy, seed: int, warmup: int, return_trace: bool = True):
    """先执行 warmup 步随机动作把机械臂推离初始位置，再交给待测策略。

    同一 seed 下 warmup 动作序列对所有模型完全相同（rng 固定），保证对照公平。
    这就是"分布变化"条件下的闭环测试：初始状态不再来自专家轨迹的分布。
    """
    thr = float(env.unwrapped.distance_threshold)
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(5000 + seed)
    for _ in range(warmup):
        obs, _r, _t, _tr, info = env.step(rng.uniform(-1.0, 1.0, 4).astype(np.float32))

    d = float(np.linalg.norm(obs["desired_goal"] - obs["achieved_goal"]))
    first = 0 if d < thr else None
    min_dist, in_tol, trace = d, int(d < thr), [d]
    steps, lat = 0, []
    while True:
        t0 = time.perf_counter()
        a = policy(obs, info)
        lat.append((time.perf_counter() - t0) * 1e3)
        obs, _r, term, trunc, info = env.step(a)
        steps += 1
        d = float(np.linalg.norm(obs["desired_goal"] - obs["achieved_goal"]))
        trace.append(d)
        min_dist = min(min_dist, d)
        in_tol += int(d < thr)
        if first is None and d < thr:
            first = steps
        if term or trunc:
            break
    return {
        "steps": steps, "first_success_step": first, "final_dist": d, "min_dist": min_dist,
        "success_end": bool(info["is_success"]), "success_ever": first is not None,
        "terminated": bool(term), "truncated": bool(trunc),
        "latency_ms": float(np.mean(lat)), "dist_trace": np.asarray(trace),
    }


def policy_linear(coef, intercept):
    def policy(obs, info=None):
        from src.bc_dataset import obs_to_input

        x = obs_to_input(obs["observation"], obs["achieved_goal"], obs["desired_goal"])[0]
        a = np.zeros(4, dtype=np.float32)
        a[:3] = np.clip(x @ coef + intercept, -1.0, 1.0)
        return a

    return policy


def fit_linear_regression():
    """对 40 条训练数据做最小二乘（不加权，和 BC 的输入完全一致）。"""
    seeds = list(TRAIN_SEEDS) + list(EXTRA_SEEDS)
    arr = episodes_to_arrays(load_episodes(seeds))
    x = np.concatenate([arr["x"], np.ones((len(arr["x"]), 1), dtype=np.float32)], axis=1)
    coef, *_ = np.linalg.lstsq(x, arr["y"], rcond=None)
    return coef[:-1], coef[-1], arr


def load_mlp(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = BCPolicy()
    model.load_state_dict(ck["model"])
    return TorchPolicy(model, ck["x_mean"], ck["x_std"]), ck.get("meta", {})


# ------------------------------------------------------------------ classify
def classify(trace: np.ndarray, seed: int) -> tuple[str, dict]:
    """失败分类：走不到位 / 方向错误 / 振荡 / 超时。"""
    start, final = float(trace[0]), float(trace[-1])
    under = trace < THR
    crossings = int(np.sum(np.diff(under.astype(int)) != 0))
    if under[-1]:
        return "success", {"crossings": crossings}
    if final > start:
        kind = "方向错误"
    elif crossings >= 2:
        kind = "振荡"
    else:
        kind = "走不到位"
    return kind, {"crossings": crossings}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=TEST_SEEDS)
    parser.add_argument("--k", type=float, default=50.0)
    parser.add_argument("--warmups", type=int, nargs="*", default=[5, 20],
                        help="随机预热步数（分布变化条件），可给多个")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    coef, intercept, lin_data = fit_linear_regression()
    print(f"[linear] 用 {len(lin_data['x'])} 个样本拟合，"
          f"系数范数 {np.linalg.norm(coef):.3f}，截距范数 {np.linalg.norm(intercept):.3f}")

    policies: list[tuple[str, object]] = [
        ("feedback", make_policy_feedback(args.k)),
        ("linear", policy_linear(coef, intercept)),
        ("random", None),  # 每回合单独构造，保证随机流随 seed 变化
    ]
    for tag in ("n20", "n40"):
        for seed in (0, 1, 2):
            path = CKPT / f"bc_{tag}_seed{seed}.pt"
            if not path.exists():
                print(f"  跳过缺失的 checkpoint {path.name}")
                continue
            pol, meta = load_mlp(path)
            policies.append((f"mlp_{tag}_s{seed}", pol))

    env = make_env()
    rows, summary = [], []
    conditions = [("std", 0)] + [(f"perturb{w}", w) for w in args.warmups]
    try:
        for cond_name, warmup in conditions:
            print(f"\n[{cond_name}] {'标准起点' if warmup == 0 else f'先随机走 {warmup} 步再接管'}")
            for name, policy in policies:
                recs = []
                for s in args.seeds:
                    pol = make_policy_random(s) if policy is None else policy
                    if warmup:
                        r = run_episode_warmup(env, pol, s, warmup)
                        lat_mean = r["latency_ms"]
                    else:
                        lat = []

                        def timed(obs, info, _pol=pol, _lat=lat):
                            t0 = time.perf_counter()
                            a = _pol(obs, info)
                            _lat.append((time.perf_counter() - t0) * 1e3)
                            return a

                        r = run_episode(env, timed, s, return_trace=True)
                        lat_mean = float(np.mean(lat))
                    kind, _info = classify(r["dist_trace"], s)
                    recs.append({"condition": cond_name, "policy": name, "test_seed": s,
                                 **{k: r[k] for k in
                                    ("steps", "first_success_step", "final_dist", "min_dist",
                                     "success_end", "success_ever", "terminated", "truncated")},
                                 "latency_ms": lat_mean, "failure": kind})
                rows.extend(recs)
                n_ok = sum(r["success_end"] for r in recs)
                fails = [r for r in recs if r["failure"] != "success"]
                counts = {}
                for r in fails:
                    counts[r["failure"]] = counts.get(r["failure"], 0) + 1
                firsts = [r["first_success_step"] for r in recs if r["first_success_step"]]
                summary.append({
                    "condition": cond_name,
                    "policy": name,
                    "success": f"{n_ok}/{len(recs)}",
                    "success_n": n_ok,
                    "n": len(recs),
                    "final_dist_mean": float(np.mean([r["final_dist"] for r in recs])),
                    "min_dist_mean": float(np.mean([r["min_dist"] for r in recs])),
                    "first_success_mean": float(np.mean(firsts)) if firsts else float("nan"),
                    "latency_ms": float(np.mean([r["latency_ms"] for r in recs])),
                    "fail_detail": ";".join(f"{k}:{v}" for k, v in sorted(counts.items())),
                    "fail_seeds": ";".join(f"{r['test_seed']}({r['failure']})" for r in fails),
                })
                print(f"  {name:<14s} {n_ok:>2d}/{len(recs)}  末距 {summary[-1]['final_dist_mean']:.4f}  "
                      f"首次达标 {summary[-1]['first_success_mean']:.2f}  "
                      f"延迟 {summary[-1]['latency_ms']:.3f} ms  {summary[-1]['fail_detail']}")
    finally:
        env.close()

    fields = ["condition", "policy", "test_seed", "steps", "first_success_step", "final_dist",
              "min_dist", "success_end", "success_ever", "terminated", "truncated",
              "latency_ms", "failure"]
    with (OUT / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with (OUT / "results_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    # ---- 容差扫描：5cm 处所有模型饱和，收紧才能看出差别 ----
    tol_rows = []
    thresholds = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    print("\n[容差扫描] 每格的分子 = final_dist < 容差的回合数（分母 20）")
    print("  " + "policy".ljust(14) + "".join(f"{t:>8}" for t in thresholds))
    for cond_name, _w in conditions:
        for name, _p in policies:
            fd = np.array([r["final_dist"] for r in rows
                           if r["condition"] == cond_name and r["policy"] == name])
            counts = {t: int((fd < t).sum()) for t in thresholds}
            tol_rows.append({"condition": cond_name, "policy": name,
                             **{f"ok_{t}": counts[t] for t in thresholds}})
            if cond_name == "std":
                print("  " + name.ljust(14) + "".join(f"{counts[t]:>5d}/20" for t in thresholds))

    with (OUT / "results_tolerance.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tol_rows[0].keys()))
        w.writeheader()
        w.writerows(tol_rows)

    print(f"\n写入 {(OUT / 'results.csv').relative_to(ROOT)}、results_summary.csv、"
          f"results_tolerance.csv")


if __name__ == "__main__":
    main()
