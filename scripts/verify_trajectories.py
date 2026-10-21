"""W5 · D2：对采集到的轨迹做逐项核对。

用法（仓库根目录）：
    ./.venv/bin/python scripts/verify_trajectories.py                # 详查 seed=0
    ./.venv/bin/python scripts/verify_trajectories.py --seed 1       # 详查 seed=1
    ./.venv/bin/python scripts/verify_trajectories.py --all          # 20 条一览

6 项检查的标准与理由见 notes/W5_逐项核对记录.md。
关键点：方向 / 比例这两项只能在"末端真的动了"的步上统计。
末端没动的步（目标超出可达范围，动作一直饱和）夹角没有定义，不能算失败。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
EPS = 1e-12
DIST_THRESHOLD = 0.05
ACTION_EPS = 0.05    # ‖a[:3]‖ 小于此值：动作接近 0，方向/比例被舍入噪声主导
MOVE_EPS = 2e-3      # ‖Δachieved‖ 小于此值：末端基本没动
STRONG_ACTION = 0.3  # 方向核对只取"明确指令"的步


def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < EPS or nv < EPS:
        return float("nan")
    return float(np.degrees(np.arccos(float(np.clip(u @ v / (nu * nv), -1.0, 1.0)))))


def check_one(path: Path) -> dict:
    z = np.load(path)
    obs_before, obs_after = z["obs_before"], z["obs_after"]
    ach_before, ach_after = z["achieved_before"], z["achieved_after"]
    action, step = z["action"], z["step"]
    term, trunc, succ = z["terminated"], z["truncated"], z["is_success"]
    desired = z["desired_goal"]
    n = len(step)

    d_ach = ach_after - ach_before
    cmd = DIST_THRESHOLD * action[:, :3]
    cmd_norm = np.linalg.norm(cmd, axis=1)
    d_norm = np.linalg.norm(d_ach, axis=1)

    a_norm = np.linalg.norm(action[:, :3], axis=1)
    moving = d_norm > MOVE_EPS
    ratio_mask = (a_norm > ACTION_EPS) & moving
    dir_mask = (a_norm > STRONG_ACTION) & moving

    ratios = d_norm[ratio_mask] / cmd_norm[ratio_mask]
    angles = np.array([angle_deg(d_ach[i], action[i, :3]) for i in np.where(dir_mask)[0]])
    # 有明确指令、末端却几乎没动 —— 目标超出可达范围的特征
    stuck = int(((a_norm > STRONG_ACTION) & ~moving).sum())

    align_diff = float(np.abs(obs_before[1:] - obs_after[:-1]).max()) if n > 1 else 0.0
    end_flags = term | trunc
    n_end = int(end_flags.sum())
    end_index = int(np.where(end_flags)[0][0]) if n_end else -1

    return {
        "file": path.name,
        "steps": n,
        "align_max_diff": align_diff,
        "step0_angle": angle_deg(d_ach[0], action[0, :3]),
        "angle_median": float(np.median(angles)) if angles.size else float("nan"),
        "angle_max": float(angles.max()) if angles.size else float("nan"),
        "n_dir_steps": int(dir_mask.sum()),
        "ratio_median": float(np.median(ratios)) if ratios.size else float("nan"),
        "ratio_p05": float(np.percentile(ratios, 5)) if ratios.size else float("nan"),
        "ratio_p95": float(np.percentile(ratios, 95)) if ratios.size else float("nan"),
        "n_ratio_steps": int(ratio_mask.sum()),
        "stuck_steps": stuck,
        "n_end_flags": n_end,
        "end_index": end_index,
        "step_ok": bool(np.array_equal(step, np.arange(n))),
        "action_in_range": bool(np.abs(action).max() <= 1.0),
        "action_abs_max": float(np.abs(action).max()),
        "terminated_any": bool(term.any()),
        "truncated_any": bool(trunc.any()),
        "dtypes_ok": bool(
            all(z[k].dtype == np.float32 for k in ("obs_before", "action", "obs_after"))
            and all(z[k].dtype == np.bool_ for k in ("terminated", "truncated", "is_success"))
        ),
        "final_dist": float(np.linalg.norm(desired - ach_after[-1])),
        "start_dist": float(np.linalg.norm(desired - ach_before[0])),
        "in_tol_count": int(succ.sum()),
        "action3_max_abs": float(np.abs(action[:, 3]).max()),
    }


def build_checks(r: dict) -> list[tuple[str, str, bool]]:
    return [
        ("1  obs_before[t] == obs_after[t-1]",
         f"max diff = {r['align_max_diff']:.1e}", r["align_max_diff"] < 1e-12),
        ("2a 首步方向（指令最明确）",
         f"{r['step0_angle']:.2f}°", r["step0_angle"] < 1.0),
        ("2b 跟踪步方向中位数",
         f"{r['angle_median']:.2f}°（{r['n_dir_steps']} 步，max {r['angle_max']:.2f}°）",
         r["angle_median"] < 5.0),
        ("3  比例 r ∈ [0.4, 0.9]",
         f"median {r['ratio_median']:.3f}（p5–p95 {r['ratio_p05']:.3f}–{r['ratio_p95']:.3f}）",
         0.4 <= r["ratio_median"] <= 0.9),
        ("4  结束标记有且只在最后一步",
         f"True 数 = {r['n_end_flags']}，位置 {r['end_index']}/{r['steps'] - 1}",
         r["n_end_flags"] == 1 and r["end_index"] == r["steps"] - 1),
        ("5  step 从 0 连续到 steps-1", "OK" if r["step_ok"] else "跳号", r["step_ok"]),
        ("6  所有 action 在 [−1,1] 内",
         f"max |a| = {r['action_abs_max']:.3f}", r["action_in_range"]),
    ]


def print_report(r: dict) -> bool:
    checks = build_checks(r)
    ok = all(c[2] for c in checks)
    print(f"\n=== {r['file']} ===")
    print(f"{'检查项':<32s}{'实测':<48s}结果")
    print("-" * 92)
    for name, detail, good in checks:
        print(f"{name:<32s}{detail:<48s}{'PASS' if good else 'FAIL'}")
    print("-" * 92)
    print(f"steps={r['steps']}  start_dist={r['start_dist']:.3f} m  "
          f"final_dist={r['final_dist']:.4f} m  在容差内 {r['in_tol_count']} 步")
    print(f"terminated={r['terminated_any']}  truncated={r['truncated_any']}  "
          f"a[3] 恒为 0：{r['action3_max_abs'] == 0.0}  dtype 一致：{r['dtypes_ok']}")
    print(f"参与方向统计 {r['n_dir_steps']} 步、比例统计 {r['n_ratio_steps']} 步；"
          f"有指令但末端没动的步 = {r['stuck_steps']}")
    if r["stuck_steps"] > 10:
        print("注意：上面这个数很大，说明该回合的目标超出当前姿态的可达范围（末端被卡住），"
              "动作持续饱和而位移≈0；这不是控制律错误。")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        paths = sorted(RAW.glob("feedback_seed*.npz"))
        print(f"{'file':<22s}{'r_med':>7s}{'ang0':>7s}{'ang_med':>9s}{'final':>9s}"
              f"{'stuck':>7s}{'align':>9s}  结果")
        all_ok = True
        for path in paths:
            r = check_one(path)
            good = all(c[2] for c in build_checks(r))
            all_ok = all_ok and good
            print(f"{r['file']:<22s}{r['ratio_median']:>7.3f}{r['step0_angle']:>7.2f}"
                  f"{r['angle_median']:>9.2f}{r['final_dist']:>9.4f}{r['stuck_steps']:>7d}"
                  f"{r['align_max_diff']:>9.1e}  {'PASS' if good else 'FAIL'}")
        print(f"\n{len(paths)} 条中 {'全部 PASS' if all_ok else '存在 FAIL（见上表）'}")
        return

    print_report(check_one(RAW / f"feedback_seed{args.seed}.npz"))


if __name__ == "__main__":
    main()
