"""PickCube 失败诊断：逐帧轨迹 + 延长回合对照。

回答两个问题，且刻意把两者分开：
  1. 失败的回合是否集中在某种目标位置 / 初始距离 / 抓取模式？
  2. 把回合上限从 50 步延长到 100 步，这些回合会不会出现静止？

第 2 问必须单独做实验，因为 50 步内的 `is_robot_static_first_step = -1`
只说明「记录的 50 步里没达到静止阈值」，对第 51 步之后没有任何信息。

诊断样本默认覆盖全部 20 个评估种子（含成功回合作对照），
否则无法区分「失败的特征」与「这个任务本身的特征」。

用法（仓库根目录）：
    ./.venv-maniskill/bin/python experiments/04_ppo_pickcube/diagnose.py \
        --checkpoint experiments/04_ppo_pickcube/runs/<run>/final_ckpt.pt \
        --max-steps 100
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401

STAGE_DIR = Path(__file__).resolve().parent
REPO = STAGE_DIR.parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from common.rollout import _scalar, build_agent, bool_metrics  # noqa: E402

BASELINE_PY = REPO / "third_party/ManiSkill/examples/baselines/ppo/ppo.py"


def vec(t) -> list[float]:
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    return np.asarray(t).reshape(-1).tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description="PickCube 失败诊断")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--env-id", default="PickCube-v1")
    ap.add_argument("--control-mode", default="pd_joint_delta_pos")
    ap.add_argument("--n-episodes", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=100,
                    help="回合上限；训练是 50，这里用 100 检验延长后是否会静止")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    out_dir = Path(args.out) if args.out else ckpt.parent / f"diagnose_{args.max_steps}step"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode="state",
        render_mode=None,
        sim_backend="physx_cuda",
        control_mode=args.control_mode,
        max_episode_steps=args.max_steps,  # 覆盖注册时的 50 步
    )
    u = env.unwrapped

    obs, _ = env.reset(seed=args.seed_start)
    print(f"实际回合上限: {getattr(env.spec, 'max_episode_steps', 'n/a')}")
    agent = build_agent(ckpt, obs.shape[1:], env.action_space.low.reshape(-1),
                        env.action_space.high.reshape(-1), device, BASELINE_PY)

    # qvel_max 才是 is_static 的口径（torch.max(torch.abs(qvel)) <= 0.2），
    # qvel_norm 是 L2 范数，仅作参考，两者不可混用
    headers = [
        "seed", "step", "obj_x", "obj_y", "obj_z", "goal_x", "goal_y", "goal_z",
        "tcp_x", "tcp_y", "tcp_z", "obj_to_goal_dist", "tcp_to_obj_dist",
        "qvel_norm", "qvel_max", "is_grasped", "is_obj_placed", "is_robot_static",
        "success",
    ]
    traj_path = out_dir / "trajectory.csv"
    traj_f = traj_path.open("w", newline="", encoding="utf-8")
    traj_w = csv.writer(traj_f)
    traj_w.writerow(headers)

    episodes: list[dict] = []

    for i in range(args.n_episodes):
        seed = args.seed_start + i
        obs, _ = env.reset(seed=seed)

        obj0 = vec(u.cube.pose.p)
        goal0 = vec(u.goal_site.pose.p)
        init_dist = float(np.linalg.norm(np.array(obj0[:2]) - np.array(goal0[:2])))

        rows: list[dict] = []
        t = 0
        terminated = truncated = False
        while not (terminated or truncated) and t < args.max_steps:
            with torch.no_grad():
                action = agent.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            t += 1

            obj = vec(u.cube.pose.p)
            goal = vec(u.goal_site.pose.p)
            tcp = vec(u.agent.tcp.pose.p)
            qvel = u.agent.robot.get_qvel()
            if isinstance(qvel, torch.Tensor):
                qvel = qvel[0, :-2].cpu().numpy()  # 排除两个夹爪关节
            qvel_norm = float(np.linalg.norm(qvel))
            qvel_max = float(np.abs(qvel).max())

            m = bool_metrics(info)
            rec = {
                "seed": seed, "step": t,
                "obj_x": obj[0], "obj_y": obj[1], "obj_z": obj[2],
                "goal_x": goal[0], "goal_y": goal[1], "goal_z": goal[2],
                "tcp_x": tcp[0], "tcp_y": tcp[1], "tcp_z": tcp[2],
                "obj_to_goal_dist": float(np.linalg.norm(
                    np.array(obj) - np.array(goal))),
                "tcp_to_obj_dist": float(np.linalg.norm(
                    np.array(tcp) - np.array(obj))),
                "qvel_norm": qvel_norm,
                "qvel_max": qvel_max,
                "is_grasped": int(m.get("is_grasped", 0)),
                "is_obj_placed": int(m.get("is_obj_placed", 0)),
                "is_robot_static": int(m.get("is_robot_static", 0)),
                "success": int(m.get("success", 0)),
            }
            rows.append(rec)
            traj_w.writerow([rec[k] for k in headers])

        def first_step(key: str, lo: int = 1, hi: int | None = None) -> int:
            hi = hi or args.max_steps
            for r in rows:
                if lo <= r["step"] <= hi and r[key]:
                    return r["step"]
            return -1

        grasp_switches = sum(
            1 for a, b in zip(rows, rows[1:]) if a["is_grasped"] != b["is_grasped"])

        ep = {
            "seed": seed,
            "goal_z": round(float(goal0[2]), 4),
            "init_obj_goal_dist": round(init_dist, 4),
            "steps_run": len(rows),
            "first_grasped": first_step("is_grasped"),
            "first_placed": first_step("is_obj_placed"),
            "first_static": first_step("is_robot_static"),
            "first_success": first_step("success"),
            "static_in_1_50": int(first_step("is_robot_static", 1, 50) > 0),
            "static_in_51_100": int(first_step("is_robot_static", 51, 100) > 0),
            "placed_in_1_50": int(first_step("is_obj_placed", 1, 50) > 0),
            "min_obj_to_goal_dist": round(
                min(r["obj_to_goal_dist"] for r in rows), 4),
            "min_tcp_to_obj_dist": round(
                min(r["tcp_to_obj_dist"] for r in rows), 4),
            "min_qvel_max_after_placed": round(
                min([r["qvel_max"] for r in rows if r["is_obj_placed"]] or [-1.0]), 4),
            "grasp_switches": grasp_switches,
            "success_ever": int(any(r["success"] for r in rows)),
        }
        episodes.append(ep)
        print(f"seed={seed} goal_z={ep['goal_z']:.3f} init_d={ep['init_obj_goal_dist']:.3f} "
              f"placed@{ep['first_placed']:>3} static@{ep['first_static']:>3} "
              f"succ={ep['success_ever']} minD={ep['min_obj_to_goal_dist']:.3f} "
              f"grasp_sw={grasp_switches}")

    traj_f.close()
    env.close()

    with (out_dir / "episode_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(episodes[0].keys()))
        w.writeheader()
        w.writerows(episodes)

    def mean_of(key: str, subset: list[dict]) -> float | None:
        vals = [e[key] for e in subset if e[key] is not None]
        return round(statistics.fmean(vals), 4) if vals else None

    ok = [e for e in episodes if e["success_ever"]]
    failed = [e for e in episodes if not e["success_ever"]]
    # 用整个 max_steps 窗口的首次到位来分组，而不是只看前 50 步，
    # 否则「51 步才到位」的回合会被误标成「从未到位」
    placed_fail = [e for e in failed if e["first_placed"] > 0]
    never_placed = [e for e in failed if e["first_placed"] == -1]

    print("\n" + "=" * 78)
    print(f"成功 {len(ok)} | 到位但失败 {len(placed_fail)} | 从未到位 {len(never_placed)}")
    print("=" * 78)
    print(f"{'分组':<14}{'n':>4}{'goal_z':>10}{'init_dist':>11}"
          f"{'min_obj2goal':>14}{'min_tcp2obj':>13}{'grasp_sw':>10}")
    for name, group in (("成功", ok), ("到位但失败", placed_fail),
                        ("从未到位", never_placed)):
        if not group:
            continue
        print(f"{name:<14}{len(group):>4}"
              f"{(mean_of('goal_z', group) or 0):>10.3f}"
              f"{(mean_of('init_obj_goal_dist', group) or 0):>11.3f}"
              f"{(mean_of('min_obj_to_goal_dist', group) or 0):>14.3f}"
              f"{(mean_of('min_tcp_to_obj_dist', group) or 0):>13.3f}"
              f"{(mean_of('grasp_switches', group) or 0):>10.1f}")

    print("\n延长到 100 步后的静止情况（qvel 用 is_static 的口径 max|qvel|，阈值 0.2）：")
    for e in placed_fail:
        print(f"  seed={e['seed']} 前50步静止={e['static_in_1_50']} "
              f"51-100步静止={e['static_in_51_100']} "
              f"到位后最小max|qvel|={e['min_qvel_max_after_placed']}")
    print(f"\n产出目录: {out_dir}")


if __name__ == "__main__":
    main()
