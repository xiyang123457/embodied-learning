"""D1：打印 FetchReach 的真实接口，结果用于填 notes/W5_接口核对表.md。

用法（仓库根目录）：
    ./.venv/bin/python scripts/check_env.py            # 文本
    ./.venv/bin/python scripts/check_env.py --json     # JSON

只打印事实，不做任何断言式结论——预期的"0.6 跟踪比"这类数必须由本脚本
在本机打出来，而不是抄别人的。
"""

from __future__ import annotations

import argparse
import json

import gymnasium as gym
import gymnasium_robotics
import numpy as np

ENV_ID = "FetchReach-v4"  # 写死，不留回退逻辑
MAX_STEPS = 50


def build_env(seed: int = 0):
    gym.register_envs(gymnasium_robotics)  # gymnasium>=1.0 必须显式注册
    env = gym.make(ENV_ID)
    obs, info = env.reset(seed=seed)
    return env, obs, info


def axis_run(env, axis: int, sign: float, magnitude: float, n_steps: int, seed: int):
    """单轴常量动作 n 步，返回每步 achieved_goal 相对起点的位移 (n_steps, 3)。"""
    obs, _ = env.reset(seed=seed)
    p0 = obs["achieved_goal"].copy()
    deltas = []
    for _ in range(n_steps):
        a = np.zeros(4, dtype=np.float32)
        a[axis] = sign * magnitude
        obs, _r, term, trunc, _info = env.step(a)
        deltas.append(obs["achieved_goal"] - p0)
        if term or trunc:
            break
    return np.asarray(deltas)


def interface_facts() -> dict:
    env, obs, info = build_env(seed=0)

    facts = {
        "env_id": ENV_ID,
        "obs_keys": sorted(obs.keys()),
        "obs_shape": {k: list(np.shape(v)) for k, v in obs.items()},
        "obs_dtype": {k: str(np.asarray(v).dtype) for k, v in obs.items()},
        "action_space": str(env.action_space),
        "action_low": [float(v) for v in env.action_space.low],
        "action_high": [float(v) for v in env.action_space.high],
        "max_episode_steps": env.spec.max_episode_steps,
        "distance_threshold": float(env.unwrapped.distance_threshold),
        "reward_type": env.unwrapped.reward_type,
        "info_keys_at_reset": sorted(info.keys()),
        "achieved_eq_obs_first3": bool(np.allclose(obs["achieved_goal"], obs["observation"][:3])),
        "zero_action_legal": bool(env.action_space.contains(np.zeros(4, dtype=np.float32))),
    }

    # info 的键在 reset 时为空，step 之后才有 is_success，必须分开记
    a = np.zeros(4, dtype=np.float32)
    _obs, _r, _term, _trunc, step_info = env.step(a)
    facts["info_keys_at_step"] = sorted(step_info.keys())

    # 抓手通道是否真的无效：同一组动作，只把 action[3] 取 +1/-1，比较整条轨迹
    def run_with_gripper(g: float) -> np.ndarray:
        o, _ = env.reset(seed=0)
        traj = []
        for _ in range(10):
            aa = np.zeros(4, dtype=np.float32)
            aa[0] = 0.3
            aa[3] = g
            o, _r, _t, _tr, _i = env.step(aa)
            traj.append(np.concatenate([o["achieved_goal"], o["observation"][3:5]]))
        return np.asarray(traj)

    traj_pos = run_with_gripper(1.0)
    traj_neg = run_with_gripper(-1.0)
    facts["gripper_channel_max_traj_diff"] = float(np.abs(traj_pos - traj_neg).max())
    facts["gripper_finger_state_max_abs"] = float(np.abs(traj_pos[:, 3:]).max())

    # 零动作时的残余漂移：位置不再被命令时，机械臂还会往 mocap 收敛一点点
    obs, _ = env.reset(seed=0)
    p0 = obs["achieved_goal"].copy()
    for _ in range(5):
        a = np.zeros(4, dtype=np.float32)
        obs, _r, _term, _trunc, _info = env.step(a)
    facts["zero_action_5step_drift"] = [round(float(v), 6) for v in (obs["achieved_goal"] - p0)]

    # ---- 跟踪比 r：单轴 a=0.5 连续 10 步 ----
    obs, _ = env.reset(seed=0)
    deltas = []
    for _ in range(10):
        a = np.zeros(4, dtype=np.float32)
        a[0] = 0.5
        prev = obs["achieved_goal"].copy()
        obs, _r, term, trunc, _info = env.step(a)
        deltas.append(float(obs["achieved_goal"][0] - prev[0]))
        if term or trunc:
            break

    cmd = 0.05 * 0.5
    facts["delta_x_per_step"] = [round(v, 4) for v in deltas]
    facts["cmd_x_per_step"] = cmd
    facts["tracking_ratio_per_step"] = [round(v / cmd, 3) for v in deltas]
    facts["tracking_ratio_first"] = round(deltas[0] / cmd, 3)
    facts["tracking_ratio_steady"] = round(float(np.median(deltas[1:])) / cmd, 3)

    # ---- r 随动作幅值怎么变：每个幅值单独 reset，取第 2–9 步中位比例 ----
    r_curve = {}
    for mag in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        d = axis_run(env, axis=0, sign=1.0, magnitude=mag, n_steps=10, seed=0)
        per = np.diff(np.concatenate([[0.0], d[:, 0]]))
        r_curve[f"{mag:.1f}"] = round(float(np.median(per[2:]) / (0.05 * mag)), 3)
    facts["tracking_ratio_by_magnitude"] = r_curve

    # ---- 饱和行为：a=1.0 连走 10 步，看每步位移是否为常数 ----
    sat = axis_run(env, axis=0, sign=1.0, magnitude=1.0, n_steps=10, seed=0)
    per_step = np.diff(np.concatenate([[0.0], sat[:, 0]]))
    facts["saturated_delta_per_step"] = [round(float(v), 4) for v in per_step]
    facts["saturated_delta_median"] = round(float(np.median(per_step[1:])), 4)
    facts["saturated_delta_std"] = round(float(np.std(per_step[1:])), 5)
    facts["saturated_ratio"] = round(float(np.median(per_step[1:]) / 0.05), 3)

    # ---- 单轴五步总位移（B2 用） ----
    for axis, sign, name in [(0, +1, "+x"), (0, -1, "-x"), (1, +1, "+y"), (2, +1, "+z")]:
        d = axis_run(env, axis=axis, sign=sign, magnitude=0.5, n_steps=5, seed=0)
        facts[f"axis5_{name}_total_delta"] = [round(float(v), 4) for v in d[-1]]
        facts[f"axis5_{name}_per_step_last"] = [round(float(v), 4) for v in (d[-1] - d[-2])]

    # ---- 一回合随机动作：回合怎么结束、is_success 是什么 ----
    obs, info = env.reset(seed=1)
    steps = 0
    while True:
        obs, _r, term, trunc, info = env.step(env.action_space.sample())
        steps += 1
        if term or trunc:
            break
    facts["random_episode"] = {
        "steps": steps,
        "terminated": bool(term),
        "truncated": bool(trunc),
        "is_success_at_end": float(info["is_success"]),
        "final_dist": round(float(np.linalg.norm(obs["desired_goal"] - obs["achieved_goal"])), 4),
    }

    env.close()
    return facts


def print_pretty(facts: dict) -> None:
    print(f"env_id                 : {facts['env_id']}")
    print(f"obs_keys               : {facts['obs_keys']}")
    print(f"obs_shape              : {facts['obs_shape']}")
    print(f"action_space           : {facts['action_space']}")
    print(f"max_episode_steps      : {facts['max_episode_steps']}")
    print(f"distance_threshold     : {facts['distance_threshold']}")
    print(f"reward_type            : {facts['reward_type']}")
    print(f"info_keys (reset/step) : {facts['info_keys_at_reset']} / {facts['info_keys_at_step']}")
    print(f"achieved_eq_obs_first3 : {facts['achieved_eq_obs_first3']}")
    print(f"zero_action_legal      : {facts['zero_action_legal']}")
    print(f"gripper traj diff      : {facts['gripper_channel_max_traj_diff']}")
    print(f"gripper finger state   : {facts['gripper_finger_state_max_abs']}")
    print(f"zero_action_5step_drift: {facts['zero_action_5step_drift']}")
    print(f"tracking_ratio first   : {facts['tracking_ratio_first']}")
    print(f"tracking_ratio steady  : {facts['tracking_ratio_steady']}")
    print(f"r by |a| (0.1..1.0)     : {facts['tracking_ratio_by_magnitude']}")
    print(f"delta_x_per_step       : {facts['delta_x_per_step']}")
    print(f"saturated per step     : {facts['saturated_delta_per_step']}")
    print(f"saturated median/std   : {facts['saturated_delta_median']} / {facts['saturated_delta_std']}")
    for name in ("+x", "-x", "+y", "+z"):
        print(f"axis5 {name:>2s} total      : {facts[f'axis5_{name}_total_delta']}")
    print(f"random_episode         : {facts['random_episode']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()

    facts = interface_facts()
    if args.json:
        print(json.dumps(facts, indent=2, ensure_ascii=False))
    else:
        print_pretty(facts)


if __name__ == "__main__":
    main()
