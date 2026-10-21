"""W5 公共模块：环境构造、三种策略、统一的回合函数。

D3（三策略基线）和 D4（轨迹采集）都复用这里的 run_episode / collect_episode，
保证两处的"成功"定义、回合终止条件完全一致。
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import gymnasium as gym
import gymnasium_robotics
import numpy as np

ENV_ID = "FetchReach-v4"
N_EPISODES = 20
SEEDS = list(range(N_EPISODES))

Policy = Callable[[Dict[str, np.ndarray], Dict], np.ndarray]


def make_env(render_mode: Optional[str] = None, max_episode_steps: int = 50):
    gym.register_envs(gymnasium_robotics)  # gymnasium>=1.0 必须显式注册
    return gym.make(ENV_ID, render_mode=render_mode, max_episode_steps=max_episode_steps)


# ---------------------------------------------------------------- policies
def policy_zero(obs, info):
    """下界：什么都不做。"""
    return np.zeros(4, dtype=np.float32)


def make_policy_random(seed: int) -> Policy:
    """稻草人对照：每回合独立的随机流，seed 由回合号决定，保证可复现。"""
    rng = np.random.default_rng(1000 + seed)

    def policy(obs, info):
        return rng.uniform(-1.0, 1.0, 4).astype(np.float32)

    return policy


def make_policy_feedback(k: float, action_dim: int = 4) -> Policy:
    """位置误差反馈：a[:3] = clip(k·(desired − achieved), −1, 1)，a[3] = 0。

    符号必须是 desired − achieved。
    """

    def policy(obs, info):
        error = obs["desired_goal"] - obs["achieved_goal"]
        a = np.zeros(action_dim, dtype=np.float32)
        a[:3] = np.clip(k * error, -1.0, 1.0)
        return a

    return policy


# ---------------------------------------------------------------- rollouts
def run_episode(env, policy: Policy, seed: int, return_trace: bool = False) -> dict:
    """跑一个回合，按两种口径同时记成功。

    first_success_step: 0 = reset 时就已经在容差内；None = 整回合没进过容差。
    success_end: 环境 `info["is_success"]` 的口径（末步是否在容差内）。
    success_ever: 回合内是否曾经进入过容差（min_dist < threshold）。
    """
    thr = float(env.unwrapped.distance_threshold)
    obs, info = env.reset(seed=seed)

    d = float(np.linalg.norm(obs["desired_goal"] - obs["achieved_goal"]))
    first_success_step = 0 if d < thr else None
    in_tol = int(d < thr)
    min_dist = d
    trace = [d]
    steps = 0

    while True:
        obs, _r, term, trunc, info = env.step(policy(obs, info))
        steps += 1
        d = float(np.linalg.norm(obs["desired_goal"] - obs["achieved_goal"]))
        trace.append(d)
        min_dist = min(min_dist, d)
        in_tol += int(d < thr)
        if first_success_step is None and d < thr:
            first_success_step = steps
        if term or trunc:
            break

    rec = {
        "seed": seed,
        "steps": steps,
        "final_dist": d,
        "min_dist": min_dist,
        "first_success_step": first_success_step,
        "frac_in_tol": in_tol / steps,
        "success_end": bool(info["is_success"]),
        "success_ever": first_success_step is not None,
        "terminated": bool(term),
        "truncated": bool(trunc),
    }
    if return_trace:
        rec["dist_trace"] = np.asarray(trace, dtype=np.float64)
    return rec


def collect_episode(env, policy: Policy, seed: int) -> dict:
    """采集一个回合的逐步数据（D4 用），字段见 notes/W5_逐项核对记录.md。"""
    obs, info = env.reset(seed=seed)
    desired = obs["desired_goal"].copy()

    buf = {
        "obs_before": [],
        "achieved_before": [],
        "action": [],
        "obs_after": [],
        "achieved_after": [],
        "step": [],
        "terminated": [],
        "truncated": [],
        "is_success": [],
    }

    t = 0
    while True:
        a = policy(obs, info)
        before = {key: np.asarray(v).copy() for key, v in obs.items()}
        obs, _r, term, trunc, info = env.step(a)
        buf["obs_before"].append(before["observation"])
        buf["achieved_before"].append(before["achieved_goal"])
        buf["action"].append(np.asarray(a))
        buf["obs_after"].append(obs["observation"])
        buf["achieved_after"].append(obs["achieved_goal"])
        buf["step"].append(t)
        buf["terminated"].append(term)
        buf["truncated"].append(trunc)
        buf["is_success"].append(bool(info["is_success"]))
        t += 1
        if term or trunc:
            break

    return {
        "desired_goal": desired.astype(np.float32),
        "episode_id": np.int32(seed),
        **{
            key: np.asarray(buf[key], dtype=np.float32)
            for key in ("obs_before", "achieved_before", "action", "obs_after", "achieved_after")
        },
        "step": np.asarray(buf["step"], dtype=np.int32),
        "terminated": np.asarray(buf["terminated"], dtype=np.bool_),
        "truncated": np.asarray(buf["truncated"], dtype=np.bool_),
        "is_success": np.asarray(buf["is_success"], dtype=np.bool_),
    }


def summarize(records: list[dict]) -> dict:
    """一组回合的汇总：两个成功率口径 + 首次达标步数 + 最终/最小距离。"""
    n = len(records)
    first = [r["first_success_step"] for r in records if r["first_success_step"]]
    return {
        "n": n,
        "success_end": sum(r["success_end"] for r in records),
        "success_ever": sum(r["success_ever"] for r in records),
        "first_success_mean": float(np.mean(first)) if first else float("nan"),
        "first_success_median": float(np.median(first)) if first else float("nan"),
        "final_dist_mean": float(np.mean([r["final_dist"] for r in records])),
        "min_dist_mean": float(np.mean([r["min_dist"] for r in records])),
    }
