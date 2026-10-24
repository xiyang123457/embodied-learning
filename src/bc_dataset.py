"""W6 · BC 数据集：把 W5 的 npz 轨迹整理成 (输入 8 维, 标签 3 维)。

输入定义（8 维）：
    0–2  desired_goal − achieved_goal     相对位置（最重要）
    3–5  observation[5:8]                 末端线速度 x/y/z
    6–7  observation[3:5]                 两指关节位移（夹爪开合）

输出（3 维）：action[:3]，抓手通道固定 0，不预测。

注意索引：gymnasium-robotics 的 observation 是
    0–2 末端位置 / 3–4 手指关节 / 5–7 末端线速度 / 8–9 手指速度
W6 计划书里写的 "observation[3:6] = 末端速度、[6:8] = 夹爪开合" 是错的，
以 W5 接口核对表打印出来的 10 维含义为准。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

REL_POS = slice(0, 3)
VEL = slice(5, 8)
FINGERS = slice(3, 5)

TRAIN_SEEDS = list(range(0, 15))          # 15 条
VAL_SEEDS = list(range(15, 20))           # 5 条
EXTRA_SEEDS = list(range(20, 40))         # 追加 20 条，用于 20 vs 40 规模对照
TEST_SEEDS = list(range(100, 120))        # 闭环测试，全程不参与训练/调参


@dataclass
class Episode:
    seed: int
    obs_before: np.ndarray      # (T,10)
    achieved_before: np.ndarray  # (T,3)
    desired_goal: np.ndarray    # (3,)
    action: np.ndarray          # (T,4)


def load_episode(seed: int, raw_dir: Path = RAW_DIR) -> Episode:
    path = raw_dir / f"feedback_seed{seed}.npz"
    if not path.exists():
        raise FileNotFoundError(f"缺少轨迹 {path}，先跑 scripts/collect_trajectories.py")
    d = np.load(path)
    return Episode(
        seed=seed,
        obs_before=d["obs_before"].astype(np.float64),
        achieved_before=d["achieved_before"].astype(np.float64),
        desired_goal=d["desired_goal"].astype(np.float64),
        action=d["action"].astype(np.float64),
    )


def load_episodes(seeds: Iterable[int], raw_dir: Path = RAW_DIR) -> list[Episode]:
    return [load_episode(s, raw_dir) for s in seeds]


def obs_to_input(obs: np.ndarray, achieved: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """单步观测 → 8 维输入。obs 可以是 (10,) 或 (T,10)。"""
    obs = np.atleast_2d(obs)
    achieved = np.atleast_2d(achieved)
    rel = desired[None, :] - achieved
    return np.concatenate([rel, obs[:, VEL], obs[:, FINGERS]], axis=1)


def episodes_to_arrays(episodes: Sequence[Episode]) -> dict:
    """把若干回合拼成训练数组，同时保留每步的距离（用于加权与统计）。"""
    xs, ys, dists, ep_ids = [], [], [], []
    for ep in episodes:
        xs.append(obs_to_input(ep.obs_before, ep.achieved_before, ep.desired_goal))
        ys.append(ep.action[:, :3])
        dists.append(np.linalg.norm(ep.desired_goal[None, :] - ep.achieved_before, axis=1))
        ep_ids.append(np.full(len(ep.action), ep.seed))
    return {
        "x": np.concatenate(xs).astype(np.float32),
        "y": np.concatenate(ys).astype(np.float32),
        "dist": np.concatenate(dists).astype(np.float32),
        "episode_id": np.concatenate(ep_ids).astype(np.int32),
    }


class Normalizer:
    """输入标准化。统计量只从训练集算，随模型一起保存。"""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    @classmethod
    def fit(cls, x: np.ndarray) -> "Normalizer":
        return cls(x.mean(0), x.std(0) + 1e-8)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {"x_mean": self.mean, "x_std": self.std}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(np.asarray(d["x_mean"]), np.asarray(d["x_std"]))


def sample_weights(dist: np.ndarray, thr: float = 0.05,
                   w_far: float = 1.0, w_near: float = 0.1) -> np.ndarray:
    """静止样本降权：已经在容差内、动作≈0 的步不要主导损失。"""
    return np.where(dist > thr, w_far, w_near).astype(np.float32)


def dataset_stats(arrays: dict) -> dict:
    dist = arrays["dist"]
    amax = np.abs(arrays["y"]).max(1)
    return {
        "n_samples": int(len(dist)),
        "n_episodes": int(len(np.unique(arrays["episode_id"]))),
        "frac_static_a": float((amax < 0.01).mean()),
        "frac_in_tol": float((dist < 0.05).mean()),
        "n_far": int((dist > 0.05).sum()),
        "dist_median": float(np.median(dist)),
    }
