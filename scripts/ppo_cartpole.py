"""阶段 1：CartPole-v1 上的 PPO 最小闭环。

对应《具身智能实验与实践学习计划（修订版）》第 3 节，产出：
  1. 随机策略基线（先看清接口再谈算法）
  2. Stable-Baselines3 PPO 短训练，保存配置 / 日志 / checkpoint
  3. 独立种子上 ≥20 回合评估，报平均回报、标准差、回合数
  4. 训练前后的策略视频
  5. 回报曲线

训练与评估严格分离：训练种子与评估种子不重叠（第 8 节）。

用法（仓库根目录）：
    ./.venv-ppo/bin/python scripts/ppo_cartpole.py
    ./.venv-ppo/bin/python scripts/ppo_cartpole.py --timesteps 50000 --tag quick
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

# CartPole 用 rgb_array 渲染时不需要真实显示会话，但 pygame 仍会初始化视频驱动
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

OUT_ROOT = Path("outputs/ppo_cartpole")
ENV_ID = "CartPole-v1"

# CartPole-v1 的"解决"标准：连续 100 回合平均回报 >= 475（满分 500）
SOLVED_RETURN = 475.0


# ---------------------------------------------------------------- 接口说明


def describe_env() -> dict:
    """打印并返回环境的接口事实，避免"不知道自己训练的是什么" 。"""
    env = gym.make(ENV_ID)
    obs, info = env.reset(seed=0)
    facts = {
        "env_id": ENV_ID,
        "observation_space": str(env.observation_space),
        "observation_dims": int(env.observation_space.shape[0]),
        "observation_meaning": [
            "小车位置 x (m), 约 [-4.8, 4.8]",
            "小车速度 v (m/s)",
            "杆角度 theta (rad), 约 [-0.42, 0.42]",
            "杆角速度 omega (rad/s)",
        ],
        "action_space": str(env.action_space),
        "action_meaning": {
            0: "向左推 (固定力)",
            1: "向右推 (固定力)",
        },
        "reward": "每存活一步 +1，无其他奖励分量",
        "terminated": "杆倾角超阈值或小车出界 → 真实终止",
        "truncated": "达到 500 步上限 → 时间截断",
        "max_episode_steps": 500,
        "first_obs_seed0": obs.tolist(),
    }
    env.close()

    print("=" * 66)
    print(f"环境: {facts['env_id']}")
    print(f"观测: {facts['observation_space']}  动作: {facts['action_space']}")
    for i, m in enumerate(facts["observation_meaning"]):
        print(f"  obs[{i}] {m}")
    print(f"奖励: {facts['reward']}")
    print(f"终止: {facts['terminated']}")
    print(f"截断: {facts['truncated']}")
    print("=" * 66)
    return facts


# ---------------------------------------------------------------- 评估


def rollout(model, n_episodes: int, seed_start: int, record: bool = False,
            record_path: Path | None = None) -> list[dict]:
    """在独立环境、独立种子上评估。返回逐回合记录。"""
    render_mode = "rgb_array" if record else None
    env = gym.make(ENV_ID, render_mode=render_mode)
    rows: list[dict] = []
    frames: list[np.ndarray] = []

    for i in range(n_episodes):
        seed = seed_start + i
        obs, _ = env.reset(seed=seed)
        total, steps, terminated, truncated = 0.0, 0, False, False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total += float(reward)
            steps += 1
            if record and i == 0:
                frames.append(env.render())
        rows.append({
            "seed": seed,
            "return": total,
            "steps": steps,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        })
    env.close()

    if record and frames and record_path is not None:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(record_path, frames, fps=30, macro_block_size=1)

    return rows


def summarize(rows: list[dict]) -> dict:
    ret = np.array([r["return"] for r in rows], dtype=float)
    return {
        "episodes": len(rows),
        "mean_return": float(ret.mean()),
        "std_return": float(ret.std(ddof=1)) if len(ret) > 1 else 0.0,
        "min_return": float(ret.min()),
        "max_return": float(ret.max()),
        "solved": int((ret >= SOLVED_RETURN).sum()),
        "truncated": int(sum(r["truncated"] for r in rows)),
    }


# ---------------------------------------------------------------- 训练


class EpisodeLogger(BaseCallback):
    """把每回合回报写进 CSV，用于画曲线（不依赖 tensorboard）。"""

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.rows: list[tuple[int, float, int]] = []
        self._cur_return = 0.0
        self._cur_len = 0

    def _on_step(self) -> bool:
        self._cur_return += float(self.locals["rewards"][0])
        self._cur_len += 1
        if self.locals["dones"][0]:
            self.rows.append((self.num_timesteps, self._cur_return, self._cur_len))
            self._cur_return = 0.0
            self._cur_len = 0
        return True

    def _on_training_end(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timesteps", "episode_return", "episode_length"])
            w.writerows(self.rows)


def train(timesteps: int, seed: int, out: Path, log_csv: Path, device: str = "cpu"):
    env = make_vec_env(ENV_ID, n_envs=4, seed=seed)
    # 默认 CPU：MlpPolicy 仅约 4.6k 参数、n_envs=4，GPU 的 kernel 启动开销
    # 远大于计算本身。用 --device cuda 可强制走 GPU 做计时对照。
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        device=device,
        n_steps=128,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        learning_rate=3e-4,
        verbose=0,
    )
    logger = EpisodeLogger(log_csv)
    t0 = time.perf_counter()
    model.learn(total_timesteps=timesteps, callback=logger, progress_bar=False)
    train_sec = time.perf_counter() - t0
    ckpt = out / "model.zip"
    model.save(ckpt)
    env.close()
    print(f"训练完成：{timesteps} 步 / {train_sec:.1f} 秒 "
          f"({timesteps / train_sec:.0f} 步/秒, device={device})")
    return model, logger, train_sec


# ---------------------------------------------------------------- 绘图


def plot_curve(log_csv: Path, out_png: Path, window: int = 20) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 系统里装了 Noto Sans CJK，必须显式指定，否则中文全变方框
    matplotlib.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    xs, ys = [], []
    with log_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            xs.append(int(row["timesteps"]))
            ys.append(float(row["episode_return"]))
    if not ys:
        return

    ys_arr = np.array(ys)
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=130)
    ax.plot(xs, ys_arr, color="#9ecae1", lw=0.9, label="每回合回报")
    if len(ys_arr) >= window:
        # 滑动平均用有效样本数，避免两端低估
        kernel = np.ones(window) / window
        smooth = np.convolve(ys_arr, kernel, mode="valid")
        ax.plot(xs[window - 1:], smooth, color="#08519c", lw=1.8,
                label=f"滑动平均 (窗口 {window})")
    ax.axhline(SOLVED_RETURN, color="#cb181d", ls="--", lw=1.2,
               label=f"CartPole-v1 解决线 {SOLVED_RETURN:.0f}")
    ax.set_xlabel("训练步数 (timesteps)")
    ax.set_ylabel("回合回报 (存活步数)")
    ax.set_title("CartPole-v1 PPO 学习曲线")
    ax.set_ylim(0, 520)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# ---------------------------------------------------------------- 主流程


def main() -> None:
    ap = argparse.ArgumentParser(description="阶段 1：CartPole-v1 PPO 最小闭环")
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0, help="训练种子")
    ap.add_argument("--n-eval", type=int, default=20, help="评估回合数")
    ap.add_argument("--eval-seed-start", type=int, default=100,
                    help="评估种子起点，必须与训练种子不重叠")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                    help="PPO 训练设备；MLP 上 GPU 通常更慢，仅供对照")
    ap.add_argument("--tag", type=str, default="", help="非默认参数时的输出子目录名")
    args = ap.parse_args()

    assert args.eval_seed_start != args.seed, "训练种子与评估种子必须分离"

    out = OUT_ROOT / args.tag if args.tag else OUT_ROOT
    out.mkdir(parents=True, exist_ok=True)

    facts = describe_env()

    # ---- 1. 随机策略基线
    class RandomPolicy:
        def __init__(self, n_actions: int, seed: int):
            self.rng = np.random.default_rng(seed)
            self.n = n_actions

        def predict(self, obs, deterministic=True):
            return self.rng.integers(self.n), None

    random_rows = rollout(RandomPolicy(2, seed=args.seed),
                          args.n_eval, args.eval_seed_start,
                          record=True, record_path=out / "video_before_random.mp4")
    random_sum = summarize(random_rows)

    # ---- 2. 训练
    model, logger, train_sec = train(args.timesteps, args.seed, out,
                                     out / "train_log.csv", device=args.device)

    # ---- 3. 独立评估
    ppo_rows = rollout(model, args.n_eval, args.eval_seed_start,
                       record=True, record_path=out / "video_after_ppo.mp4")
    ppo_sum = summarize(ppo_rows)

    # ---- 3b. 阶段门槛：从保存的 checkpoint 重新加载后仍能评估出同样结果
    reloaded = PPO.load(out / "model.zip", device=args.device)
    reload_rows = rollout(reloaded, args.n_eval, args.eval_seed_start)
    reload_sum = summarize(reload_rows)
    same = all(abs(a["return"] - b["return"]) < 1e-9
               for a, b in zip(reload_rows, ppo_rows))
    if not same:
        print("[警告] 重载后的评估结果与内存模型不完全一致，检查 save/load 路径")

    # ---- 4. 落盘
    with (out / "eval_random.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "return", "steps", "terminated", "truncated"])
        w.writeheader()
        w.writerows(random_rows)
    with (out / "eval_ppo.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "return", "steps", "terminated", "truncated"])
        w.writeheader()
        w.writerows(ppo_rows)

    config = {
        "env_id": ENV_ID,
        "timesteps": args.timesteps,
        "train_seed": args.seed,
        "eval_seeds": [args.eval_seed_start, args.eval_seed_start + args.n_eval - 1],
        "n_eval_episodes": args.n_eval,
        "n_envs": 4,
        "device": args.device,
        "train_seconds": round(train_sec, 1),
        "steps_per_second": round(args.timesteps / train_sec, 1),
        "policy": "MlpPolicy",
        "n_steps": 128,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "learning_rate": 3e-4,
        "torch": __import__("torch").__version__,
        "sb3": __import__("stable_baselines3").__version__,
        "gymnasium": gym.__version__,
        "numpy": np.__version__,
    }
    (out / "config.json").write_text(
        json.dumps({"env_facts": facts, "train": config}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    plot_curve(out / "train_log.csv", out / "learning_curve.png")

    result = {"random": random_sum, "ppo": ppo_sum, "ppo_reloaded": reload_sum}
    (out / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 66)
    print(f"{'策略':<14}{'平均回报':>10}{'标准差':>10}{'最小':>8}{'最大':>8}{'达标':>8}")
    for name, s in (("随机", random_sum), ("PPO", ppo_sum), ("PPO重载", reload_sum)):
        print(f"{name:<14}{s['mean_return']:>10.1f}{s['std_return']:>10.1f}"
              f"{s['min_return']:>8.0f}{s['max_return']:>8.0f}"
              f"{s['solved']:>5}/{s['episodes']:<2}")
    print("=" * 66)
    print(f"输出目录: {out}")
    print(f"回合数: {args.n_eval}   训练种子: {args.seed}   "
          f"评估种子: {args.eval_seed_start}-{args.eval_seed_start + args.n_eval - 1}")


if __name__ == "__main__":
    main()
