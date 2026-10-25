"""W7：4×4 网格 MDP、值迭代与表格 Q-learning。

本模块只依赖 NumPy。环境与教材常用 FrozenLake 一致：
- 状态 0..15，起点 0，终点 15；
- 洞 5/7/11/12 与终点都是终止状态；
- 动作 0=左、1=下、2=右、3=上；
- 滑面时，意图动作和两个垂直方向各 1/3。

奖励显式定义为 r(s, a, s')，避免把“落到终点的奖励”错误地写在动作上。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from numpy.typing import NDArray

GRID_SIZE = 4
N_STATES = GRID_SIZE * GRID_SIZE
N_ACTIONS = 4
START_STATE = 0
GOAL_STATE = 15
HOLES = (5, 7, 11, 12)
TERMINAL_STATES = (*HOLES, GOAL_STATE)

ACTION_NAMES = ("left", "down", "right", "up")
ACTION_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))
ACTION_SYMBOLS = ("←", "↓", "→", "↑")

RewardMode = Literal["sparse", "hacked"]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def manhattan_to_goal(state: int) -> int:
    """返回状态到终点的 Manhattan 距离。"""
    row, col = divmod(int(state), GRID_SIZE)
    goal_row, goal_col = divmod(GOAL_STATE, GRID_SIZE)
    return abs(row - goal_row) + abs(col - goal_col)


def move(state: int, action: int) -> int:
    """执行一个动作；撞墙时停在原地。"""
    row, col = divmod(int(state), GRID_SIZE)
    dr, dc = ACTION_DELTAS[int(action)]
    next_row = min(GRID_SIZE - 1, max(0, row + dr))
    next_col = min(GRID_SIZE - 1, max(0, col + dc))
    return next_row * GRID_SIZE + next_col


def build_transition(slippery: bool = False) -> FloatArray:
    """构建 P[s, a, s']，形状 (16, 4, 16)。

    终止状态用自环表示，便于矩阵运算；训练时到达终止状态即结束，
    不会真的在自环上继续采样。
    """
    transition = np.zeros((N_STATES, N_ACTIONS, N_STATES), dtype=np.float64)
    for state in range(N_STATES):
        if state in TERMINAL_STATES:
            transition[state, :, state] = 1.0
            continue
        for action in range(N_ACTIONS):
            intended = [action]
            if slippery:
                intended.extend(((action - 1) % N_ACTIONS, (action + 1) % N_ACTIONS))
            for actual_action in intended:
                next_state = move(state, actual_action)
                # 边界处多个实际动作会落到同一个格子，概率在此自然累加。
                transition[state, action, next_state] += 1.0 / len(intended)
    return transition


def build_reward(reward_mode: RewardMode = "sparse") -> FloatArray:
    """构建 R[s, a, s']，形状 (16, 4, 16)。"""
    reward = np.zeros((N_STATES, N_ACTIONS, N_STATES), dtype=np.float64)

    if reward_mode == "sparse":
        reward[:, :, GOAL_STATE] = 1.0
    elif reward_mode == "hacked":
        current_distance = np.array([manhattan_to_goal(s) for s in range(N_STATES)])
        next_distance = current_distance
        closer = next_distance[None, None, :] < current_distance[:, None, None]
        reward = np.broadcast_to(closer, reward.shape).astype(np.float64) * 0.1
    else:
        raise ValueError(f"未知奖励模式：{reward_mode!r}")

    # 终止状态不再产生转移奖励。
    reward[list(TERMINAL_STATES), :, :] = 0.0
    return reward


def build_mdp(
    slippery: bool = False,
    reward_mode: RewardMode = "sparse",
) -> tuple[FloatArray, FloatArray]:
    return build_transition(slippery), build_reward(reward_mode)


def q_from_v(V: FloatArray, P: FloatArray, R: FloatArray, gamma: float) -> FloatArray:
    """由 V 计算一步前瞻的 Q。"""
    return np.einsum("sat,sat->sa", P, R + gamma * V[None, None, :])


def policy_from_q(Q: FloatArray) -> NDArray[np.int64]:
    """贪心策略；终止状态记为 -1。"""
    policy = Q.argmax(axis=1).astype(np.int64)
    policy[list(TERMINAL_STATES)] = -1
    return policy


@dataclass(frozen=True)
class ValueIterationResult:
    V: FloatArray
    Q: FloatArray
    policy: NDArray[np.int64]
    iterations: int
    history: list[tuple[int, float, float]]  # (iteration, delta, V[start])
    converged: bool


def value_iteration(
    P: FloatArray,
    R: FloatArray,
    gamma: float = 0.99,
    tol: float = 1e-12,
    max_iterations: int = 10_000,
) -> ValueIterationResult:
    """同步值迭代：先算完整 V_new，再整体替换。"""
    V = np.zeros(N_STATES, dtype=np.float64)
    history: list[tuple[int, float, float]] = []
    converged = False
    iterations = 0
    Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)

    for iteration in range(1, max_iterations + 1):
        Q = q_from_v(V, P, R, gamma)
        V_new = Q.max(axis=1)
        V_new[list(TERMINAL_STATES)] = 0.0
        delta = float(np.max(np.abs(V_new - V)))
        V = V_new
        history.append((iteration, delta, float(V[START_STATE])))
        iterations = iteration
        if delta < tol:
            converged = True
            break

    # 用收敛后的 V 重新算一遍 Q，保证 policy 与最终价值对应。
    Q = q_from_v(V, P, R, gamma)
    policy = policy_from_q(Q)
    return ValueIterationResult(V, Q, policy, iterations, history, converged)


def policy_evaluation(
    P: FloatArray,
    R: FloatArray,
    policy: NDArray[np.int64],
    gamma: float = 0.99,
    tol: float = 1e-12,
    max_iterations: int = 10_000,
) -> FloatArray:
    """固定策略的无限时域折扣价值 V^π。"""
    V = np.zeros(N_STATES, dtype=np.float64)
    safe_policy = np.asarray(policy, dtype=np.int64).copy()
    safe_policy[list(TERMINAL_STATES)] = 0

    for _ in range(max_iterations):
        Q = q_from_v(V, P, R, gamma)
        V_new = Q[np.arange(N_STATES), safe_policy]
        V_new[list(TERMINAL_STATES)] = 0.0
        delta = float(np.max(np.abs(V_new - V)))
        V = V_new
        if delta < tol:
            return V
    raise RuntimeError("策略评估未收敛；请检查 gamma、tol 或终止状态。")


def finite_horizon_policy_metrics(
    P: FloatArray,
    R: FloatArray,
    policy: NDArray[np.int64],
    horizon: int,
    gamma: float = 0.99,
) -> tuple[float, float]:
    """精确计算固定策略在 H 步内的折扣价值与不折扣成功率。

    这个函数用于解释“价值 != 成功率”，不参与 Q-learning 训练。
    """
    safe_policy = np.asarray(policy, dtype=np.int64).copy()
    safe_policy[list(TERMINAL_STATES)] = 0
    expected_reward = np.einsum("sat,sat->sa", P, R)

    distribution = np.zeros(N_STATES, dtype=np.float64)
    distribution[START_STATE] = 1.0
    discounted_value = 0.0
    success_probability = 0.0

    for step in range(horizon):
        reward_now = float(
            np.sum(distribution * expected_reward[np.arange(N_STATES), safe_policy])
        )
        discounted_value += (gamma**step) * reward_now

        next_distribution = distribution @ P[np.arange(N_STATES), safe_policy, :]
        success_probability += float(next_distribution[GOAL_STATE])
        next_distribution[list(TERMINAL_STATES)] = 0.0
        distribution = next_distribution

    return discounted_value, success_probability


def q_target(
    reward: float,
    next_q_max: float,
    gamma: float,
    terminated: bool,
    truncated: bool,
    bootstrap_truncated: bool = True,
) -> float:
    """TD 目标；把终止语义单独封装，便于测试和做错误对照。"""
    if bootstrap_truncated:
        should_stop = terminated
    else:
        should_stop = terminated or truncated
    return reward if should_stop else reward + gamma * next_q_max


class GridWorld:
    """最小的 Gym 风格网格环境。"""

    def __init__(
        self,
        slippery: bool = False,
        horizon: int = 100,
        reward_mode: RewardMode = "sparse",
        seed: int | None = None,
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon 必须为正整数")
        self.slippery = bool(slippery)
        self.horizon = int(horizon)
        self.reward_mode = reward_mode
        self.P, self.R = build_mdp(self.slippery, self.reward_mode)
        self.rng = np.random.default_rng(seed)
        self.state = START_STATE
        self.steps = 0
        self.terminated = False
        self.truncated = False

    def reset(self) -> int:
        self.state = START_STATE
        self.steps = 0
        self.terminated = False
        self.truncated = False
        return self.state

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, object]]:
        if self.terminated or self.truncated:
            raise RuntimeError("回合已结束，请先 reset()。")
        if action not in range(N_ACTIONS):
            raise ValueError(f"动作必须是 0..3，收到 {action!r}")

        probabilities = self.P[self.state, int(action)]
        next_state = int(self.rng.choice(N_STATES, p=probabilities))
        reward = float(self.R[self.state, int(action), next_state])
        self.state = next_state
        self.steps += 1

        self.terminated = next_state in TERMINAL_STATES
        self.truncated = (self.steps >= self.horizon) and not self.terminated
        info = {
            "success": next_state == GOAL_STATE,
            "distance": manhattan_to_goal(next_state),
        }
        return next_state, reward, self.terminated, self.truncated, info


@dataclass(frozen=True)
class QLearningResult:
    Q: FloatArray
    seed: int
    horizon: int
    epsilon_final: float
    train_success: BoolArray
    train_returns_undiscounted: FloatArray
    train_returns_discounted: FloatArray

    @property
    def q0_max(self) -> float:
        return float(self.Q[START_STATE].max())

    @property
    def train_success_last500(self) -> float:
        return float(self.train_success[-500:].mean())


def q_learning(
    slippery: bool = False,
    reward_mode: RewardMode = "sparse",
    horizon: int = 100,
    episodes: int = 6_000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    decay_fraction: float = 0.6,
    seed: int = 0,
    bootstrap_truncated: bool = True,
) -> QLearningResult:
    """表格 Q-learning；时间截断默认仍自举。"""
    if episodes <= 0:
        raise ValueError("episodes 必须为正整数")
    if not 0.0 < decay_fraction <= 1.0:
        raise ValueError("decay_fraction 必须在 (0, 1] 内")

    env = GridWorld(
        slippery=slippery,
        horizon=horizon,
        reward_mode=reward_mode,
        seed=seed,
    )
    exploration_rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
    successes = np.zeros(episodes, dtype=np.bool_)
    returns_undiscounted = np.zeros(episodes, dtype=np.float64)
    returns_discounted = np.zeros(episodes, dtype=np.float64)

    decay_episodes = max(1, int(round(decay_fraction * episodes)))
    epsilon_final = epsilon_end

    for episode in range(episodes):
        progress = min(episode / decay_episodes, 1.0)
        epsilon = epsilon_start + progress * (epsilon_end - epsilon_start)
        epsilon_final = epsilon
        state = env.reset()
        done = False
        step = 0

        while not done:
            if exploration_rng.random() < epsilon:
                action = int(exploration_rng.integers(N_ACTIONS))
            else:
                action = int(np.argmax(Q[state]))

            next_state, reward, terminated, truncated, info = env.step(action)
            target = q_target(
                reward=reward,
                next_q_max=float(Q[next_state].max()),
                gamma=gamma,
                terminated=terminated,
                truncated=truncated,
                bootstrap_truncated=bootstrap_truncated,
            )
            Q[state, action] += alpha * (target - Q[state, action])

            returns_undiscounted[episode] += reward
            returns_discounted[episode] += (gamma**step) * reward
            successes[episode] = successes[episode] or bool(info["success"])

            state = next_state
            step += 1
            done = terminated or truncated

    return QLearningResult(
        Q=Q,
        seed=seed,
        horizon=horizon,
        epsilon_final=epsilon_final,
        train_success=successes,
        train_returns_undiscounted=returns_undiscounted,
        train_returns_discounted=returns_discounted,
    )


@dataclass(frozen=True)
class EvaluationResult:
    success_rate: float
    successes: int
    episodes: int
    avg_return: float
    avg_steps: float


def evaluate_policy(
    policy: NDArray[np.int64],
    slippery: bool = False,
    horizon: int = 100,
    reward_mode: RewardMode = "sparse",
    episode_seeds: Iterable[int] = range(10_000, 10_500),
    gamma: float = 0.99,
) -> EvaluationResult:
    """用独立种子和 ε=0 评估策略；成功率只看任务，不看训练奖励。"""
    success_count = 0
    episode_count = 0
    total_return = 0.0
    total_steps = 0
    safe_policy = np.asarray(policy, dtype=np.int64)

    for episode_seed in episode_seeds:
        env = GridWorld(
            slippery=slippery,
            horizon=horizon,
            reward_mode=reward_mode,
            seed=int(episode_seed),
        )
        state = env.reset()
        done = False
        step = 0
        while not done:
            action = int(safe_policy[state])
            state, reward, terminated, truncated, info = env.step(action)
            total_return += (gamma**step) * reward
            step += 1
            done = terminated or truncated
            if info["success"]:
                success_count += 1
        total_steps += step
        episode_count += 1

    return EvaluationResult(
        success_rate=success_count / episode_count,
        successes=success_count,
        episodes=episode_count,
        avg_return=total_return / episode_count,
        avg_steps=total_steps / episode_count,
    )


def evaluate_q(
    Q: FloatArray,
    slippery: bool = False,
    horizon: int = 100,
    reward_mode: RewardMode = "sparse",
    episode_seeds: Iterable[int] = range(10_000, 10_500),
    gamma: float = 0.99,
) -> EvaluationResult:
    return evaluate_policy(
        policy=Q.argmax(axis=1).astype(np.int64),
        slippery=slippery,
        horizon=horizon,
        reward_mode=reward_mode,
        episode_seeds=episode_seeds,
        gamma=gamma,
    )
