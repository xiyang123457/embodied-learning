"""W7：网格 MDP、值迭代、Q-learning 与终止语义的最小测试。"""

from __future__ import annotations

import numpy as np
import pytest

from src.mdp_gridworld import (
    GOAL_STATE,
    N_STATES,
    TERMINAL_STATES,
    GridWorld,
    build_mdp,
    build_reward,
    build_transition,
    evaluate_policy,
    finite_horizon_policy_metrics,
    move,
    q_target,
    value_iteration,
)


def test_move_clips_at_walls() -> None:
    assert move(0, 0) == 0  # 左上角向左
    assert move(0, 3) == 0  # 左上角向上
    assert move(3, 2) == 3  # 右上角向右
    assert move(12, 1) == 12  # 左下角向下
    assert move(0, 2) == 1
    assert move(0, 1) == 4


def test_transition_rows_are_distributions() -> None:
    for slippery in (False, True):
        P = build_transition(slippery)
        assert P.shape == (N_STATES, 4, N_STATES)
        assert np.allclose(P.sum(axis=2), 1.0)
        assert np.all(P >= 0.0)


def test_non_slippery_transition() -> None:
    P = build_transition(slippery=False)
    assert P[0, 2, 1] == 1.0  # 从 0 向右
    assert P[14, 2, GOAL_STATE] == 1.0  # 从中右向右进终点


def test_slippery_transition_and_duplicate_probabilities() -> None:
    P = build_transition(slippery=True)
    # 从 0 向右：意图到 1，向上撞墙留在 0，向下到 4。
    assert np.allclose(P[0, 2, [0, 1, 4]], [1 / 3, 1 / 3, 1 / 3])
    # 从 0 向左：意图撞墙留在 0，向上也留在 0，向下到 4 -> 0 的概率累加为 2/3。
    assert P[0, 0, 0] == pytest.approx(2 / 3)
    assert P[0, 0, 4] == pytest.approx(1 / 3)


def test_terminal_states_are_absorbing_and_have_zero_reward() -> None:
    P = build_transition(slippery=True)
    R = build_reward("sparse")
    for state in TERMINAL_STATES:
        assert np.allclose(P[state], np.eye(N_STATES)[state])
        assert np.allclose(R[state], 0.0)


def test_sparse_and_hacked_rewards() -> None:
    R = build_reward("sparse")
    assert R[14, 2, GOAL_STATE] == 1.0
    assert R[0, 2, 1] == 0.0

    Rh = build_reward("hacked")
    assert Rh[0, 2, 1] == pytest.approx(0.1)  # 从距离 6 到 5
    assert Rh[1, 0, 0] == 0.0  # 从距离 5 到 6，没有奖励
    assert np.allclose(Rh[list(TERMINAL_STATES)], 0.0)


def test_value_iteration_matches_reference_values() -> None:
    deterministic = value_iteration(*build_mdp(False))
    slippery = value_iteration(*build_mdp(True))

    assert deterministic.iterations == 7
    assert deterministic.V[0] == pytest.approx(0.9509900499, abs=1e-10)
    assert deterministic.converged

    assert slippery.iterations == 704
    assert slippery.V[0] == pytest.approx(0.5420259320, abs=1e-10)
    assert slippery.converged


def test_deterministic_policy_is_six_step_shortest_path() -> None:
    P, R = build_mdp(False)
    result = value_iteration(P, R)
    path = []
    state = 0
    for _ in range(6):
        action = int(result.policy[state])
        path.append(action)
        state = move(state, action)
    assert path == [1, 1, 2, 1, 2, 2]
    assert state == GOAL_STATE


def test_value_is_not_success_probability() -> None:
    P, R = build_mdp(False)
    result = value_iteration(P, R)
    value, success = finite_horizon_policy_metrics(P, R, result.policy, horizon=6)
    assert value == pytest.approx(0.9509900499, abs=1e-10)
    assert success == pytest.approx(1.0)


def test_q_target_separates_terminated_and_truncated() -> None:
    # 时间截断时，正确实现仍保留未来价值。
    assert q_target(1.0, 5.0, 0.9, terminated=False, truncated=True) == pytest.approx(5.5)
    # 混成 done 时丢掉未来项。
    assert q_target(
        1.0, 5.0, 0.9, terminated=False, truncated=True, bootstrap_truncated=False
    ) == pytest.approx(1.0)
    # 真正终止时永远不 bootstrap。
    assert q_target(1.0, 5.0, 0.9, terminated=True, truncated=False) == pytest.approx(1.0)


def test_environment_marks_goal_hole_and_timeout() -> None:
    goal_env = GridWorld(slippery=False, horizon=10, seed=0)
    goal_env.state = 14
    _, reward, terminated, truncated, info = goal_env.step(2)
    assert reward == 1.0
    assert terminated and not truncated
    assert info["success"] is True

    hole_env = GridWorld(slippery=False, horizon=10, seed=0)
    hole_env.state = 4
    _, _, terminated, truncated, _ = hole_env.step(2)
    assert terminated and not truncated

    timeout_env = GridWorld(slippery=False, horizon=1, seed=0)
    _, _, terminated, truncated, _ = timeout_env.step(2)
    assert not terminated and truncated


def test_non_slippery_greedy_policy_reaches_goal_in_every_episode() -> None:
    P, R = build_mdp(False)
    result = value_iteration(P, R)
    evaluation = evaluate_policy(
        result.policy,
        slippery=False,
        horizon=100,
        episode_seeds=range(10_000, 10_020),
    )
    assert evaluation.successes == evaluation.episodes == 20
