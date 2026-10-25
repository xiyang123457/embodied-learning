"""W7：一条命令复现值迭代、Q-learning、终止语义和奖励设计实验。

用法（仓库根目录）：
    ./.venv/bin/python scripts/run_w7.py

产物：
    outputs/w7/summary.csv          逐实验、逐种子的原始结果
    outputs/w7/vi_convergence.csv   值迭代每轮 delta 与 V(起点)
    outputs/w7/vi_convergence.png   值迭代收敛曲线
    outputs/w7/q_learning.png       Q-learning 学习曲线与最终 Q0
    outputs/w7/policies.png         不滑/滑面的最优策略
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mdp_gridworld import (  # noqa: E402
    ACTION_DELTAS,
    GOAL_STATE,
    HOLES,
    N_STATES,
    START_STATE,
    TERMINAL_STATES,
    QLearningResult,
    ValueIterationResult,
    build_mdp,
    evaluate_q,
    q_learning,
    value_iteration,
)

OUT = ROOT / "outputs" / "w7"
SEEDS = (0, 1, 2)
TEST_SEEDS = range(10_000, 10_500)
GAMMA = 0.99
ALPHA = 0.1
TRAIN_EPISODES = 6_000
HORIZONS = {"D2": 100, "C1": 20, "C2": 50}
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


@dataclass(frozen=True)
class Scenario:
    slippery: bool
    vi: ValueIterationResult

    @property
    def environment(self) -> str:
        return "slippery" if self.slippery else "deterministic"


def run_scenarios() -> dict[bool, Scenario]:
    scenarios: dict[bool, Scenario] = {}
    for slippery in (False, True):
        P, R = build_mdp(slippery=slippery, reward_mode="sparse")
        result = value_iteration(P, R, gamma=GAMMA, tol=1e-12)
        if not result.converged:
            raise RuntimeError("值迭代未收敛")
        scenarios[slippery] = Scenario(slippery=slippery, vi=result)
        print(
            f"[VI] {scenarios[slippery].environment:<13} "
            f"iterations={result.iterations:4d}  V(0)={result.V[START_STATE]:.10f}"
        )
    return scenarios


def run_q_experiment(
    *,
    experiment: str,
    scenario: Scenario,
    horizon: int,
    seed: int,
    reward_mode: str = "sparse",
    bootstrap_truncated: bool = True,
    termination_handling: str = "correct",
) -> tuple[dict[str, object], QLearningResult]:
    result = q_learning(
        slippery=scenario.slippery,
        reward_mode=reward_mode,  # type: ignore[arg-type]
        horizon=horizon,
        episodes=TRAIN_EPISODES,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon_start=1.0,
        epsilon_end=0.05,
        decay_fraction=0.6,
        seed=seed,
        bootstrap_truncated=bootstrap_truncated,
    )
    evaluation = evaluate_q(
        result.Q,
        slippery=scenario.slippery,
        horizon=horizon,
        reward_mode=reward_mode,  # type: ignore[arg-type]
        episode_seeds=TEST_SEEDS,
        gamma=GAMMA,
    )
    vi_value = float(scenario.vi.V[START_STATE])
    row: dict[str, object] = {
        "experiment": experiment,
        "environment": scenario.environment,
        "horizon": horizon,
        "seed": seed,
        "policy": "tabular_q_learning",
        "reward_design": reward_mode,
        "termination_handling": termination_handling,
        "train_episodes": TRAIN_EPISODES,
        "train_success_last500": result.train_success_last500,
        "avg_undiscounted_return_last500": float(
            result.train_returns_undiscounted[-500:].mean()
        ),
        "avg_discounted_return_last500": float(
            result.train_returns_discounted[-500:].mean()
        ),
        "q0_max": result.q0_max,
        "vi_iterations": scenario.vi.iterations,
        "vi_v0": vi_value,
        "q0_abs_error": abs(result.q0_max - vi_value),
        "greedy_successes": evaluation.successes,
        "greedy_episodes": evaluation.episodes,
        "greedy_success": evaluation.success_rate,
        "greedy_avg_return": evaluation.avg_return,
        "greedy_avg_steps": evaluation.avg_steps,
    }
    print(
        f"[Q ] {experiment:<15} {scenario.environment:<13} H={horizon:<3} "
        f"seed={seed} train={result.train_success_last500:.3f} "
        f"greedy={evaluation.successes}/{evaluation.episodes} "
        f"Q0={result.q0_max:.4f} err={row['q0_abs_error']:.4f}"
    )
    return row, result


def save_vi_history(scenarios: dict[bool, Scenario]) -> None:
    path = OUT / "vi_convergence.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["environment", "iteration", "delta", "v_start"])
        for scenario in scenarios.values():
            for iteration, delta, v_start in scenario.vi.history:
                writer.writerow([scenario.environment, iteration, delta, v_start])


def save_summary(rows: list[dict[str, object]]) -> None:
    path = OUT / "summary.csv"
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_q_curves(q_results: dict[tuple[str, int], QLearningResult]) -> None:
    """保存 D2 每个回合的训练行为，保证学习曲线有原始底表。"""
    path = OUT / "q_training_curves.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "environment",
                "seed",
                "episode",
                "train_success",
                "return_undiscounted",
                "return_discounted",
            ]
        )
        for (environment, seed), result in sorted(q_results.items()):
            for episode in range(len(result.train_success)):
                writer.writerow(
                    [
                        environment,
                        seed,
                        episode + 1,
                        int(result.train_success[episode]),
                        float(result.train_returns_undiscounted[episode]),
                        float(result.train_returns_discounted[episode]),
                    ]
                )


def plot_vi(scenarios: dict[bool, Scenario]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    color_map = {False: COLORS[0], True: COLORS[1]}

    for slippery, scenario in scenarios.items():
        color = color_map[slippery]
        history = np.asarray(scenario.vi.history, dtype=np.float64)
        label = scenario.environment
        axes[0].semilogy(history[:, 0], history[:, 1], color=color, lw=1.8, label=label)
        axes[0].scatter(
            scenario.vi.iterations,
            history[-1, 1],
            color=color,
            s=28,
            zorder=3,
        )
        axes[1].plot(history[:, 0], history[:, 2], color=color, lw=1.8, label=label)
        axes[1].scatter(
            scenario.vi.iterations,
            history[-1, 2],
            color=color,
            s=28,
            zorder=3,
        )
        axes[1].annotate(
            f"{scenario.vi.iterations} iters\nV={history[-1, 2]:.4f}",
            xy=(scenario.vi.iterations, history[-1, 2]),
            xytext=(-18 if slippery else 10, 18 if slippery else -28),
            textcoords="offset points",
            color=color,
            fontsize=8,
        )

    axes[0].axhline(1e-12, color="0.5", ls=":", lw=1.0, label="tol = 1e-12")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("max |V_new - V|")
    axes[0].set_title("Convergence residual")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(frameon=False)

    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("V(start)")
    axes[1].set_title("Value at start state")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle("Value iteration: synchronous updates, gamma=0.99", y=1.02)
    fig.tight_layout()
    path = OUT / "vi_convergence.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _rolling_mean(values: np.ndarray, window: int = 100) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / window
    smooth = np.convolve(values, kernel, mode="valid")
    return np.arange(window, len(values) + 1), smooth


def plot_q_learning(
    scenarios: dict[bool, Scenario],
    q_results: dict[tuple[str, int], QLearningResult],
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3))
    color_map = {False: COLORS[0], True: COLORS[1]}

    for slippery in (False, True):
        color = color_map[slippery]
        label = "slippery" if slippery else "deterministic"
        seed_curves = []
        for seed in SEEDS:
            result = q_results[(label, seed)]
            x, smooth = _rolling_mean(result.train_success, window=100)
            seed_curves.append(smooth)
            axes[0].plot(x, smooth, color=color, alpha=0.28, lw=1.0)
        mean_curve = np.mean(seed_curves, axis=0)
        axes[0].plot(x, mean_curve, color=color, lw=2.2, label=f"{label} (mean of 3 seeds)")

    axes[0].set_xlabel("training episode")
    axes[0].set_ylabel("rolling success rate (100 episodes)")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title("Training behavior: epsilon-greedy, epsilon -> 0.05")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    for group_index, slippery in enumerate((False, True)):
        color = color_map[slippery]
        label = "slippery" if slippery else "deterministic"
        q0_values = np.array([q_results[(label, seed)].q0_max for seed in SEEDS])
        jitter = np.linspace(-0.08, 0.08, len(SEEDS))
        axes[1].scatter(
            group_index + jitter,
            q0_values,
            color=color,
            s=42,
            zorder=3,
            label=f"{label}: Q0 per seed",
        )
        vi_value = scenarios[slippery].vi.V[START_STATE]
        axes[1].hlines(
            vi_value,
            group_index - 0.25,
            group_index + 0.25,
            color=color,
            ls="--",
            lw=1.4,
            label=f"{label}: VI = {vi_value:.4f}",
        )
        axes[1].text(
            group_index,
            0.97,
            f"Q0 mean = {q0_values.mean():.4f}",
            transform=axes[1].get_xaxis_transform(),
            ha="center",
            va="top",
            color=color,
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=color, alpha=0.8),
        )

    all_values = [
        scenarios[slippery].vi.V[START_STATE] for slippery in (False, True)
    ] + [
        q_results[(label, seed)].q0_max
        for label in ("deterministic", "slippery")
        for seed in SEEDS
    ]
    axes[1].set_ylim(min(all_values) - 0.05, max(all_values) + 0.14)
    axes[1].set_xticks([0, 1], ["deterministic", "slippery"])
    axes[1].set_xlim(-0.5, 1.5)
    axes[1].set_ylabel("Q0 max after training")
    axes[1].set_title("Learned value vs model-based truth")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7.5)

    fig.suptitle("Tabular Q-learning: 6000 episodes x 3 seeds", y=1.02)
    fig.tight_layout()
    path = OUT / "q_learning.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_policies(scenarios: dict[bool, Scenario]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6))
    for ax, slippery in zip(axes, (False, True)):
        scenario = scenarios[slippery]
        values = scenario.vi.V
        policy = scenario.vi.policy

        for state in range(N_STATES):
            row, col = divmod(state, 4)
            if state in HOLES:
                ax.add_patch(plt.Rectangle((col - 0.48, row - 0.48), 0.96, 0.96,
                                           color="0.35"))
                ax.text(col, row, "H", ha="center", va="center", color="white",
                        fontsize=11, fontweight="bold")
            elif state == GOAL_STATE:
                ax.add_patch(plt.Rectangle((col - 0.48, row - 0.48), 0.96, 0.96,
                                           color="#7fbf7f"))
                ax.text(col, row, "G", ha="center", va="center", color="black",
                        fontsize=11, fontweight="bold")
            else:
                if state == START_STATE:
                    ax.add_patch(plt.Circle((col, row), 0.42, color="#dbe9f6", zorder=0))
                dr, dc = ACTION_DELTAS[int(policy[state])]
                ax.annotate(
                    "",
                    xy=(col + 0.30 * dc, row + 0.30 * dr),
                    xytext=(col - 0.25 * dc, row - 0.25 * dr),
                    arrowprops=dict(arrowstyle="-|>", color="#1f4e79", lw=1.8),
                )
                prefix = "S " if state == START_STATE else ""
                ax.text(
                    col - 0.43,
                    row - 0.39,
                    f"{prefix}{values[state]:.3f}",
                    fontsize=6.8,
                    color="0.2",
                    ha="left",
                    va="bottom",
                )

        ax.set_xlim(-0.55, 3.55)
        ax.set_ylim(3.55, -0.55)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{'Slippery' if slippery else 'Deterministic'} | "
            f"{scenario.vi.iterations} iterations | V(S)={values[START_STATE]:.3f}"
        )
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Optimal policy from value iteration (numbers are V(s))", y=1.02)
    fig.tight_layout()
    path = OUT / "policies.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenarios = run_scenarios()
    save_vi_history(scenarios)

    rows: list[dict[str, object]] = []
    q_results: dict[tuple[str, int], QLearningResult] = {}

    print("\n=== D2: value iteration vs Q-learning (H=100) ===")
    for slippery in (False, True):
        scenario = scenarios[slippery]
        for seed in SEEDS:
            row, result = run_q_experiment(
                experiment="D2_model_vs_modelfree",
                scenario=scenario,
                horizon=HORIZONS["D2"],
                seed=seed,
            )
            rows.append(row)
            q_results[(scenario.environment, seed)] = result

    print("\n=== C1: terminated vs truncated target (H=20) ===")
    for seed in SEEDS:
        for bootstrap, handling in ((True, "correct"), (False, "mixed_done")):
            row, _ = run_q_experiment(
                experiment="C1_termination_semantics",
                scenario=scenarios[True],
                horizon=HORIZONS["C1"],
                seed=seed,
                bootstrap_truncated=bootstrap,
                termination_handling=handling,
            )
            rows.append(row)

    print("\n=== C2: sparse vs hacked reward (H=50) ===")
    for seed in SEEDS:
        for reward_mode in ("sparse", "hacked"):
            row, _ = run_q_experiment(
                experiment="C2_reward_design",
                scenario=scenarios[True],
                horizon=HORIZONS["C2"],
                seed=seed,
                reward_mode=reward_mode,
            )
            rows.append(row)

    save_summary(rows)
    save_q_curves(q_results)
    vi_path = plot_vi(scenarios)
    q_path = plot_q_learning(scenarios, q_results)
    policy_path = plot_policies(scenarios)

    print(f"\n写入 {OUT / 'summary.csv'}")
    print(f"写入 {OUT / 'q_training_curves.csv'}")
    print(f"写入 {vi_path}")
    print(f"写入 {q_path}")
    print(f"写入 {policy_path}")


if __name__ == "__main__":
    main()
