"""阶段实验共用的运行封装与评估工具。

放在 experiments/common 而不是根 src/，是因为这些工具只服务 experiments/ 下的
阶段实验，依赖 third_party 里的 ManiSkill baseline，与 W2–W7 的 src/ 没有关系。
"""

__all__ = ["runner", "rollout"]
