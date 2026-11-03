# experiments

《具身智能实验与实践学习计划（修订版）》各阶段的实验代码与产物。

## 为什么单独建这个目录

根目录的 `scripts/`、`src/` 属于 W2–W7 的学习主线（每个脚本几十行、共用一套依赖）。
到了阶段 2 之后，每个阶段都有**自己的虚拟环境、自己的第三方依赖、自己的输出目录**，
再往 `scripts/` 里堆会让「哪个脚本该用哪个 venv」变得不可查。

所以从阶段 2 起：

| 目录 | 内容 |
|---|---|
| `scripts/` | 只保留 W2–W7 的学习脚本，不再新增 |
| `experiments/` | 阶段实验，每个阶段一个子目录，自带环境说明、配置、产物 |

## 目录约定

每个阶段目录结构一致：

```
<阶段>/
├── README.md          # 怎么装、怎么跑、已知坑
├── report.md          # 结果与分析（纯核对类阶段可省）
├── configs/           # 实验参数（JSON，配置驱动，不靠命令行口口相传）
├── run.py / evaluate.py  # 薄壳入口，真正逻辑在 ../common/；路径基于 __file__，不依赖 cwd
├── runs/ 或 outputs/  # 实验产物
```

产物入库原则：

- 进 git：`config`、`meta`、日志、评估 CSV/JSON、`final_ckpt.pt`、少量视频、图
- 不进 git（见根 `.gitignore`）：中间 `ckpt_*.pt`、tensorboard events、`third_party/`、各阶段 venv

## 阶段索引

| 阶段 | 目录 | 环境 | 状态 | 报告 |
|---|---|---|---|---|
| 1 CartPole PPO | `01_ppo_cartpole/` | `.venv-ppo` | 完成（20/20） | [report.md](01_ppo_cartpole/report.md) |
| 2 ManiSkill 接口核对 | `02_maniskill_probe/` | `.venv-maniskill` | 完成 | 见该目录 README 的实测结论 |
| 3a PushCube PPO | `03_ppo_pushcube/` | `.venv-maniskill` | 完成（19/20） | [report.md](03_ppo_pushcube/report.md) |
| 3b PickCube PPO | `04_ppo_pickcube/` | `.venv-maniskill` | 部分成功（11/20，受预算限制） | [report.md](04_ppo_pickcube/report.md) |
| 4 ACT（Transfer Cube） | 未开始 | `.venv-act`（规划） | — | — |
| 5 数据量消融 | 未开始 | 依赖阶段 4 | — | — |

`common/` 放跨阶段共用的代码：`runner.py`（配置驱动的运行封装与留痕）、
`rollout.py`（独立评估、分解指标、慢放视频）。各阶段目录只保留自己的 `configs/` 和薄壳入口，
避免把 `run.py` / `evaluate.py` 复制一遍。

## 环境对照表

| 环境 | Python | 关键依赖 | 用于 |
|---|---|---|---|
| `.venv` | 3.12 | mujoco 3.11.0 / gymnasium-robotics 1.4.2 / torch 2.14.1+cpu | W5–W7（**不要动**） |
| `.venv-ppo` | 3.12 | torch 2.15.0+cu132 / SB3 2.9.2 | 阶段 1 |
| `.venv-maniskill` | 3.12 | mani_skill 3.0.1 / sapien 3.0.3 / torch 2.15.0 / tensorboard | 阶段 2、3 |

重建 `.venv-ppo`：

```bash
/usr/bin/python3.12 -m venv .venv-ppo
./.venv-ppo/bin/pip install -r requirements-ppo.txt
```

重建 `.venv-maniskill`：

```bash
/usr/bin/python3.12 -m venv .venv-maniskill
./.venv-maniskill/bin/pip install torch mani_skill tensorboard
```

## 跨阶段通用规范

沿用计划书第 8 节，落到工程上就是：

1. **训练与评估分离**：评估用独立环境实例、独立种子列表（默认 `100–119`），绝不与训练种子重叠。
2. **配置驱动**：所有超参写在 `configs/*.json`，每次 run 存一份快照，命令行只传配置名。
3. **每次 run 留痕**：`meta.json` 记录命令、包版本、GPU、主仓库与第三方仓库 commit、耗时、退出码。
4. **先冒烟后长跑**：小并行数、短步数先验证 reset / 奖励 / 成功判定 / checkpoint / 视频，再按预算拉长。
5. **区分三种结论**：「跑通接口」「完成评估」「复现基线」不是一回事，报告里必须写清是哪一种。
