# 阶段 3 第二步：PickCube-v1 PPO（抓取任务）

对应计划书第 5 节「PushCube → PickCube」。使用 ManiSkill 官方 PPO baseline，不改其代码。

环境：`.venv-maniskill`。前置的 `third_party/ManiSkill` 检出方式见
[`../03_ppo_pushcube/README.md`](../03_ppo_pushcube/README.md)。

## 与 PushCube 的关键差异

| 项 | PushCube | **PickCube** |
|---|---|---|
| 成功距离 | 2D < 10 cm | **3D ≤ 2.5 cm** |
| 静止要求 | 无 | **有**（qvel < 0.2） |
| 目标位置 | 固定相对偏移 | **独立随机**，z 在方块上方 0–30 cm |
| 是否需要抓取 | 否 | 基本必须 |

**两者的成功率不能直接比较。**

## 训练

```bash
# 冒烟：64 环境 / 51200 步，约 35 秒，只验证链路
./.venv-maniskill/bin/python experiments/04_ppo_pickcube/run.py --config smoke

# 主训练：2048 环境 / 200 万步，约 140 秒
./.venv-maniskill/bin/python experiments/04_ppo_pickcube/run.py --config train

# 预算对照：600 万步，约 487 秒
./.venv-maniskill/bin/python experiments/04_ppo_pickcube/run.py \
  --config train --set total_timesteps=6000000 tag=train6m
```

## 评估

```bash
./.venv-maniskill/bin/python experiments/04_ppo_pickcube/evaluate.py \
  --checkpoint experiments/04_ppo_pickcube/runs/<run>/final_ckpt.pt --n-episodes 20
```

除 `success` 外还会记录 `is_grasped` / `is_obj_placed` / `is_robot_static`
三个分解指标，`per_episode.csv` 里每个都有 `_ever` 与 `_end` 两列，
用来区分「没抓到」「抓到了没搬过去」「搬到了没停稳」三种失败。

## 图表

```bash
# 两个预算一起传，会额外生成 compare_eval.png
./.venv-maniskill/bin/python experiments/common/plot.py \
  experiments/04_ppo_pickcube/runs/train_20261103_095601_seed1 \
  experiments/04_ppo_pickcube/runs/train6m_20261103_095904_seed1
```

产出到 `runs/plots/`：每个 run 的 `_eval.png`（success + return 双轴）、
`_sps.png`（吞吐）、`_breakdown.png`（is_grasped / is_obj_placed /
is_robot_static / success 的比率），以及多 run 时的 `compare_eval.png`。

若 run 目录下存在 `diagnose_*/`，还会额外生成：

| 文件 | 内容 |
|---|---|
| `_diagnose_<N>step_goal_height.png` | 左：每个种子的目标高度与成败散点；右：按高度分层的成功率 |
| `_diagnose_<N>step_traj_success_seedM.png` | 成功回合的轨迹：obj→goal 距离 + max\|qvel\|，绿底标出方块在圈内的区间 |
| `_diagnose_<N>step_traj_failure_seedM.png` | 失败回合的同一组曲线，用来看「到位」与「静止」是否错开 |

## 当前结论（详见 [report.md](report.md)）

| 预算 | 评估 | 总成功 | `goal_z < 0.15` | `goal_z ≥ 0.15` | 耗时 |
|---|---|---|---|---|---|
| 2M | 20 回合 | 0/20 (0%) | — | — | 140 秒 |
| 6M | 50 回合 | **27/50 (54%)** | 23/24 (95.8%) | 4/26 (15.4%) | 487 秒 |
| 10M | 50 回合 × 2 批 | **50/50 (100%)** | 24/24、22/22 | 26/26、28/28 | 646 秒 |

- **加预算一次性解决了问题**：6M 高目标 15.4% → 10M 100%（两批互不重叠的评估种子，共 100 回合全中）。
- **2M 不够是预算问题，不是方法问题**：2M 时 `is_grasped` 已 95%，但 `is_obj_placed` 为 0。
- **「失败集中在某个条件」不等于「该条件是能力边界」**：6M 下高目标明显更难，
  但那只是训练量不足，10M 后差距完全消失。
- **return 在 4M 步前涨了 7 倍而 success 始终为 0**，再次说明不能用 return 代替 success。

## 失败诊断

```bash
./.venv-maniskill/bin/python experiments/04_ppo_pickcube/diagnose.py \
  --checkpoint experiments/04_ppo_pickcube/runs/<run>/final_ckpt.pt --max-steps 100
```

逐帧记录 `obj` / `goal` / `tcp` 位置、`obj_to_goal_dist`、`max|qvel|` 与三个判定，
输出 `trajectory.csv` 与 `episode_summary.csv`（含首次到位/首次静止步数）。

结论（6M 模型）：

| 目标高度 | 20 种子 / 100 步 | 50 种子 / 50 步 |
|---|---|---|
| `goal_z < 0.15` | 9/9 (100%) | **23/24 (95.8%)** |
| `goal_z ≥ 0.15` | 2/11 (18%) | **4/26 (15.4%)** |

两档置信区间不重叠（低 [80%, 99%]，高 [6%, 34%]），所以在 6M 预算下这个差距是真实的。
**但它不是能力边界**：同一配置训练到 10M 后，两档都变成 100%。

- 初始距离与抓取质量在成功/失败组之间几乎无差异（0.108–0.117、0.009–0.016），
  所以不是「够不着」或「抓不稳」。
- 延长到 100 步后成功数仍是 11/20，**时间不是瓶颈**。
- 失败回合的典型轨迹是「方块在圈内时手臂停不下来，等停下来方块已被推出圈外」。

**因此 55% 这个整体数字掩盖了「低目标全中、高目标基本不行」的结构，
后续评估必须按目标高度分层。**

## 已知坑

与阶段 3 相同（tyro 用 kebab-case、`runs/` 相对 cwd、需要 tensorboard、`sim_backend` 固定为
`physx_cuda`、`info` 里的值是 cuda tensor 必须先 `.cpu()`）。
