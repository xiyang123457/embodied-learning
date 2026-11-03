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

## 当前结论（详见 [report.md](report.md)）

| 预算 | `success` | `is_grasped` | `is_obj_placed` | `is_robot_static` |
|---|---|---|---|---|
| 2M（140 秒） | **0/20** | 19/20 | 0/20 | 20/20 |
| 6M（487 秒） | **11/20 (55%)** | 20/20 | 16/20 | 11/20 |

- **2M 不够是预算问题，不是方法问题**：2M 时 `is_grasped` 已 95%，但 `is_obj_placed` 为 0。
- **失败中 5/9 是「搬到位但没停稳」**，不是「没搬过去」（4/9）。
- **return 在 4M 步前涨了 7 倍而 success 始终为 0**，再次说明不能用 return 代替 success。

## 已知坑

与阶段 3 相同（tyro 用 kebab-case、`runs/` 相对 cwd、需要 tensorboard、`sim_backend` 固定为
`physx_cuda`、`info` 里的值是 cuda tensor 必须先 `.cpu()`）。
