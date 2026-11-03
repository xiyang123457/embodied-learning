# 阶段 3：PushCube-v1 PPO

对应计划书第 5 节。使用 **ManiSkill 官方 PPO baseline**（不修改其代码），只在本目录做参数固化、运行留痕与独立评估。

环境：`.venv-maniskill`。

## 前置：拉取官方 baseline

第三方代码不进 git（`third_party/` 已忽略），需要时按固定 tag 重新检出：

```bash
git clone --depth 1 --branch v3.0.1 --filter=blob:none --sparse \
  https://github.com/haosulab/ManiSkill.git third_party/ManiSkill
cd third_party/ManiSkill && git sparse-checkout set examples/baselines
```

本次使用的 commit 记录在每次 run 的 `meta.json` 里（`a4a4f92`）。

## 训练

```bash
# 先看会执行什么命令（不运行）
./.venv-maniskill/bin/python experiments/03_ppo_pushcube/run.py --config smoke --dry-run

# 冒烟：64 环境 / 51200 步，约 34 秒，只验证链路
./.venv-maniskill/bin/python experiments/03_ppo_pushcube/run.py --config smoke

# 主训练：2048 环境 / 200 万步，约 4 分钟
./.venv-maniskill/bin/python experiments/03_ppo_pushcube/run.py --config train
```

临时改参数（不写回配置文件）：

```bash
./.venv-maniskill/bin/python experiments/03_ppo_pushcube/run.py --config train --set num_envs=1024
```

每次 run 落在 `runs/<tag>_<时间>_seed<N>/`，其中：

| 文件 | 内容 |
|---|---|
| `config_snapshot.json` | 本次实际生效的完整配置 |
| `meta.json` | 命令、包版本、GPU、主仓库与 baseline commit、起止时间、耗时、退出码 |
| `stdout.log` | 训练与评估全过程输出 |
| `final_ckpt.pt` | 最终权重（中间 `ckpt_*.pt` 不入 git） |
| `videos/` | 官方 baseline 自动录制的评估回合 |
| `artifacts.txt` | 本次产物清单 |

## 评估

官方 baseline 的 `--evaluate` 用**训练种子**重置评估环境，不满足计划书「评估种子与训练种子不重叠」，
所以单独用 `evaluate.py` 做一次：

```bash
./.venv-maniskill/bin/python experiments/03_ppo_pushcube/evaluate.py \
  --checkpoint experiments/03_ppo_pushcube/runs/<run>/final_ckpt.pt \
  --n-episodes 20 --seed-start 100
```

产出 `per_episode.csv`、`summary.json`，以及两段**慢放**视频：首个成功回合与首个失败回合。

视频为什么默认 4 fps：成功回合只要 11–19 步就结束，成功即 `terminated`，无法继续录；
按 20 fps 播放不到 1 秒，看不清「靠近 → 推动 → 方块进目标圈」的过程。
所以默认降到 4 fps，首帧定格 5 帧、末帧定格 10 帧，并在每帧左上角叠加
`step / reward / success`。用 `--video-fps` 可调整速度。

## 图表

官方 baseline 只写 tensorboard events 和 stdout，不产图，所以单独用公共绘图脚本
从 `stdout.log` 与 `eval/per_episode.csv` 生成（不依赖 tensorboard）：

```bash
./.venv-maniskill/bin/python experiments/common/plot.py \
  experiments/03_ppo_pushcube/runs/<run>
```

产出到 `runs/plots/`：

| 文件 | 内容 |
|---|---|
| `<run>_eval.png` | 评估曲线：success_once 与 return 随训练步数变化（双轴） |
| `<run>_sps.png` | 吞吐曲线：每个迭代的 steps/s |
| `<run>_breakdown.png` | 独立评估的指标分解柱状图 |

## 配置项

`configs/smoke.json` / `configs/train.json` 的键直接映射到官方脚本参数（`_` → `-`）。
`tag` 与 `description` 是本项目自用键，不会传给 baseline。

注意官方默认值不是常见配置：`gamma=0.8`、`gae_lambda=0.9`、`num_steps=50`（等于 episode 上限），
`train.json` 沿用官方 README 的 PushCube 示例（`num_envs=2048`、`num_steps=20`、`update_epochs=8`）。

## 已知坑

1. **tyro 的 CLI 是 kebab-case**（`--num-envs`），布尔是 `--capture-video` / `--no-capture-video`，不接受 `--x True`。`run.py` 已处理。
2. **官方脚本的 `runs/` 相对 cwd 生成**，所以 `run.py` 强制 `cwd=本阶段目录`，否则产物会散落到调用处。
3. **需要 `tensorboard`**（`SummaryWriter`），pip 版 `mani_skill` 不带。
4. **`sim_backend` 硬编码 `physx_cuda`**，单环境评估会慢，但保证与训练同一物理后端。

结果与分析见 [report.md](report.md)。
