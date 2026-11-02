# 阶段 1：CartPole PPO 最小闭环

对应计划书第 3 节。目的是把「随机基线 → 训练 → 保存 → 独立评估 → 视频 → 曲线」整条管线跑通。

环境：`.venv-ppo`（Python 3.12 + torch 2.15.0+cu132 + SB3 2.9.2），见 `requirements-ppo.txt`。

## 运行

```bash
# 正式：100k 步，约 340 秒（CPU）
./.venv-ppo/bin/python experiments/01_ppo_cartpole/ppo_cartpole.py

# 冒烟：3k 步，约 12 秒
./.venv-ppo/bin/python experiments/01_ppo_cartpole/ppo_cartpole.py --timesteps 3000 --n-eval 3 --tag smoke

# 设备对照（MLP 上 GPU 通常更慢，仅供验证）
./.venv-ppo/bin/python experiments/01_ppo_cartpole/ppo_cartpole.py --timesteps 20000 --device cuda --tag gpu_bench
```

输出默认落在本目录 `outputs/`，不依赖调用时的工作目录。

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--timesteps` | 100000 | 训练总步数 |
| `--seed` | 0 | 训练种子 |
| `--n-eval` | 20 | 评估回合数 |
| `--eval-seed-start` | 100 | 评估种子起点，脚本会断言它不等于训练种子 |
| `--device` | cpu | `cpu` / `cuda` |
| `--tag` | 空 | 非默认参数时的输出子目录名 |

## 产物

`outputs/`：`learning_curve.png`、`train_log.csv`、`eval_random.csv`、`eval_ppo.csv`、
`video_before_random.mp4`、`video_after_ppo.mp4`、`model.zip`、`config.json`、`summary.json`。

## 结果摘要

随机 21.0 ± 8.6（0/20）→ PPO 500.0 ± 0.0（20/20），checkpoint 重载后逐回合完全一致。

完整分析与局限见 [report.md](report.md)，其中两条值得记住：

- 标准差 0.0 是撞到 `max_episode_steps` 上限造成的，**不是策略稳定**，20 回合全撞顶就失去区分力。
- 曲线末端滑动平均 421 而评估 500：训练期指标与评估期指标的口径不同。
