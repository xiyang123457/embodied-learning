# 阶段 4：ACT 双臂 Transfer Cube（`sim_transfer_cube_scripted`）

对应计划书第 6 节。使用 ACT 官方仓库。相对官方代码只有**路径配置补丁**和**训练循环稳定性补丁**，
算法与超参未改（见文末「补丁三」）。

环境：`.venv-act`（Python 3.11 + mujoco 2.3.7 + dm_control 1.0.14 + torch 2.15.0+cu132）。

> 目录编号 05 是实验顺序，对应计划书的**阶段 4**。

## 前置：克隆 ACT 并打两处补丁

`third_party/` 不进 git，重建时需重放以下步骤。本次使用 commit **742c753**。

```bash
git clone --depth 1 https://github.com/tonyzhaozh/act.git third_party/act
```

`third_party/act/constants.py` 原仓库里有两处需要改：

1. `DATA_DIR = '<put your data dir here>'` 是占位符，改为基于文件位置推导：

```python
import os
import pathlib

_ACT_DIR = pathlib.Path(__file__).resolve().parent      # third_party/act
_REPO_ROOT = _ACT_DIR.parent.parent                     # 仓库根目录
DATA_DIR = str(_REPO_ROOT / 'experiments' / '05_act_transfer_cube' / 'data')
```

2. `sim_transfer_cube_scripted` 的 `num_episodes` 固定为 50，改为可用环境变量覆盖：

```python
'num_episodes': int(os.environ.get('ACT_NUM_EPISODES', 50)),
```

`constants.py` 其余部分未改动；`utils.py` / `imitate_episodes.py` 的改动见下文「补丁三」。

## 环境

```bash
conda create -y -p .venv-act python=3.11
./.venv-act/bin/pip install "numpy<2" mujoco==2.3.7 dm_control==1.0.14
./.venv-act/bin/pip install torch torchvision pyquaternion pyyaml h5py \
    matplotlib einops opencv-python-headless ipython tqdm
cd third_party/act/detr && ../../../.venv-act/bin/pip install -e .
```

实测：安装 torch 时 numpy 会被升到 **2.4.6**，与 mujoco 2.3.7 / dm_control 1.0.14 仍可正常工作
（已用 `warnings.simplefilter("error")` 验证加载与步进）。

## 数据生成

```bash
cd third_party/act
MUJOCO_GL=egl ../../.venv-act/bin/python record_sim_episodes.py \
  --task_name sim_transfer_cube_scripted \
  --dataset_dir ../../experiments/05_act_transfer_cube/data/sim_transfer_cube_scripted \
  --num_episodes 50
```

脚本化策略 **50/50 成功**，数据 **18 GB**。

## 训练

```bash
cd third_party/act
ACT_NUM_EPISODES=25 MUJOCO_GL=egl ../../.venv-act/bin/python imitate_episodes.py \
  --task_name sim_transfer_cube_scripted \
  --ckpt_dir ../../experiments/05_act_transfer_cube/runs/<tag> \
  --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 \
  --batch_size 4 --dim_feedforward 3200 --num_epochs 200 --lr 1e-5 --seed 0
```

## 评估

```bash
cd third_party/act
ACT_NUM_EPISODES=25 MUJOCO_GL=egl ../../.venv-act/bin/python imitate_episodes.py \
  --task_name sim_transfer_cube_scripted \
  --ckpt_dir ../../experiments/05_act_transfer_cube/runs/<tag> \
  --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 \
  --batch_size 4 --dim_feedforward 3200 --num_epochs 200 --lr 1e-5 --seed 0 \
  --eval --temporal_agg
```

固定跑 **50 次 rollout**（`num_rollouts = 50` 硬编码），输出成功率与分档 reward。

## 已知坑

1. **内存是硬约束**：ACT 的 `load_data` 会把全部图像读进 RAM。
   单条 episode 352 MB（400 帧 × 480×640×3），50 条 18 GB，本机 32 GB 可用约 19 GB，
   训练 50 条极可能 OOM。**因此训练用 25 条（9 GB），磁盘上的 50 条留给阶段 5 消融。**
2. **每个 epoch 的梯度更新次数很少**：`EpisodicDataset` 按 episode 采样，
   25 条 × 0.8 = 20 个训练 episode，`batch_size=4` 时每 epoch 只有 **5 次更新**。
   所以 2000 epochs 并非"很多"，官方默认就是这么训练的。
3. **checkpoint 很大**：单个 321 MB，一次训练存 6 个 ≈ 1.9 GB，已加入 `.gitignore`。
4. **必须设 `MUJOCO_GL=egl`**，否则无头环境会尝试初始化 GLFW 窗口。
5. 评估会生成 50 个 `video*.mp4`（约 300 MB），已忽略；代表性案例复制到 `clips/`。

## 补丁三：训练循环与 DataLoader（2026-11-03 追加）

背景：2000 epoch 长跑时整机明显卡顿。实测定位到两个与训练相关的成因——worker 进程每个
epoch 重建（PID 每 2~4 秒变化，每次 fork 约 1.9 GB）、以及每 100 epoch 落盘 336 MB 时的阻塞。
算法与超参未动。

`utils.py` `load_data()`：worker 常驻，不再每个 epoch 重建。

```python
num_workers = int(os.environ.get('ACT_NUM_WORKERS', 2))
loader_kwargs = {'persistent_workers': True, 'prefetch_factor': 2} if num_workers > 0 else {}
train_dataloader = DataLoader(..., num_workers=num_workers, **loader_kwargs)
val_dataloader = DataLoader(..., num_workers=num_workers, **loader_kwargs)
```

> 注意：`num_workers` 从 1 改为 2 会改变 worker 侧的 `np.random.choice` 序列（起始帧采样），
> 因此**与旧 run 不是逐位可复现**，新旧对比只能看同一指标的趋势。

`imitate_episodes.py`：

| 改动 | 性质 |
|---|---|
| worker 常驻（`persistent_workers=True`，`num_workers=2`） | **提速 + 消除卡顿尖峰** |
| `forward_pass` 的 `.cuda()` 加 `non_blocking=True` | 提速（与 `pin_memory=True` 配合） |
| `save_ckpt_atomic()`：写 `.tmp` 再 `os.replace` | 健壮性：中断不产生截断 ckpt |
| best 权重改存 `v.detach().cpu().clone()` | 避免显存里多留一份约 336 MB |
| 每 epoch 归约后立即转 `float` | **清理，不是提速** |

最后一项的说明：原实现 `train_history[nb*epoch : nb*(epoch+1)]` 的切片恰好是当前 epoch
（`nb = len(train_dataloader)`），计算量本来就是常数，不存在随 epoch 增长的重算；
保留的张量也只有 0-dim 标量，显存增量在几十 MB 量级。改它是为了让历史列表不再持有
CUDA 张量，**不要把它记成性能收益**。

## 当前结果

| 训练量 | 数据 | 成功率 | 平均 return |
|---|---|---|---|
| 200 epochs | 25 条 | **26%（13/50）** | 164.32 |

详见 [report.md](report.md)。这是短训练结果（官方默认 2000 epochs），不是最终基线。
