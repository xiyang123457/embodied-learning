# 阶段 2：ManiSkill 接口核对与后端实测

对应计划书第 4 节。在写任何训练代码之前，先把「这个环境到底吃什么、吐什么」钉死。

环境：`.venv-maniskill`（mani_skill 3.0.1 / sapien 3.0.3 / torch 2.15.0 / Vulkan 渲染）。

## 脚本

### `probe_env.py` — 接口核对

```bash
./.venv-maniskill/bin/python experiments/02_maniskill_probe/probe_env.py
./.venv-maniskill/bin/python experiments/02_maniskill_probe/probe_env.py --env PickCube-v1
```

产出 `outputs/PushCube-v1_facts.json`、渲染图 `PushCube-v1_rgb.png`、随机动作视频 `PushCube-v1_random.mp4`。

### `bench_backend.py` — CPU vs GPU 仿真吞吐

```bash
./.venv-maniskill/bin/python experiments/02_maniskill_probe/bench_backend.py --steps 100
```

产出 `outputs/backend_bench.json`。

## 实测结论（PushCube-v1，mani_skill 3.0.1）

| 项 | 值 |
|---|---|
| 机器人 | `panda` |
| 默认控制模式 | `pd_joint_delta_pos`（另有 10 种可选） |
| 动作 | `Box(-1, 1, (8,))`：7 个关节位置增量 + 1 个夹爪 |
| 观测（state） | 扁平 `(num_envs, 35)` |
| 观测（state_dict） | `agent.qpos(9) + agent.qvel(9) + extra.tcp_pose(7) + goal_pos(3) + obj_pose(7)` |
| 奖励模式 | `normalized_dense`，每步是潜力变化，不是「存活 +1」 |
| 成功判定 | `info["success"]`，**当步**口径；成功时 `terminated=True` |
| episode 上限 | **50 步**（`elapsed_steps=50` 时 `truncated=True`），ManiSkill 3.0.1 不把它暴露成属性，只能实测 |
| 渲染 | 无头 Vulkan 可跑，`512×512×3` |

动作语义实测（其余维固定为 0，只动一维看 qpos 增量）：

| 动作 | qpos 变化 |
|---|---|
| `joint_i = +1` | 对应关节转动约 +0.025 rad |
| `gripper = +1` | **无变化**（初始已张开到上限） |
| `gripper = -1` | 两指各 -0.0196（闭合） |
| 全零 | 夹爪也 -0.0098 |

**第 8 维不是简单的「+1 张开」，而是夹爪目标位置比例**，可见变化取决于当前夹爪状态。

## CPU vs GPU 仿真后端

| backend | num_envs | env_steps/s | 显存 |
|---|---|---|---|
| cpu | 1 | 1865.7 | — |
| gpu | 1 | 100.9 | 1.27 GB |
| gpu | 64 | 5672.3 | 1.27 GB |
| gpu | 256 | 21532.8 | 1.34 GB |
| gpu | 1024 | 68156.1 | 1.53 GB |

分界线在 `num_envs ≈ 32–64`：

- **单环境 GPU 比 CPU 慢约 18 倍**（PhysX/CUDA 上下文的一次性开销，其中 1.27 GB 是与并行数无关的固定占用）
- **1024 环境 GPU 快 36.5 倍**
- 8 GB 显存很充裕：1024 个 PushCube 环境平均每个只占约 0.26 MB

所以：接口核对/调试用 `cpu`，训练用 `gpu` + 大批量。

## 已知坑（都是实测撞出来的）

1. **`sim_backend` 的取值**：`probe_env.py` 用 `"gpu"`，官方 baseline 硬编码 `"physx_cuda"`，两者都能工作，但不要混用。
2. **`GPU PhysX can only be enabled once before any other code involving PhysX`**：同一进程内不能先初始化 CPU PhysX 再切 GPU，必须分进程（`bench_backend.py` 因此用 `subprocess` 逐个跑）。
3. **`render()` 与 `step()` 的返回值是 cuda tensor**，`np.asarray()` 会直接抛 `TypeError`，必须先 `.cpu()`。
4. **批量模式下 `action_space.shape` 是 `(num_envs, act_dim)`**，取 `shape[0]` 会拿到 `num_envs`。
5. **`max_episode_steps` 不是属性**：`env.spec.max_episode_steps` 与 `unwrapped._max_episode_steps` 都是 `None`，只能跑到 `truncated=True` 反推。
