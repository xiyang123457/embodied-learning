# embodied-learning

具身智能学习仓库。W1–W8 逐步把「一个点」做成「一个能感知、能决策的系统」。

| 周 | 做了什么 | 目录 |
|---|---|---|
| W2 | 二维点 + 比例反馈：证明「随机 vs 反馈」的差距由 `Kp·DT` 决定 | `scripts/first_loop.py` |
| W3 | 坐标变换与二连杆正运动学：`T_target_source` 命名 + 下标消去 | `src/transforms.py`、`src/two_link.py` |
| W4 | 雅可比、奇异位形、单关节 PD 调参 | `src/jacobian.py`、`src/pd_control.py` |
| W5 | MuJoCo `FetchReach-v4` 接口核对、位置基线、专家轨迹采集 | `scripts/w5_core.py` 等 |
| W6 | 行为克隆：20/40 条轨迹、线性回归与 MLP 对照、2 mm 闭环判据 | `src/bc_dataset.py`、`src/bc_model.py` |
| W7 | 4×4 网格 MDP：值迭代、表格 Q-learning、终止语义与 reward hacking 对照 | `src/mdp_gridworld.py`、`scripts/run_w7.py` |
| 阶段 1 | CartPole PPO 最小闭环：随机 0/20 → PPO 20/20，含 checkpoint 重载验证 | `experiments/01_ppo_cartpole/` |
| 阶段 2 | ManiSkill 接口核对 + CPU/GPU 仿真后端实测（单环境慢 18 倍，1024 环境快 36 倍） | `experiments/02_maniskill_probe/` |
| 阶段 3 | PushCube PPO：19/20 成功；发现 return 是反向指标 | `experiments/03_ppo_pushcube/` |

**W2 的起点实验**：一个不看状态的随机策略和一个只看位置误差的比例控制策略，
在完全相同的 30 个起点/目标上，到达成功率分别是 0/30 和 30/30；而这个差距**完全由
一个常数（`Kp·DT`）决定**，不是"算法好不好"。

## 环境

按周选择环境：

| 用于 | 环境 | 依赖 |
|---|---|---|
| **W2–W4** | Windows + conda，Python 3.10.19（`pytorch` 环境） | NumPy + Matplotlib + pytest，不需要 GPU |
| **W5** | Ubuntu 24.04 + 系统 Python 3.12（venv） | mujoco 3.11.0 / gymnasium 1.4.0 / gymnasium-robotics 1.4.2 |
| **W6–W7** | 复用 W5 的 Ubuntu + Python 3.12 venv | W6 需要 CPU 版 PyTorch；W7 只用 NumPy + Matplotlib |
| **阶段 1** | 独立 `.venv-ppo`（系统 Python 3.12） | torch 2.15.0+cu132 / SB3 2.9.2 / gymnasium 1.4.0，见 `requirements-ppo.txt` |
| **阶段 2–3** | 独立 `.venv-maniskill`（系统 Python 3.12） | mani_skill 3.0.1 / sapien 3.0.3 / torch 2.15.0 / tensorboard；渲染需 Vulkan |

W2–W4：

```
conda activate pytorch
python -m pip install -r requirements.txt
```

W5 的装法见下文「W5」一节。**`mujoco` 必须 < 3.12**，否则 `gym.make` 会无信息报错。
W6 另需 CPU 版 PyTorch；W7 不需要新增依赖。

## 运行

**必须在仓库根目录运行**（`scripts/first_loop.py` 里的 `OUT = Path("outputs/first_loop")`
是相对路径，在 `scripts/` 里跑会把结果建到 `scripts/outputs/`）：

```powershell
python scripts/first_loop.py
```

参数化运行（对应指南附录A.2 的四个修改）：

```powershell
python scripts/first_loop.py --kp 0.5          # 修改1 增益扫描
python scripts/first_loop.py --dt 0.6          # 修改2 控制周期扫描
python scripts/first_loop.py --policy open     # 修改3 开环
python scripts/first_loop.py --noise 0.05      # 修改4 观测噪声（单位：米）
```

`--help` 看全部参数。非默认参数会自动落到 `outputs/first_loop/<标签>/`，不覆盖基准结果。

## 评估方法

| 判据 | 定义 | 文件 |
|---|---|---|
| `ever_success` | 100 步内**任意一步**误差 < 0.02 m | `outputs/first_loop/summary.csv` |
| `held_success` | **最后 5 步**误差全部 < 0.02 m | 同上 |
| `final_error_m` | 第 100 步执行完之后的误差（米） | 同上 |
| 误差曲线 | `next_error_m` 随 `step` 变化 | `outputs/first_loop/*_trace.csv` |

**报结果时必须同时报 `held_success` 和 `final_error_m`。** 只看成功率会漏掉"误差收敛到
什么程度"这一层信息——`Kp=0.5` 时成功率 4/30，而 `Kp=1.0` 时 30/30，两者的区别不在
代码，在步数和误差量级。

## W3：坐标变换与正运动学

`src/transforms.py` 是后续几周共用的底座：全部变换按 `T_target_source` 命名，
表示「把 source 系里的点变到 target 系」。复合法则是下标相邻消去。

```powershell
python -m pytest tests/ -q          # 12 个测试函数 20 个用例（transforms 9 + jacobian 11）
python scripts/camera_to_end.py     # P4 验收题：相机 → 基座 → 末端
python scripts/w3_p4_visualize.py   # 出乘法链的图
```

`src/two_link.py` 是平面二连杆正运动学加工具偏移。`scripts/w3_p3_visualize.py` 出三姿态图，
`scripts/w3_d2_visualize.py` 出 D2 的四张图。

教程见 `notes/W3_坐标变换与正运动学_教程.md`（同目录有离线 HTML 版）。

## W4：雅可比、奇异位形与单关节 PD

`src/jacobian.py` 提供解析雅可比、`det_ana`、`cond_num`、`ik_velocity`；
`src/pd_control.py` 是单关节 PD 仿真（含阻尼 `b` 与常值扰动 `τ_g`）。

```powershell
python scripts/w4_d1_visualize.py   # 雅可比两列的几何含义
python scripts/w4_d2_visualize.py   # 奇异位形、逆解爆炸
python scripts/w4_d3_visualize.py   # Kp/Kd 扫描与 ζ 地图
```

关键结论（细节见 `notes/W4_奇异位形.md`、`notes/W4_PD结论.md`）：

- **`det(J) = l₁l₂sin(θ₂)`**：奇异只取决于 `θ₂`，即 `θ₂ = 0`（伸直）或 `π`（折叠）。
  实测 `det_ana(1.2) = 0.932039`，与 `sin(1.2)` 一致。
- **`det` 看不出"离奇异多近"，条件数才行**：固定 `θ₁=0`，`θ₂` 从 `π/2` 降到 `1e-4`，
  `cond(J)` 从 **2.62** 涨到 **5.0e4**；`θ₂ = 0` 时 `inf`。
- **单关节 PD 是二阶系统，只有 P 会一直振荡**（像没阻尼的弹簧），必须加 D。
  W2 的一阶模型只要 P 就够，这是两者的本质区别。
- **`ζ = (Kd + b) / (2√(Kp·I))`，`Kp` 在分母的平方根里**：`Kp` 涨 4 倍，`ζ` 掉一半。
  实测 `Kp = 25/100/400` → `ζ = 2.15/1.08/0.54`，超调 `0 / 0 / 12.84%`。
  **含 `b` 的版本才对得上实测**（不含 `b` 时理论超调 16.5%，明显偏大）。
- **纯 PD 在有常值扰动时静差 = `τ_g/Kp`，永不为 0**——这是 PID 里 I 项存在的理由。

> ⚠️ 模型是 `I·q̈ = Kp(q_d−q) + Kd(0−q̇) − b·q̇ + τ_g`，**忽略了 `M(q)`、`C(q,q̇)q̇` 和真实重力项 `g(q)`**。
> 用来把 PD 本身看清楚，不是机器人仿真。写报告必须标注这一点。

## W5：FetchReach 接口与位置基线

环境：Ubuntu 24.04 + **系统 Python 3.12**（`/usr/bin/python3.12 -m venv .venv`，不用 conda 的 3.14），
mujoco 3.11.0 / gymnasium 1.4.0 / gymnasium-robotics 1.4.2。**mujoco 不能升到 ≥3.12**：
`mjtJoint` 变成 pybind11 enum 后 `gym.make("FetchReach-v4")` 直接无信息 `AssertionError`。

```bash
./.venv/bin/python scripts/check_env.py                 # 打印接口事实（填核对表用）
./.venv/bin/python scripts/fetchreach_baseline.py       # 断言 + 单轴 + 扫 k + 三策略 20 回合
./.venv/bin/python scripts/collect_trajectories.py      # 20 条专家轨迹 → data/raw/*.npz
./.venv/bin/python scripts/verify_trajectories.py       # 6 项逐项核对（--all 查 20 条）
./.venv/bin/python scripts/plot_w5.py                   # 两张图 → outputs/w5/
```

关键实测结论（细节见 `notes/W5_接口核对表.md`、`notes/W5_逐项核对记录.md`）：

- 动作 4 维是「末端目标位移比例」，内部 `mocap += 0.05·a[:3]`；第 4 维抓手完全无效。
- **命令量 ≠ 实际位移**：跟踪比 r ≈ 0.60–0.65，随指令幅值增大略降。
- 50 步上限是 `truncated`，`terminated` 恒 False；`is_success` 是**当步**口径。
- 容差 0.05 m 偏宽松：k=1..100 的 `success_ever` 都是 20/20，靠首次达标步数区分
  （k=50 最好，2.55 步；k=100 越过 `0.05·r·k<2` 后进入 0.02–0.03 m 极限环）。
- 个别目标 z < ≈0.417 m 在初始姿态下不可达，残差约 2 cm（工作空间限制，非增益问题）。

## W6：行为克隆（BC）对照实验

```bash
./.venv/bin/python scripts/train_bc.py --sanity            # 32 样本过拟合检查（loss 7.4e-7）
./.venv/bin/python scripts/train_bc.py --data-size 20      # 15 训练 / 5 验证，3 个种子
./.venv/bin/python scripts/train_bc.py --data-size 40      # 35 训练 / 5 验证
./.venv/bin/python scripts/eval_closed_loop.py             # 8 个模型 × 20 测试回合（seed 100–119）
./.venv/bin/python scripts/plot_w6.py                      # 两张图 → outputs/w6/
```

输入 8 维（相对位置 3 + 末端速度 `obs[5:8]` 3 + 夹爪 `obs[3:5]` 2），输出 3 维动作，抓手固定 0。
报告见 `notes/W6_BC对照报告.md`，**零基础讲义见 `notes/W6_BC讲义.md`**（含读图方法、局限、自测题、汇报预案）。

关键结论：

- 数据偏置：专家 2.5 步就到目标，而回合跑满 50 步 → **94.7% 样本是近零动作**，
  距目标 ≥5 cm 的只有 93 个（40 条轨迹）。
- 5 cm 判据下**反馈专家 / 线性回归 / MLP-BC 全部 20/20**，该基准已饱和、没有区分力。
- 判据收紧到 **2 mm** 才分得开：线性回归 1/20，MLP 19/20，专家 19/20
  → 专家主体近似线性，非线性集中在 clip 饱和区。
- 数据 20→40 条让验证 loss 减半（0.15→0.07），闭环指标不变：瓶颈是分布，不是数量。

## W7：网格 MDP 与表格 Q-learning

本周不碰 MuJoCo，在 4×4 网格上实现纯 NumPy 的环境、值迭代和表格 Q-learning：

```bash
./.venv/bin/python scripts/run_w7.py       # 跑全部实验，生成 CSV + 三张图 → outputs/w7/
./.venv/bin/python -m pytest -q            # 全仓 32 个用例
```

环境有 16 个状态、4 个动作，洞和终点为终止状态，另有步数截断；奖励只在到达终点时为 1，折扣因子 `γ=0.99`。实验包含三组对照：

- **D2 有模型 vs 无模型**：值迭代与 Q-learning 比较起点价值；
- **C1 终止语义**：`truncated` 是否仍 bootstrap；
- **C2 奖励设计**：稀疏终点奖励 vs “变近 +0.1”的距离塑形奖励。

关键结果：

- 值迭代：不滑 **7 轮 / V(0)=0.950990**，滑面 **704 轮 / V(0)=0.542026**。
- Q-learning：不滑三种子 `Q0=0.950990`，纯贪心 500/500；滑面 `Q0` 三种子均值 **0.5264**，与值迭代均值误差 **0.0157**，纯贪心成功率 **366/500**。
- 终止语义错误：滑面 `Q0` 均值从 **0.5664** 降到 **0.1684**，仅为参考值的 **31.1%**，但 H=20 成功率没有同步归零。
- 奖励设计错误：训练不折扣回报从 **0.4167** 升到 **1.1877**，任务成功率却从 **52.8%** 降到 **0%**。

报告见 `notes/W7_RL报告.md`，**零基础 + 专业双轨讲义见 `notes/W7_RL讲义.md`**，PPO/扩散策略概念见 `notes/W7_D4_PPO与扩散概念.md`。图和原始数据在 `outputs/w7/`。

## 阶段 1：CartPole PPO 最小闭环

按《具身智能实验与实践学习计划（修订版）》第 3 节执行，独立环境 `.venv-ppo`
（Python 3.12 + torch 2.15.0+cu132 + SB3 2.9.2），**不动 W5–W7 的 `.venv`**：

```bash
./.venv-ppo/bin/python experiments/01_ppo_cartpole/ppo_cartpole.py          # 100k 步，约 340 秒
./.venv-ppo/bin/python experiments/01_ppo_cartpole/ppo_cartpole.py --timesteps 3000 --n-eval 3 --tag smoke
./.venv-ppo/bin/python experiments/01_ppo_cartpole/ppo_cartpole.py --timesteps 20000 --device cuda --tag gpu_bench
```

产物在 `experiments/01_ppo_cartpole/outputs/`：曲线、逐回合评估 CSV、训练前后视频、
`model.zip`、`config.json`。

关键结论（详见 [`experiments/01_ppo_cartpole/report.md`](experiments/01_ppo_cartpole/report.md)）：

- 随机策略 **21.0 ± 8.6（0/20）**，PPO **500.0 ± 0.0（20/20）**，从 checkpoint 重载后逐回合结果**完全一致**。
- **标准差 0.0 是撞顶造成的，不是策略稳定**：500 是 `max_episode_steps` 截断上限，
  20 回合全撞顶就失去区分力——与 W6「5 cm 判据饱和」是同一类陷阱。
- 曲线末端滑动平均 421 ≠ 评估 500：窗口 20 仍包含 82.7k 步稳定之前的低分回合，
  **训练期指标与评估期指标口径不同，报告时两个都要给**。
- 40k–80k 之间经历约 3 次崩塌-恢复，**"某次评估 500"不能当作收敛证据**，要看连续多回合。
- PPO 跑 CPU：`MlpPolicy` 仅约 4.6k 参数，GPU 的 kernel 启动开销大于计算本身；
  GPU 能力已在阶段 0 单独验收（`capability (12,0)` + 反向传播通过 + 峰值 288 MB）。

## 阶段 2–3：ManiSkill 与 PushCube PPO

从阶段 2 起，实验代码不再放进 `scripts/`，统一收进 `experiments/`：每个阶段自带 README、
配置、运行封装与产物，避免「这个脚本该用哪个 venv」变得不可查。
总索引见 [`experiments/README.md`](experiments/README.md)。

```bash
# 阶段 2：接口核对 + CPU/GPU 后端实测
./.venv-maniskill/bin/python experiments/02_maniskill_probe/probe_env.py
./.venv-maniskill/bin/python experiments/02_maniskill_probe/bench_backend.py

# 阶段 3：PushCube PPO（官方 baseline，配置驱动，每次 run 留 meta.json）
./.venv-maniskill/bin/python experiments/03_ppo_pushcube/run.py --config smoke
./.venv-maniskill/bin/python experiments/03_ppo_pushcube/run.py --config train
./.venv-maniskill/bin/python experiments/03_ppo_pushcube/evaluate.py \
  --checkpoint experiments/03_ppo_pushcube/runs/<run>/final_ckpt.pt --n-episodes 20
```

关键结论：

- **ManiSkill 该不该用 GPU，取决于并行数**：单环境 GPU 比 CPU 慢 18 倍，1024 环境快 36.5 倍，
  分界线在 `num_envs ≈ 32–64`；8 GB 显存跑 1024 个 PushCube 只占 1.53 GB。这与 CartPole PPO 的结论不矛盾——
  那边瓶颈是 CPU 串行采样，这边瓶颈是物理仿真本身。
- **PushCube 的 return 是反向指标**：成功的 19 个回合 return 只有 3.8–4.9，唯一失败的回合是 23.4。
  因为 `normalized_dense` 是势函数变化，失败策略熬满 50 步反而累积更多奖励。**必须按 `success` 判定。**
- 独立评估 **19/20（95%）**，但 20 回合的 Wilson 置信区间是 **[76%, 99%]**；训练内 8 回合评估的
  1.0 → 0.625 跳变更可能是抽样噪声，而不是策略崩了。
- 这属于「在官方 baseline 上完成自己的训练与评估」，**不等于复现官方成功率**——
  官方 README 没给 PushCube 的百分比。

详细分析见 [`experiments/03_ppo_pushcube/report.md`](experiments/03_ppo_pushcube/report.md)。

## 目录

```
configs/        参数（暂未使用；W5 的参数仍写在脚本里）
src/            可复用实现：transforms.py、two_link.py、jacobian.py、pd_control.py、
                bc_dataset.py、bc_model.py、mdp_gridworld.py
scripts/        W2–W7 的入口脚本，按周分组（阶段实验不放这里）：
                W2  first_loop.py、plot_error_curves.py、plot_position_curves.py、summarize_runs.py
                W3  camera_to_end.py、w3_*.py
                W4  w4_*.py
                W5  w5_core.py、check_env.py、fetchreach_baseline.py、
                    collect_trajectories.py、verify_trajectories.py、plot_w5.py
                W6  train_bc.py、eval_closed_loop.py、plot_w6.py
                W7  run_w7.py
                另：md_to_standalone_html.py 是笔记转离线 HTML 的工具
experiments/    阶段实验（计划书第 2 节起），每个阶段自带 README、report、configs 与产物：
                01_ppo_cartpole/   CartPole PPO（.venv-ppo）
                02_maniskill_probe/ 接口核对与 CPU/GPU 后端实测（.venv-maniskill）
                03_ppo_pushcube/  PushCube PPO，含 run.py / evaluate.py 与官方 baseline 封装
tests/          pytest 单测：test_transforms.py（9 用例）、test_jacobian.py（11 用例）、
                test_mdp_gridworld.py（12 用例），pythonpath=. 由 pytest.ini 配置
data/raw/       原始数据（只读）：W5 的 20 条专家轨迹 feedback_seed*.npz
data/processed/ 处理后数据
outputs/        W2–W7 的实验结果：first_loop/ 按参数分子目录、w5/ 与 w6/ 放 CSV、模型和图、
                w7/ 放 W7 的 summary.csv、训练曲线、收敛记录和三张图、
                figures/ 放 W2–W4 的图
notes/          教程、手算记录、答辩笔记、周报、W7 报告与双轨讲义（阶段报告放各自 experiments 目录）
third_party/    第三方仓库（不进 git）：ManiSkill sparse checkout，供阶段 3 的官方 PPO baseline
```

## 说明

各周的模型假设不同，引用结论时不要跨周套用：

- **W2**：无惯性、无碰撞、无关节的二维点教学模型，动作单位 m/s，`DT` 是控制周期。
  输出全部来自数值模拟，**不能解释为真实机器人性能**（指南附录A 原文）。
- **W4**：单关节简化模型，忽略 `M(q)`、`C(q,q̇)q̇` 与真实重力项 `g(q)`，
  只用来观察 PD 本身的行为。
- **W5**：MuJoCo `FetchReach-v4`，动作是**末端目标位移比例**（内部 `mocap += 0.05·a[:3]`），
  不是关节力矩。跟踪比约 0.60–0.65，所以"命令量 ≠ 实际位移"。
- **W6**：在同一 FetchReach 数据上比较行为克隆，输入含相对位置、末端速度和夹爪状态；
  结论受专家线性和静止样本比例限制，不代表真实机器人泛化能力。
- **W7**：4×4 离散 MDP，纯 NumPy 教学环境；`V(0)` 是折扣期望回报，**不是成功率**。
  结论不能外推到连续控制、视觉策略或深度 RL。
