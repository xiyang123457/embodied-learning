# embodied-learning

具身智能学习仓库。W1–W8 逐步把「一个点」做成「一个能感知、能决策的系统」。

| 周 | 做了什么 | 目录 |
|---|---|---|
| W2 | 二维点 + 比例反馈：证明「随机 vs 反馈」的差距由 `Kp·DT` 决定 | `scripts/first_loop.py` |
| W3 | 坐标变换与二连杆正运动学：`T_target_source` 命名 + 下标消去 | `src/transforms.py`、`src/two_link.py` |
| W4 | 雅可比、奇异位形、单关节 PD 调参 | `src/jacobian.py`、`src/pd_control.py` |
| W5 | MuJoCo `FetchReach-v4` 接口核对、位置基线、专家轨迹采集 | `scripts/w5_core.py` 等 |

**W2 的起点实验**：一个不看状态的随机策略和一个只看位置误差的比例控制策略，
在完全相同的 30 个起点/目标上，到达成功率分别是 0/30 和 30/30；而这个差距**完全由
一个常数（`Kp·DT`）决定**，不是"算法好不好"。

## 环境

分两套，按周选：

| 用于 | 环境 | 依赖 |
|---|---|---|
| **W2–W4** | Windows + conda，Python 3.10.19（`pytorch` 环境） | NumPy + Matplotlib + pytest，不需要 GPU |
| **W5** | Ubuntu 24.04 + 系统 Python 3.12（venv） | mujoco 3.11.0 / gymnasium 1.4.0 / gymnasium-robotics 1.4.2 |

W2–W4：

```
conda activate pytorch
python -m pip install -r requirements.txt
```

W5 的装法见下文「W5」一节。**`mujoco` 必须 < 3.12**，否则 `gym.make` 会无信息报错。

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

## 目录

```
configs/        参数（暂未使用；W5 的参数仍写在脚本里，等 W6 训练阶段再抽出来）
src/            可复用实现：transforms.py、transforms_drill.py、two_link.py、
                jacobian.py、pd_control.py
scripts/        入口脚本，按周分组：
                W2  first_loop.py、plot_error_curves.py、plot_position_curves.py、summarize_runs.py
                W3  camera_to_end.py、w3_*.py
                W4  w4_*.py
                W5  w5_core.py、check_env.py、fetchreach_baseline.py、
                    collect_trajectories.py、verify_trajectories.py、plot_w5.py
                另：md_to_standalone_html.py 是笔记转离线 HTML 的工具
tests/          pytest 单测：test_transforms.py（9 用例）、test_jacobian.py（11 用例），
                pythonpath=. 由 pytest.ini 配置
data/raw/       原始数据（只读）：W5 的 20 条专家轨迹 feedback_seed*.npz
data/processed/ 处理后数据
outputs/        实验结果：first_loop/ 按参数分子目录、w5/ 放 CSV 与两张图、
                figures/ 放 W2–W4 的图
notes/          教程、手算记录、答辩笔记、周报
```

## 说明

三周的模型假设不同，引用结论时不要跨周套用：

- **W2**：无惯性、无碰撞、无关节的二维点教学模型，动作单位 m/s，`DT` 是控制周期。
  输出全部来自数值模拟，**不能解释为真实机器人性能**（指南附录A 原文）。
- **W4**：单关节简化模型，忽略 `M(q)`、`C(q,q̇)q̇` 与真实重力项 `g(q)`，
  只用来观察 PD 本身的行为。
- **W5**：MuJoCo `FetchReach-v4`，动作是**末端目标位移比例**（内部 `mocap += 0.05·a[:3]`），
  不是关节力矩。跟踪比约 0.60–0.65，所以"命令量 ≠ 实际位移"。
