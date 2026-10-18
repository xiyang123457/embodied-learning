# embodied-learning

具身智能学习仓库。W1–W8 逐步把「一个点」做成「一个能感知、能决策的系统」。

**本实验验证什么**：一个不看状态的随机策略和一个只看位置误差的比例控制策略，
在完全相同的 30 个起点/目标上，到达成功率分别是 0/30 和 30/30；而这个差距**完全由
一个常数（`Kp·DT`）决定**，不是"算法好不好"。

## 环境

Windows + conda，Python 3.10.19（`pytorch` 环境）。本实验只依赖 NumPy 和 Matplotlib，
不需要 GPU、不需要仿真器。

```
conda activate pytorch
python -m pip install -r requirements.txt
```

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

## 目录

```
configs/      参数（暂空，W5 起用）
src/          可复用实现（暂空）
scripts/      入口脚本：first_loop.py
tests/        小型自动检查
data/raw/     原始数据（只读）
data/processed/ 处理后数据
outputs/      实验结果，按参数分子目录
notes/        索引笔记、答辩笔记、周报
```

## 说明

本实验是**无惯性、无碰撞、无关节**的二维点教学模型，输出全部来自数值模拟，
不能解释为真实机器人性能（指南附录A 原文）。动作单位是 m/s，`DT` 是控制周期。
