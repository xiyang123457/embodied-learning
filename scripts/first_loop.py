"""附录A 第一个可运行实验：二维点的速度控制（无惯性、无碰撞、无关节的教学模型）。

用途
    复现《具身智能入门学习路线与实践指南》附录A 的 30 种子对照实验，并支持
    A.2 要求的四个修改（增益 / 控制周期 / 开环 / 观测噪声），全部用命令行参数切换，
    不需要改代码常量。

输入
    CLI 参数（全部可选，默认值 = 附录A 原文）：
        --kp      float  比例增益，默认 2.0
        --dt      float  控制时间间隔（秒），默认 0.05
        --policy  str    both / feedback / open，默认 both（原文行为：random + feedback）
        --noise   float  观测噪声标准差（米），默认 0.0
        --seeds   int    种子数，默认 30（即 seed 0..29）
        --tag     str    输出子目录名，默认按参数自动生成

输出
    outputs/first_loop/summary.csv            2*30 = 60 行
        policy, seed, ever_success, held_success, final_error_m
    outputs/first_loop/{policy}_trace.csv     seed=0 那一回合的 100 行轨迹
        step, px_m, py_m, gx_m, gy_m, ax_mps, ay_mps, next_error_m
    非默认参数时自动落到子目录，例如 outputs/first_loop/kp0.5/summary.csv。

跑完应看到的自检数字（30 种子 / 100 步 / 容差 0.02 m，numpy 2.2.6）
    python scripts/first_loop.py
        random: final-window success 0/30        （ever 可能是 1/30，正常）
        feedback: final-window success 30/30
        feedback 最终误差均值 1.51e-05 m
    python scripts/first_loop.py --kp 0.5       → held 4/30，  最终误差 4.29e-02 m
    python scripts/first_loop.py --kp 0.2       → held 0/30，  最终误差 1.97e-01 m
    python scripts/first_loop.py --dt 0.6       → held 30/30， 最终误差 0（Kp*DT=1.2 过冲但收敛）
    python scripts/first_loop.py --dt 1.0       → held 0/30，  最终误差 4.03e-01 m（Kp*DT=2 等幅振荡）
    python scripts/first_loop.py --policy open  → held 0/30，  最终误差 3.76 m
    python scripts/first_loop.py --noise 0.05   → ever 30/30 但 held 14/30，最终误差 1.51e-02 m
"""

from pathlib import Path
# pathlib：跨平台处理路径。OUT 是相对路径，必须在仓库根目录运行，否则输出会建到
# 调用者所在目录（在 scripts/ 里跑就会变成 scripts/outputs/，这是本仓库最容易踩的坑）。

import csv
# csv：写 summary.csv / trace.csv。不用 pandas 是因为附录A 只依赖 NumPy，保持零额外依赖。

import argparse
# argparse：把 hyperparameter 从代码常量提到命令行，A.2 的 17 组对照才不用改代码重跑。

import numpy as np
# numpy：唯一必需的数值库。随机数生成、向量运算、范数计算全部来自它。


# ==== 1. 常量与命令行参数 ====

DT, STEPS, TOL = 0.05, 100, 0.02
# DT：    float，控制时间间隔 = 0.05 秒（即 20 Hz）
#         为什么是 0.05：附录A 原文取值；它同时是"仿真步长"和"策略决策间隔"，
#         真实系统里这两个是分开的数（指南 5.3 节），附录A 是简化。
# STEPS： int，一回合固定 100 步 → 100 * 0.05 = 5 秒模拟时间
#         为什么固定步数而不是固定时长：比较两种策略时回合长度必须一致。
# TOL：   float，到达容差 = 0.02 米
#         为什么是 0.02：约等于一个指尖宽度，且这里是教学模型，不是精度指标。

OUT = Path("outputs/first_loop")
# OUT：   Path，基准输出目录（相对路径）
#         为什么用相对路径：附录A 原文如此；代价是强依赖"从仓库根目录运行"。
OUT.mkdir(parents=True, exist_ok=True)
# mkdir(parents=True, exist_ok=True)：创建多级目录，已存在不报错。
#   坑：不加 parents=True 时 outputs 不存在会抛 FileNotFoundError。


def parse_args():
    """解析命令行参数。返回值：argparse.Namespace，属性与 --xxx 同名。"""
    parser = argparse.ArgumentParser(description="附录A 到点实验（可参数化版本）")
    # 默认值 = 附录A 原文，保证"不带参数运行"与原文完全等价。
    parser.add_argument("--kp", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--policy", choices=["both", "feedback", "open"], default="both")
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def out_dir(args):
    """把参数映射成输出目录。

    默认参数 → outputs/first_loop（P1 要求的固定位置）
    非默认参数 → outputs/first_loop/<自动标签>，避免下一次运行覆盖上一次
    （指南 A.1 原文要求"做参数对照时先复制或更改输出目录"）。
    """
    parts = []
    # parts：list[str]，非默认参数的名字，如 ["kp0.5"] 或 ["dt0.6", "noise0.05"]
    #   为什么要拼标签：A.2 要跑 17 组，同名文件互相覆盖后无法回溯是哪组配置。
    if args.kp != 2.0:
        parts.append(f"kp{args.kp:g}")
    if args.dt != DT:
        parts.append(f"dt{args.dt:g}")
    if args.policy != "both":
        parts.append(args.policy)
    if args.noise != 0.0:
        parts.append(f"noise{args.noise:g}")
    tag = args.tag or "_".join(parts)
    return OUT / tag if tag else OUT


# ==== 2. 场景与策略 ====

def reset(seed):
    """生成一个回合的起点和目标。返回 (position, goal)，都是 shape (2,) 的 float64。"""
    rng = np.random.default_rng(seed)
    # rng：Generator，默认 PCG64 位生成器
    #   为什么不用 np.random.seed：default_rng 是 NumPy 新接口，不会污染全局随机状态，
    #   多个生成器可以并存（本函数和 run() 就各用一个）。
    position = rng.uniform(-0.5, 0.5, size=2)
    goal = rng.uniform(-0.5, 0.5, size=2)
    # position / goal：ndarray shape (2,)，单位米，取值 [-0.5, 0.5]
    #   例：array([0.123, -0.456])。为什么是 1 米见方的工作空间：
    #   附录A 原文；配合 DT=0.05 和 clip(±1.0)，200 步内一定能走到任何目标。
    return position, goal


def feedback(position, goal, rng):
    """比例控制：朝目标方向给速度指令。返回 shape (2,) 的 ndarray，单位 m/s。"""
    return 2.0 * (goal - position)
    # 为什么乘 2.0：比例增益 Kp，单位 1/s。误差 (米) × Kp (1/s) = 速度 (米/秒)。
    #   2.0 * 0.05 = 0.10 → 每步误差只收缩到 0.90 倍，100 步后 0.9^100 ≈ 2.7e-5。


def make_feedback(kp):
    """闭包：返回一个把 kp 固定进去的 feedback 函数，签名与 feedback(position, goal, rng) 一致。"""
    def _feedback(position, goal, rng):
        return kp * (goal - position)
    return _feedback


def random_action(position, goal, rng):
    """随机策略：不看状态，每步在 [-1, 1]^2 里均匀抽速度。返回 shape (2,) 的 ndarray。"""
    return rng.uniform(-1.0, 1.0, size=2)
    # 为什么是这个策略：作为"无信息"基线。每轴单步标准差 = 0.05/sqrt(3) ≈ 0.029 m，
    #   100 步独立累加 → 位置标准差 ≈ 0.29 m，远大于 0.02 m 容差，所以几乎不可能成功。


def open_loop(position, goal, rng):
    """开环策略：不看当前状态，永远朝同一方向给恒定速度。返回 array([0.5, 0.5])。"""
    return np.array([0.5, 0.5])
    # 为什么是 0.5：A.2 第三项修改指定的固定动作；0.5 m/s * 5 s = 2.5 m/轴位移，
    #   必然冲出工作空间，所以它"看起来能动"但换起点就全崩。


# ==== 3. 单回合仿真 ====

def run(policy, seed, dt=DT, sigma=0.0):
    """跑一个回合。返回 (ever, held, final_error, trace)。

    ever    : int，1 = 100 步内任意一步误差 < TOL
    held    : int，1 = 最后 5 步误差全部 < TOL
    final_error : float，第 100 步之后的误差（米）
    trace   : list，每行 [step, *before, *goal, *action, error]
    """
    position, goal = reset(seed)
    # 注意 rng 的种子加 10000：场景流和策略流必须分开
    #   否则"改动作"会连带改掉起点目标，30 个种子的场景就不一致了（答辩第 5 问）。
    rng = np.random.default_rng(seed + 10000)

    errors, trace = [], []
    # errors：list[float]，每步执行完之后的误差，长度 STEPS=100
    # trace： list[list]，喂给 csv.writer 的行，列序与表头严格对应

    for step in range(STEPS):
        before = position.copy()
        # before：ndarray (2,)，执行这一步之前的位置。必须 copy，
        #   否则 position 被重新赋值后 trace 里全是最终位置。

        # 观测噪声只加给"策略看到的东西"，误差仍然用真实 position 计算
        #   —— 这就是指南第 4 天说的"区分传感误差与系统状态"。
        #   sigma == 0 时不消耗 rng，保证与附录A 原文逐位一致。
        obs = before if sigma == 0.0 else before + rng.normal(0.0, sigma, size=2)

        action = np.asarray(policy(obs, goal, rng))
        # asarray：把 list / tuple 统一成 ndarray，下一步的 shape 检查才有意义
        #   坑：如果策略返回 (1, 2)，shape 检查能抓住；返回标量则会在减法时广播出 (2,)。

        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Action must be a finite 2D vector")
        # 为什么显式查：NaN/Inf 一旦进入积分，后面 100 步全是 NaN，
        #   np.any(errors < TOL) 会静默返回 False，看上去像"策略不行"而不是"代码有 bug"。

        action = np.clip(action, -1.0, 1.0)
        # clip：速度饱和，±1.0 m/s 是执行器上限
        #   坑：删掉这行 Kp=2.0 时动作最大能到 2.8 m/s，D1 的 1.5e-5 就对不上了。

        position = before + dt * action
        # 这一行就是"控制周期"本身：速度 (m/s) × 时间 (s) = 位移 (m)。
        #   改 dt 等于改"策略多久决策一次"，没有任何惯性项，所以位置直接跳过去。

        error = float(np.linalg.norm(goal - position))
        # norm：欧氏距离，单位米。注意是**走完这一步之后**的误差 → 列名 next_error_m。

        errors.append(error)
        trace.append([step, *before, *goal, *action, error])

    ever = int(np.any(np.array(errors) < TOL))
    held = int(np.all(np.array(errors[-5:]) < TOL))
    # ever 和 held 不是一回事：ever 是"中途碰到过"，held 是"最后站得住"。
    #   Kp=0.5 时两者都是 4/30（从没进过容差）；noise=0.05 时 ever 30/30 但 held 14/30
    #   （进去过，站不住）—— 只报一个数会得到相反结论。

    return ever, held, errors[-1], trace


# ==== 4. 写出 CSV ====

def save_csv(path, header, rows):
    """写 CSV。path: Path；header: list[str]；rows: list[list]。无返回值。"""
    with path.open("w", newline="", encoding="utf-8") as f:
        # newline=""：交给 csv 模块处理换行，否则 Windows 下每行之间会多一个空行。
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# ==== 5. 主循环 ====

def main():
    args = parse_args()

    # 运行开始先打印最终生效的配置（指南 5.2 节要求），失败时贴日志就能复现。
    print(f"config: kp={args.kp:g} dt={args.dt:g} policy={args.policy} "
          f"noise={args.noise:g}m seeds={args.seeds}")

    if args.policy == "both":
        # 原文顺序：random 在前，feedback 在后，与附录A 的打印顺序一致。
        policies = {"random": random_action, "feedback": make_feedback(args.kp)}
    elif args.policy == "open":
        policies = {"open": open_loop}
    else:
        policies = {"feedback": make_feedback(args.kp)}

    dest = out_dir(args)
    dest.mkdir(parents=True, exist_ok=True)

    summary = []
    # summary：list[list]，每行 [policy, seed, ever, held, final_error]，共 len(policies)*seeds 行

    for name, policy in policies.items():
        results = []
        for seed in range(args.seeds):
            ever, held, final_error, trace = run(policy, seed, dt=args.dt, sigma=args.noise)
            results.append(held)
            summary.append([name, seed, ever, held, final_error])
            if seed == 0:
                # 只存 seed=0 的轨迹：够画曲线，又不会把 outputs 撑到 30 倍。
                save_csv(dest / f"{name}_trace.csv",
                         ["step", "px_m", "py_m", "gx_m", "gy_m",
                          "ax_mps", "ay_mps", "next_error_m"], trace)
        print(f"{name}: final-window success {sum(results)}/{args.seeds}")

    save_csv(dest / "summary.csv",
             ["policy", "seed", "ever_success", "held_success", "final_error_m"], summary)
    print(f"Saved to: {dest.resolve()}")


if __name__ == "__main__":
    main()
