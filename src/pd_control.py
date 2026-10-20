"""单关节 PD 控制仿真：二阶系统，返回超调 / 上升时间 / 稳态误差。

用途
    W2 的附录A 是一阶系统（p ← p + DT·a），只有 P 就能单调收敛。
    真实关节有质量，是二阶系统，只有 P 会像没有阻尼的弹簧一样一直振荡，
    必须加 D 当刹车。本模块把这件事跑成三个数字。

模型（简化，写报告时必须标注这三条忽略）
        I·q̈ = Kp·(q_d − q) + Kd·(0 − q̇) − b·q̇ + τ_g
    忽略：惯性矩阵 M(q) 随位形变化、科氏力 C(q,q̇)q̇、真实重力项 g(q)。
    用简化模型是为了把 PD 本身看清楚，不是仿真机器人。

跑完应看到的自检数字（I=0.1, b=0.5, qd=1.0, dt=1e-3, T=2s, Kd=6.3）
    Kp=25   zeta=1.99  超调  0.00%   上升 0.606s
    Kp=100  zeta=1.00  超调  0.00%   上升 0.136s
    Kp=400  zeta=0.50  超调 12.73%   上升 0.034s
    Kd=2    zeta=0.32  超调 25.6%
    有 tau_g=-0.5 时稳态误差 = tau_g/Kp，Kp=100 约 0.005 rad

怎么验证跑对了
    python scripts/w4_d3_visualize.py
    控制台打印的六组指标与上面逐个对上。
"""

import numpy as np

I_DEFAULT, B_DEFAULT = 0.1, 0.5
QD_DEFAULT, DT_DEFAULT, T_DEFAULT = 1.0, 1e-3, 2.0
# I_DEFAULT：float，转动惯量，kg·m²。0.1 是个小型关节的量级，
#   关键不在准不准，而在让 ω_n = sqrt(Kp/I) 落在几十 rad/s，2 秒内能走完
# B_DEFAULT：float，粘性阻尼。设成 0 会让「Kd 该给多少」这个问题过于理想，
#   真实关节总有摩擦，保留 0.5 更接近实际
# DT_DEFAULT：float，积分步长 1 ms。显式欧拉要求 dt < 2/ω_n，
#   Kp=400 时 ω_n≈63，1 ms 留了 30 倍余量
# T_DEFAULT：float，仿真 2 秒。ts[-1] 之后取 0.2 s 算稳态误差


def simulate(kp, kd, tau_g=0.0, I=I_DEFAULT, b=B_DEFAULT, qd=QD_DEFAULT,
             q0=0.0, dt=DT_DEFAULT, T=T_DEFAULT):
    """跑一次仿真，返回 (ts, qs)。初速度取 0。"""
    n = int(round(T / dt))
    ts = np.arange(n + 1) * dt
    qs = np.empty(n + 1)
    qs[0] = q0
    q, qdot = q0, 0.0

    for k in range(n):
        tau = kp * (qd - q) + kd * (0.0 - qdot)
        # tau：float，控制力矩。P 项拉向目标，D 项按「当前速度」反向刹车。
        #   why D 的目标是 0 而不是 qd：D 项抑制的是速度本身，不是位置误差
        qddot = (tau - b * qdot + tau_g) / I
        # 半隐式欧拉：先更新速度再更新位置。
        #   为什么不用显式欧拉（先位置后速度）：同样步长下稳定性差一倍，
        #   Kp=400 时会数值发散，看起来像「控制器不稳」，其实是积分器不稳
        qdot += qddot * dt
        q += qdot * dt
        qs[k + 1] = q

    return ts, qs


def metrics(ts, qs, qd=QD_DEFAULT, tail=0.2):
    """返回 (超调百分比, 上升时间 s, 稳态误差 rad)。"""
    overshoot = max(0.0, float(qs.max() - qd) / qd) * 100.0
    # 超调：冲过目标多少。用 max(0, ...) 是因为过阻尼时从下方逼近，永远不过冲

    idx = np.argmax(qs >= 0.9 * qd)
    # argmax 返回第一个 True 的位置。若从未到过 0.9，返回 0
    rise = float(ts[idx]) if qs[idx] >= 0.9 * qd else float("nan")
    # 兜底判断：argmax 在「全 False」时会返回 0，那个 0 不是真正的到达时刻

    tail_mask = ts >= ts[-1] - tail
    ess = float(qd - qs[tail_mask].mean())
    # 稳态误差：最后 0.2 秒的均值与目标的差。
    #   为什么用工均值而不是最后一点：单点受数值抖动影响，而且「停在哪」本来就该看一段
    return overshoot, rise, ess


def zeta(kp, kd, I=I_DEFAULT, b=B_DEFAULT):
    """阻尼比。Kd 记在分子，Kp 在分母的平方根里，所以 Kp 变大必须同步加 Kd。

    返回值不套 float()：kp 传数组时这里也要能算，出图时用它扫一整条曲线。
    标量输入得到的是 np.float64，格式化打印和比较都不受影响。
    """
    return (kd + b) / (2.0 * np.sqrt(kp * I))
    # 分子是 Kd + b：粘性阻尼和 D 项干的是同一件事，都对速度施加反向力矩，
    #   所以算阻尼比时要把 b 加进去，否则理论值和实测会差 10%


def sweep(kp_list, kd_list, **kwargs):
    """对每个 (kp, kd) 跑一遍，返回 [(kp, kd, 超调, 上升, 稳态), ...]。"""
    rows = []
    for kp in kp_list:
        for kd in kd_list:
            ts, qs = simulate(kp, kd, **kwargs)
            rows.append((kp, kd) + metrics(ts, qs, kwargs.get("qd", QD_DEFAULT)))
    return rows
