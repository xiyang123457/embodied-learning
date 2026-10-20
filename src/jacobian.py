"""平面二连杆雅可比：解析解、有限差分、行列式、条件数。

用途
    雅可比回答一个问题：关节速度转一圈，末端速度变多少。
        ẋ = J(θ) · θ̇
    J 的第 j 列 = 只有第 j 个关节以 1 rad/s 转动、其他关节不动时，末端的线速度。
    本模块同时给出解析解和中心差分两个版本，后者用来验证前者。

输入输出
    全部函数接受 theta 为长度 2 的序列 (theta1, theta2)，单位弧度。
    J_ana / J_fd 返回 (2,2) ndarray，行是 x/y 速度分量，列是关节。
    det_ana 返回 float；cond_num 返回 float（奇异处为 inf）。

跑完应看到的自检数字（l1 = l2 = 1）
    J_ana((0, 0))            -> [[0, 0], [2, 1]]      伸直位形，x 方向速度为 0
    J_ana((pi/4, pi/2))      -> [[-0.707, 0.707], [0.707, 0.707]]
    max|J_ana - J_fd| < 1e-9（三个位形都是一般位形下的量级）
    det_ana((0, 1.2))        -> 0.932039              等于 sin(1.2)
    det_ana((0, 0))          -> 0                     伸直即奇异
    det_ana((0, pi))         -> 0                     折叠即奇异

怎么验证跑对了
    python -m pytest tests/test_jacobian.py -q
    约 6 个用例全绿。
"""

import numpy as np

from src.two_link import fk
# fk：正运动学，返回 (x, y) 的标量元组。有限差分要对它连续调四次。
#   why 不在本模块重写一遍 fk：差分的全部意义是「验证解析式和我自己的 fk 一致」，
#       如果这里另写一份，验的就是两个不相干的东西。


# ==== 1. 解析解 ====
# 对 W3 的 x = l1 cosθ1 + l2 cos(θ1+θ2) 逐列求偏导。
#   第一列：两项都含 θ1，所以两项都要导
#   第二列：θ2 只出现在第二项的 (θ1+θ2) 里，所以只有一项
#   「第二列写出来和第一列完全一样」是最常见的错，那是漏了这一点。

def J_ana(theta, l1=1.0, l2=1.0):
    """返回 (2,2) 解析雅可比。"""
    t1, t2 = float(theta[0]), float(theta[1])
    s1, c1 = np.sin(t1), np.cos(t1)
    s12, c12 = np.sin(t1 + t2), np.cos(t1 + t2)

    return np.array([
        [-l1 * s1 - l2 * s12, -l2 * s12],
        [l1 * c1 + l2 * c12, l2 * c12],
    ])
    # 第 0 行是 x 方向速度，第 1 行是 y 方向速度
    #   坑：把行和列写反的话，矩阵转置后依然是个合法 2x2，diff 检查也照样能过，
    #       但物理含义全错。所以测试里必须有 (0,0) 的代入手算做定海神针。


# ==== 2. 有限差分 ====
# 中心差分 [f(θ+ε) - f(θ-ε)] / (2ε)，误差 O(ε²)。
#   同样的代码量，比前向差分 [f(θ+ε) - f(θ)] / ε 的 O(ε) 多拿 6 个数量级精度。
#   ε 取 1e-6：再小会被浮点舍入吃掉（f(θ+1e-12) - f(θ) 全是噪声）。

def J_fd(theta, eps=1e-6, l1=1.0, l2=1.0):
    """返回 (2,2) 中心差分的数值雅可比。"""
    theta = np.asarray(theta, dtype=float)
    J = np.zeros((2, 2))

    for j in range(2):
        d = np.zeros(2)
        d[j] = eps
        # d：ndarray (2,)，只在第 j 个关节上加扰动，这就是「其他关节全部不动」
        plus = np.array(fk(theta[0] + d[0], theta[1] + d[1], l1, l2))
        minus = np.array(fk(theta[0] - d[0], theta[1] - d[1], l1, l2))
        J[:, j] = (plus - minus) / (2.0 * eps)
        # J[:, j] 整列覆盖：一列就是「只动第 j 个关节时末端的位移速度」
    return J


# ==== 3. 行列式与条件数 ====
# det(J) = l1·l2·sin(θ2)，简单到可疑，但可以自己推：
#   两列叉积 = l1l2[sin(θ1+θ2)cosθ1 - sinθ1 cos(θ1+θ2)] = l1l2·sin(θ2)
# 这个式子把「奇异」这件事压缩成一行：det = 0 等价于 sin(θ2) = 0，
#   也就是 θ2 = 0（伸直）或 θ2 = π（折叠）。

def det_ana(theta2, l1=1.0, l2=1.0):
    """返回解析行列式 l1·l2·sin(theta2)。"""
    return float(l1 * l2 * np.sin(float(theta2)))


def cond_num(theta, l1=1.0, l2=1.0):
    """返回 J 的条件数。奇异位形返回 inf，这是正常结果不是错误。"""
    return float(np.linalg.cond(J_ana(theta, l1, l2)))
    # np.linalg.cond -> float，最大奇异值 / 最小奇异值
    #   含义：输入误差被放大几倍。cond=500 表示关节角 0.1% 的误差会让末端速度偏 50%
    #   坑：θ2 恰好为 0 时返回 inf，numpy 不会抛异常，别以为算错了


def ik_velocity(xdot, theta, l1=1.0, l2=1.0):
    """速度级逆解：给定末端速度，返回关节速度 J⁻¹·ẋ。

    奇异附近会给出荒谬值（θ2=1e-4 时约 22360 rad/s），这是物理上不可执行，
    不是代码 bug。真实系统要加阻尼，见 notes/W4_奇异位形.md。
    """
    return np.linalg.solve(J_ana(theta, l1, l2), np.asarray(xdot, dtype=float))
    # solve 而不是 inv@：解线性方程组的数值稳定性更好，少一次显式求逆
