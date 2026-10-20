"""W3 原理验证：一个变换的方向由它的名字决定。

用途
    只验证三件事，全部围绕同一个判断：这个变换是从哪个系到哪个系。
    (1) R_y 的 sin 符号位置，以及写反之后错成什么样
    (2) T_base_cam @ T_cam_obj 合法，T_cam_base @ T_cam_obj 非法，两者差多少
    (3) 便宜求逆 [[Rᵀ, -Rᵀt], [0,1]] 与 np.linalg.inv 一致，但与 T.T 不是一回事
    二连杆正运动学和工具偏移不在这里实现，那是本周 P3 的交付内容。

输入
    无。

输出
    三节打印加一节自检。

跑完应该看到的自检数字（Python 3.10.19 / numpy 2.2.6）
    R_y(90) @ (0,0,1)        = (1, 0, 0)
    错符号版 @ (0,0,1)        = (-1, 0, 0)
    正确复合                  = (1, 1, 0)
    错误复合                  = (0, 0, 0)      <- 看着合法，其实毫无意义
    max|inverse(T) @ T - I|   = 0
    max|inverse(T) - T.T|     = 1
"""

import numpy as np


def rot_y(t):
    """绕 y 轴转 t 弧度。全篇唯一一个 sin 符号反着放的矩阵（z→x→z 循环）。"""
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(t):
    """绕 z 轴转 t 弧度。二维 [[c,-s],[s,c]] 的直接推广。"""
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_T(R, t):
    """拼 4x4 齐次矩阵 [[R, t], [0, 1]]。第 4 维把「乘法加加法」压成纯乘法。"""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inverse(T):
    """便宜求逆 [[Rᵀ, -Rᵀt], [0,1]]。写法错成 -t 是最常见的 bug。"""
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


# ==== 1. R_y 的符号自检 ====
# 绕 y 转是 z→x→z 的循环，所以 z 轴正方向应该被转到 x 轴正方向
wrong_y = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
print("== 1. R_y 的符号自检 ==")
# round 到 6 位：cos(pi/2) 是 6.12e-17 不是 0，不 round 会盖住真正的分量
print("  R_y(90) @ (0,0,1)   =", np.round(rot_y(np.pi / 2) @ [0, 0, 1], 6))
print("  错符号版 @ (0,0,1)   =", np.round(wrong_y @ [0, 0, 1], 6))

# ==== 2. 下标消去：同一组矩阵，两种乘法 ====
# 相机装在基座右侧 1 m 且绕 z 转 90 度；物体在相机正前方 1 m
T_base_cam = make_T(rot_z(np.pi / 2), [1.0, 0.0, 0.0])
T_cam_obj = make_T(np.eye(3), [1.0, 0.0, 0.0])
right = T_base_cam @ T_cam_obj
# 下标 base←cam 接 cam←obj，中间 cam 消掉，结果 base←obj，合法
wrong = inverse(T_base_cam) @ T_cam_obj
# 下标 cam←base 接 cam←obj，中间对不上，这个乘积没有几何意义，但程序不报错

print("\n== 2. 下标消去 ==")
print("  T_base_cam @ T_cam_obj =", right[:3, 3])
print("  T_cam_base @ T_cam_obj =", wrong[:3, 3])

# ==== 3. 便宜求逆与转置的区别 ====
# 平移为 0 时两者相同；只要平移不为 0，T.T 就一定不等于 T 的逆
diff_inv = np.abs(inverse(T_base_cam) @ T_base_cam - np.eye(4)).max()
diff_t = np.abs(inverse(T_base_cam) - T_base_cam.T).max()
print("\n== 3. 便宜求逆 vs 转置 ==")
print(f"  max|inverse(T) @ T - I| = {diff_inv:.6g}")
print(f"  max|inverse(T) - T.T|   = {diff_t:.6g}")

# ==== 4. 自检 ====
checks = [
    ("R_y(90) 把 z 轴转到 x 轴", np.allclose(rot_y(np.pi / 2) @ [0, 0, 1], [1, 0, 0], atol=1e-12)),
    ("正确复合 = (1,1,0)", np.allclose(right[:3, 3], [1, 1, 0], atol=1e-12)),
    ("错误复合 = (0,0,0)", np.allclose(wrong[:3, 3], [0, 0, 0], atol=1e-12)),
    ("求逆往返 = 单位阵", diff_inv < 1e-15),
    ("T.T 不是逆", not np.allclose(inverse(T_base_cam), T_base_cam.T)),
]
print("\n== 4. 自检 ==")
for name, ok in checks:
    print(("  PASS  " if ok else "  FAIL  ") + name)
print("\n全绿:", all(ok for _, ok in checks))
