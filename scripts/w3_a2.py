"""W3 · D1 A2/A3：二维旋转平移的矩阵验算（非交付物，只用于验证手算）。

运行（必须在仓库根目录）：
    python scripts/w3_a2.py

验证两件事：
  1. A2  —— 正序（先转后平移）应得 (1,1)，反序（先平移后旋转）应得 (0,2)
  2. A3  —— 逆变换回原点、RᵀR=I 且 det=+1、两个顺序相乘结果不同
"""

import numpy as np

# ==== 1. 二维基本块 ====
# 逆时针 90°：(x, y) -> (-y, x)
R90 = np.array([[0.0, -1.0],
                [1.0,  0.0]])
I2 = np.eye(2)


def hom(R, t):
    """拼 3x3 齐次矩阵 [[R, t], [0, 1]]"""
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2] = t
    return T


# ==== 2. 两个顺序 ====
R_h = hom(R90, [0.0, 0.0])   # 纯旋转
T_t = hom(I2, [1.0, 0.0])    # 纯平移 +x 1m

# 矩阵从右往左作用：写在右边的先作用
T_pos = T_t @ R_h   # 正序：先转 90°，再平移 +x
T_neg = R_h @ T_t   # 反序：先平移 +x，再转 90°

p = np.array([1.0, 0.0, 1.0])   # 齐次坐标下的 (1, 0)

# ==== 3. A2：矩阵验算 ====
print("== A2 矩阵验算（对照手算） ==")
print("  正序 T_t @ R_h :", (T_pos @ p)[:2], "  手算 (1, 1)")
print("  反序 R_h @ T_t :", (T_neg @ p)[:2], "  手算 (0, 2)")
print("  cos(pi/2)     :", np.cos(np.pi / 2))
print("                   == 0 ?", np.cos(np.pi / 2) == 0, "  ← False 是正常的，是浮点 6.12e-17")

# ==== 4. A3：三个小实验 ====
print("\n== A3 三个实验 ==")
print("  1 逆变换回原点 :", (np.linalg.inv(T_pos) @ T_pos @ p)[:2])
print("  2 R^T R = I    :", np.allclose(R90.T @ R90, np.eye(2)),
      " det(R) =", round(float(np.linalg.det(R90)), 15))
print("  3 两序相乘结果 :")
print("     T_pos =", T_pos.tolist())
print("     T_neg =", T_neg.tolist())
print("     相等?       :", np.allclose(T_pos, T_neg), "  ← False 才对，矩阵乘法不交换")

# ==== 5. 自检 ====
checks = [
    ("正序 = (1, 1)",        np.allclose((T_pos @ p)[:2], [1.0, 1.0])),
    ("反序 = (0, 2)",        np.allclose((T_neg @ p)[:2], [0.0, 2.0])),
    ("cos(pi/2) 不是 0",     np.cos(np.pi / 2) != 0),
    ("逆变换回到 (1, 0)",    np.allclose((np.linalg.inv(T_pos) @ T_pos @ p)[:2], [1.0, 0.0], atol=1e-12)),
    ("R^T R = I",            np.allclose(R90.T @ R90, np.eye(2))),
    ("det(R) = +1",          np.allclose(np.linalg.det(R90), 1.0)),
    ("乘法不交换",           not np.allclose(T_pos, T_neg)),
]

print("\n== 自检 ==")
for name, ok in checks:
    print(("  PASS  " if ok else "  FAIL  ") + name)
print("\n全绿:", all(ok for _, ok in checks))
