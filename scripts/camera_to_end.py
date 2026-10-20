"""W3 · P4：相机看到目标，算出它在末端坐标系里的位置。

用途
    这是《指南》5.4 节的验收原题，也是 W3 的毕业设计。
    给出相机到基座、末端到基座两个变换，算出目标相对末端的位置，
    再用逆变换把原坐标恢复回来。

    顺带把三个常被跳过的细节真的跑一遍：
    Q1  把 T_base_cam 写成 T_cam_base，结果会怎样
    Q2  inverse(T) 和 T.T 是一回事吗
    Q3  这个算例里哪些量是传感器测的、哪些是标定得到的

输入
    无。场景全部写死在 SCENE 里，数值特意选成能手算的。

输出
    控制台打印两段：正向换算、三个问题的验证，最后是自检结论。

跑完应看到的自检数字
    目标在基座系      (1.0, 1.0, 0.0)      手算：相机在 (1,0,0)，光轴转 90° 指向 +y
    目标在末端系      (0.5, 0.5, 0.0)      手算：减去末端 (0.5,0.5,0)
    Q1 错写法给出     (0.0, 0.0, 0.0)      恰好是基座原点，看着像答案
    Q2 |T^-1 - T.T|   0.5                 不是一回事，差的就是平移那一段
    往返误差          6.12e-17              浮点尾巴，用 allclose 判
"""

import sys
from pathlib import Path

import numpy as np

# 让 `python scripts/camera_to_end.py` 能找到 src 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.transforms import apply, compose, inverse, make_T, rot_z
# apply：把点送过去并返回 (N,3)，这里只用它复制一个原点。
# compose：按下标消去拼乘法链，是本脚本的主角。
# inverse：把 T_base_end 反过来变成 T_end_base。
# make_T / rot_z：由「位置 + 绕 z 的角度」拼出 4x4，避免手抄 16 个数。


# ==== 1. 场景 ====
# 所有数字选成 0、1、0.5，目的是让每一步都能在纸上核对。
CAM_POS = [1.0, 0.0, 0.0]
# CAM_POS：list[float]，相机原点在基座系里的位置，单位米
#   why 放在 +x 方向 1 米：这样「相机光轴指向哪」和「目标在哪」能直接用符号看出来
CAM_YAW = np.pi / 2
# CAM_YAW：float，相机绕 z 轴转过的角度，弧度
#   why 取 90°：转完之后相机的 +x 会指向基座的 +y，方向变化一眼可辨

OBJ_IN_CAM = [1.0, 0.0, 0.0]
# OBJ_IN_CAM：list[float]，目标在相机系里的位置
#   why (1,0,0)：正好在相机正前方 1 米，这是视觉系统最常报告的形式

END_POS = [0.5, 0.5, 0.0]
# END_POS：list[float]，末端原点在基座系里的位置
#   why (0.5,0.5)：与相机错开，避免两个坐标系重合导致错误被掩盖
END_YAW = 0.0
# END_YAW：float，末端绕 z 的转角
#   why 取 0：让末端与基座姿态一致，于是减法就能算出相对位置，手算最省事


# ==== 2. 两个已知变换 ====
# 命名规则：T_target_source 表示把 source 系里的点变到 target 系。
#   所以「相机在基座系里的位姿」叫 T_base_cam，不是 T_cam_base。

T_base_cam = make_T(rot_z(CAM_YAW), CAM_POS)
# T_base_cam：(4,4)，把相机系里的点变到基座系。由手眼标定得到
T_base_end = make_T(rot_z(END_YAW), END_POS)
# T_base_end：(4,4)，把末端系里的点变到基座系。由关节角经正运动学算出
T_cam_obj = make_T(np.eye(3), OBJ_IN_CAM)
# T_cam_obj：(4,4)，把目标变到相机系。由相机测量得到，只有平移没有旋转，
#   因为「目标位置」本身就是在相机系里表示的，不存在额外的姿态


# ==== 3. 第一步：相机系 -> 基座系 ====
# 下标 base←cam 接 cam←obj，中间的 cam 消掉，结果 base←obj，合法。
T_base_obj = compose(T_base_cam, T_cam_obj)
obj_in_base = apply(T_base_obj, [0.0, 0.0, 0.0])[0]
# apply(T, [0,0,0]) -> (1,3)：目标点的位置就是 T 的平移列，
#   因为坐标原点变换过去正好落在 T[:3,3] 上

# ==== 4. 第二步：基座系 -> 末端系 ====
# 下标 end←base 接 base←obj，结果 end←obj。这里必须用逆变换，
#   T_base_end 的方向是 base←end，要往反方向走就得先反过来。
T_end_obj = compose(inverse(T_base_end), T_base_obj)
obj_in_end = apply(T_end_obj, [0.0, 0.0, 0.0])[0]


# ==== 5. 三个问题的验证 ====
# Q1：把 T_base_cam 写成 T_cam_base 会怎样？
#   下标 cam←base 接 cam←obj，中间的 base 和 cam 对不上，乘积没有几何意义。
#   但 numpy 不会报错，它返回一个数字。
wrong = compose(inverse(T_base_cam), T_cam_obj)
wrong_pos = apply(wrong, [0.0, 0.0, 0.0])[0]

# Q2：inverse(T) 和 T.T 是一回事吗？
#   只有平移为 0 时才相等。T.T 会把平移挪到最后一列，语义变成「对偶变换」。
diff_transpose = float(np.abs(inverse(T_base_end) - T_base_end.T).max())

# Q3：哪些是传感器测的、哪些是标定得到的？见下方 CLASSES。

CLASSES = [
    ("T_base_cam", "手眼标定", "相机装好后离线标一次，之后视为常数"),
    ("T_base_end", "关节角 + 正运动学", "每走一步都要重算"),
    ("T_cam_obj", "相机测量", "每帧都变，且带检测噪声"),
    ("T_end_tool", "机械设计参数", "图纸给定，永远不变"),
]
# CLASSES：list[tuple[str, str, str]]，四个变换各自由什么决定
#   why 单独列出来：这四个量的「更新频率」和「误差来源」完全不同，
#       把它们当成同一类东西处理，就会在需要实时更新的地方用了标定值


# ==== 6. 打印与自检 ====

def main():
    np.set_printoptions(precision=6, suppress=True)

    print("== 正向：相机系 -> 基座系 ==")
    print("  T_base_obj = T_base_cam @ T_cam_obj")
    print(f"  目标在基座系 = {obj_in_base}")
    print("  手算核对：相机在 (1,0,0)，绕 z 转 90° 后相机的 +x 指向基座的 +y，")
    print("            所以物体在 (1,0,0) + (0,1,0) = (1,1,0)")

    print("\n== 反向：基座系 -> 末端系 ==")
    print("  T_end_obj = inverse(T_base_end) @ T_base_obj")
    print(f"  目标在末端系 = {obj_in_end}")
    print("  手算核对：末端在 (0.5,0.5,0) 且无旋转，所以 (1,1,0)-(0.5,0.5,0) = (0.5,0.5,0)")

    print("\n== Q1：写成 T_cam_base @ T_cam_obj 会怎样 ==")
    print(f"  得到 {wrong_pos}")
    print("  恰好是基座原点。不报错、不是 NaN，看着像个正常答案，所以最难查。")

    print("\n== Q2：inverse(T) 与 T.T 是一回事吗 ==")
    print(f"  max|inverse(T_base_end) - T_base_end.T| = {diff_transpose:.6g}")
    print("  平移为 0 时两者才相等。本例末端平移 (0.5,0.5,0)，差 0.5 米。")

    print("\n== Q3：四个变换各自由什么决定 ==")
    for name, src, note in CLASSES:
        print(f"  {name:<12} {src:<20} {note}")

    print("\n== 自检 ==")
    checks = [
        ("目标在基座系 = (1,1,0)", np.allclose(obj_in_base, [1.0, 1.0, 0.0], atol=1e-12)),
        ("目标在末端系 = (0.5,0.5,0)", np.allclose(obj_in_end, [0.5, 0.5, 0.0], atol=1e-12)),
        ("Q1 错写法 = (0,0,0)", np.allclose(wrong_pos, [0.0, 0.0, 0.0], atol=1e-12)),
        ("Q2 T.T 不等于 T^-1", diff_transpose > 1e-6),
        ("逆变换往返可恢复", np.allclose(
            apply(T_end_obj, apply(inverse(T_end_obj), obj_in_end)),
            obj_in_end, atol=1e-12)),
    ]
    for name, ok in checks:
        print(("  PASS  " if ok else "  FAIL  ") + name)
    print("\n全绿:", all(ok for _, ok in checks))


if __name__ == "__main__":
    main()
