"""阶段 2：ManiSkill 接口核对与渲染验证。

对应计划书第 4 节：
  - 在固定版本中确认 PushCube-v1 / PickCube-v1 的实际任务 ID 与可用控制模式
  - 记录观测字段、动作维度与含义、奖励、成功判定、episode 上限
  - 验证渲染链路（SAPIEN 走 Vulkan，与 W5 的 MuJoCo 是两套独立栈）
  - 保存一张渲染图作为"能启动并保存仿真画面"的证据

用法（仓库根目录）：
    ./.venv-maniskill/bin/python scripts/check_maniskill.py
    ./.venv-maniskill/bin/python scripts/check_maniskill.py --env PickCube-v1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# 首次运行会下载资产，禁止交互式提问以免脚本挂住
os.environ.setdefault("MS_SKIP_ASSET_DOWNLOAD_PROMPT", "1")

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

OUT = Path(__file__).resolve().parent / "outputs"


def tree(obj, depth: int = 0, max_depth: int = 3) -> dict:
    """把嵌套 dict 的 shape/dtype 摊平成可打印结构。"""
    if depth > max_depth:
        return {"_": type(obj).__name__}
    if isinstance(obj, dict):
        return {k: tree(v, depth + 1, max_depth) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return {"shape": list(obj.shape), "dtype": str(obj.dtype)}
    if torch_is_tensor(obj):
        return {"shape": list(obj.shape), "dtype": str(obj.dtype), "device": str(obj.device)}
    return {"type": type(obj).__name__, "value": str(obj)[:80]}


def torch_is_tensor(obj) -> bool:
    try:
        import torch

        return isinstance(obj, torch.Tensor)
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="ManiSkill 接口核对")
    ap.add_argument("--env", type=str, default="PushCube-v1")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    import mani_skill
    import sapien
    import torch

    print("=" * 70)
    print(f"mani_skill {mani_skill.__version__} | sapien {sapien.__version__} | "
          f"torch {torch.__version__} | cuda {torch.cuda.is_available()}")
    print("=" * 70)

    # ---------- 1. state 模式：纯粹的接口事实
    env = gym.make(args.env, obs_mode="state", render_mode=None)
    u = env.unwrapped

    facts: dict = {
        "mani_skill_version": mani_skill.__version__,
        "sapien_version": sapien.__version__,
        "torch_version": torch.__version__,
        "env_id": env.spec.id if env.spec else args.env,
    }

    for attr in ("obs_mode", "control_mode", "reward_mode", "sim_backend", "num_envs"):
        if hasattr(u, attr):
            facts[attr] = str(getattr(u, attr))
    # episode 上限挂在 spec 上，不在 unwrapped 上
    facts["max_episode_steps"] = getattr(env.spec, "max_episode_steps", None)

    print(f"环境 ID            : {facts['env_id']}")
    print(f"obs_mode           : {facts.get('obs_mode')}")
    print(f"control_mode       : {facts.get('control_mode')}")
    print(f"reward_mode        : {facts.get('reward_mode')}")
    print(f"sim_backend        : {facts.get('sim_backend')}")
    print(f"max_episode_steps  : {facts.get('max_episode_steps')}")
    print(f"num_envs           : {facts.get('num_envs')}")
    print(f"action_space       : {env.action_space}")
    if hasattr(env.action_space, "shape"):
        facts["action_shape"] = list(env.action_space.shape)

    # 可用控制模式（ManiSkill 的 agent 类自带支持列表）
    try:
        robot_uids = str(getattr(u, "robot_uids", "?"))
        facts["robot_uids"] = robot_uids
        print(f"robot_uids         : {robot_uids}")
        if hasattr(u, "agent") and hasattr(u.agent, "supported_control_modes"):
            modes = list(u.agent.supported_control_modes)
            facts["supported_control_modes"] = modes
            print(f"支持的控制模式     : {modes}")
    except Exception as e:  # noqa: BLE001
        print(f"[提示] 控制模式枚举失败: {e}")

    # ---------- 2. 随机动作跑几步，看观测/奖励/成功判定
    obs, info = env.reset(seed=0)
    facts["obs_structure"] = tree(obs)
    facts["info_keys"] = sorted(info.keys())
    print("\n观测结构:")
    print(json.dumps(facts["obs_structure"], ensure_ascii=False, indent=2)[:1200])
    print(f"info keys: {facts['info_keys']}")

    rng = np.random.default_rng(0)
    steps = 0
    total_reward = 0.0
    last_info = {}
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(np.asarray(reward).reshape(-1)[0])
        steps += 1
        last_info = info
        if bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0]):
            break

    facts["random_steps"] = steps
    facts["random_total_reward"] = total_reward
    facts["info_after_step"] = {
        k: (str(v)[:120] if not isinstance(v, (int, float, bool)) else v)
        for k, v in last_info.items()
    }
    print(f"\n随机动作 {steps} 步: 累计奖励 {total_reward:.3f}")
    print(f"step 后的 info: {json.dumps(facts['info_after_step'], ensure_ascii=False)}")
    env.close()

    # ---------- 2b. state_dict 模式：把 35 维扁平向量拆成可读字段
    try:
        denv = gym.make(args.env, obs_mode="state_dict", render_mode=None)
        dobs, _ = denv.reset(seed=0)
        facts["state_dict_structure"] = tree(dobs, max_depth=4)
        print("\nstate_dict 观测字段:")
        print(json.dumps(facts["state_dict_structure"], ensure_ascii=False, indent=2)[:1500])
        denv.close()
    except Exception as e:  # noqa: BLE001
        facts["state_dict_error"] = f"{type(e).__name__}: {e}"
        print(f"[提示] state_dict 模式不可用: {e}")

    # ---------- 2c. 动作语义：每一维到底控制什么
    print("\n" + "-" * 70)
    print("动作语义探测（其余维固定为 0，只动一维，看 qpos 变化）")
    try:
        import torch as _t

        aenv = gym.make(args.env, obs_mode="state_dict", render_mode=None)
        robot = aenv.unwrapped.agent.robot

        def qpos_delta(vec: np.ndarray) -> list[float]:
            aenv.reset(seed=0)
            q0 = robot.get_qpos().clone()
            a = _t.as_tensor(vec, dtype=q0.dtype, device=q0.device).reshape(1, -1)
            aenv.step(a)
            return [round(float(x), 5) for x in (robot.get_qpos() - q0)[0].cpu().numpy()]

        acts = {
            "all_zero": np.zeros(8),
            "joint0_+1": np.array([1.0] + [0.0] * 7),
            "joint1_+1": np.array([0.0, 1.0] + [0.0] * 6),
            "gripper_+1": np.array([0.0] * 7 + [1.0]),
            "gripper_-1": np.array([0.0] * 7 + [-1.0]),
        }
        deltas = {k: qpos_delta(v) for k, v in acts.items()}
        facts["action_semantics_qpos_delta"] = deltas
        for k, v in deltas.items():
            nz = {i: x for i, x in enumerate(v) if abs(x) > 1e-6}
            print(f"  {k:<12} -> qpos 变化 {nz if nz else '全部为 0'}")
        aenv.close()
    except Exception as e:  # noqa: BLE001
        facts["action_semantics_error"] = f"{type(e).__name__}: {e}"
        print(f"[提示] 动作语义探测失败: {e}")

    # ---------- 3. 渲染验证（Vulkan）
    print("\n" + "-" * 70)
    print("渲染验证（obs_mode=rgb, render_mode=rgb_array）")
    render_ok = False
    try:
        renv = gym.make(args.env, obs_mode="rgb", render_mode="rgb_array")
        obs, info = renv.reset(seed=0)
        frame = renv.render()
        # ManiSkill 的 render() 返回的是 CUDA tensor，必须先搬回 host
        if torch_is_tensor(frame):
            frame = frame.cpu().numpy()
        arr = np.asarray(frame)
        if arr.ndim == 4:  # (num_envs, H, W, 3)
            arr = arr[0]
        img_path = OUT / f"{args.env}_rgb.png"
        import imageio.v2 as imageio

        imageio.imwrite(img_path, arr.astype(np.uint8))
        render_ok = True
        facts["render_image"] = str(img_path)
        facts["render_shape"] = list(arr.shape)
        print(f"渲染成功: {img_path} shape={arr.shape}")
        print(f"obs(rgb) 结构: {json.dumps(tree(obs), ensure_ascii=False)[:400]}")
        renv.close()
    except Exception as e:  # noqa: BLE001
        facts["render_error"] = f"{type(e).__name__}: {e}"
        print(f"[失败] 渲染不可用: {type(e).__name__}: {e}")
        print("排查方向: vulkaninfo / nvidia_icd.json / EGL 库 / 无头会话设置")

    facts["render_ok"] = render_ok

    # ---------- 4. 录制随机动作视频（阶段门槛："可以重置场景并录制运动"）
    print("\n" + "-" * 70)
    print("录制随机动作视频（最多 50 步）")
    try:
        venv = gym.make(args.env, obs_mode="rgb", render_mode="rgb_array")
        venv.reset(seed=0)
        frames: list[np.ndarray] = []
        total_r = 0.0
        succ = False
        for _ in range(50):
            obs, r, te, tr, info = venv.step(venv.action_space.sample())
            total_r += float(np.asarray(r).reshape(-1)[0])
            f = venv.render()
            if torch_is_tensor(f):
                f = f.cpu().numpy()
            f = np.asarray(f)
            if f.ndim == 4:
                f = f[0]
            frames.append(f.astype(np.uint8))
            succ = bool(np.asarray(info["success"]).reshape(-1)[0]) or succ
            if bool(np.asarray(te).reshape(-1)[0]) or bool(np.asarray(tr).reshape(-1)[0]):
                break
        vid = OUT / f"{args.env}_random.mp4"
        imageio.mimsave(vid, frames, fps=20, macro_block_size=1)
        facts["random_video"] = str(vid)
        facts["random_video_frames"] = len(frames)
        facts["random_video_return"] = round(total_r, 4)
        facts["random_video_success"] = succ
        venv.close()
        print(f"  已保存 {vid}（{len(frames)} 帧，累计奖励 {total_r:.3f}，成功={succ}）")
    except Exception as e:  # noqa: BLE001
        facts["random_video_error"] = f"{type(e).__name__}: {e}"
        print(f"[失败] 录制失败: {e}")

    (OUT / f"{args.env}_facts.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n接口事实已保存: {OUT / f'{args.env}_facts.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
