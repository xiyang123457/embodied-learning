"""把 ACT 评估输出的 mp4v 视频转成 H.264，使其能在 IDE / 浏览器中播放。

背景（实测）：
  ACT 的 visualize_episodes.save_videos 用 cv2.VideoWriter + fourcc='mp4v' 写视频。
  本机 OpenCV 的 H264 fourcc 会走 v4l2 硬件编码器并失败，所以只能编出 mp4v；
  而 mp4v（MPEG-4 Part 2）不被 VS Code 的 <video> 预览和主流浏览器解码，
  表现为"在加载视频时出错"。另外 cv2 写出的 mp4 把 moov 放在文件末尾，
  流式播放器无法定位索引。

做法：
  用 ffmpeg 重编码为 libx264 (avc1) + yuv420p + faststart，原地替换。
  已经是 h264 的文件自动跳过，可重复执行。

用法：
  ./.venv-ppo/bin/python experiments/05_act_transfer_cube/fix_videos.py <目录或文件...>
  ./.venv-ppo/bin/python experiments/05_act_transfer_cube/fix_videos.py --dry-run <目录>
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def find_ffmpeg(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        pass
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise SystemExit(
        "找不到 ffmpeg。用 ./.venv-ppo/bin/python 运行（该环境装了 imageio-ffmpeg），"
        "或用 --ffmpeg 指定路径。"
    )


def codec_of(ffmpeg: str, path: Path) -> str:
    """用 ffmpeg -i 的输出来判断编码，够用即可。"""
    r = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)],
                       capture_output=True, text=True, errors="ignore")
    text = (r.stderr or "") + (r.stdout or "")
    low = text.lower()
    if "h264" in low or "avc1" in low:
        return "h264"
    if "mp4v" in low or "mpeg4" in low:
        return "mp4v"
    return "unknown"


def transcode(ffmpeg: str, src: Path, dst: Path) -> None:
    # 显式 -f mp4：临时文件名不以 .mp4 结尾，否则 ffmpeg 无法推断容器格式
    subprocess.run(
        [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(src),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
         "-movflags", "+faststart", "-f", "mp4",
         str(dst)],
        check=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="把 mp4v 视频转成 H.264")
    ap.add_argument("paths", nargs="*", type=Path, help="视频文件或目录；默认当前目录")
    ap.add_argument("--dry-run", action="store_true", help="只列出要处理的文件")
    ap.add_argument("--ffmpeg", default=None, help="显式指定 ffmpeg 可执行文件")
    args = ap.parse_args()

    ffmpeg = find_ffmpeg(args.ffmpeg)

    targets: list[Path] = []
    for p in (args.paths or [Path(".")]):
        if p.is_dir():
            targets += sorted(p.glob("*.mp4"))
        elif p.suffix.lower() == ".mp4":
            targets.append(p)

    if not targets:
        print("没有找到 mp4 文件")
        return 0

    fixed = skipped = failed = 0
    for src in targets:
        c = codec_of(ffmpeg, src)
        if c == "h264":
            skipped += 1
            continue

        before = src.stat().st_size / 1024**2
        if args.dry_run:
            print(f"[dry] {src.name:<16} {c:<7} {before:6.1f} MB")
            continue

        tmp = src.with_name(src.name + ".tmp")
        try:
            transcode(ffmpeg, src, tmp)
            tmp.replace(src)
            after = src.stat().st_size / 1024**2
            fixed += 1
            print(f"[ok] {src.name:<16} {c} -> h264  {before:6.1f} -> {after:5.1f} MB")
        except subprocess.CalledProcessError as e:
            tmp.unlink(missing_ok=True)
            failed += 1
            print(f"[fail] {src.name}: {e}", file=sys.stderr)

    print(f"\n修好 {fixed} 个，跳过 {skipped} 个（已是 h264），失败 {failed} 个")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
