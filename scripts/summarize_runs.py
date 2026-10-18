"""汇总 outputs/first_loop 下所有配置目录的 summary.csv，打印成一张对照表。

用途
    把 A.2 四个修改的所有运行结果并排放，避免在 14 个目录里手工抄数字。
输入
    无参数。扫描 outputs/first_loop/ 下所有含 summary.csv 的子目录（含基准目录本身）。
输出
    stdout 一张 Markdown 表；同时写 outputs/first_loop/A4_summary.md。
    列：tag / policy / n / ever / held / mean_final_error_m
    tag = 目录名（基准目录显示为 baseline）。
"""

from pathlib import Path
# pathlib：跨平台路径拼接，用 rglob 找 summary.csv。

import csv
# csv：读 summary.csv。数值列读出来是字符串，必须自己 float()。

import numpy as np
# numpy：取均值。只有一行统计，也可以用 statistics，但项目里已经在用 numpy。

BASE = Path("outputs/first_loop")
# BASE：Path，基准输出目录（相对路径，必须在仓库根目录运行）
#   为什么相对路径：与 first_loop.py 保持一致，避免两套路径规则。


def collect():
    """返回 {(tag, policy): [ever, held, final_error, n]}，直接按 tag 排序。"""
    rows = {}
    # rows：dict，键是 (目录名, 策略名)，值是 [ever 合计, held 合计, 误差列表]
    #   为什么用 dict 而不是先拼 list：同一目录里可能有两个策略（基准目录就是 random+feedback）。
    for csv_path in sorted(BASE.rglob("summary.csv")):
        tag = csv_path.parent.relative_to(BASE).as_posix()
        if tag == ".":
            tag = "baseline"
        with csv_path.open(encoding="utf-8") as f:
            for rec in csv.DictReader(f):
                # DictReader：把表头映射成 key；所有值仍是 str，
                #   坑：直接对 "held_success" 求和会把字符串拼接起来。
                key = (tag, rec["policy"])
                slot = rows.setdefault(key, [0, 0, []])
                slot[0] += int(rec["ever_success"])
                slot[1] += int(rec["held_success"])
                slot[2].append(float(rec["final_error_m"]))
    return rows


def main():
    rows = collect()
    lines = ["| tag | policy | n | ever | held | mean_final_error_m |",
             "|---|---|---|---|---|---|"]
    for (tag, policy), (ever, held, errs) in sorted(rows.items()):
        lines.append(f"| {tag} | {policy} | {len(errs)} | {ever} | {held} | "
                     f"{np.mean(errs):.4g} |")
        # :.4g 保留 4 位有效数字：1.5e-05 和 0.1973 才能在同一列里都看清。
    text = "\n".join(lines)
    print(text)
    out = BASE / "A4_summary.md"
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved to: {out.resolve()}")


if __name__ == "__main__":
    main()
