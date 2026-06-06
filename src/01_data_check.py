"""
煤巷顶板位移 —— 01 数据质量检查与字段标准化
=============================================
功能：
  1. 读取原始 CSV（自动检测 BOM / 编码）
  2. 中文字段 -> 标准英文字段映射
  3. 检查缺失值、重复行、字段类型
  4. 输出清洗后的标准字段数据（dataset_checked.csv）
  5. 生成数据质量检查报告（Markdown + CSV 附表）
  描述性统计使用 pandas / numpy 计算。
"""

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 将 src 目录加入 path，以便直接运行此脚本时也能 import config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    TABLE_DIR,
    REPORT_DIR,
    ensure_dirs,
)

# ---------------------------------------------------------------------------
# 中文字段 -> 标准英文字段 映射表
# ---------------------------------------------------------------------------
CN_TO_EN: dict[str, str] = {
    "围岩裂隙发育程度": "fracture_degree",
    "围岩裂隙裂隙发育程度": "fracture_degree",
    "煤层强度": "coal_strength",
    "底板强度": "floor_strength",
    "顶板强度": "roof_strength",
    "埋深": "depth",
    "毛宽": "width",
    "巷道宽度": "width",
    "锚杆支护面积": "bolt_area",
    "锚索密度": "anchor_density",
    "顶板位移": "roof_displacement",
}

# 标准字段列表（按顺序）
STANDARD_FIELDS = [
    "id",
    "fracture_degree",
    "coal_strength",
    "floor_strength",
    "roof_strength",
    "depth",
    "width",
    "bolt_area",
    "anchor_density",
    "roof_displacement",
]

# 需要做描述性统计的数值字段（排除 id）
NUMERIC_FIELDS = STANDARD_FIELDS[1:]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    # 确保输出目录存在
    ensure_dirs()

    # ---- 1. 读取原始数据 ----
    raw_path = DATA_RAW_DIR / "data_raw.csv"
    if not raw_path.exists():
        print(f"[错误] 找不到原始数据文件: {raw_path}")
        sys.exit(1)

    rows_raw: list[dict[str, str]] = []
    header_raw: list[str] = []
    encoding_used: str = "utf-8-sig"
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb2312"]:
        try:
            with open(raw_path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                header_raw = list(reader.fieldnames or [])
                rows_raw = list(reader)
            encoding_used = enc
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        print("[错误] 无法以任何已知编码读取原始数据文件。")
        sys.exit(1)

    print(f"[信息] 使用编码: {encoding_used}")
    print(f"[信息] 读取原始数据: {len(rows_raw)} 行, {len(header_raw)} 列")
    print(f"[信息] 原始字段: {header_raw}")

    # ---- 2. 字段映射 ----
    mapped_header: list[str] = []
    unmapped: list[str] = []

    for idx, col in enumerate(header_raw):
        col_stripped = col.strip()
        if idx == 0:
            mapped_header.append("id")
            continue
        en = CN_TO_EN.get(col_stripped)
        if en:
            mapped_header.append(en)
        else:
            unmapped.append(col_stripped)
            mapped_header.append(col_stripped)

    if unmapped:
        print(f"[警告] 以下字段未在映射表中找到，将保留原名: {unmapped}")

    # 重命名
    rows_renamed: list[dict[str, str]] = []
    for row in rows_raw:
        new_row: dict[str, str] = {}
        for old_key, new_key in zip(header_raw, mapped_header):
            val = row.get(old_key, "")
            if val is None:
                val = ""
            new_row[new_key] = val
        rows_renamed.append(new_row)

    # ---- 3. 检查标准字段是否齐全 ----
    current_fields = set(mapped_header)
    required_fields = set(STANDARD_FIELDS)
    missing_fields = required_fields - current_fields
    if missing_fields:
        print(f"[错误] 缺少必要字段: {missing_fields}")
        sys.exit(1)

    extra_fields = current_fields - required_fields
    if extra_fields:
        print(f"[信息] 存在额外字段: {extra_fields}")

    print(f"[信息] 标准化字段: {STANDARD_FIELDS}")
    print(f"[信息] 样本量: {len(rows_renamed)}")

    # ---- 4. 缺失值检查 ----
    missing_counts: dict[str, int] = {f: 0 for f in STANDARD_FIELDS}
    for row in rows_renamed:
        for f in STANDARD_FIELDS:
            val = row.get(f, "").strip()
            if val == "":
                missing_counts[f] += 1

    print("\n[缺失值统计]")
    for f in STANDARD_FIELDS:
        print(f"  {f}: {missing_counts[f]}")

    # ---- 5. 类型转换 ----
    rows_clean: list[dict[str, Any]] = []
    type_issues: list[str] = []
    for i, row in enumerate(rows_renamed, start=1):
        clean_row: dict[str, Any] = {}
        for f in STANDARD_FIELDS:
            val = row.get(f, "").strip()
            if val == "":
                clean_row[f] = None
                continue
            if f == "id":
                try:
                    clean_row[f] = int(val)
                except ValueError:
                    clean_row[f] = val
            else:
                try:
                    clean_row[f] = float(val)
                except ValueError:
                    type_issues.append(
                        f"  行 {i}, 字段 {f}: 无法转为数值 -> '{val}'"
                    )
                    clean_row[f] = None
        rows_clean.append(clean_row)

    if type_issues:
        print(f"\n[警告] 类型转换问题 ({len(type_issues)} 处):")
        for issue in type_issues[:20]:
            print(issue)

    # ---- 6. 重复行检查 ----
    # 6a. 完整重复行（含 id）
    seen_full: set[str] = set()
    dup_full_indices: list[int] = []
    for i, row in enumerate(rows_clean, start=1):
        key = "|".join(
            str(row.get(f, "")).strip() if row.get(f) is not None else ""
            for f in STANDARD_FIELDS
        )
        if key in seen_full:
            dup_full_indices.append(i)
        else:
            seen_full.add(key)

    # 6b. 去掉 id 的重复
    seen_noid: set[str] = set()
    dup_noid_indices: list[int] = []
    for i, row in enumerate(rows_clean, start=1):
        key = "|".join(
            str(row.get(f, "")).strip() if row.get(f) is not None else ""
            for f in STANDARD_FIELDS[1:]
        )
        if key in seen_noid:
            dup_noid_indices.append(i)
        else:
            seen_noid.add(key)

    print(f"\n[重复行检查]")
    print(f"  完整重复行（含 id）: {len(dup_full_indices)}")
    print(f"  去掉 id 后重复行: {len(dup_noid_indices)}")

    # ---- 7. 描述性统计（使用 pandas） ----
    print(f"\n[描述性统计]")
    df_clean = pd.DataFrame(rows_clean)
    df_numeric = df_clean[NUMERIC_FIELDS].apply(pd.to_numeric, errors="coerce")

    desc_stats: dict[str, dict[str, float]] = {}
    for f in NUMERIC_FIELDS:
        series = df_numeric[f].dropna()
        if len(series) == 0:
            print(f"  {f}: 无有效数据")
            continue

        stats = {
            "count": int(series.count()),
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std(ddof=1)),
            "skew": float(series.skew()),
            "kurt": float(series.kurt()),
        }
        desc_stats[f] = stats
        print(
            f"  {f}: n={stats['count']}, min={stats['min']:.4f}, "
            f"max={stats['max']:.4f}, mean={stats['mean']:.4f}, "
            f"median={stats['median']:.4f}, std={stats['std']:.4f}, "
            f"skew={stats['skew']:.4f}, kurt={stats['kurt']:.4f}"
        )

    # ---- 8. roof_displacement 专项统计 ----
    print(f"\n[roof_displacement 专项统计]")
    if "roof_displacement" in desc_stats:
        rd = desc_stats["roof_displacement"]
        print(f"  最小值: {rd['min']:.4f}")
        print(f"  最大值: {rd['max']:.4f}")
        print(f"  均值: {rd['mean']:.4f}")
        print(f"  中位数: {rd['median']:.4f}")
        print(f"  标准差: {rd['std']:.4f}")
        print(f"  偏度: {rd['skew']:.4f}")
        print(f"  峰度: {rd['kurt']:.4f}")
    else:
        print("  无有效数据")

    # ---- 9. 输出文件 ----
    # 9a. dataset_checked.csv
    checked_path = DATA_PROCESSED_DIR / "dataset_checked.csv"
    with open(checked_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(STANDARD_FIELDS)
        for row in rows_clean:
            writer.writerow(
                [
                    row.get(f, "") if row.get(f) is not None else ""
                    for f in STANDARD_FIELDS
                ]
            )
    print(f"\n[输出] 标准化数据: {checked_path}")

    # 9b. missing_value_summary.csv
    missing_path = TABLE_DIR / "missing_value_summary.csv"
    with open(missing_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["字段", "缺失数量", "缺失比例(%)"])
        total = len(rows_clean)
        for f in STANDARD_FIELDS:
            cnt = missing_counts[f]
            pct = round(cnt / total * 100, 2) if total > 0 else 0.0
            writer.writerow([f, cnt, pct])
    print(f"[输出] 缺失值汇总: {missing_path}")

    # 9c. duplicate_samples.csv
    dup_path = TABLE_DIR / "duplicate_samples.csv"
    with open(dup_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["重复类型", "行号（从1开始）"])
        for idx in dup_full_indices:
            writer.writerow(["完整重复(含id)", idx])
        for idx in dup_noid_indices:
            writer.writerow(["去掉id后重复", idx])
    print(f"[输出] 重复样本: {dup_path}")

    # 9d. descriptive_statistics.csv
    desc_path = TABLE_DIR / "descriptive_statistics.csv"
    stat_order = ["count", "min", "max", "mean", "median", "std", "skew", "kurt"]
    with open(desc_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variable"] + stat_order)
        for f in NUMERIC_FIELDS:
            if f in desc_stats:
                writer.writerow(
                    [f] + [round(desc_stats[f][s], 6) for s in stat_order]
                )
    print(f"[输出] 描述性统计: {desc_path}")

    # 9e. Markdown 报告
    report_path = REPORT_DIR / "01_data_check_report.md"
    md_lines: list[str] = []
    md_lines.append("# 煤巷顶板位移数据质量检查报告\n\n")

    md_lines.append("## 1. 基本信息\n\n")
    md_lines.append(f"- 原始文件: `{raw_path.name}`\n")
    md_lines.append(f"- 样本量: {len(rows_clean)}\n")
    md_lines.append(f"- 字段数量: {len(STANDARD_FIELDS)}（标准化后）\n")
    md_lines.append(f"- 原始字段: {', '.join(header_raw)}\n")
    md_lines.append(f"- 标准化字段: {', '.join(STANDARD_FIELDS)}\n\n")

    md_lines.append("## 2. 字段映射\n\n")
    md_lines.append("| 原始字段 | 标准字段 |\n")
    md_lines.append("|---------|--------|\n")
    for orig, mapped in zip(header_raw, mapped_header):
        md_lines.append(f"| {orig} | {mapped} |\n")
    md_lines.append("\n")

    md_lines.append("## 3. 缺失值统计\n\n")
    md_lines.append("| 字段 | 缺失数量 | 缺失比例(%) |\n")
    md_lines.append("|------|---------|-----------|\n")
    total = len(rows_clean)
    for f in STANDARD_FIELDS:
        cnt = missing_counts[f]
        pct = round(cnt / total * 100, 2) if total > 0 else 0.0
        md_lines.append(f"| {f} | {cnt} | {pct} |\n")
    md_lines.append("\n")

    md_lines.append("## 4. 重复行检查\n\n")
    md_lines.append(f"- 完整重复行（含 id）: {len(dup_full_indices)}\n")
    md_lines.append(f"- 去掉 id 后重复行: {len(dup_noid_indices)}\n\n")

    md_lines.append("## 5. 字段类型\n\n")
    md_lines.append("| 字段 | 类型 |\n")
    md_lines.append("|------|------|\n")
    for f in STANDARD_FIELDS:
        ftype = "整数/字符串" if f == "id" else "数值(float)"
        md_lines.append(f"| {f} | {ftype} |\n")
    md_lines.append("\n")

    md_lines.append("## 6. 描述性统计\n\n")
    md_lines.append(
        "| variable | count | min | max | mean | median | std | skew | kurt |\n"
    )
    md_lines.append(
        "|----------|-------|-----|-----|------|--------|-----|------|------|\n"
    )
    for f in NUMERIC_FIELDS:
        if f in desc_stats:
            s = desc_stats[f]
            md_lines.append(
                f"| {f} | {s['count']} | {s['min']:.4f} | {s['max']:.4f} | "
                f"{s['mean']:.4f} | {s['median']:.4f} | {s['std']:.4f} | "
                f"{s['skew']:.4f} | {s['kurt']:.4f} |\n"
            )
    md_lines.append("\n")

    md_lines.append("## 7. roof_displacement 专项统计\n\n")
    if "roof_displacement" in desc_stats:
        rd = desc_stats["roof_displacement"]
        md_lines.append(f"- 最小值: {rd['min']:.4f}\n")
        md_lines.append(f"- 最大值: {rd['max']:.4f}\n")
        md_lines.append(f"- 均值: {rd['mean']:.4f}\n")
        md_lines.append(f"- 中位数: {rd['median']:.4f}\n")
        md_lines.append(f"- 标准差: {rd['std']:.4f}\n")
        md_lines.append(f"- 偏度: {rd['skew']:.4f}\n")
        md_lines.append(f"- 峰度: {rd['kurt']:.4f}\n")
    else:
        md_lines.append("无有效数据。\n")

    with open(report_path, "w", encoding="utf-8-sig") as f:
        f.write("".join(md_lines))
    print(f"[输出] 数据质量检查报告: {report_path}")

    print("\n[完成] 01_data_check.py 执行完毕。")


if __name__ == "__main__":
    main()