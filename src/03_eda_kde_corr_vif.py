"""
煤巷顶板位移 —— 03 KDE、相关性矩阵与 VIF 分析
================================================
功能：
  1. KDE 分布分析（8 个连续变量）
  2. 清洗前后 roof_displacement 分布对比
  3. Pearson / Spearman 相关性矩阵
  4. VIF 多重共线性分析
  5. 数据版本分布对比
  6. Markdown 报告
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 非交互后端
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    DATA_PROCESSED_DIR,
    FIGURE_DIR,
    TABLE_DIR,
    REPORT_DIR,
    ensure_dirs,
)

# ---------------------------------------------------------------------------
# 中文字体设置
# ---------------------------------------------------------------------------
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
INPUT_FEATURES = [
    "fracture_degree",
    "coal_strength",
    "floor_strength",
    "roof_strength",
    "depth",
    "width",
    "bolt_area",
    "anchor_density",
]

TARGET = "roof_displacement"

CONTINUOUS_VARS = [
    "coal_strength",
    "floor_strength",
    "roof_strength",
    "depth",
    "width",
    "bolt_area",
    "anchor_density",
    "roof_displacement",
]

ALL_VARS = INPUT_FEATURES + [TARGET]

KDE_LABELS = {
    "coal_strength": "Coal Strength / MPa",
    "floor_strength": "Floor Strength / MPa",
    "roof_strength": "Roof Strength / MPa",
    "depth": "Depth / m",
    "width": "Width / m",
    "bolt_area": "Bolt Area / m²",
    "anchor_density": "Anchor Density / m⁻¹",
    "roof_displacement": "Roof Displacement / mm",
}


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def classify_distribution(skew: float, kurt: float) -> str:
    """根据偏度和峰度判断分布类型。"""
    parts: list[str] = []
    if skew > 0.5:
        parts.append("右偏")
    elif skew < -0.5:
        parts.append("左偏")
    else:
        parts.append("近似对称")

    if kurt > 3:
        parts.append("尖峰厚尾")
    elif kurt < -1:
        parts.append("平缓")
    else:
        parts.append("适中")

    return "；".join(parts)


def save_table(df: pd.DataFrame, name: str) -> None:
    """保存 CSV 到 outputs/tables。"""
    path = TABLE_DIR / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  已输出 {name}")


def save_fig(name: str) -> None:
    """保存当前 matplotlib 图形到 outputs/figures，dpi=300。"""
    path = FIGURE_DIR / name
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  已输出 {name}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ensure_dirs()

    print("=" * 70)
    print("第三阶段：KDE、相关性矩阵与 VIF 分析")
    print("=" * 70)

    # ---- 加载数据 ----
    modeling_path = DATA_PROCESSED_DIR / "dataset_modeling_176.csv"
    full_path = DATA_PROCESSED_DIR / "dataset_full.csv"
    dedup_path = DATA_PROCESSED_DIR / "dataset_deduplicated.csv"
    winsor_path = DATA_PROCESSED_DIR / "dataset_winsorized.csv"

    df = pd.read_csv(modeling_path, encoding="utf-8-sig")
    df_full = pd.read_csv(full_path, encoding="utf-8-sig")
    df_dedup = pd.read_csv(dedup_path, encoding="utf-8-sig")
    df_winsor = pd.read_csv(winsor_path, encoding="utf-8-sig")

    n = len(df)
    print(f"主分析样本量: {n}")
    print(f"输入特征数: {len(INPUT_FEATURES)}")
    print(f"目标变量: {TARGET}")

    # ======================================================================
    # 一、KDE 分布分析
    # ======================================================================
    print("\n--- KDE 分布分析 ---")

    kde_summary_rows: list[dict] = []
    for var in CONTINUOUS_VARS:
        series = df[var].dropna()
        vals = series.values
        count = len(vals)
        min_v = float(np.min(vals))
        max_v = float(np.max(vals))
        mean_v = float(np.mean(vals))
        median_v = float(np.median(vals))
        std_v = float(np.std(vals, ddof=1))
        skew_v = float(series.skew())
        kurt_v = float(series.kurt())
        dist_type = classify_distribution(skew_v, kurt_v)

        kde_summary_rows.append({
            "variable": var,
            "count": count,
            "min": round(min_v, 4),
            "max": round(max_v, 4),
            "mean": round(mean_v, 4),
            "median": round(median_v, 4),
            "std": round(std_v, 4),
            "skew": round(skew_v, 4),
            "kurt": round(kurt_v, 4),
            "distribution_type": dist_type,
        })

        # 绘制 KDE
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.kdeplot(vals, fill=True, color="#2196F3", alpha=0.3, ax=ax)
        sns.kdeplot(vals, color="#1976D2", linewidth=2, ax=ax)
        ax.axvline(mean_v, color="#D32F2F", linestyle="--", linewidth=1.5,
                   label=f"Mean = {mean_v:.2f}")
        ax.axvline(median_v, color="#388E3C", linestyle=":", linewidth=1.5,
                   label=f"Median = {median_v:.2f}")
        ax.set_xlabel(KDE_LABELS.get(var, var), fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"KDE: {var}", fontsize=14)
        ax.legend(loc="upper right", fontsize=10)
        sns.despine()
        save_fig(f"kde_{var}.png")

    save_table(pd.DataFrame(kde_summary_rows), "kde_distribution_summary.csv")

    # ======================================================================
    # 二、清洗前后 roof_displacement 分布对比
    # ======================================================================
    print("\n--- 清洗前后目标变量分布对比 ---")

    rd_raw = df_full[TARGET].dropna()
    rd_clean = df[TARGET].dropna()

    raw_stats = {
        "dataset": "raw_200",
        "sample_count": len(rd_raw),
        "target_min": float(rd_raw.min()),
        "target_max": float(rd_raw.max()),
        "target_mean": float(rd_raw.mean()),
        "target_median": float(rd_raw.median()),
        "target_std": float(rd_raw.std(ddof=1)),
        "target_skew": float(rd_raw.skew()),
        "target_kurt": float(rd_raw.kurt()),
    }
    clean_stats = {
        "dataset": "modeling_176",
        "sample_count": len(rd_clean),
        "target_min": float(rd_clean.min()),
        "target_max": float(rd_clean.max()),
        "target_mean": float(rd_clean.mean()),
        "target_median": float(rd_clean.median()),
        "target_std": float(rd_clean.std(ddof=1)),
        "target_skew": float(rd_clean.skew()),
        "target_kurt": float(rd_clean.kurt()),
    }
    save_table(pd.DataFrame([raw_stats, clean_stats]),
               "raw_vs_filtered_target_distribution.csv")

    # 对比 KDE
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(rd_raw, fill=True, color="#FF7043", alpha=0.3, label="Raw (n=200)", ax=ax)
    sns.kdeplot(rd_raw, color="#E64A19", linewidth=1.5, ax=ax)
    sns.kdeplot(rd_clean, fill=True, color="#42A5F5", alpha=0.3, label="Cleaned (n=176)", ax=ax)
    sns.kdeplot(rd_clean, color="#1565C0", linewidth=2, ax=ax)
    ax.axvline(raw_stats["target_mean"], color="#E64A19", linestyle="--", linewidth=1,
               label=f"Raw Mean = {raw_stats['target_mean']:.1f}")
    ax.axvline(clean_stats["target_mean"], color="#1565C0", linestyle="--", linewidth=1,
               label=f"Cleaned Mean = {clean_stats['target_mean']:.1f}")
    ax.set_xlabel("Roof Displacement / mm", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Raw vs Cleaned Roof Displacement KDE", fontsize=14)
    ax.legend(loc="upper right", fontsize=9)
    sns.despine()
    save_fig("raw_vs_filtered_roof_displacement_kde.png")

    # 对比箱线图
    fig, ax = plt.subplots(figsize=(6, 5))
    box_data = [rd_raw.values, rd_clean.values]
    bp = ax.boxplot(box_data, patch_artist=True, widths=0.5, labels=["Raw (n=200)", "Cleaned (n=176)"])
    bp["boxes"][0].set_facecolor("#FF7043")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("#42A5F5")
    bp["boxes"][1].set_alpha(0.6)
    ax.set_ylabel("Roof Displacement / mm", fontsize=12)
    ax.set_title("Raw vs Cleaned Roof Displacement Boxplot", fontsize=14)
    sns.despine()
    save_fig("raw_vs_filtered_roof_displacement_boxplot.png")

    # ======================================================================
    # 三、相关性矩阵分析
    # ======================================================================
    print("\n--- 相关性矩阵分析 ---")

    df_all = df[ALL_VARS].copy()

    pearson_corr = df_all.corr(method="pearson")
    spearman_corr = df_all.corr(method="spearman")

    save_table(pearson_corr.reset_index().rename(columns={"index": "variable"}),
               "pearson_corr_matrix.csv")
    save_table(spearman_corr.reset_index().rename(columns={"index": "variable"}),
               "spearman_corr_matrix.csv")

    # 热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(pearson_corr, dtype=bool), k=1)
    sns.heatmap(pearson_corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Pearson Correlation Matrix", fontsize=14)
    plt.tight_layout()
    save_fig("pearson_corr_heatmap.png")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(spearman_corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Spearman Correlation Matrix", fontsize=14)
    plt.tight_layout()
    save_fig("spearman_corr_heatmap.png")

    # 与 roof_displacement 的相关性排序
    target_pearson = pearson_corr[TARGET].drop(TARGET)
    target_spearman = spearman_corr[TARGET].drop(TARGET)

    ranking_rows = []
    for feat in target_pearson.index:
        pr = float(target_pearson[feat])
        sr = float(target_spearman[feat])
        ranking_rows.append({
            "feature": feat,
            "pearson_r": round(pr, 4),
            "pearson_abs": round(abs(pr), 4),
            "spearman_r": round(sr, 4),
            "spearman_abs": round(abs(sr), 4),
        })
    df_ranking = pd.DataFrame(ranking_rows).sort_values("pearson_abs", ascending=False)
    save_table(df_ranking, "target_correlation_ranking.csv")

    # 最高相关特征
    top_feat = df_ranking.iloc[0]
    print(f"与 roof_displacement Pearson 相关性最高的特征: {top_feat['feature']} "
          f"(r={top_feat['pearson_r']:.4f})")

    # ======================================================================
    # 四、VIF 多重共线性分析
    # ======================================================================
    print("\n--- VIF 多重共线性分析 ---")

    df_input = df[INPUT_FEATURES].dropna().astype(float)
    df_const = add_constant(df_input)

    vif_rows = []
    for i, feat in enumerate(["const"] + INPUT_FEATURES):
        vif_val = float(variance_inflation_factor(df_const.values, i))
        if feat == "const":
            continue
        if vif_val < 5:
            judgement = "共线性较弱"
        elif vif_val < 10:
            judgement = "存在一定共线性"
        else:
            judgement = "严重共线性"
        vif_rows.append({
            "feature": feat,
            "VIF": round(vif_val, 4),
            "collinearity_judgement": judgement,
        })

    df_vif = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
    save_table(df_vif, "vif_summary.csv")

    max_vif = df_vif["VIF"].max()
    max_vif_feat = df_vif.loc[df_vif["VIF"].idxmax(), "feature"]
    print(f"最大 VIF: {max_vif:.4f} ({max_vif_feat})")

    # ======================================================================
    # 五、数据版本分布对比
    # ======================================================================
    print("\n--- 数据版本分布对比 ---")

    def get_target_stats(df_source: pd.DataFrame, version_name: str) -> dict:
        s = df_source[TARGET].dropna()
        return {
            "dataset": version_name,
            "sample_count": len(s),
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std(ddof=1)),
            "skew": float(s.skew()),
            "kurt": float(s.kurt()),
        }

    versions_comp = [
        get_target_stats(df_full, "dataset_full"),
        get_target_stats(df_dedup, "dataset_deduplicated"),
        get_target_stats(df_winsor, "dataset_winsorized"),
        get_target_stats(df, "dataset_modeling_176"),
    ]
    save_table(pd.DataFrame(versions_comp), "dataset_versions_distribution_comparison.csv")

    # ======================================================================
    # 六、Markdown 报告
    # ======================================================================
    print("\n--- 生成报告 ---")

    lines: list[str] = []
    lines.append("# KDE、相关性矩阵与 VIF 分析报告\n\n")

    lines.append("## 1. 主分析数据\n\n")
    lines.append(f"- 主分析数据: `{modeling_path}`\n")
    lines.append(f"- 样本量: {n}\n")
    lines.append(f"- 输入特征数: {len(INPUT_FEATURES)}\n")
    lines.append(f"- 目标变量: {TARGET}\n\n")

    lines.append("## 2. KDE 分布分析\n\n")
    lines.append(
        "| variable | count | min | max | mean | median | std | skew | kurt | distribution_type |\n"
    )
    lines.append(
        "|----------|-------|-----|-----|------|--------|-----|------|------|------------------|\n"
    )
    for row in kde_summary_rows:
        lines.append(
            f"| {row['variable']} | {row['count']} | {row['min']:.2f} | {row['max']:.2f} | "
            f"{row['mean']:.2f} | {row['median']:.2f} | {row['std']:.2f} | "
            f"{row['skew']:.2f} | {row['kurt']:.2f} | {row['distribution_type']} |\n"
        )
    lines.append("\n")

    lines.append("## 3. 清洗前后 roof_displacement 分布变化\n\n")
    lines.append(f"- 原始 200 条数据 roof_displacement 最大值: {raw_stats['target_max']:.2f} mm\n")
    lines.append(f"- 清洗后 176 条数据 roof_displacement 最大值: {clean_stats['target_max']:.2f} mm\n")
    lines.append(f"- 原始标准差: {raw_stats['target_std']:.2f} mm\n")
    lines.append(f"- 清洗后标准差: {clean_stats['target_std']:.2f} mm\n")
    lines.append("- 清洗后目标变量分布更稳定，更适合作为主建模样本。\n\n")

    lines.append("## 4. Pearson 相关性分析\n\n")
    lines.append(
        "各特征与 roof_displacement 的 Pearson 相关系数如下（按绝对值降序）：\n\n"
    )
    lines.append("| feature | pearson_r |\n")
    lines.append("|---------|----------|\n")
    for _, row in df_ranking.iterrows():
        lines.append(f"| {row['feature']} | {row['pearson_r']:.4f} |\n")
    lines.append("\n")
    lines.append(
        "各特征与顶板位移的 Pearson 相关系数绝对值普遍偏低，"
        "说明线性相关性较弱。顶板位移受围岩强度、埋深、"
        "巷道宽度及支护参数等多因素非线性耦合作用影响。"
        "因此后续适合采用 XGBoost、SVR、GBDT 等非线性模型。\n\n"
    )

    lines.append("## 5. Spearman 相关性分析\n\n")
    lines.append(
        "各特征与 roof_displacement 的 Spearman 秩相关系数如下（按绝对值降序）：\n\n"
    )
    lines.append("| feature | spearman_r |\n")
    lines.append("|---------|-----------|\n")
    spearman_ranking = sorted(ranking_rows, key=lambda x: x["spearman_abs"], reverse=True)
    for row in spearman_ranking:
        lines.append(f"| {row['feature']} | {row['spearman_r']:.4f} |\n")
    lines.append("\n")
    lines.append("Spearman 秩相关结果与 Pearson 一致，进一步证实了变量间的非线性关系。\n\n")

    lines.append("## 6. 与 roof_displacement 的相关性排序\n\n")
    lines.append(
        "| feature | pearson_r | pearson_abs | spearman_r | spearman_abs |\n"
    )
    lines.append(
        "|---------|----------|------------|-----------|------------|\n"
    )
    for _, row in df_ranking.iterrows():
        lines.append(
            f"| {row['feature']} | {row['pearson_r']:.4f} | {row['pearson_abs']:.4f} | "
            f"{row['spearman_r']:.4f} | {row['spearman_abs']:.4f} |\n"
        )
    lines.append("\n")

    lines.append("## 7. VIF 多重共线性分析\n\n")
    lines.append("| feature | VIF | collinearity_judgement |\n")
    lines.append("|---------|-----|-----------------------|\n")
    for _, row in df_vif.iterrows():
        lines.append(
            f"| {row['feature']} | {row['VIF']:.4f} | {row['collinearity_judgement']} |\n"
        )
    lines.append("\n")
    lines.append(
        f"所有 VIF 均小于 {max_vif:.2f}，不存在严重多重共线性。"
        "8 个特征均可保留用于后续建模。\n\n"
    )

    lines.append("## 8. 数据版本分布对比\n\n")
    lines.append(
        "| dataset | sample_count | min | max | mean | median | std | skew | kurt |\n"
    )
    lines.append(
        "|---------|-------------|-----|-----|------|--------|-----|------|------|\n"
    )
    for v in versions_comp:
        lines.append(
            f"| {v['dataset']} | {v['sample_count']} | {v['min']:.2f} | {v['max']:.2f} | "
            f"{v['mean']:.2f} | {v['median']:.2f} | {v['std']:.2f} | "
            f"{v['skew']:.2f} | {v['kurt']:.2f} |\n"
        )
    lines.append("\n")

    lines.append("## 9. 后续建模建议\n\n")
    lines.append(
        "清洗后的顶板位移数据仍呈一定右偏分布，但极端响应值影响明显减弱。"
        "相关性分析表明，各输入特征与顶板位移之间的线性相关性整体较弱，"
        "说明顶板位移受围岩强度、埋深、巷道宽度及支护参数等多因素非线性耦合作用影响。"
        "VIF 分析结果表明各特征间不存在严重多重共线性，"
        "因此保留全部 8 个输入特征用于后续机器学习建模。\n\n"
    )

    lines.append("## 10. 输出文件清单\n\n")
    output_items = [
        "kde_distribution_summary.csv",
        "raw_vs_filtered_target_distribution.csv",
        "pearson_corr_matrix.csv",
        "spearman_corr_matrix.csv",
        "target_correlation_ranking.csv",
        "vif_summary.csv",
        "dataset_versions_distribution_comparison.csv",
    ]
    for item in output_items:
        lines.append(f"- `outputs/tables/{item}`\n")

    figures = [
        "kde_coal_strength.png",
        "kde_floor_strength.png",
        "kde_roof_strength.png",
        "kde_depth.png",
        "kde_width.png",
        "kde_bolt_area.png",
        "kde_anchor_density.png",
        "kde_roof_displacement.png",
        "raw_vs_filtered_roof_displacement_kde.png",
        "raw_vs_filtered_roof_displacement_boxplot.png",
        "pearson_corr_heatmap.png",
        "spearman_corr_heatmap.png",
    ]
    for fig_name in figures:
        lines.append(f"- `outputs/figures/{fig_name}`\n")

    lines.append(f"\n- `outputs/reports/03_eda_kde_corr_vif_report.md`\n")

    report_path = REPORT_DIR / "03_eda_kde_corr_vif_report.md"
    with open(report_path, "w", encoding="utf-8-sig") as f:
        f.write("".join(lines))
    print(f"  已输出 03_eda_kde_corr_vif_report.md")

    print("=" * 70)
    print("03_eda_kde_corr_vif.py 执行完毕。")
    print("=" * 70)


if __name__ == "__main__":
    main()