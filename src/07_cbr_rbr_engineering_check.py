"""
煤巷顶板位移 —— 07 CBR-RBR 混合推理工程校核
================================================
功能：
  1. CBR 案例推理：检索 Top-5 相似历史案例
  2. RBR 规则推理：位移风险等级、支护参数校核
  3. CBR-RBR 综合校核
  4. 图表、表格、报告
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    DATA_PROCESSED_DIR,
    FIGURE_DIR,
    TABLE_DIR,
    REPORT_DIR,
    ensure_dirs,
)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

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
TEST_SIZE = 0.2
DEFAULT_BEST_SEED = 198
CBR_TOP_K = 5

ALL_FEATURE_NAMES = INPUT_FEATURES + [TARGET]


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABLE_DIR / name, index=False, encoding="utf-8-sig")
    print(f"  已输出 {name}")


def save_fig(name: str) -> None:
    plt.savefig(FIGURE_DIR / name, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  已输出 {name}")


def main() -> None:
    ensure_dirs()

    print("=" * 70)
    print("第七阶段：CBR-RBR 混合推理工程校核")
    print("=" * 70)

    # ---- 读取稳定种子 ----
    best_seed_path = TABLE_DIR / "best_stable_split_seed.csv"
    if best_seed_path.exists():
        best_seed = int(pd.read_csv(best_seed_path, encoding="utf-8-sig")["seed"].iloc[0])
    else:
        best_seed = DEFAULT_BEST_SEED

    # ---- 加载数据 ----
    df = pd.read_csv(DATA_PROCESSED_DIR / "dataset_modeling_176.csv", encoding="utf-8-sig")
    X_all = df[INPUT_FEATURES].values
    y_all = df[TARGET].values
    n = len(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, random_state=best_seed
    )
    n_test = len(X_test)
    print(f"测试样本量: {n_test}")
    print(f"CBR Top-K: {CBR_TOP_K}")

    # ---- 读取预测值 ----
    pred_path = TABLE_DIR / "residual_correction_test_predictions.csv"
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path, encoding="utf-8-sig")
        if "selected_final_pred" in pred_df.columns:
            y_pred = pred_df["selected_final_pred"].values
        else:
            y_pred = pred_df["base_pred"].values
        y_true = pred_df["y_true"].values
    else:
        y_pred = np.zeros(n_test)
        y_true = y_test

    # 构建测试集 DataFrame
    df_test = pd.DataFrame({"test_sample_index": range(n_test), "y_true": y_true, "y_pred": y_pred})
    for i, feat in enumerate(INPUT_FEATURES):
        df_test[feat] = X_test[:, i]
    df_test["prediction_error"] = np.abs(y_true - y_pred)

    # ======================================================================
    # 一、CBR 案例推理
    # ======================================================================
    print("\n--- CBR 案例推理 ---")

    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)
    X_test_scaled = scaler.transform(X_test)
    case_base_df = df[ALL_FEATURE_NAMES].copy()

    cbr_similar_rows: list[dict] = []
    cbr_summary_rows: list[dict] = []

    for test_idx in range(n_test):
        test_vec = X_test_scaled[test_idx:test_idx + 1]
        distances = euclidean_distances(test_vec, X_all_scaled).flatten()

        sorted_idx = np.argsort(distances)
        top_indices = []
        for idx in sorted_idx:
            if len(top_indices) >= CBR_TOP_K:
                break
            if distances[idx] < 1e-8:
                if np.allclose(X_all[idx], X_test[test_idx]):
                    continue
            top_indices.append(idx)

        top_displacements = []
        top_similarities = []
        for rank, case_idx in enumerate(top_indices[:CBR_TOP_K], 1):
            sim = 1.0 / (1.0 + float(distances[case_idx]))
            top_similarities.append(sim)
            top_displacements.append(float(y_all[case_idx]))

            cbr_similar_rows.append({
                "test_sample_index": test_idx,
                "case_rank": rank,
                "similar_case_index": int(case_idx),
                "similarity": round(sim, 6),
                "distance": round(float(distances[case_idx]), 6),
                "case_roof_displacement": float(y_all[case_idx]),
                "case_fracture_degree": float(case_base_df.iloc[case_idx]["fracture_degree"]),
                "case_coal_strength": float(case_base_df.iloc[case_idx]["coal_strength"]),
                "case_floor_strength": float(case_base_df.iloc[case_idx]["floor_strength"]),
                "case_roof_strength": float(case_base_df.iloc[case_idx]["roof_strength"]),
                "case_depth": float(case_base_df.iloc[case_idx]["depth"]),
                "case_width": float(case_base_df.iloc[case_idx]["width"]),
                "case_bolt_area": float(case_base_df.iloc[case_idx]["bolt_area"]),
                "case_anchor_density": float(case_base_df.iloc[case_idx]["anchor_density"]),
            })

        top5_mean = float(np.mean(top_displacements))
        top5_max = float(np.max(top_displacements))
        top5_min = float(np.min(top_displacements))
        top5_mean_sim = float(np.mean(top_similarities))

        if top5_max >= 150:
            risk_hint = "相似案例中存在高位移响应，建议重点复核"
        elif top5_mean >= 100:
            risk_hint = "相似案例平均响应较高，建议加强支护校核"
        else:
            risk_hint = "相似案例响应处于常规范围"

        cbr_summary_rows.append({
            "test_sample_index": test_idx,
            "y_true": y_test[test_idx],
            "y_pred": y_pred[test_idx],
            "top5_mean_displacement": round(top5_mean, 4),
            "top5_max_displacement": round(top5_max, 4),
            "top5_min_displacement": round(top5_min, 4),
            "top5_mean_similarity": round(top5_mean_sim, 6),
            "similar_case_risk_hint": risk_hint,
        })

    save_table(pd.DataFrame(cbr_similar_rows), "cbr_similar_cases.csv")
    save_table(pd.DataFrame(cbr_summary_rows), "cbr_case_summary.csv")

    # ======================================================================
    # 二、RBR 规则推理
    # ======================================================================
    print("\n--- RBR 规则推理 ---")

    rbr_rows: list[dict] = []
    rule_trigger_counts: dict[str, int] = {}

    def add_rule(name: str) -> None:
        rule_trigger_counts[name] = rule_trigger_counts.get(name, 0) + 1

    for test_idx in range(n_test):
        row = df_test.iloc[test_idx]
        yp = row["y_pred"]
        depth_v = row["depth"]
        width_v = row["width"]
        bolt_v = row["bolt_area"]
        anchor_v = row["anchor_density"]
        roof_s = row["roof_strength"]
        coal_s = row["coal_strength"]
        yt = row["y_true"]
        err = row["prediction_error"]

        triggered: list[str] = []

        # 初始风险等级
        if yp < 50:
            risk_level = "低风险"
        elif yp < 100:
            risk_level = "中风险"
        elif yp < 150:
            risk_level = "较高风险"
        else:
            risk_level = "高风险"

        # 埋深规则
        if depth_v >= 700:
            triggered.append("depth>=700: 埋深较大，关注深部高地应力影响")
            add_rule("depth_high")
        if depth_v >= 800:
            if risk_level in ["低风险", "中风险"]:
                risk_level = "较高风险"

        # 宽度规则
        if width_v >= 6.5:
            triggered.append("width>=6.5: 巷道宽度较大，围岩暴露范围增加")
            add_rule("width_high")
        if width_v <= 3.0:
            triggered.append("width<=3.0: 巷道宽度偏小，边界工况")
            add_rule("width_low")

        # 支护规则
        support_check = "支护参数满足当前设计要求"
        if bolt_v <= 0.55:
            triggered.append("bolt_area<=0.55: 锚杆支护面积偏低")
            add_rule("bolt_area_low")
        if anchor_v <= 0.8:
            triggered.append("anchor_density<=0.8: 锚索密度偏低")
            add_rule("anchor_density_low")
        if yp >= 100 and anchor_v <= 1.0:
            support_check = "预测位移较大且锚索密度偏低，建议加强锚索支护"
            triggered.append("displacement>=100 & anchor<=1.0: 建议加强锚索支护")
            add_rule("support_anchor_weak")
        if yp >= 100 and bolt_v <= 0.64:
            support_check = "预测位移较大且锚杆支护面积偏低，建议提高锚杆支护参数"
            triggered.append("displacement>=100 & bolt_area<=0.64: 建议提高锚杆支护")
            add_rule("support_bolt_weak")

        # 围岩强度规则
        if roof_s <= 30:
            triggered.append("roof_strength<=30: 顶板强度偏低，加强复核")
            add_rule("roof_strength_low")
        if coal_s <= 5:
            triggered.append("coal_strength<=5: 煤层强度偏低，关注变形风险")
            add_rule("coal_strength_low")

        # 误差复核
        error_check = "预测误差较大，建议人工复核" if err >= 30 else "预测误差处于可接受范围"

        # 工程建议
        if risk_level == "低风险":
            eng_suggestion = "预测顶板位移较小，现有支护参数基本满足控制要求。"
        elif risk_level == "中风险":
            eng_suggestion = "预测顶板位移处于中等水平，建议保持现有支护并加强监测。"
        elif risk_level == "较高风险":
            eng_suggestion = "预测顶板位移较大，建议复核支护参数并适当加强锚杆锚索支护。"
        else:
            eng_suggestion = "预测顶板位移较高，应进行重点工程复核，建议加强支护并提高监测频率。"

        rbr_rows.append({
            "test_sample_index": test_idx,
            "y_true": yt,
            "y_pred": yp,
            "prediction_error": err,
            "depth": depth_v,
            "width": width_v,
            "bolt_area": bolt_v,
            "anchor_density": anchor_v,
            "roof_strength": roof_s,
            "coal_strength": coal_s,
            "initial_risk_level": "低风险" if yp < 50 else ("中风险" if yp < 100 else ("较高风险" if yp < 150 else "高风险")),
            "final_risk_level": risk_level,
            "support_check_result": support_check,
            "error_check": error_check,
            "triggered_rules": "; ".join(triggered) if triggered else "无",
            "engineering_suggestion": eng_suggestion,
        })

    save_table(pd.DataFrame(rbr_rows), "rbr_rule_check_results.csv")

    # ======================================================================
    # 三、CBR-RBR 综合校核
    # ======================================================================
    print("\n--- CBR-RBR 综合校核 ---")

    df_cbr = pd.DataFrame(cbr_summary_rows)
    df_rbr = pd.DataFrame(rbr_rows)
    df_merged = df_rbr.merge(
        df_cbr[["test_sample_index", "top5_mean_displacement", "top5_max_displacement",
                "top5_min_displacement", "top5_mean_similarity", "similar_case_risk_hint"]],
        on="test_sample_index", how="left"
    )

    conclusions = []
    for _, row in df_merged.iterrows():
        fr = row["final_risk_level"]
        hint = str(row["similar_case_risk_hint"])
        err_chk = str(row["error_check"])
        if fr == "高风险":
            conclusions.append("预测结果显示该样本存在较高顶板变形风险，应进行重点工程复核。")
        elif fr == "较高风险":
            conclusions.append("预测结果显示该样本顶板位移偏大，建议结合相似案例和支护参数进行复核。")
        elif "高位移" in hint:
            conclusions.append("相似案例中存在高位移响应，应提高校核等级。")
        elif "预测误差较大" in err_chk:
            conclusions.append("该样本预测误差较大，建议纳入后续误差分析。")
        else:
            conclusions.append("预测结果与相似案例和规则校核基本一致，可作为工程参考。")
    df_merged["final_check_conclusion"] = conclusions

    save_table(df_merged, "cbr_rbr_final_engineering_check.csv")

    # ======================================================================
    # 四、统计汇总
    # ======================================================================
    print("\n--- 风险等级统计 ---")

    risk_counts = df_merged["final_risk_level"].value_counts()
    risk_order = ["低风险", "中风险", "较高风险", "高风险"]
    risk_summary_rows = []
    for rl in risk_order:
        cnt = int(risk_counts.get(rl, 0))
        risk_summary_rows.append({
            "risk_level": rl,
            "sample_count": cnt,
            "percentage": round(cnt / n_test * 100, 1),
        })
        print(f"  {rl}: {cnt}")

    save_table(pd.DataFrame(risk_summary_rows), "cbr_rbr_risk_summary.csv")

    rule_summary_rows = [
        {"rule": k, "trigger_count": v}
        for k, v in sorted(rule_trigger_counts.items(), key=lambda x: x[1], reverse=True)
    ]
    save_table(pd.DataFrame(rule_summary_rows), "cbr_rbr_rule_trigger_summary.csv")

    # ======================================================================
    # 五、图表
    # ======================================================================
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#4CAF50", "#FFC107", "#FF9800", "#F44336"]
    cnts = [risk_counts.get(rl, 0) for rl in risk_order]
    bars = ax.bar(risk_order, cnts, color=colors, width=0.5)
    for bar, val in zip(bars, cnts):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha="center", fontsize=12)
    ax.set_xlabel("Risk Level", fontsize=12)
    ax.set_ylabel("Sample Count", fontsize=12)
    ax.set_title("Risk Level Distribution of Test Samples", fontsize=14)
    plt.tight_layout()
    save_fig("cbr_rbr_risk_level_distribution.png")

    risk_color_map = {"低风险": "#4CAF50", "中风险": "#FFC107", "较高风险": "#FF9800", "高风险": "#F44336"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for rl in risk_order:
        mask = df_merged["final_risk_level"] == rl
        if mask.any():
            ax.scatter(df_merged.loc[mask, "y_pred"], df_merged.loc[mask, "y_true"],
                       color=risk_color_map[rl], s=50, alpha=0.8, edgecolors="k",
                       linewidth=0.3, label=rl)
    min_v = min(df_merged["y_pred"].min(), df_merged["y_true"].min())
    max_v = max(df_merged["y_pred"].max(), df_merged["y_true"].max())
    ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.5, label="Perfect Fit")
    ax.set_xlabel("Predicted Displacement / mm", fontsize=12)
    ax.set_ylabel("True Displacement / mm", fontsize=12)
    ax.set_title("Prediction vs True Value (Colored by Risk Level)", fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig("cbr_rbr_prediction_vs_risk.png")

    # ======================================================================
    # 六、报告
    # ======================================================================
    lines: list[str] = []
    lines.append("# CBR-RBR 混合推理工程校核报告\n\n")

    lines.append("## 1. CBR-RBR 模块目的\n\n")
    lines.append(
        "CBR-RBR 混合推理机制将历史相似案例检索（Case-Based Reasoning）与"
        "显式工程规则（Rule-Based Reasoning）相结合，"
        "一方面利用 CBR 为预测结果提供相似工程案例依据，"
        "另一方面利用 RBR 对预测位移、埋深、巷道宽度及支护参数进行规则化校核。"
        "该机制不替代预测模型，而是对预测结果进行工程合理性复核。\n\n"
    )

    lines.append("## 2. CBR 案例推理方法\n\n")
    lines.append(f"- 案例库: 全部 {n} 条历史样本\n")
    lines.append("- 特征空间: 8 个输入特征经 StandardScaler 标准化\n")
    lines.append("- 距离度量: 欧氏距离\n")
    lines.append("- 相似度: similarity = 1 / (1 + distance)\n")
    lines.append(f"- 检索: 对每个测试样本检索 Top-{CBR_TOP_K} 相似案例\n\n")

    lines.append("## 3. RBR 规则推理方法\n\n")
    lines.append("### 3.1 位移风险等级\n\n")
    lines.append("| 预测位移 (mm) | 风险等级 |\n")
    lines.append("|:-----------:|:------:|\n")
    lines.append("| < 50 | 低风险 |\n")
    lines.append("| 50-100 | 中风险 |\n")
    lines.append("| 100-150 | 较高风险 |\n")
    lines.append("| >= 150 | 高风险 |\n\n")

    lines.append("### 3.2 支护参数校核规则\n\n")
    lines.append("- 锚杆支护面积 <= 0.55: 建议复核锚杆支护强度\n")
    lines.append("- 锚索密度 <= 0.8: 建议复核锚索补强措施\n")
    lines.append("- 预测位移 >= 100 且锚索密度 <= 1.0: 建议加强锚索支护\n")
    lines.append("- 预测位移 >= 100 且锚杆支护面积 <= 0.64: 建议提高锚杆支护参数\n\n")

    lines.append("### 3.3 工程地质条件校核\n\n")
    lines.append("- 埋深 >= 700: 关注深部高地应力影响\n")
    lines.append("- 巷道宽度 >= 6.5: 复核顶板控制效果\n")
    lines.append("- 顶板强度 <= 30: 加强顶板稳定性复核\n")
    lines.append("- 煤层强度 <= 5: 关注围岩变形风险\n\n")

    lines.append("## 4. 风险等级分布\n\n")
    lines.append("| 风险等级 | 样本数量 | 占比 |\n")
    lines.append("|---------|:------:|:----:|\n")
    for r in risk_summary_rows:
        lines.append(f"| {r['risk_level']} | {r['sample_count']} | {r['percentage']}% |\n")
    lines.append("\n")

    lines.append("## 5. 规则触发统计\n\n")
    lines.append("| 规则 | 触发次数 |\n")
    lines.append("|------|:------:|\n")
    for r in rule_summary_rows:
        lines.append(f"| {r['rule']} | {r['trigger_count']} |\n")
    lines.append("\n")

    lines.append("## 6. 典型样本校核示例\n\n")
    sample_indices = []
    for rl in risk_order:
        mask = df_merged["final_risk_level"] == rl
        if mask.any():
            sample_indices.append(int(df_merged[mask].iloc[0]["test_sample_index"]))
    for si in sample_indices[:3]:
        row = df_merged[df_merged["test_sample_index"] == si].iloc[0]
        lines.append(f"### 测试样本 {si}\n\n")
        lines.append(f"- 预测位移: {row['y_pred']:.1f} mm\n")
        lines.append(f"- 真实位移: {row['y_true']:.1f} mm\n")
        lines.append(f"- 风险等级: {row['final_risk_level']}\n")
        lines.append(f"- 支护校核: {row['support_check_result']}\n")
        lines.append(f"- 相似案例风险提示: {row['similar_case_risk_hint']}\n")
        lines.append(f"- 综合结论: {row['final_check_conclusion']}\n\n")

    lines.append("## 7. 结论\n\n")
    lines.append(
        "CBR-RBR 混合推理机制将历史相似案例检索与显式工程规则相结合，"
        "一方面利用 CBR 为预测结果提供相似工程案例依据，"
        "另一方面利用 RBR 对预测位移、埋深、巷道宽度及支护参数进行规则化校核。"
        "该机制能够提高机器学习预测结果的工程可解释性，"
        "并为支护参数调整和现场风险复核提供辅助依据。\n\n"
    )

    lines.append("## 8. 输出文件清单\n\n")
    for fname in [
        "cbr_similar_cases.csv", "cbr_case_summary.csv",
        "rbr_rule_check_results.csv", "cbr_rbr_final_engineering_check.csv",
        "cbr_rbr_risk_summary.csv", "cbr_rbr_rule_trigger_summary.csv",
        "cbr_rbr_risk_level_distribution.png", "cbr_rbr_prediction_vs_risk.png",
        "07_cbr_rbr_engineering_check_report.md",
    ]:
        lines.append(f"- `{fname}`\n")

    with open(REPORT_DIR / "07_cbr_rbr_engineering_check_report.md", "w", encoding="utf-8-sig") as f:
        f.write("".join(lines))
    print(f"  已输出 07_cbr_rbr_engineering_check_report.md")

    print("=" * 70)
    print("07_cbr_rbr_engineering_check.py 执行完毕。")
    print("=" * 70)


if __name__ == "__main__":
    main()