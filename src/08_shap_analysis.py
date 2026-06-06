"""
煤巷顶板位移 —— 08 SHAP 模型解释与特征影响分析
=================================================
功能：
  1. 加载 final_base_xgboost 计算 SHAP 值
  2. 全局特征重要性排序
  3. 局部样本解释
  4. 与相关性分析对比
  5. 图表、表格、报告
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import shap

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    DATA_PROCESSED_DIR,
    FIGURE_DIR,
    TABLE_DIR,
    MODEL_DIR,
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
MODEL_RANDOM_STATE = 42

FEATURE_CN = {
    "fracture_degree": "Fracture Degree",
    "coal_strength": "Coal Strength / MPa",
    "floor_strength": "Floor Strength / MPa",
    "roof_strength": "Roof Strength / MPa",
    "depth": "Depth / m",
    "width": "Width / m",
    "bolt_area": "Bolt Area / m2",
    "anchor_density": "Anchor Density",
}


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
    print("第八阶段：SHAP 模型解释与特征影响分析")
    print("=" * 70)

    # ---- 读取稳定种子 ----
    best_seed_path = TABLE_DIR / "best_stable_split_seed.csv"
    if best_seed_path.exists():
        best_seed = int(pd.read_csv(best_seed_path, encoding="utf-8-sig")["seed"].iloc[0])
    else:
        best_seed = DEFAULT_BEST_SEED
    print(f"稳定划分种子: {best_seed}")

    # ---- 加载数据 ----
    df = pd.read_csv(DATA_PROCESSED_DIR / "dataset_modeling_176.csv", encoding="utf-8-sig")
    X = df[INPUT_FEATURES].values
    y = df[TARGET].values
    n = len(df)
    print(f"主建模样本量: {n}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=best_seed
    )
    n_train = len(X_train)
    n_test = len(X_test)
    print(f"训练集样本量: {n_train}")
    print(f"测试集样本量: {n_test}")
    print(f"解释模型: XGBoost")

    # ---- 加载或训练模型 ----
    model_path = MODEL_DIR / "final_base_xgboost.joblib"
    if model_path.exists():
        model = joblib.load(model_path)
        print("[信息] 加载 final_base_xgboost.joblib")
    else:
        model = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", random_state=MODEL_RANDOM_STATE, verbosity=0,
        )
        model.fit(X_train, y_train)
        print("[信息] 未找到 saved model，重新训练 baseline XGBoost")

    # ---- SHAP Explainer ----
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # ---- 基础预测 ----
    y_test_pred = model.predict(X_test)

    # ======================================================================
    # 一、SHAP 特征重要性
    # ======================================================================
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # 方向分析
    direction_rows = []
    for i, feat in enumerate(INPUT_FEATURES):
        fvals = X_test[:, i]
        svals = shap_values[:, i]
        corr = np.corrcoef(fvals, svals)[0, 1] if np.std(fvals) > 0 else 0.0
        if corr > 0.2:
            direction = "特征值增大整体倾向于提高预测顶板位移"
        elif corr < -0.2:
            direction = "特征值增大整体倾向于降低预测顶板位移"
        else:
            direction = "影响方向呈非线性或交互特征"
        direction_rows.append({
            "feature": feat,
            "mean_abs_shap": round(float(mean_abs_shap[i]), 6),
            "mean_feature_value": round(float(np.mean(fvals)), 4),
            "correlation_with_shap": round(float(corr), 4),
            "direction_hint": direction,
        })

    df_importance = pd.DataFrame(direction_rows).sort_values("mean_abs_shap", ascending=False)
    df_importance["importance_rank"] = range(1, len(df_importance) + 1)
    save_table(df_importance, "shap_feature_importance.csv")

    # 打印 Top 5
    print("\nTop 5 SHAP 特征重要性:")
    top5 = df_importance.head(5)
    for _, row in top5.iterrows():
        print(f"  {row['importance_rank']}. {row['feature']}: mean_abs_shap = {row['mean_abs_shap']:.4f}")

    # ======================================================================
    # 二、局部 SHAP 解释（误差最大的 5 个样本）
    # ======================================================================
    errors = np.abs(y_test - y_test_pred)
    top_error_idx = np.argsort(errors)[-5:][::-1]

    local_rows = []
    for idx in top_error_idx:
        sv = shap_values[idx]
        sorted_idx = np.argsort(np.abs(sv))[::-1]
        local_rows.append({
            "sample_index": int(idx),
            "y_true": round(float(y_test[idx]), 4),
            "y_pred": round(float(y_test_pred[idx]), 4),
            "prediction_error": round(float(errors[idx]), 4),
            "top_positive_feature": INPUT_FEATURES[np.argmax(sv)],
            "top_positive_shap": round(float(np.max(sv)), 6),
            "top_negative_feature": INPUT_FEATURES[np.argmin(sv)],
            "top_negative_shap": round(float(np.min(sv)), 6),
            "top_1_feature": INPUT_FEATURES[sorted_idx[0]],
            "top_1_shap": round(float(sv[sorted_idx[0]]), 6),
            "top_2_feature": INPUT_FEATURES[sorted_idx[1]] if len(sorted_idx) > 1 else "",
            "top_2_shap": round(float(sv[sorted_idx[1]]), 6) if len(sorted_idx) > 1 else 0,
            "top_3_feature": INPUT_FEATURES[sorted_idx[2]] if len(sorted_idx) > 2 else "",
            "top_3_shap": round(float(sv[sorted_idx[2]]), 6) if len(sorted_idx) > 2 else 0,
        })
    save_table(pd.DataFrame(local_rows), "shap_local_explanations_top_error_samples.csv")

    # ======================================================================
    # 三、图表
    # ======================================================================
    # 1. SHAP beeswarm
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=INPUT_FEATURES, show=False)
    save_fig("shap_summary_beeswarm.png")

    # 2. SHAP bar 重要性
    fig = plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test, feature_names=INPUT_FEATURES, plot_type="bar", show=False)
    save_fig("shap_feature_importance_bar.png")

    # 3-5. Dependence plots (Top 3)
    top3_features = df_importance.head(3)
    for rank, (_, row) in enumerate(top3_features.iterrows(), 1):
        feat_name = row["feature"]
        feat_idx = INPUT_FEATURES.index(feat_name)
        fig = plt.figure(figsize=(7, 5))
        shap.dependence_plot(
            feat_idx, shap_values, X_test,
            feature_names=INPUT_FEATURES, show=False
        )
        save_fig(f"shap_dependence_top{rank}.png")

    # 6. 手动 mean_abs_shap 柱状图
    fig, ax = plt.subplots(figsize=(8, 5))
    feats = df_importance["feature"].tolist()
    vals = df_importance["mean_abs_shap"].tolist()
    ax.barh(feats[::-1], vals[::-1], color="#42A5F5")
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("SHAP Feature Importance (Mean Absolute SHAP)", fontsize=14)
    plt.tight_layout()
    save_fig("shap_mean_abs_importance.png")

    # ======================================================================
    # 四、与相关性对比
    # ======================================================================
    corr_path = TABLE_DIR / "target_correlation_ranking.csv"
    pearson_ranking = ""
    if corr_path.exists():
        df_corr = pd.read_csv(corr_path, encoding="utf-8-sig")
        pearson_top = df_corr.sort_values("pearson_abs", ascending=False)["feature"].iloc[0]
        shap_top = df_importance.iloc[0]["feature"]
        same = pearson_top == shap_top
        pearson_ranking = df_corr.to_string(index=False)

    # ======================================================================
    # 五、报告
    # ======================================================================
    lines: list[str] = []
    lines.append("# SHAP 模型解释与特征影响分析报告\n\n")

    lines.append("## 1. 分析目的\n\n")
    lines.append(
        "SHAP (SHapley Additive exPlanations) 用于解释机器学习模型对各特征的贡献。"
        "本文基于基础 XGBoost 模型进行 SHAP 分析，"
        "揭示各输入特征对顶板位移预测结果的影响机制。\n\n"
    )

    lines.append("## 2. 解释对象\n\n")
    lines.append("- 模型: XGBoost（final_base_xgboost.joblib）\n")
    lines.append("- 选择原因: XGBoost 是最终基础预测模型，SHAP TreeExplainer 对树模型解释效果稳定\n")
    lines.append("- 数据: 测试集 36 条样本\n\n")

    lines.append("## 3. 全局特征重要性排序\n\n")
    lines.append("| Rank | Feature | Mean ABS SHAP | Direction |\n")
    lines.append("|------|---------|:-------------:|----------|\n")
    for _, row in df_importance.iterrows():
        lines.append(
            f"| {row['importance_rank']} | {row['feature']} | {row['mean_abs_shap']:.4f} | {row['direction_hint']} |\n"
        )
    lines.append("\n")

    lines.append("## 4. Top 5 重要特征解释\n\n")
    for _, row in df_importance.head(5).iterrows():
        lines.append(
            f"### {row['importance_rank']}. {row['feature']}\n\n"
            f"- Mean |SHAP|: {row['mean_abs_shap']:.4f}\n"
            f"- 影响方向: {row['direction_hint']}\n"
            f"- 特征均值: {row['mean_feature_value']:.4f}\n\n"
        )

    lines.append("## 5. 局部样本解释\n\n")
    lines.append("以下为测试集中预测误差最大的 5 个样本的 SHAP 局部解释：\n\n")
    lines.append("| Sample | y_true | y_pred | Error | Top 1 Feat | Top 1 SHAP | Top 2 Feat | Top 2 SHAP |\n")
    lines.append("|--------|--------|--------|-------|-----------|:----------:|-----------|:----------:|\n")
    for lr in local_rows:
        lines.append(
            f"| {lr['sample_index']} | {lr['y_true']:.1f} | {lr['y_pred']:.1f} | {lr['prediction_error']:.1f} | "
            f"{lr['top_1_feature']} | {lr['top_1_shap']:.3f} | {lr['top_2_feature']} | {lr['top_2_shap']:.3f} |\n"
        )
    lines.append("\n")

    lines.append("## 6. SHAP 与相关性矩阵对比\n\n")
    lines.append(
        "相关性矩阵反映的是单变量线性关系，"
        "SHAP 反映的是模型内部非线性贡献。"
        "若二者排序不完全一致，说明顶板位移预测受到非线性和特征交互影响。\n\n"
    )
    if corr_path.exists() and pearson_top and not (pearson_top == shap_top):
        lines.append(
            f"Pearson 相关性最高特征为 {pearson_top}，"
            f"SHAP 重要性最高特征为 {shap_top}，"
            "二者不一致，进一步证实了变量间存在非线性耦合效应。\n\n"
        )
    else:
        lines.append(
            f"Pearson 相关性最高特征与 SHAP 重要性最高特征均为 {shap_top}，"
            "说明该特征在线性和非线性维度上均对顶板位移有重要影响。\n\n"
        )

    lines.append("## 7. 结论\n\n")
    lines.append(
        "SHAP 分析结果表明，不同输入变量对顶板位移预测结果的贡献存在明显差异。"
        "与传统相关性分析不同，SHAP 能够反映非线性模型中各特征对单个样本预测结果的边际贡献。"
        "特征重要性排序进一步说明顶板位移受围岩强度、埋深、巷道宽度及支护参数等多因素耦合作用影响，"
        "为后续支护参数优化和工程风险识别提供了可解释依据。\n\n"
    )

    lines.append("## 8. 输出文件清单\n\n")
    for fname in [
        "shap_feature_importance.csv",
        "shap_local_explanations_top_error_samples.csv",
        "shap_summary_beeswarm.png",
        "shap_feature_importance_bar.png",
        "shap_dependence_top1.png",
        "shap_dependence_top2.png",
        "shap_dependence_top3.png",
        "shap_mean_abs_importance.png",
        "08_shap_analysis_report.md",
    ]:
        lines.append(f"- `{fname}`\n")

    with open(REPORT_DIR / "08_shap_analysis_report.md", "w", encoding="utf-8-sig") as f:
        f.write("".join(lines))
    print(f"  已输出 08_shap_analysis_report.md")

    print("=" * 70)
    print("08_shap_analysis.py 执行完毕。")
    print("=" * 70)


if __name__ == "__main__":
    main()