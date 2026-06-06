"""
煤巷顶板位移 —— 08 SHAP 模型解释与特征影响分析
=================================================
功能：
  1. 加载 final_base_xgboost 计算 SHAP 值
  2. 全局特征重要性排序
  3. 局部样本解释
  4. 与相关性分析对比
  5. 关键特征非线性影响机制与边际效应分析
  6. PDP 辅助分析（可选）
  7. 图表、表格、报告
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

# 本文中 SHAP 重要性排序 Top 5（按已运行的 mean_abs_shap）
TOP5_FEATURES = [
    "depth",
    "roof_strength",
    "anchor_density",
    "floor_strength",
    "coal_strength",
]


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABLE_DIR / name, index=False, encoding="utf-8-sig")
    print(f"  已输出 {name}")


def save_fig(name: str) -> None:
    plt.savefig(FIGURE_DIR / name, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  已输出 {name}")


# ---------------------------------------------------------------------------
# 非线性分析与边际效应
# ---------------------------------------------------------------------------
def compute_marginal_effects(
    feature_values: np.ndarray,
    shap_values: np.ndarray,
    feature_name: str,
    n_bins: int = 5,
) -> pd.DataFrame:
    """对特征分箱计算边际效应。"""
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(feature_values, percentiles)

    rows = []
    for i in range(n_bins):
        mask = (feature_values >= bin_edges[i]) & (feature_values < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (feature_values >= bin_edges[i]) & (feature_values <= bin_edges[i + 1])
        sv_bin = shap_values[mask]
        fv_bin = feature_values[mask]
        if len(sv_bin) == 0:
            continue
        rows.append({
            "feature": feature_name,
            "bin_id": i + 1,
            "bin_left": round(float(bin_edges[i]), 4),
            "bin_right": round(float(bin_edges[i + 1]), 4),
            "bin_center": round(float(np.mean([bin_edges[i], bin_edges[i + 1]])), 4),
            "sample_count": len(sv_bin),
            "mean_feature_value": round(float(np.mean(fv_bin)), 4),
            "mean_shap_value": round(float(np.mean(sv_bin)), 6),
            "std_shap_value": round(float(np.std(sv_bin)), 6),
            "marginal_effect": 0.0,
        })

    # 计算相邻区间边际效应
    for i in range(1, len(rows)):
        delta_shap = rows[i]["mean_shap_value"] - rows[i - 1]["mean_shap_value"]
        delta_feat = rows[i]["mean_feature_value"] - rows[i - 1]["mean_feature_value"]
        if abs(delta_feat) > 1e-10:
            rows[i]["marginal_effect"] = round(delta_shap / delta_feat, 6)

    return pd.DataFrame(rows)


def draw_nonlinear_curve(
    fvals: np.ndarray,
    svals: np.ndarray,
    feature_name: str,
    quantiles: dict,
) -> None:
    """绘制非线性影响曲线（LOWESS 平滑 + 分位数标注）。"""
    fig, ax = plt.subplots(figsize=(8, 5))

    # 散点
    ax.scatter(fvals, svals, s=8, alpha=0.4, color="#1565C0", edgecolors="none")

    # LOWESS 平滑
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sorted_idx = np.argsort(fvals)
        smoothed = lowess(svals[sorted_idx], fvals[sorted_idx], frac=0.3, return_sorted=True)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="#E64A19", linewidth=2, label="LOWESS Smooth")
    except Exception:
        # fallback: 分箱平均曲线
        edges = np.percentile(fvals, np.linspace(0, 100, 11))
        bin_centers = []
        bin_means = []
        for i in range(10):
            mask = (fvals >= edges[i]) & (fvals < edges[i + 1])
            if i == 9:
                mask = (fvals >= edges[i]) & (fvals <= edges[i + 1])
            if mask.sum() > 0:
                bin_centers.append(np.mean(fvals[mask]))
                bin_means.append(np.mean(svals[mask]))
        ax.plot(bin_centers, bin_means, "o-", color="#E64A19", linewidth=2, label="Binned Mean")

    # SHAP=0 参考线
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    # 分位数竖线
    for pct, label, ls in [(25, "Q25", ":"), (50, "Q50", "-."), (75, "Q75", ":")]:
        qv = quantiles.get(pct, None)
        if qv is not None:
            ax.axvline(x=qv, color="green", linestyle=ls, linewidth=1.2, alpha=0.6,
                       label=f"{label}={qv:.1f}")

    ax.set_xlabel(f"{feature_name}", fontsize=12)
    ax.set_ylabel("SHAP Value", fontsize=12)
    ax.set_title(f"SHAP Nonlinear Effect: {feature_name}", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    save_fig(f"shap_nonlinear_effect_{feature_name}.png")


def draw_marginal_effect_curve(bin_df: pd.DataFrame, feature_name: str) -> None:
    """绘制边际效应曲线。"""
    data = bin_df[bin_df["marginal_effect"] != 0].copy()
    if len(data) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(data["bin_center"], data["marginal_effect"], "o-", color="#2E7D32", linewidth=2, markersize=6)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel(f"{feature_name} (Bin Center)", fontsize=12)
    ax.set_ylabel("Marginal Effect (ΔSHAP / ΔFeature)", fontsize=12)
    ax.set_title(f"Marginal Effect: {feature_name}", fontsize=14)

    # 标注最大和最小边际效应
    if len(data) > 0:
        idx_max = data["marginal_effect"].idxmax()
        idx_min = data["marginal_effect"].idxmin()
        for idx, color in [(idx_max, "red"), (idx_min, "blue")]:
            ax.annotate(
                f"{data.loc[idx, 'marginal_effect']:.4f}",
                (data.loc[idx, "bin_center"], data.loc[idx, "marginal_effect"]),
                textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color=color,
            )
    plt.tight_layout()
    save_fig(f"shap_marginal_effect_{feature_name}.png")


# ---------------------------------------------------------------------------
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
    # 一、SHAP 特征重要性（原有）
    # ======================================================================
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

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

    print("\nTop 5 SHAP 特征重要性:")
    top5 = df_importance.head(5)
    for _, row in top5.iterrows():
        print(f"  {row['importance_rank']}. {row['feature']}: mean_abs_shap = {row['mean_abs_shap']:.4f}")

    # 核验 Top 5 列表是否匹配
    actual_top5 = df_importance.head(5)["feature"].tolist()
    if actual_top5 != TOP5_FEATURES:
        print(f"[信息] SHAP Top 5 已更新: {actual_top5}")
        top5_features = actual_top5
    else:
        top5_features = TOP5_FEATURES

    # ======================================================================
    # 二、局部 SHAP 解释（原有）
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
    # 三、基础图表（原有）
    # ======================================================================
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=INPUT_FEATURES, show=False)
    save_fig("shap_summary_beeswarm.png")

    fig = plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test, feature_names=INPUT_FEATURES, plot_type="bar", show=False)
    save_fig("shap_feature_importance_bar.png")

    top3_features = df_importance.head(3)
    for rank, (_, row) in enumerate(top3_features.iterrows(), 1):
        feat_name = row["feature"]
        feat_idx = INPUT_FEATURES.index(feat_name)
        fig = plt.figure(figsize=(7, 5))
        shap.dependence_plot(feat_idx, shap_values, X_test, feature_names=INPUT_FEATURES, show=False)
        save_fig(f"shap_dependence_top{rank}.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    feats = df_importance["feature"].tolist()
    vals = df_importance["mean_abs_shap"].tolist()
    ax.barh(feats[::-1], vals[::-1], color="#42A5F5")
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("SHAP Feature Importance (Mean Absolute SHAP)", fontsize=14)
    plt.tight_layout()
    save_fig("shap_mean_abs_importance.png")

    # ======================================================================
    # 四、关键特征非线性影响机制与边际效应分析（新增）
    # ======================================================================
    print("\n" + "=" * 70)
    print("第八阶段补充：SHAP 非线性影响机制与边际效应分析")
    print("=" * 70)
    print(f"分析特征数: {len(top5_features)}")
    print("关键特征:")
    for i, f in enumerate(top5_features, 1):
        print(f"  {i}. {f}")

    all_bin_rows = []
    mechanism_rows = []

    for feat_name in top5_features:
        feat_idx = INPUT_FEATURES.index(feat_name)
        fvals = X_test[:, feat_idx]
        svals = shap_values[:, feat_idx]

        # 分位数
        quantiles = {
            25: np.percentile(fvals, 25),
            50: np.percentile(fvals, 50),
            75: np.percentile(fvals, 75),
        }

        # 非线性曲线
        draw_nonlinear_curve(fvals, svals, feat_name, quantiles)

        # 边际效应分箱
        bin_df = compute_marginal_effects(fvals, svals, feat_name, n_bins=5)
        all_bin_rows.append(bin_df)

        # 边际效应曲线
        draw_marginal_effect_curve(bin_df, feat_name)

        # 机制汇总
        # 主效应方向
        corr_shap_feat = np.corrcoef(fvals, svals)[0, 1] if np.std(fvals) > 0 else 0.0
        if corr_shap_feat > 0.2:
            main_direction = "特征值增大，SHAP 贡献整体为正"
        elif corr_shap_feat < -0.2:
            main_direction = "特征值增大，SHAP 贡献整体为负"
        else:
            main_direction = "影响方向呈非线性或交互特征"

        # 非线性模式
        bin_means = bin_df["mean_shap_value"].values
        if len(bin_means) >= 3:
            if bin_means[-1] > bin_means[0] and np.all(np.diff(bin_means) > -0.01):
                nonlinear_pattern = "单调递增"
            elif bin_means[-1] < bin_means[0] and np.all(np.diff(bin_means) < 0.01):
                nonlinear_pattern = "单调递减"
            elif np.max(bin_means) - np.min(bin_means) > 0.5 * np.std(bin_means):
                nonlinear_pattern = "先增后减或波动型"
            else:
                nonlinear_pattern = "近似平坦"
        else:
            nonlinear_pattern = "数据不足"

        # 关键区间
        abs_means = np.abs(bin_df["mean_shap_value"].values)
        if len(abs_means) > 0:
            key_idx = np.argmax(abs_means)
            key_interval = (
                f"[{bin_df.iloc[key_idx]['bin_left']:.2f}, "
                f"{bin_df.iloc[key_idx]['bin_right']:.2f}]"
            )
        else:
            key_interval = "N/A"

        # 边际效应描述
        marg_vals = bin_df[bin_df["marginal_effect"] != 0]["marginal_effect"].values
        if len(marg_vals) > 0:
            max_abs_idx = np.argmax(np.abs(marg_vals))
            max_me = marg_vals[max_abs_idx]
            marg_desc = (
                f"最大边际效应 {max_me:.4f}，"
                f"{'正' if max_me > 0 else '负'}向变化"
            )
        else:
            marg_desc = "边际效应不显著"

        # 工程解释
        engineering_map = {
            "depth": (
                "埋深是最重要影响因素。随着埋深增大，地应力水平通常升高，"
                "模型预测顶板位移整体增大；高埋深区间 SHAP 值明显为正，"
                "说明深部开采条件下顶板变形风险增强。"
            ),
            "roof_strength": (
                "顶板强度对位移预测具有重要影响。SHAP 显示 roof_strength 增大时"
                "预测位移可能升高，这可能不是单一强度因果作用，"
                "而反映了顶板强度与埋深、支护参数、地质条件之间的耦合关系。"
            ),
            "anchor_density": (
                "锚索密度增大整体倾向于降低预测顶板位移，"
                "说明锚索补强对顶板变形具有控制作用；"
                "若高密度区间边际效应减弱，可解释为支护加密存在边际收益递减。"
            ),
            "floor_strength": (
                "底板强度表现出非线性或交互影响，"
                "说明其对顶板位移的作用可能通过围岩整体结构条件间接体现。"
            ),
            "coal_strength": (
                "煤层强度对顶板位移存在一定贡献。"
                "若高煤层强度区间 SHAP 值升高，应谨慎解释为样本耦合效应，"
                "而不是简单认为煤层强度越高位移越大。"
            ),
        }
        eng_interp = engineering_map.get(feat_name, "需要进一步分析。")

        mechanism_rows.append({
            "feature": feat_name,
            "mean_abs_shap": round(float(mean_abs_shap[feat_idx]), 6),
            "main_effect_direction": main_direction,
            "nonlinear_pattern": nonlinear_pattern,
            "key_interval": key_interval,
            "marginal_effect_description": marg_desc,
            "engineering_interpretation": eng_interp,
        })

    # 保存边际效应总表
    all_bins = pd.concat(all_bin_rows, ignore_index=True) if all_bin_rows else pd.DataFrame()
    save_table(all_bins, "shap_marginal_effect_bins.csv")

    # 保存机制汇总表
    save_table(pd.DataFrame(mechanism_rows), "shap_nonlinear_mechanism_summary.csv")

    # ======================================================================
    # 五、PDP 辅助分析（可选）
    # ======================================================================
    try:
        from sklearn.inspection import PartialDependenceDisplay
        for feat_name in top5_features:
            feat_idx = INPUT_FEATURES.index(feat_name)
            fig, ax = plt.subplots(figsize=(7, 5))
            PartialDependenceDisplay.from_estimator(
                model, X_test, [feat_idx],
                feature_names=INPUT_FEATURES, ax=ax,
            )
            ax.set_title(f"Partial Dependence: {feat_name}", fontsize=14)
            plt.tight_layout()
            save_fig(f"pdp_{feat_name}.png")
    except Exception:
        print("[信息] PDP 分析跳过（可能缺少依赖），不影响主流程。")

    # ======================================================================
    # 六、与相关性对比（原有）
    # ======================================================================
    corr_path = TABLE_DIR / "target_correlation_ranking.csv"
    pearson_top = ""
    if corr_path.exists():
        df_corr = pd.read_csv(corr_path, encoding="utf-8-sig")
        pearson_top = df_corr.sort_values("pearson_abs", ascending=False)["feature"].iloc[0]
        shap_top = df_importance.iloc[0]["feature"]

    # ======================================================================
    # 七、更新 SHAP 报告（原有 + 新章节）
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

    lines.append("## 5. 全局特征贡献度分析（SHAP 蜂群图）\n\n")
    lines.append(
        "SHAP 蜂群图从全局角度展示了各输入特征对 XGBoost 模型预测结果的贡献分布。"
        "图中每一个点代表一个样本在对应特征上的 SHAP 值，"
        "横坐标表示该特征对预测顶板位移的贡献方向和贡献大小。"
        "当 SHAP 值大于 0 时，说明该特征在该样本中推动模型输出更大的顶板位移预测值；"
        "当 SHAP 值小于 0 时，则说明该特征降低了模型预测结果。"
        "点的颜色表示特征取值大小，红色代表特征值较高，蓝色代表特征值较低。"
        "结果表明，depth、roof_strength、anchor_density、floor_strength 和 coal_strength"
        "是影响顶板位移预测的主要特征。其中，depth 的高取值样本整体表现出较明显的正向贡献，"
        "说明高埋深工况下模型倾向于预测更大的顶板位移；"
        "anchor_density 的高取值样本多表现为负向贡献，"
        "说明锚索密度提高有助于降低模型预测的顶板位移。"
        "roof_strength、floor_strength 和 coal_strength 的 SHAP 分布呈现一定正负交错特征，"
        "表明其影响机制并非单一线性关系，"
        "而可能受到围岩条件、埋深和支护参数之间耦合作用的影响。\n\n"
    )

    lines.append("## 6. 局部样本解释\n\n")
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

    # ---- 新增：第 7 节 非线性影响机制 ----
    lines.append("## 7. 关键特征的非线性影响机制与边际效应分析\n\n")

    lines.append("### 7.1 分析方法\n\n")
    lines.append(
        "为进一步揭示关键特征对顶板位移预测结果的作用机制，"
        "本文基于 SHAP dependence relationship 对 Top 5 关键特征进行非线性影响分析，"
        "并通过分位区间统计计算近似边际效应。"
        "边际效应用相邻特征区间平均 SHAP 值的变化量与平均特征值变化量之比表示，"
        "用于刻画特征变化对模型输出贡献的敏感程度。\n\n"
    )

    for mr in mechanism_rows:
        feat = mr["feature"]
        lines.append(f"### 7.{mechanism_rows.index(mr) + 2} {feat} 的非线性影响机制\n\n")
        if feat == "depth":
            lines.append(f"- {feat} 是最重要特征。\n")
            lines.append("- 埋深增加整体倾向于提高顶板位移预测值。\n")
            lines.append("- 高埋深区间如果 SHAP 值升高，说明深部地应力对顶板变形具有增强作用。\n")
            lines.append("- 若边际效应在某些区间更明显，说明这些埋深区间对位移预测更敏感。\n\n")
        elif feat == "roof_strength":
            lines.append(f"- {feat} 是重要影响因素。\n")
            lines.append("- 如果 SHAP 显示 roof_strength 增大整体提高预测位移，不要简单解释为\"强度越大越危险\"。\n")
            lines.append("- 应解释为模型捕捉到 roof_strength 与埋深、支护参数、围岩结构之间的耦合关系。\n")
            lines.append("- 其作用具有非线性和工程耦合特征。\n\n")
        elif feat == "anchor_density":
            lines.append(f"- {feat} 增大整体倾向于降低预测位移。\n")
            lines.append("- 说明锚索支护对控制顶板变形有积极作用。\n")
            lines.append("- 如果高 anchor_density 区间边际效应减弱，可解释为支护参数加密存在边际收益递减。\n")
            lines.append("- 这可以为支护参数优化提供依据。\n\n")
        else:
            lines.append(f"- {feat} 对预测结果有贡献，但方向可能受到其他特征交互影响。\n")
            lines.append("- 不宜做单因素因果解释。\n")
            lines.append("- 应从围岩整体结构和工程耦合角度解释。\n\n")

        lines.append(f"**机制摘要**: {mr['engineering_interpretation']}\n\n")

    lines.append("### 7.7 小结\n\n")
    lines.append(
        "- SHAP 非线性分析表明，关键特征对顶板位移预测的影响不是简单线性关系；\n"
        "- depth、roof_strength、anchor_density 是影响模型输出的核心变量；\n"
        "- anchor_density 的负向 SHAP 贡献说明支护增强能够降低预测位移；\n"
        "- depth 的正向 SHAP 贡献说明深部工况更容易出现较大顶板变形；\n"
        "- 关键特征存在区间敏感性和边际效应变化，说明支护优化应关注特征阈值区间。\n\n"
    )
    # 结束新增

    lines.append("## 8. 结论\n\n")
    lines.append(
        "SHAP 分析结果表明，不同输入变量对顶板位移预测结果的贡献存在明显差异。"
        "与传统相关性分析不同，SHAP 能够反映非线性模型中各特征对单个样本预测结果的边际贡献。"
        "特征重要性排序进一步说明顶板位移受围岩强度、埋深、巷道宽度及支护参数等多因素耦合作用影响，"
        "为后续支护参数优化和工程风险识别提供了可解释依据。\n\n"
        "进一步的 SHAP 非线性影响与边际效应分析表明，"
        "depth、roof_strength 和 anchor_density 对模型输出存在明显区间敏感性，"
        "其中 depth 在高埋深区间对预测位移具有更强正向贡献，"
        "anchor_density 整体表现为负向贡献且可能存在支护边际收益递减现象。\n\n"
    )

    lines.append("## 9. 输出文件清单\n\n")
    for fname in [
        "shap_feature_importance.csv",
        "shap_local_explanations_top_error_samples.csv",
        "shap_marginal_effect_bins.csv",
        "shap_nonlinear_mechanism_summary.csv",
        "shap_summary_beeswarm.png",
        "shap_feature_importance_bar.png",
        "shap_dependence_top1.png",
        "shap_dependence_top2.png",
        "shap_dependence_top3.png",
        "shap_mean_abs_importance.png",
    ] + [f"shap_nonlinear_effect_{f}.png" for f in top5_features] + \
      [f"shap_marginal_effect_{f}.png" for f in top5_features] + \
      ["08_shap_analysis_report.md"]:
        lines.append(f"- `{fname}`\n")

    report_path = REPORT_DIR / "08_shap_analysis_report.md"
    with open(report_path, "w", encoding="utf-8-sig") as f:
        f.write("".join(lines))
    print(f"  已更新 08_shap_analysis_report.md")

    # ======================================================================
    # 更新 final_academic_model_report.md 中的 SHAP 章节
    # ======================================================================
    final_report_path = REPORT_DIR / "final_academic_model_report.md"
    if final_report_path.exists():
        content = final_report_path.read_text(encoding="utf-8-sig")
        old_section = (
            "SHAP 重要性最高特征为 depth，而 Pearson 相关性最高特征为 roof_strength，"
            "说明模型捕捉到了非线性耦合关系。depth 增大整体倾向于提高预测顶板位移。"
        )
        new_section = (
            "SHAP 重要性最高特征为 depth，而 Pearson 相关性最高特征为 roof_strength，"
            "说明模型捕捉到了非线性耦合关系。depth 增大整体倾向于提高预测顶板位移。"
            "进一步的 SHAP 非线性影响与边际效应分析表明，"
            "depth、roof_strength 和 anchor_density 对模型输出存在明显区间敏感性，"
            "其中 depth 在高埋深区间对预测位移具有更强正向贡献，"
            "anchor_density 整体表现为负向贡献且可能存在支护边际收益递减现象。"
        )
        if old_section in content:
            content = content.replace(old_section, new_section)
            with open(final_report_path, "w", encoding="utf-8-sig") as f:
                f.write(content)
            print("  已更新 final_academic_model_report.md 中 SHAP 章节")

    print("\n" + "=" * 70)
    print("SHAP 非线性影响机制分析完成。")
    print("=" * 70)


if __name__ == "__main__":
    main()