"""
煤巷顶板位移 —— 04 基础预测模型对比
=====================================
功能：
  1. 11 个基础模型训练与测试集评估
  2. 10 折交叉验证
  3. 模型对比图表
  4. 最优模型保存
  5. Markdown 报告
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

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    DATA_PROCESSED_DIR,
    FIGURE_DIR,
    TABLE_DIR,
    MODEL_DIR,
    REPORT_DIR,
    ensure_dirs,
)

# ---------------------------------------------------------------------------
# 中文字体
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

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 10


# ---------------------------------------------------------------------------
# SMAPE
# ---------------------------------------------------------------------------
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """对称平均绝对百分比误差，分母为 0 时安全处理。"""
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1e-10, denom)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABLE_DIR / name, index=False, encoding="utf-8-sig")
    print(f"  已输出 {name}")


def save_fig(name: str) -> None:
    plt.savefig(FIGURE_DIR / name, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  已输出 {name}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ensure_dirs()

    print("=" * 70)
    print("第四阶段：基础预测模型对比")
    print("=" * 70)

    # ---- 加载数据 ----
    df = pd.read_csv(DATA_PROCESSED_DIR / "dataset_modeling_176.csv", encoding="utf-8-sig")
    X = df[INPUT_FEATURES].values
    y = df[TARGET].values
    n = len(df)
    print(f"主建模样本量: {n}")

    # ---- 划分 ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    n_train = len(X_train)
    n_test = len(X_test)
    print(f"训练集样本量: {n_train}")
    print(f"测试集样本量: {n_test}")
    print(f"划分随机种子: {RANDOM_STATE}")

    # ---- 保存划分信息 ----
    split_info = {
        "random_state": RANDOM_STATE,
        "train_size": n_train,
        "test_size": n_test,
        "target_train_mean": float(np.mean(y_train)),
        "target_test_mean": float(np.mean(y_test)),
        "target_train_std": float(np.std(y_train, ddof=1)),
        "target_test_std": float(np.std(y_test, ddof=1)),
        "target_train_min": float(np.min(y_train)),
        "target_test_min": float(np.min(y_test)),
        "target_train_max": float(np.max(y_train)),
        "target_test_max": float(np.max(y_test)),
    }
    save_table(pd.DataFrame([split_info]), "baseline_train_test_split_info.csv")

    # ---- 模型定义 ----
    models: list[dict] = [
        {
            "name": "Dummy Regressor",
            "short_name": "DummyRegressor",
            "pipeline": Pipeline([
                ("model", DummyRegressor(strategy="mean"))
            ]),
        },
        {
            "name": "Ridge Regression",
            "short_name": "Ridge",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]),
        },
        {
            "name": "ElasticNet",
            "short_name": "ElasticNet",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
            ]),
        },
        {
            "name": "SVR (RBF)",
            "short_name": "SVR",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1)),
            ]),
        },
        {
            "name": "ANN Neural Network",
            "short_name": "ANN",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", MLPRegressor(
                    hidden_layer_sizes=(32, 16), activation="relu",
                    solver="adam", max_iter=3000, random_state=RANDOM_STATE
                )),
            ]),
        },
        {
            "name": "Random Forest",
            "short_name": "RandomForest",
            "pipeline": Pipeline([
                ("model", RandomForestRegressor(
                    n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
                )),
            ]),
        },
        {
            "name": "Gradient Boosting",
            "short_name": "GradientBoosting",
            "pipeline": Pipeline([
                ("model", GradientBoostingRegressor(
                    n_estimators=300, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
                )),
            ]),
        },
        {
            "name": "AdaBoost",
            "short_name": "AdaBoost",
            "pipeline": Pipeline([
                ("model", AdaBoostRegressor(
                    n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE
                )),
            ]),
        },
        {
            "name": "XGBoost",
            "short_name": "XGBoost",
            "pipeline": Pipeline([
                ("model", XGBRegressor(
                    n_estimators=300, learning_rate=0.05, max_depth=3,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=RANDOM_STATE, objective="reg:squarederror",
                    verbosity=0,
                )),
            ]),
        },
        {
            "name": "LightGBM",
            "short_name": "LightGBM",
            "pipeline": Pipeline([
                ("model", LGBMRegressor(
                    n_estimators=300, learning_rate=0.05, max_depth=3,
                    random_state=RANDOM_STATE, verbose=-1,
                )),
            ]),
        },
        {
            "name": "CatBoost",
            "short_name": "CatBoost",
            "pipeline": Pipeline([
                ("model", CatBoostRegressor(
                    iterations=300, learning_rate=0.05, depth=3,
                    random_seed=RANDOM_STATE, verbose=False,
                )),
            ]),
        },
    ]

    # ---- 训练 & 评估 ----
    results_rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {"y_true": y_test}

    print(f"\n{'Model':<25s} {'Train R2':>8s} {'Test R2':>8s} {'Test MAE':>8s} {'Test RMSE':>8s}")
    print("-" * 65)

    for m in models:
        name = m["name"]
        pipe = m["pipeline"]
        pipe.fit(X_train, y_train)

        # 训练集
        y_train_pred = pipe.predict(X_train)
        train_r2 = float(r2_score(y_train, y_train_pred))
        train_mae = float(mean_absolute_error(y_train, y_train_pred))
        train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))

        # 测试集
        y_test_pred = pipe.predict(X_test)
        test_r2 = float(r2_score(y_test, y_test_pred))
        test_mae = float(mean_absolute_error(y_test, y_test_pred))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
        test_smape = smape(y_test, y_test_pred)

        # 10 折 CV
        cv_r2 = cross_val_score(pipe, X, y, cv=CV_FOLDS, scoring="r2")
        cv_rmse = cross_val_score(pipe, X, y, cv=CV_FOLDS, scoring="neg_root_mean_squared_error")

        results_rows.append({
            "model": name,
            "short_name": m["short_name"],
            "train_R2": round(train_r2, 4),
            "train_MAE": round(train_mae, 4),
            "train_RMSE": round(train_rmse, 4),
            "test_R2": round(test_r2, 4),
            "test_MAE": round(test_mae, 4),
            "test_RMSE": round(test_rmse, 4),
            "test_SMAPE": round(test_smape, 4),
            "CV_R2_mean": round(float(np.mean(cv_r2)), 4),
            "CV_R2_std": round(float(np.std(cv_r2)), 4),
            "CV_RMSE_mean": round(float(-np.mean(cv_rmse)), 4),
            "CV_RMSE_std": round(float(np.std(cv_rmse)), 4),
        })

        predictions[m["short_name"]] = y_test_pred

        print(f"{name:<25s} {train_r2:>8.4f} {test_r2:>8.4f} {test_mae:>8.4f} {test_rmse:>8.4f}")

    df_results = pd.DataFrame(results_rows)
    save_table(df_results, "baseline_model_results.csv")

    # ---- 最优模型 ----
    best_idx = df_results["test_RMSE"].idxmin()
    best_row = df_results.iloc[best_idx]
    best_name = best_row["model"]
    best_short = best_row["short_name"]
    print(f"\n最优基础模型: {best_name}")
    print(f"  测试集 R2:  {best_row['test_R2']:.4f}")
    print(f"  测试集 MAE: {best_row['test_MAE']:.4f}")
    print(f"  测试集 RMSE: {best_row['test_RMSE']:.4f}")

    save_table(pd.DataFrame([best_row]), "baseline_best_model.csv")

    # ---- 预测值保存 ----
    pred_df = pd.DataFrame()
    pred_df["sample_index"] = range(len(y_test))
    pred_df["y_true"] = y_test
    for m in models:
        pred_df[m["short_name"] + "_pred"] = predictions[m["short_name"]]
    pred_df["best_model_pred"] = predictions[best_short]
    save_table(pred_df, "baseline_predictions.csv")

    # ---- 保存最优模型 ----
    best_pipe = models[best_idx]["pipeline"]
    joblib.dump(best_pipe, MODEL_DIR / "baseline_best_model.joblib")
    print(f"  已输出 baseline_best_model.joblib")
    joblib.dump(INPUT_FEATURES, MODEL_DIR / "feature_columns.joblib")
    print(f"  已输出 feature_columns.joblib")

    # ======================================================================
    # 图表
    # ======================================================================
    model_labels = [r["short_name"] for r in results_rows]

    # R2 对比
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(model_labels))
    w = 0.3
    ax.bar(x - w/2, [r["train_R2"] for r in results_rows], w, label="Train R2", color="#42A5F5")
    ax.bar(x + w/2, [r["test_R2"] for r in results_rows], w, label="Test R2", color="#FF7043")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("R2 Score", fontsize=12)
    ax.set_title("Baseline Model R2 Comparison", fontsize=14)
    ax.legend(fontsize=10)
    sns_despine = plt.gcf()
    plt.tight_layout()
    save_fig("baseline_model_r2_comparison.png")

    # RMSE 对比
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w/2, [r["train_RMSE"] for r in results_rows], w, label="Train RMSE", color="#66BB6A")
    ax.bar(x + w/2, [r["test_RMSE"] for r in results_rows], w, label="Test RMSE", color="#FF7043")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("RMSE / mm", fontsize=12)
    ax.set_title("Baseline Model RMSE Comparison", fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig("baseline_model_rmse_comparison.png")

    # MAE 对比
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w/2, [r["train_MAE"] for r in results_rows], w, label="Train MAE", color="#AB47BC")
    ax.bar(x + w/2, [r["test_MAE"] for r in results_rows], w, label="Test MAE", color="#FF7043")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("MAE / mm", fontsize=12)
    ax.set_title("Baseline Model MAE Comparison", fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig("baseline_model_mae_comparison.png")

    # 最优模型散点图
    fig, ax = plt.subplots(figsize=(6, 6))
    best_pred = predictions[best_short]
    ax.scatter(y_test, best_pred, alpha=0.6, edgecolors="k", linewidth=0.3, color="#1565C0")
    lims = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Fit")
    ax.set_xlabel("True Roof Displacement / mm", fontsize=12)
    ax.set_ylabel("Predicted Roof Displacement / mm", fontsize=12)
    ax.set_title(f"Best Baseline: {best_name}\n"
                 f"Test R2={best_row['test_R2']:.4f}, RMSE={best_row['test_RMSE']:.2f} mm",
                 fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig("baseline_best_model_prediction_scatter.png")

    # ======================================================================
    # 报告
    # ======================================================================
    lines: list[str] = []
    lines.append("# 基础预测模型对比报告\n\n")

    lines.append("## 1. 数据来源\n\n")
    lines.append(f"- 数据文件: `data/processed/dataset_modeling_176.csv`\n")
    lines.append(f"- 样本量: {n}\n")
    lines.append(f"- 输入特征 ({len(INPUT_FEATURES)} 个): {', '.join(INPUT_FEATURES)}\n")
    lines.append(f"- 目标变量: {TARGET}\n\n")

    lines.append("## 2. 训练/测试集划分\n\n")
    lines.append(f"- 划分方式: `train_test_split(test_size={TEST_SIZE}, random_state={RANDOM_STATE})`\n")
    lines.append(f"- 训练集: {n_train} 条\n")
    lines.append(f"- 测试集: {n_test} 条\n")
    lines.append(f"- 当前步骤使用 random_state={RANDOM_STATE} 作为基础模型对比划分，\n")
    lines.append(f"  后续通过样本空间方差控制划分优化模块搜索更稳定的划分种子。\n\n")

    lines.append("## 3. 模型列表\n\n")
    lines.append("| 序号 | 模型 | 说明 |\n")
    lines.append("|-----|------|------|\n")
    for i, m in enumerate(models, 1):
        lines.append(f"| {i} | {m['name']} | - |\n")
    lines.append("\n")

    lines.append("## 4. 训练集与测试集结果\n\n")
    lines.append(
        "| Model | Train R2 | Train MAE | Train RMSE | Test R2 | Test MAE | Test RMSE | Test SMAPE |\n"
    )
    lines.append(
        "|-------|----------|-----------|------------|---------|----------|-----------|------------|\n"
    )
    for r in results_rows:
        lines.append(
            f"| {r['model']} | {r['train_R2']:.4f} | {r['train_MAE']:.2f} | {r['train_RMSE']:.2f} | "
            f"{r['test_R2']:.4f} | {r['test_MAE']:.2f} | {r['test_RMSE']:.2f} | {r['test_SMAPE']:.2f}% |\n"
        )
    lines.append("\n")

    lines.append("## 5. 10 折交叉验证结果\n\n")
    lines.append("| Model | CV R2 Mean | CV R2 Std | CV RMSE Mean | CV RMSE Std |\n")
    lines.append("|-------|-----------|-----------|-------------|------------|\n")
    for r in results_rows:
        lines.append(
            f"| {r['model']} | {r['CV_R2_mean']:.4f} | {r['CV_R2_std']:.4f} | "
            f"{r['CV_RMSE_mean']:.2f} | {r['CV_RMSE_std']:.2f} |\n"
        )
    lines.append("\n")

    lines.append("## 6. 最优基础模型\n\n")
    lines.append(f"- 最优模型: **{best_name}**\n")
    lines.append(f"- 测试集 R2: {best_row['test_R2']:.4f}\n")
    lines.append(f"- 测试集 MAE: {best_row['test_MAE']:.2f} mm\n")
    lines.append(f"- 测试集 RMSE: {best_row['test_RMSE']:.2f} mm\n")
    lines.append(f"- 测试集 SMAPE: {best_row['test_SMAPE']:.2f}%\n\n")

    lines.append("## 7. 过拟合分析\n\n")
    train_r2_best = best_row["train_R2"]
    test_r2_best = best_row["test_R2"]
    if train_r2_best - test_r2_best > 0.15:
        lines.append(
            f"最优模型训练集 R2 ({train_r2_best:.4f}) 与测试集 R2 ({test_r2_best:.4f}) "
            f"差距较大，存在一定过拟合现象，需在贝叶斯优化中加入正则化参数搜索。\n\n"
        )
    else:
        lines.append(
            f"最优模型训练集 R2 ({train_r2_best:.4f}) 与测试集 R2 ({test_r2_best:.4f}) "
            f"差距在可接受范围内。\n\n"
        )

    lines.append("## 8. 后续优化方向\n\n")
    lines.append(
        "基础模型对比结果表明，非线性集成模型整体优于线性模型，"
        "说明顶板位移与围岩强度、埋深、巷道宽度及支护参数之间存在明显非线性关系。"
        "考虑到基础模型参数尚未经过系统优化，后续采用贝叶斯优化方法对表现较优的模型进行超参数寻优，"
        "并进一步构建残差修正网络以降低局部样本预测误差。\n\n"
    )

    lines.append("## 9. 输出文件清单\n\n")
    for fname in [
        "baseline_model_results.csv",
        "baseline_best_model.csv",
        "baseline_predictions.csv",
        "baseline_train_test_split_info.csv",
        "baseline_model_r2_comparison.png",
        "baseline_model_rmse_comparison.png",
        "baseline_model_mae_comparison.png",
        "baseline_best_model_prediction_scatter.png",
        "baseline_best_model.joblib",
        "feature_columns.joblib",
        "04_baseline_model_report.md",
    ]:
        lines.append(f"- `{fname}`\n")

    with open(REPORT_DIR / "04_baseline_model_report.md", "w", encoding="utf-8-sig") as f:
        f.write("".join(lines))
    print(f"  已输出 04_baseline_model_report.md")

    print("=" * 70)
    print("04_baseline_models.py 执行完毕。")
    print("=" * 70)


if __name__ == "__main__":
    main()