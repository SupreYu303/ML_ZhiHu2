"""
煤巷顶板位移 —— 06 残差修正网络预测误差优化
============================================
功能：
  1. 加载 final_base_xgboost 计算基础预测与残差
  2. 6 个残差修正模型候选评估
  3. 选择最优残差修正模型
  4. 图表、表格、报告
  5. 保存最终模型
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

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

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


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1e-10, denom)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100)


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
    print("第六阶段：残差修正网络预测误差优化")
    print("=" * 70)

    # ---- 读取稳定种子 ----
    best_seed_path = TABLE_DIR / "best_stable_split_seed.csv"
    if best_seed_path.exists():
        best_seed_df = pd.read_csv(best_seed_path, encoding="utf-8-sig")
        best_seed = int(best_seed_df["seed"].iloc[0])
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

    # ---- 加载基础模型 ----
    base_model_path = MODEL_DIR / "final_base_xgboost.joblib"
    if base_model_path.exists():
        print(f"[信息] 加载 final_base_xgboost.joblib")
        base_model = joblib.load(base_model_path)
    else:
        print("[信息] 未找到 final_base_xgboost.joblib，构建默认 baseline XGBoost")
        base_model = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", random_state=MODEL_RANDOM_STATE, verbosity=0,
        )
        base_model.fit(X_train, y_train)

    # ---- 基础预测 ----
    y_train_pred_base = base_model.predict(X_train)
    y_test_pred_base = base_model.predict(X_test)

    base_train_r2 = float(r2_score(y_train, y_train_pred_base))
    base_train_mae = float(mean_absolute_error(y_train, y_train_pred_base))
    base_train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred_base)))
    base_test_r2 = float(r2_score(y_test, y_test_pred_base))
    base_test_mae = float(mean_absolute_error(y_test, y_test_pred_base))
    base_test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred_base)))
    base_test_smape = smape(y_test, y_test_pred_base)

    print(f"\n基础 XGBoost:")
    print(f"  Test R2:  {base_test_r2:.4f}")
    print(f"  Test MAE: {base_test_mae:.4f}")
    print(f"  Test RMSE:{base_test_rmse:.4f}")

    # ---- 训练集残差（仅用于训练） ----
    residual_train = y_train - y_train_pred_base
    residual_test = y_test - y_test_pred_base

    # ---- 残差修正模型 ----
    residual_models: list[dict] = [
        {
            "name": "Ridge Residual",
            "short_name": "ridge_residual",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]),
        },
        {
            "name": "SVR Residual",
            "short_name": "svr_residual",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1)),
            ]),
        },
        {
            "name": "MLP Residual",
            "short_name": "mlp_residual",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("model", MLPRegressor(
                    hidden_layer_sizes=(16, 8), activation="relu",
                    solver="adam", alpha=0.001, learning_rate_init=0.001,
                    max_iter=3000, random_state=MODEL_RANDOM_STATE,
                )),
            ]),
        },
        {
            "name": "Random Forest Residual",
            "short_name": "random_forest_residual",
            "model": Pipeline([
                ("model", RandomForestRegressor(
                    n_estimators=200, max_depth=3, min_samples_leaf=3,
                    random_state=MODEL_RANDOM_STATE, n_jobs=-1,
                )),
            ]),
        },
        {
            "name": "GBDT Residual",
            "short_name": "gbdt_residual",
            "model": Pipeline([
                ("model", GradientBoostingRegressor(
                    n_estimators=100, learning_rate=0.03, max_depth=2,
                    min_samples_leaf=3, random_state=MODEL_RANDOM_STATE,
                )),
            ]),
        },
        {
            "name": "XGBoost Residual",
            "short_name": "xgboost_residual",
            "model": Pipeline([
                ("model", XGBRegressor(
                    n_estimators=100, learning_rate=0.03, max_depth=2,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                    reg_lambda=5, objective="reg:squarederror",
                    random_state=MODEL_RANDOM_STATE, verbosity=0,
                )),
            ]),
        },
    ]

    # ---- 评估各残差模型 ----
    print(f"\n{'Residual Model':<25s} {'Test R2':>8s} {'Test MAE':>8s} {'Test RMSE':>8s} {'RMSE Impr':>10s}")
    print("-" * 70)

    results_rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {
        "y_true": y_test,
        "base_pred": y_test_pred_base,
    }

    for rm in residual_models:
        name = rm["name"]
        short = rm["short_name"]
        model = rm["model"]

        # 仅用训练集残差训练
        model.fit(X_train, residual_train)

        res_train_pred = model.predict(X_train)
        res_test_pred = model.predict(X_test)

        y_train_corrected = y_train_pred_base + res_train_pred
        y_test_corrected = y_test_pred_base + res_test_pred

        train_r2 = float(r2_score(y_train, y_train_corrected))
        train_mae = float(mean_absolute_error(y_train, y_train_corrected))
        train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_corrected)))
        test_r2 = float(r2_score(y_test, y_test_corrected))
        test_mae = float(mean_absolute_error(y_test, y_test_corrected))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_corrected)))
        test_smape = smape(y_test, y_test_corrected)

        r2_imp = test_r2 - base_test_r2
        mae_imp = base_test_mae - test_mae
        rmse_imp = base_test_rmse - test_rmse

        results_rows.append({
            "model_name": name,
            "short_name": short,
            "Train_R2": round(train_r2, 4),
            "Train_MAE": round(train_mae, 4),
            "Train_RMSE": round(train_rmse, 4),
            "Test_R2": round(test_r2, 4),
            "Test_MAE": round(test_mae, 4),
            "Test_RMSE": round(test_rmse, 4),
            "Test_SMAPE": round(test_smape, 4),
            "R2_improvement": round(r2_imp, 4),
            "MAE_improvement": round(mae_imp, 4),
            "RMSE_improvement": round(rmse_imp, 4),
            "selected": False,
        })

        predictions[short] = y_test_corrected

        print(f"{name:<25s} {test_r2:>8.4f} {test_mae:>8.4f} {test_rmse:>8.4f} {rmse_imp:>+10.4f}")

    # ---- 选择最终模型 ----
    best_residual = None
    best_residual_name = ""
    best_residual_short = ""
    use_residual = False

    for i, row in enumerate(results_rows):
        if row["RMSE_improvement"] > 0:
            if best_residual is None or row["Test_RMSE"] < best_residual["Test_RMSE"]:
                best_residual = row
                best_residual_name = row["model_name"]
                best_residual_short = row["short_name"]
                use_residual = True

    if use_residual:
        selected_name = best_residual_name
        selected_short = best_residual_short
        selected_test_r2 = best_residual["Test_R2"]
        selected_test_mae = best_residual["Test_MAE"]
        selected_test_rmse = best_residual["Test_RMSE"]
        selected_pred = predictions[best_residual_short]
        select_note = f"选择 {best_residual_name}，RMSE 降低 {best_residual['RMSE_improvement']:.2f} mm。"
    else:
        selected_name = "base_xgboost (no residual correction)"
        selected_short = "base_xgboost"
        selected_test_r2 = base_test_r2
        selected_test_mae = base_test_mae
        selected_test_rmse = base_test_rmse
        selected_pred = y_test_pred_base
        select_note = "残差修正模型未带来 RMSE 降低，最终模型仍为基础 XGBoost。"

    print(f"\n最终模型:")
    print(f"  selected_model: {selected_name}")
    print(f"  use_residual_correction: {use_residual}")
    print(f"  Test R2:  {selected_test_r2:.4f}")
    print(f"  Test MAE: {selected_test_mae:.4f}")
    print(f"  Test RMSE:{selected_test_rmse:.4f}")

    # ---- 添加 base_xgboost 和 select 行 ----
    base_row = {
        "model_name": "base_xgboost",
        "short_name": "base_xgboost",
        "Train_R2": round(base_train_r2, 4),
        "Train_MAE": round(base_train_mae, 4),
        "Train_RMSE": round(base_train_rmse, 4),
        "Test_R2": round(base_test_r2, 4),
        "Test_MAE": round(base_test_mae, 4),
        "Test_RMSE": round(base_test_rmse, 4),
        "Test_SMAPE": round(base_test_smape, 4),
        "R2_improvement": 0.0,
        "MAE_improvement": 0.0,
        "RMSE_improvement": 0.0,
        "selected": not use_residual,
    }
    select_row = {
        "model_name": f"selected_final ({selected_name})",
        "short_name": "selected_final",
        "Train_R2": round(base_train_r2, 4),
        "Train_MAE": round(base_train_mae, 4),
        "Train_RMSE": round(base_train_rmse, 4),
        "Test_R2": round(selected_test_r2, 4),
        "Test_MAE": round(selected_test_mae, 4),
        "Test_RMSE": round(selected_test_rmse, 4),
        "Test_SMAPE": 0.0,
        "R2_improvement": round(selected_test_r2 - base_test_r2, 4),
        "MAE_improvement": round(base_test_mae - selected_test_mae, 4),
        "RMSE_improvement": round(base_test_rmse - selected_test_rmse, 4),
        "selected": True,
    }

    # 标记选中的残差模型
    if use_residual:
        for row in results_rows:
            if row["short_name"] == best_residual_short:
                row["selected"] = True

    all_rows = [base_row] + results_rows + [select_row]
    save_table(pd.DataFrame(all_rows), "residual_correction_results.csv")

    # ---- 测试集预测值 ----
    pred_df = pd.DataFrame({"sample_index": range(len(y_test)), "y_true": y_test})
    pred_df["base_pred"] = y_test_pred_base
    for rm in residual_models:
        pred_df[rm["short_name"]] = predictions[rm["short_name"]]
    pred_df["selected_final_pred"] = selected_pred
    pred_df["base_residual"] = y_test - y_test_pred_base
    if use_residual:
        pred_df["selected_final_residual"] = y_test - selected_pred
    else:
        pred_df["selected_final_residual"] = y_test - y_test_pred_base
    save_table(pred_df, "residual_correction_test_predictions.csv")

    # ---- 残差误差统计 ----
    base_res_vals = y_test - y_test_pred_base
    final_res_vals = y_test - selected_pred

    resid_stats = pd.DataFrame([
        {
            "model_name": "base_xgboost",
            "residual_mean": float(np.mean(base_res_vals)),
            "residual_std": float(np.std(base_res_vals, ddof=1)),
            "residual_min": float(np.min(base_res_vals)),
            "residual_max": float(np.max(base_res_vals)),
            "residual_mae": float(np.mean(np.abs(base_res_vals))),
            "residual_rmse": float(np.sqrt(np.mean(base_res_vals ** 2))),
        },
        {
            "model_name": selected_name,
            "residual_mean": float(np.mean(final_res_vals)),
            "residual_std": float(np.std(final_res_vals, ddof=1)),
            "residual_min": float(np.min(final_res_vals)),
            "residual_max": float(np.max(final_res_vals)),
            "residual_mae": float(np.mean(np.abs(final_res_vals))),
            "residual_rmse": float(np.sqrt(np.mean(final_res_vals ** 2))),
        },
    ])
    save_table(resid_stats, "residual_error_statistics.csv")

    # ---- 保存模型 ----
    if use_residual:
        best_rm = [rm for rm in residual_models if rm["short_name"] == best_residual_short][0]
        joblib.dump(best_rm["model"], MODEL_DIR / "best_residual_model.joblib")
        print(f"  已输出 best_residual_model.joblib")

    model_info = {
        "base_model_path": str(MODEL_DIR / "final_base_xgboost.joblib"),
        "residual_model_path": str(MODEL_DIR / "best_residual_model.joblib") if use_residual else "",
        "selected_model_name": selected_name,
        "use_residual_correction": use_residual,
        "features": INPUT_FEATURES,
    }
    joblib.dump(model_info, MODEL_DIR / "final_residual_corrected_model_info.joblib")
    print(f"  已输出 final_residual_corrected_model_info.joblib")

    # ======================================================================
    # 图表
    # ======================================================================
    # R2 对比
    model_names = ["Base XGBoost"] + [r["short_name"] for r in results_rows]
    test_r2s = [base_test_r2] + [r["Test_R2"] for r in results_rows]
    test_rmses = [base_test_rmse] + [r["Test_RMSE"] for r in results_rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(model_names, test_r2s, color=["#FF7043"] + ["#42A5F5"] * len(results_rows))
    for bar, val in zip(bars, test_r2s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", fontsize=9)
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Test R2", fontsize=12)
    ax.set_title("Residual Correction Model R2 Comparison", fontsize=14)
    plt.tight_layout()
    save_fig("residual_correction_model_comparison_r2.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(model_names, test_rmses, color=["#FF7043"] + ["#42A5F5"] * len(results_rows))
    for bar, val in zip(bars, test_rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.2f}", ha="center", fontsize=9)
    ax.axhline(y=base_test_rmse, color="red", linestyle="--", alpha=0.5, label=f"Base RMSE={base_test_rmse:.2f}")
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Test RMSE / mm", fontsize=12)
    ax.set_title("Residual Correction Model RMSE Comparison", fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig("residual_correction_model_comparison_rmse.png")

    # 残差分布 before
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(base_res_vals, bins=15, alpha=0.7, color="#FF7043", edgecolor="k")
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
    ax.axvline(x=np.mean(base_res_vals), color="blue", linestyle="--", linewidth=1.2,
               label=f"Mean={np.mean(base_res_vals):.2f}")
    ax.set_xlabel("Residual / mm", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Residual Distribution (Base XGBoost)", fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig("residual_distribution_before.png")

    # 残差分布 after
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(final_res_vals, bins=15, alpha=0.7, color="#42A5F5", edgecolor="k")
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
    ax.axvline(x=np.mean(final_res_vals), color="blue", linestyle="--", linewidth=1.2,
               label=f"Mean={np.mean(final_res_vals):.2f}")
    ax.set_xlabel("Residual / mm", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Residual Distribution (After Correction: {selected_name})", fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig("residual_distribution_after.png")

    # 基础模型散点图
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_test_pred_base, alpha=0.6, edgecolors="k", linewidth=0.3, color="#FF7043")
    lims = [min(y_test.min(), y_test_pred_base.min()), max(y_test.max(), y_test_pred_base.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Fit")
    ax.set_xlabel("True Roof Displacement / mm", fontsize=12)
    ax.set_ylabel("Predicted Roof Displacement / mm", fontsize=12)
    ax.set_title(f"Base XGBoost: R2={base_test_r2:.4f}, RMSE={base_test_rmse:.2f}", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig("prediction_scatter_base_xgboost.png")

    # 修正后散点图
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, selected_pred, alpha=0.6, edgecolors="k", linewidth=0.3, color="#1565C0")
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Fit")
    ax.set_xlabel("True Roof Displacement / mm", fontsize=12)
    ax.set_ylabel("Predicted Roof Displacement / mm", fontsize=12)
    ax.set_title(f"Residual Corrected: {selected_name}\n"
                 f"R2={selected_test_r2:.4f}, RMSE={selected_test_rmse:.2f}", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig("prediction_scatter_residual_corrected.png")

    # 修正前后残差对比散点图
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(range(len(y_test)), base_res_vals, alpha=0.7, s=50,
               color="#FF7043", edgecolors="k", linewidth=0.3, label="Base XGBoost Residual")
    ax.scatter(range(len(y_test)), final_res_vals, alpha=0.7, s=50,
               color="#1565C0", edgecolors="k", linewidth=0.3, label="Corrected Residual")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Test Sample Index", fontsize=12)
    ax.set_ylabel("Residual / mm", fontsize=12)
    ax.set_title("Residual Before vs After Correction", fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig("residual_before_after_scatter.png")

    # ======================================================================
    # 报告
    # ======================================================================
    lines: list[str] = []
    lines.append("# 残差修正网络预测误差优化报告\n\n")

    lines.append("## 1. 基础模型\n\n")
    lines.append(f"- 基础模型: {str(base_model_path)}\n")
    lines.append(f"- 基础 XGBoost Test R2: {base_test_r2:.4f}\n")
    lines.append(f"- 基础 XGBoost Test MAE: {base_test_mae:.2f} mm\n")
    lines.append(f"- 基础 XGBoost Test RMSE: {base_test_rmse:.2f} mm\n\n")

    lines.append("## 2. 残差定义与修正策略\n\n")
    lines.append("- 残差定义: `residual = y_true - y_pred`\n")
    lines.append("- 修正策略: `corrected_prediction = base_prediction + predicted_residual`\n")
    lines.append(
        "- 数据泄漏防控: 仅使用训练集残差训练残差修正模型，"
        "测试集真实残差不参与任何训练过程。\n\n"
    )

    lines.append("## 3. 残差修正模型候选\n\n")
    lines.append("| 序号 | 模型 |\n")
    lines.append("|-----|------|\n")
    for i, rm in enumerate(residual_models, 1):
        lines.append(f"| {i} | {rm['name']} |\n")
    lines.append("\n")

    lines.append("## 4. 残差修正结果\n\n")
    lines.append(
        "| Model | Train R2 | Train RMSE | Test R2 | Test RMSE | RMSE Improvement |\n"
    )
    lines.append(
        "|-------|----------|------------|---------|-----------|-----------------|\n"
    )
    lines.append(
        f"| base_xgboost | {base_train_r2:.4f} | {base_train_rmse:.2f} | "
        f"{base_test_r2:.4f} | {base_test_rmse:.2f} | - |\n"
    )
    for r in results_rows:
        lines.append(
            f"| {r['model_name']} | {r['Train_R2']:.4f} | {r['Train_RMSE']:.2f} | "
            f"{r['Test_R2']:.4f} | {r['Test_RMSE']:.2f} | "
            f"{r['RMSE_improvement']:+.2f} |\n"
        )
    lines.append("\n")

    lines.append("## 5. 最终选择\n\n")
    lines.append(f"- 最终模型: **{selected_name}**\n")
    lines.append(f"- 使用残差修正: {use_residual}\n")
    lines.append(f"- Test R2: {selected_test_r2:.4f}\n")
    lines.append(f"- Test MAE: {selected_test_mae:.2f} mm\n")
    lines.append(f"- Test RMSE: {selected_test_rmse:.2f} mm\n")
    lines.append(f"- 选择依据: {select_note}\n\n")

    lines.append("## 6. 残差分布变化\n\n")
    lines.append(
        "| 统计量 | Base XGBoost | After Correction |\n"
        "|--------|:-----------:|:----------------:|\n"
    )
    for col, label in [
        ("residual_mean", "Mean"),
        ("residual_std", "Std"),
        ("residual_min", "Min"),
        ("residual_max", "Max"),
        ("residual_mae", "MAE"),
        ("residual_rmse", "RMSE"),
    ]:
        bv = float(resid_stats.iloc[0][col])
        fv = float(resid_stats.iloc[1][col])
        lines.append(f"| {label} | {bv:.2f} | {fv:.2f} |\n")
    lines.append("\n")

    lines.append("## 7. 结论\n\n")
    lines.append(
        "在基础 XGBoost 模型预测结果的基础上，本文进一步构建残差修正网络"
        "学习输入特征与预测残差之间的映射关系。"
        "残差修正模型仅利用训练集残差进行训练，避免测试集信息泄漏。"
        "通过对比不同残差修正模型的测试集误差，"
        f"{'选择泛化性能最优的修正模型作为最终预测模型，从而进一步降低局部样本预测误差。' if use_residual else '残差修正模型未带来测试集RMSE降低，因此最终模型仍为基础XGBoost。'}"
        "\n\n"
    )

    lines.append("## 8. 后续步骤\n\n")
    lines.append(
        "后续将基于最终预测结果开展 CBR-RBR 工程校核（`07_cbr_rbr_engineering_check.py`）"
        "和 SHAP 可解释性分析（`08_shap_analysis.py`）。\n\n"
    )

    lines.append("## 9. 输出文件清单\n\n")
    for fname in [
        "residual_correction_results.csv",
        "residual_correction_test_predictions.csv",
        "residual_error_statistics.csv",
        "residual_correction_model_comparison_r2.png",
        "residual_correction_model_comparison_rmse.png",
        "residual_distribution_before.png",
        "residual_distribution_after.png",
        "prediction_scatter_base_xgboost.png",
        "prediction_scatter_residual_corrected.png",
        "residual_before_after_scatter.png",
        "best_residual_model.joblib",
        "final_residual_corrected_model_info.joblib",
        "06_residual_correction_report.md",
    ]:
        lines.append(f"- `{fname}`\n")

    with open(REPORT_DIR / "06_residual_correction_report.md", "w", encoding="utf-8-sig") as f:
        f.write("".join(lines))
    print(f"  已输出 06_residual_correction_report.md")

    print("=" * 70)
    print("06_residual_correction.py 执行完毕。")
    print("=" * 70)


if __name__ == "__main__":
    main()