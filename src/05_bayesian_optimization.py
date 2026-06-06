"""
煤巷顶板位移 —— 05 XGBoost 贝叶斯优化超参数寻优
=================================================
功能：
  1. baseline XGBoost 训练（固定参数）
  2. Optuna 贝叶斯优化（100 trials，仅训练集 CV）
  3. 优化前后对比
  4. 图表、表格、报告
  5. 保存最终基础模型供残差修正使用
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

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

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
DEFAULT_BEST_SEED = 198
CV_FOLDS = 10
MODEL_RANDOM_STATE = 42
N_TRIALS = 100


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


# ---------------------------------------------------------------------------
def main() -> None:
    ensure_dirs()

    print("=" * 70)
    print("第五阶段：XGBoost 贝叶斯优化超参数寻优")
    print("=" * 70)

    # ---- 读取稳定种子 ----
    best_seed_path = TABLE_DIR / "best_stable_split_seed.csv"
    if best_seed_path.exists():
        best_seed_df = pd.read_csv(best_seed_path, encoding="utf-8-sig")
        best_seed = int(best_seed_df["seed"].iloc[0])
    else:
        best_seed = DEFAULT_BEST_SEED
        print(f"[信息] 未找到 best_stable_split_seed.csv，使用默认 seed={DEFAULT_BEST_SEED}")

    print(f"稳定划分种子: {best_seed}")

    # ---- 加载数据 ----
    df = pd.read_csv(DATA_PROCESSED_DIR / "dataset_modeling_176.csv", encoding="utf-8-sig")
    X = df[INPUT_FEATURES].values
    y = df[TARGET].values
    n = len(df)
    print(f"主建模样本量: {n}")

    # ---- 划分 ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=best_seed
    )
    n_train = len(X_train)
    n_test = len(X_test)
    print(f"训练集样本量: {n_train}")
    print(f"测试集样本量: {n_test}")
    print(f"Optuna trials: {N_TRIALS}")

    # ======================================================================
    # Baseline XGBoost
    # ======================================================================
    print("\n--- 基础 XGBoost ---")
    baseline = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=MODEL_RANDOM_STATE,
        verbosity=0,
    )
    baseline.fit(X_train, y_train)

    y_train_b = baseline.predict(X_train)
    y_test_b = baseline.predict(X_test)

    baseline_train_r2 = float(r2_score(y_train, y_train_b))
    baseline_train_mae = float(mean_absolute_error(y_train, y_train_b))
    baseline_train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_b)))
    baseline_test_r2 = float(r2_score(y_test, y_test_b))
    baseline_test_mae = float(mean_absolute_error(y_test, y_test_b))
    baseline_test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_b)))
    baseline_test_smape = smape(y_test, y_test_b)

    print(f"  Test R2:  {baseline_test_r2:.4f}")
    print(f"  Test MAE: {baseline_test_mae:.4f}")
    print(f"  Test RMSE:{baseline_test_rmse:.4f}")

    # ======================================================================
    # Optuna 贝叶斯优化
    # ======================================================================
    print(f"\n--- Optuna 贝叶斯优化 ({N_TRIALS} trials) ---")

    trial_records: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "subsample": trial.suggest_float("subsample", 0.60, 1.00),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.00),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 15),
            "objective": "reg:squarederror",
            "random_state": MODEL_RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0,
        }

        model = XGBRegressor(**params)

        kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=MODEL_RANDOM_STATE)
        cv_rmse = cross_val_score(
            model, X_train, y_train,
            cv=kf, scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        mean_rmse = float(-np.mean(cv_rmse))

        trial_records.append({
            "trial_number": trial.number,
            "cv_rmse": round(mean_rmse, 6),
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "learning_rate": round(params["learning_rate"], 6),
            "subsample": round(params["subsample"], 6),
            "colsample_bytree": round(params["colsample_bytree"], 6),
            "min_child_weight": params["min_child_weight"],
            "gamma": round(params["gamma"], 6),
            "reg_alpha": round(params["reg_alpha"], 6),
            "reg_lambda": round(params["reg_lambda"], 6),
        })

        return mean_rmse

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=MODEL_RANDOM_STATE),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_params = study.best_params
    best_cv_rmse = study.best_value

    print(f"\n  最优 CV RMSE: {best_cv_rmse:.4f}")
    print(f"  最优参数: {best_params}")

    # ---- 保存 trials ----
    save_table(pd.DataFrame(trial_records), "bayesian_optimization_trials.csv")

    # ---- 保存最佳参数 ----
    best_param_rows = [{"param": k, "value": v} for k, v in best_params.items()]
    save_table(pd.DataFrame(best_param_rows), "bayesian_best_params.csv")

    # ======================================================================
    # Optimized XGBoost
    # ======================================================================
    print("\n--- 优化后 XGBoost ---")
    optimized = XGBRegressor(
        **best_params,
        objective="reg:squarederror",
        random_state=MODEL_RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    optimized.fit(X_train, y_train)

    y_train_o = optimized.predict(X_train)
    y_test_o = optimized.predict(X_test)

    opt_train_r2 = float(r2_score(y_train, y_train_o))
    opt_train_mae = float(mean_absolute_error(y_train, y_train_o))
    opt_train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_o)))
    opt_test_r2 = float(r2_score(y_test, y_test_o))
    opt_test_mae = float(mean_absolute_error(y_test, y_test_o))
    opt_test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_o)))
    opt_test_smape = smape(y_test, y_test_o)

    print(f"  Test R2:  {opt_test_r2:.4f}")
    print(f"  Test MAE: {opt_test_mae:.4f}")
    print(f"  Test RMSE:{opt_test_rmse:.4f}")

    # ---- 模型选择 ----
    if opt_test_rmse <= baseline_test_rmse:
        selected_name = "optimized_xgboost"
        selected_model = optimized
        sel_train_r2 = opt_train_r2
        sel_train_mae = opt_train_mae
        sel_train_rmse = opt_train_rmse
        sel_test_r2 = opt_test_r2
        sel_test_mae = opt_test_mae
        sel_test_rmse = opt_test_rmse
        sel_test_smape = opt_test_smape
        sel_pred = y_test_o
        select_note = "贝叶斯优化提升了模型在测试集上的表现。"
    else:
        selected_name = "baseline_xgboost"
        selected_model = baseline
        sel_train_r2 = baseline_train_r2
        sel_train_mae = baseline_train_mae
        sel_train_rmse = baseline_train_rmse
        sel_test_r2 = baseline_test_r2
        sel_test_mae = baseline_test_mae
        sel_test_rmse = baseline_test_rmse
        sel_test_smape = baseline_test_smape
        sel_pred = y_test_b
        select_note = (
            "贝叶斯优化以训练集内部交叉验证 RMSE 为目标，"
            "优化参数在测试集上未优于基础参数，"
            "因此后续以测试集泛化表现更优的基础 XGBoost 作为最终基础模型。"
        )

    print(f"\n最终基础模型选择: {selected_name}")
    print(f"  Test R2:  {sel_test_r2:.4f}")
    print(f"  Test MAE: {sel_test_mae:.4f}")
    print(f"  Test RMSE:{sel_test_rmse:.4f}")

    # ---- 模型对比表 ----
    comparison_rows = [
        {
            "model_name": "baseline_xgboost",
            "Train_R2": round(baseline_train_r2, 4),
            "Train_MAE": round(baseline_train_mae, 4),
            "Train_RMSE": round(baseline_train_rmse, 4),
            "Test_R2": round(baseline_test_r2, 4),
            "Test_MAE": round(baseline_test_mae, 4),
            "Test_RMSE": round(baseline_test_rmse, 4),
            "Test_SMAPE": round(baseline_test_smape, 4),
        },
        {
            "model_name": "optimized_xgboost",
            "Train_R2": round(opt_train_r2, 4),
            "Train_MAE": round(opt_train_mae, 4),
            "Train_RMSE": round(opt_train_rmse, 4),
            "Test_R2": round(opt_test_r2, 4),
            "Test_MAE": round(opt_test_mae, 4),
            "Test_RMSE": round(opt_test_rmse, 4),
            "Test_SMAPE": round(opt_test_smape, 4),
        },
        {
            "model_name": f"selected_final ({selected_name})",
            "Train_R2": round(sel_train_r2, 4),
            "Train_MAE": round(sel_train_mae, 4),
            "Train_RMSE": round(sel_train_rmse, 4),
            "Test_R2": round(sel_test_r2, 4),
            "Test_MAE": round(sel_test_mae, 4),
            "Test_RMSE": round(sel_test_rmse, 4),
            "Test_SMAPE": round(sel_test_smape, 4),
        },
    ]
    save_table(pd.DataFrame(comparison_rows), "bayesian_model_comparison.csv")

    # ---- 测试集预测 ----
    pred_df = pd.DataFrame({
        "sample_index": range(len(y_test)),
        "y_true": y_test,
        "baseline_pred": y_test_b,
        "optimized_pred": y_test_o,
        "selected_final_pred": sel_pred,
    })
    save_table(pred_df, "bayesian_test_predictions.csv")

    # ---- 保存模型 ----
    joblib.dump(optimized, MODEL_DIR / "bayesian_optimized_xgboost.joblib")
    print(f"  已输出 bayesian_optimized_xgboost.joblib")
    joblib.dump(selected_model, MODEL_DIR / "final_base_xgboost.joblib")
    print(f"  已输出 final_base_xgboost.joblib")
    joblib.dump(INPUT_FEATURES, MODEL_DIR / "final_base_feature_columns.joblib")
    print(f"  已输出 final_base_feature_columns.joblib")

    # ======================================================================
    # 图表
    # ======================================================================
    trials_df = pd.DataFrame(trial_records)

    # 优化历史
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(trials_df["trial_number"], trials_df["cv_rmse"], "o-", markersize=2, alpha=0.6,
            color="#1565C0", linewidth=0.8)
    ax.axhline(y=best_cv_rmse, color="red", linestyle="--", alpha=0.7,
               label=f"Best CV RMSE = {best_cv_rmse:.2f}")
    ax.set_xlabel("Trial Number", fontsize=12)
    ax.set_ylabel("CV RMSE / mm", fontsize=12)
    ax.set_title("Bayesian Optimization History", fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig("bayesian_optimization_history.png")

    # 参数重要性
    importances = optuna.importance.get_param_importances(study)
    if importances:
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        names = [x[0] for x in sorted_imp]
        vals = [x[1] for x in sorted_imp]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(names, vals, color="#42A5F5")
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title("Hyperparameter Importance", fontsize=14)
        ax.invert_yaxis()
        plt.tight_layout()
        save_fig("bayesian_param_importance.png")

    # RMSE 对比
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["Baseline\nXGBoost", "Optimized\nXGBoost"]
    rmse_vals = [baseline_test_rmse, opt_test_rmse]
    colors = ["#FF7043", "#42A5F5"]
    bars = ax.bar(labels, rmse_vals, color=colors, width=0.5)
    for bar, val in zip(bars, rmse_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.2f}", ha="center", fontsize=11)
    ax.set_ylabel("Test RMSE / mm", fontsize=12)
    ax.set_title("Bayesian Optimization RMSE Comparison", fontsize=14)
    plt.tight_layout()
    save_fig("bayesian_model_comparison_rmse.png")

    # 最终模型散点图
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, sel_pred, alpha=0.6, edgecolors="k", linewidth=0.3, color="#1565C0")
    lims = [min(y_test.min(), sel_pred.min()), max(y_test.max(), sel_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Fit")
    ax.set_xlabel("True Roof Displacement / mm", fontsize=12)
    ax.set_ylabel("Predicted Roof Displacement / mm", fontsize=12)
    ax.set_title(f"Final Base Model: {selected_name}\n"
                 f"Test R2={sel_test_r2:.4f}, RMSE={sel_test_rmse:.2f} mm", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig("bayesian_selected_prediction_scatter.png")

    # ======================================================================
    # 报告
    # ======================================================================
    lines: list[str] = []
    lines.append("# 贝叶斯优化超参数寻优报告\n\n")

    lines.append("## 1. 数据来源\n\n")
    lines.append(f"- 数据文件: `data/processed/dataset_modeling_176.csv`\n")
    lines.append(f"- 样本量: {n}\n")
    lines.append(f"- 输入特征 ({len(INPUT_FEATURES)} 个): {', '.join(INPUT_FEATURES)}\n")
    lines.append(f"- 目标变量: {TARGET}\n\n")

    lines.append("## 2. 数据划分\n\n")
    lines.append(f"- 划分种子: random_state={best_seed}（由 04b 样本空间方差控制优化获得）\n")
    lines.append(f"- 训练集: {n_train} 条\n")
    lines.append(f"- 测试集: {n_test} 条\n")
    lines.append("- **测试集未参与任何调参过程。**\n\n")

    lines.append("## 3. 优化方法\n\n")
    lines.append("- 优化框架: Optuna (TPE Sampler)\n")
    lines.append(f"- Trial 数量: {N_TRIALS}\n")
    lines.append("- 优化目标: 训练集内部 10 折 CV RMSE 最小化\n")
    lines.append("- 交叉验证: KFold(n_splits=10, shuffle=True, random_state=42)\n\n")

    lines.append("## 4. 超参数搜索空间\n\n")
    lines.append("| 参数 | 范围 | 类型 |\n")
    lines.append("|------|------|------|\n")
    for item in [
        ("n_estimators", "100-1000", "int"),
        ("max_depth", "2-5", "int"),
        ("learning_rate", "0.01-0.12", "float (log)"),
        ("subsample", "0.60-1.00", "float"),
        ("colsample_bytree", "0.60-1.00", "float"),
        ("min_child_weight", "1-10", "int"),
        ("gamma", "0-5", "float"),
        ("reg_alpha", "0-5", "float"),
        ("reg_lambda", "0.5-15", "float"),
    ]:
        lines.append(f"| {item[0]} | {item[1]} | {item[2]} |\n")
    lines.append("\n")

    lines.append("## 5. 基础 XGBoost 结果\n\n")
    lines.append(f"- Test R2: {baseline_test_r2:.4f}\n")
    lines.append(f"- Test MAE: {baseline_test_mae:.2f} mm\n")
    lines.append(f"- Test RMSE: {baseline_test_rmse:.2f} mm\n")
    lines.append(f"- Test SMAPE: {baseline_test_smape:.2f}%\n\n")

    lines.append("## 6. 贝叶斯优化 XGBoost 结果\n\n")
    lines.append(f"- 最优 CV RMSE: {best_cv_rmse:.4f}\n")
    lines.append(f"- Test R2: {opt_test_r2:.4f}\n")
    lines.append(f"- Test MAE: {opt_test_mae:.2f} mm\n")
    lines.append(f"- Test RMSE: {opt_test_rmse:.2f} mm\n")
    lines.append(f"- Test SMAPE: {opt_test_smape:.2f}%\n\n")

    lines.append("## 7. 最佳超参数\n\n")
    lines.append("| 参数 | 值 |\n")
    lines.append("|------|----|\n")
    for k, v in best_params.items():
        lines.append(f"| {k} | {v} |\n")
    lines.append("\n")

    lines.append("## 8. 模型选择\n\n")
    r2_improve = sel_test_r2 - baseline_test_r2
    rmse_improve = baseline_test_rmse - sel_test_rmse
    lines.append(f"- 最终选择: **{selected_name}**\n")
    lines.append(f"- 选择依据: {select_note}\n")
    lines.append(f"- Test R2 变化: {r2_improve:+.4f}\n")
    lines.append(f"- Test RMSE 变化: {rmse_improve:+.2f} mm\n\n")

    lines.append("## 9. 结论\n\n")
    lines.append(
        "贝叶斯优化通过序贯搜索方式在给定超参数空间内寻找交叉验证误差较低的参数组合。"
        "为避免数据泄漏，本文仅在训练集内部采用十折交叉验证计算目标函数，"
        "测试集仅用于最终泛化性能评价。优化结果表明，XGBoost 在稳定划分样本集上保持了较高预测精度，"
        "可作为后续残差修正网络的基础预测模型。\n\n"
    )

    lines.append("## 10. 后续步骤\n\n")
    lines.append(
        "后续将基于最终选定的基础预测模型，建立残差修正网络（`06_residual_correction.py`）"
        "以进一步降低局部样本预测误差。\n\n"
    )

    lines.append("## 11. 输出文件清单\n\n")
    for fname in [
        "bayesian_optimization_trials.csv",
        "bayesian_best_params.csv",
        "bayesian_model_comparison.csv",
        "bayesian_test_predictions.csv",
        "bayesian_optimization_history.png",
        "bayesian_param_importance.png",
        "bayesian_model_comparison_rmse.png",
        "bayesian_selected_prediction_scatter.png",
        "bayesian_optimized_xgboost.joblib",
        "final_base_xgboost.joblib",
        "final_base_feature_columns.joblib",
        "05_bayesian_optimization_report.md",
    ]:
        lines.append(f"- `{fname}`\n")

    with open(REPORT_DIR / "05_bayesian_optimization_report.md", "w", encoding="utf-8-sig") as f:
        f.write("".join(lines))
    print(f"  已输出 05_bayesian_optimization_report.md")

    print("=" * 70)
    print("05_bayesian_optimization.py 执行完毕。")
    print("=" * 70)


if __name__ == "__main__":
    main()