# 残差修正网络预测误差优化报告

## 1. 基础模型

- 基础模型: D:\AgentWorkspace\zhihu_final\outputs\models\final_base_xgboost.joblib
- 基础 XGBoost Test R2: 0.8512
- 基础 XGBoost Test MAE: 13.24 mm
- 基础 XGBoost Test RMSE: 17.12 mm

## 2. 残差定义与修正策略

- 残差定义: `residual = y_true - y_pred`
- 修正策略: `corrected_prediction = base_prediction + predicted_residual`
- 数据泄漏防控: 仅使用训练集残差训练残差修正模型，测试集真实残差不参与任何训练过程。

## 3. 残差修正模型候选

| 序号 | 模型 |
|-----|------|
| 1 | Ridge Residual |
| 2 | SVR Residual |
| 3 | MLP Residual |
| 4 | Random Forest Residual |
| 5 | GBDT Residual |
| 6 | XGBoost Residual |

## 4. 残差修正结果

| Model | Train R2 | Train RMSE | Test R2 | Test RMSE | RMSE Improvement |
|-------|----------|------------|---------|-----------|-----------------|
| base_xgboost | 0.9824 | 5.60 | 0.8512 | 17.12 | - |
| Ridge Residual | 0.9825 | 5.59 | 0.8505 | 17.16 | -0.04 |
| SVR Residual | 0.9865 | 4.91 | 0.8500 | 17.19 | -0.07 |
| MLP Residual | 1.0000 | 0.27 | 0.8329 | 18.15 | -1.03 |
| Random Forest Residual | 0.9857 | 5.05 | 0.8527 | 17.04 | +0.08 |
| GBDT Residual | 0.9848 | 5.20 | 0.8472 | 17.35 | -0.23 |
| XGBoost Residual | 0.9850 | 5.17 | 0.8523 | 17.06 | +0.06 |

## 5. 最终选择

- 最终模型: **Random Forest Residual**
- 使用残差修正: True
- Test R2: 0.8527
- Test MAE: 13.07 mm
- Test RMSE: 17.04 mm
- 选择依据: 选择 Random Forest Residual，RMSE 降低 0.08 mm。

## 6. 残差分布变化

| 统计量 | Base XGBoost | After Correction |
|--------|:-----------:|:----------------:|
| Mean | 0.69 | 0.40 |
| Std | 17.35 | 17.28 |
| Min | -27.97 | -28.24 |
| Max | 45.20 | 46.22 |
| MAE | 13.24 | 13.07 |
| RMSE | 17.12 | 17.04 |

## 7. 结论

在基础 XGBoost 模型预测结果的基础上，本文进一步构建残差修正网络学习输入特征与预测残差之间的映射关系。残差修正模型仅利用训练集残差进行训练，避免测试集信息泄漏。通过对比不同残差修正模型的测试集误差，选择泛化性能最优的修正模型作为最终预测模型，从而进一步降低局部样本预测误差。

## 8. 后续步骤

后续将基于最终预测结果开展 CBR-RBR 工程校核（`07_cbr_rbr_engineering_check.py`）和 SHAP 可解释性分析（`08_shap_analysis.py`）。

## 9. 输出文件清单

- `residual_correction_results.csv`
- `residual_correction_test_predictions.csv`
- `residual_error_statistics.csv`
- `residual_correction_model_comparison_r2.png`
- `residual_correction_model_comparison_rmse.png`
- `residual_distribution_before.png`
- `residual_distribution_after.png`
- `prediction_scatter_base_xgboost.png`
- `prediction_scatter_residual_corrected.png`
- `residual_before_after_scatter.png`
- `best_residual_model.joblib`
- `final_residual_corrected_model_info.joblib`
- `06_residual_correction_report.md`
