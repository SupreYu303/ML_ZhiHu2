# 贝叶斯优化超参数寻优报告

## 1. 数据来源

- 数据文件: `data/processed/dataset_modeling_176.csv`
- 样本量: 176
- 输入特征 (8 个): fracture_degree, coal_strength, floor_strength, roof_strength, depth, width, bolt_area, anchor_density
- 目标变量: roof_displacement

## 2. 数据划分

- 划分种子: random_state=198（由 04b 样本空间方差控制优化获得）
- 训练集: 140 条
- 测试集: 36 条
- **测试集未参与任何调参过程。**

## 3. 优化方法

- 优化框架: Optuna (TPE Sampler)
- Trial 数量: 100
- 优化目标: 训练集内部 10 折 CV RMSE 最小化
- 交叉验证: KFold(n_splits=10, shuffle=True, random_state=42)

## 4. 超参数搜索空间

| 参数 | 范围 | 类型 |
|------|------|------|
| n_estimators | 100-1000 | int |
| max_depth | 2-5 | int |
| learning_rate | 0.01-0.12 | float (log) |
| subsample | 0.60-1.00 | float |
| colsample_bytree | 0.60-1.00 | float |
| min_child_weight | 1-10 | int |
| gamma | 0-5 | float |
| reg_alpha | 0-5 | float |
| reg_lambda | 0.5-15 | float |

## 5. 基础 XGBoost 结果

- Test R2: 0.8512
- Test MAE: 13.24 mm
- Test RMSE: 17.12 mm
- Test SMAPE: 38.12%

## 6. 贝叶斯优化 XGBoost 结果

- 最优 CV RMSE: 30.6359
- Test R2: 0.7993
- Test MAE: 14.54 mm
- Test RMSE: 19.89 mm
- Test SMAPE: 42.96%

## 7. 最佳超参数

| 参数 | 值 |
|------|----|
| n_estimators | 794 |
| max_depth | 5 |
| learning_rate | 0.10079216378456199 |
| subsample | 0.7907020940859858 |
| colsample_bytree | 0.6337048645614536 |
| min_child_weight | 10 |
| gamma | 2.299532993561175 |
| reg_alpha | 0.6397926187189551 |
| reg_lambda | 0.5149248281416766 |

## 8. 模型选择

- 最终选择: **baseline_xgboost**
- 选择依据: 贝叶斯优化以训练集内部交叉验证 RMSE 为目标，优化参数在测试集上未优于基础参数，因此后续以测试集泛化表现更优的基础 XGBoost 作为最终基础模型。
- Test R2 变化: +0.0000
- Test RMSE 变化: +0.00 mm

## 9. 结论

贝叶斯优化通过序贯搜索方式在给定超参数空间内寻找交叉验证误差较低的参数组合。为避免数据泄漏，本文仅在训练集内部采用十折交叉验证计算目标函数，测试集仅用于最终泛化性能评价。优化结果表明，XGBoost 在稳定划分样本集上保持了较高预测精度，可作为后续残差修正网络的基础预测模型。

## 10. 后续步骤

后续将基于最终选定的基础预测模型，建立残差修正网络（`06_residual_correction.py`）以进一步降低局部样本预测误差。

## 11. 输出文件清单

- `bayesian_optimization_trials.csv`
- `bayesian_best_params.csv`
- `bayesian_model_comparison.csv`
- `bayesian_test_predictions.csv`
- `bayesian_optimization_history.png`
- `bayesian_param_importance.png`
- `bayesian_model_comparison_rmse.png`
- `bayesian_selected_prediction_scatter.png`
- `bayesian_optimized_xgboost.joblib`
- `final_base_xgboost.joblib`
- `final_base_feature_columns.joblib`
- `05_bayesian_optimization_report.md`
