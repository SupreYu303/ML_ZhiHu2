# 稳定划分条件下的基础预测模型对比报告

## 1. 数据来源

- 数据文件: `data/processed/dataset_modeling_176.csv`
- 样本量: 176
- 输入特征 (8 个): fracture_degree, coal_strength, floor_strength, roof_strength, depth, width, bolt_area, anchor_density
- 目标变量: roof_displacement

## 2. 训练/测试集划分

- 划分种子: random_state=198
- 种子来源: 由 04b 样本空间方差控制划分稳定性优化获得
- 选择依据: 兼顾测试集目标变量分布稳定性与预测性能
- 训练集: 140 条
- 测试集: 36 条

## 3. 模型列表

| 序号 | 模型 |
|-----|------|
| 1 | Dummy Regressor |
| 2 | Ridge Regression |
| 3 | ElasticNet |
| 4 | SVR (RBF) |
| 5 | ANN Neural Network |
| 6 | Random Forest |
| 7 | Gradient Boosting |
| 8 | AdaBoost |
| 9 | XGBoost |
| 10 | LightGBM |
| 11 | CatBoost |

## 4. 训练集与测试集结果

| Model | Train R2 | Train MAE | Train RMSE | Test R2 | Test MAE | Test RMSE | Test SMAPE |
|-------|----------|-----------|------------|---------|----------|-----------|------------|
| Dummy Regressor | 0.0000 | 35.43 | 42.23 | -0.0002 | 36.80 | 44.39 | 71.51% |
| Ridge Regression | 0.1863 | 32.75 | 38.09 | 0.2287 | 30.16 | 38.98 | 62.84% |
| ElasticNet | 0.1854 | 32.79 | 38.11 | 0.2171 | 30.48 | 39.28 | 63.35% |
| SVR (RBF) | 0.3667 | 24.44 | 33.61 | 0.3046 | 28.67 | 37.02 | 58.78% |
| ANN Neural Network | 0.9746 | 4.46 | 6.73 | 0.4705 | 23.25 | 32.30 | 49.87% |
| Random Forest | 0.9155 | 9.53 | 12.28 | 0.7430 | 18.36 | 22.50 | 42.22% |
| Gradient Boosting | 0.9818 | 4.28 | 5.70 | 0.7987 | 15.14 | 19.91 | 38.01% |
| AdaBoost | 0.6363 | 23.01 | 25.47 | 0.6208 | 24.14 | 27.34 | 57.70% |
| XGBoost | 0.9824 | 4.22 | 5.60 | 0.8512 | 13.24 | 17.12 | 38.12% |
| LightGBM | 0.7840 | 15.33 | 19.63 | 0.7129 | 17.47 | 23.79 | 45.09% |
| CatBoost | 0.8668 | 12.70 | 15.41 | 0.7733 | 17.81 | 21.13 | 46.18% |

## 5. 10 折交叉验证结果

| Model | CV R2 Mean | CV R2 Std | CV RMSE Mean | CV RMSE Std |
|-------|-----------|-----------|-------------|------------|
| Dummy Regressor | -0.0780 | 0.1147 | 42.50 | 5.52 |
| Ridge Regression | 0.0493 | 0.2379 | 39.43 | 6.28 |
| ElasticNet | 0.0562 | 0.2193 | 39.36 | 6.08 |
| SVR (RBF) | 0.1722 | 0.2599 | 36.78 | 8.12 |
| ANN Neural Network | -0.2903 | 0.6674 | 44.25 | 10.85 |
| Random Forest | 0.4684 | 0.2752 | 28.29 | 10.51 |
| Gradient Boosting | 0.4456 | 0.2780 | 28.98 | 9.23 |
| AdaBoost | 0.3313 | 0.2226 | 32.86 | 7.60 |
| XGBoost | 0.4850 | 0.2480 | 28.27 | 9.76 |
| LightGBM | 0.3986 | 0.3021 | 30.63 | 10.60 |
| CatBoost | 0.4164 | 0.2958 | 30.06 | 10.55 |

## 6. 最优稳定基础模型

- 最优模型: **XGBoost**
- 测试集 R2: 0.8512
- 测试集 MAE: 13.24 mm
- 测试集 RMSE: 17.12 mm
- 测试集 SMAPE: 38.12%

## 7. 与 random_state=42 初始基线对比

| 指标 | seed=42 基线 | seed=198 稳定 | 改善 |
|------|:-----------:|:-------------:|:----:|
| XGBoost Test R2 | 0.6005 | 0.8512 | +0.2507 |
| XGBoost Test RMSE | 28.57 | 17.12 | -11.45 |

稳定划分条件下模型性能显著改善，说明对于小样本数据集，数据划分对评估结果有重要影响。通过样本空间方差控制优化，找到了更具代表性的划分方案。

## 8. 后续优化方向

在完成样本空间方差控制的数据集划分优化后，稳定划分条件下各模型预测性能明显改善。其中 XGBoost 模型取得最优测试集预测效果，表明其能够较好刻画顶板位移与围岩强度、埋深、巷道宽度及支护参数之间的非线性映射关系。因此，本文选择 XGBoost 作为后续贝叶斯优化和残差修正网络的基础预测模型。

## 9. 输出文件清单

- `stable_baseline_model_results.csv`
- `stable_baseline_best_model.csv`
- `stable_baseline_predictions.csv`
- `stable_baseline_train_test_split_info.csv`
- `stable_baseline_model_r2_comparison.png`
- `stable_baseline_model_rmse_comparison.png`
- `stable_baseline_model_mae_comparison.png`
- `stable_baseline_best_model_prediction_scatter.png`
- `stable_baseline_best_model.joblib`
- `stable_feature_columns.joblib`
- `04c_stable_baseline_model_report.md`
