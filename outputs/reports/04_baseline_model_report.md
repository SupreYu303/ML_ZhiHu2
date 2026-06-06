# 基础预测模型对比报告

## 1. 数据来源

- 数据文件: `data/processed/dataset_modeling_176.csv`
- 样本量: 176
- 输入特征 (8 个): fracture_degree, coal_strength, floor_strength, roof_strength, depth, width, bolt_area, anchor_density
- 目标变量: roof_displacement

## 2. 训练/测试集划分

- 划分方式: `train_test_split(test_size=0.2, random_state=42)`
- 训练集: 140 条
- 测试集: 36 条
- 当前步骤使用 random_state=42 作为基础模型对比划分，
  后续通过样本空间方差控制划分优化模块搜索更稳定的划分种子。

## 3. 模型列表

| 序号 | 模型 | 说明 |
|-----|------|------|
| 1 | Dummy Regressor | - |
| 2 | Ridge Regression | - |
| 3 | ElasticNet | - |
| 4 | SVR (RBF) | - |
| 5 | ANN Neural Network | - |
| 6 | Random Forest | - |
| 7 | Gradient Boosting | - |
| 8 | AdaBoost | - |
| 9 | XGBoost | - |
| 10 | LightGBM | - |
| 11 | CatBoost | - |

## 4. 训练集与测试集结果

| Model | Train R2 | Train MAE | Train RMSE | Test R2 | Test MAE | Test RMSE | Test SMAPE |
|-------|----------|-----------|------------|---------|----------|-----------|------------|
| Dummy Regressor | 0.0000 | 35.21 | 42.00 | -0.0005 | 37.62 | 45.22 | 74.52% |
| Ridge Regression | 0.2024 | 31.90 | 37.51 | 0.2227 | 32.66 | 39.86 | 69.53% |
| ElasticNet | 0.2013 | 31.94 | 37.54 | 0.2193 | 32.79 | 39.94 | 69.60% |
| SVR (RBF) | 0.4157 | 23.75 | 32.11 | 0.2336 | 30.09 | 39.57 | 63.98% |
| ANN Neural Network | 0.9444 | 6.81 | 9.90 | 0.1837 | 30.97 | 40.84 | 71.87% |
| Random Forest | 0.9273 | 8.90 | 11.32 | 0.5138 | 23.65 | 31.52 | 50.87% |
| Gradient Boosting | 0.9883 | 3.50 | 4.54 | 0.5638 | 22.80 | 29.86 | 46.96% |
| AdaBoost | 0.6544 | 22.16 | 24.69 | 0.3530 | 30.62 | 36.36 | 67.88% |
| XGBoost | 0.9837 | 4.17 | 5.36 | 0.6005 | 22.17 | 28.57 | 46.81% |
| LightGBM | 0.8000 | 14.79 | 18.78 | 0.4644 | 26.47 | 33.09 | 52.92% |
| CatBoost | 0.8894 | 11.58 | 13.97 | 0.4995 | 24.58 | 31.98 | 55.31% |

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

## 6. 最优基础模型

- 最优模型: **XGBoost**
- 测试集 R2: 0.6005
- 测试集 MAE: 22.17 mm
- 测试集 RMSE: 28.57 mm
- 测试集 SMAPE: 46.81%

## 7. 过拟合分析

最优模型训练集 R2 (0.9837) 与测试集 R2 (0.6005) 差距较大，存在一定过拟合现象，需在贝叶斯优化中加入正则化参数搜索。

## 8. 后续优化方向

基础模型对比结果表明，非线性集成模型整体优于线性模型，说明顶板位移与围岩强度、埋深、巷道宽度及支护参数之间存在明显非线性关系。考虑到基础模型参数尚未经过系统优化，后续采用贝叶斯优化方法对表现较优的模型进行超参数寻优，并进一步构建残差修正网络以降低局部样本预测误差。

## 9. 输出文件清单

- `baseline_model_results.csv`
- `baseline_best_model.csv`
- `baseline_predictions.csv`
- `baseline_train_test_split_info.csv`
- `baseline_model_r2_comparison.png`
- `baseline_model_rmse_comparison.png`
- `baseline_model_mae_comparison.png`
- `baseline_best_model_prediction_scatter.png`
- `baseline_best_model.joblib`
- `feature_columns.joblib`
- `04_baseline_model_report.md`
