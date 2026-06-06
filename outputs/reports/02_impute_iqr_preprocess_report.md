# 样本清洗与工程适用域筛选报告

## 1. 原始数据

- 原始样本量: 200
- 输入文件: `D:\AgentWorkspace\zhihu_final\data\processed\dataset_checked.csv`

## 2. 清洗后主建模数据

- 清洗后样本量: 176
- 输入文件: `D:\AgentWorkspace\zhihu_final\data\processed\Paper_Filtered_Data.csv`

## 3. 缺失值检查

- 当前数据无缺失值，XGBoost 多重插补模块未触发。

## 4. IQR 异常值识别（基于原始 200 条）

- 顶板位移 Q1: 22.25
- 顶板位移 Q3: 95.00
- IQR: 72.75
- IQR 异常上限: 204.125 mm
- 顶板位移 > 204.125 mm 的 14 条样本被识别为统计异常响应样本。

## 5. 人工工程复核剔除

- IQR 极端响应样本: 14 条
- 工程适用域边界/敏感性复核样本: 10 条
- 总剔除样本: 24 条

被剔除样本不视为错误数据，而是作为工程异常响应或适用域外样本保留解释。

## 6. Paper_Filtered_Data.csv 验证

- 样本量是否为 176: True
- 字段数是否为 9: True
- 是否无缺失值: True
- 无 id 列: True
- roof_displacement 最大值: 179.00 (参考 179.00)
- roof_displacement 标准差: 42.80 (参考 42.80)
- 目标变量稳定性: 原始 std=78.04, 清洗后 std=42.80

## 7. 清洗前后顶板位移分布变化

- 原始最大值: 454.00 mm
- 清洗后最大值: 179.00 mm
- 原始标准差: 78.04 mm
- 清洗后标准差: 42.80 mm

## 8. 敏感性验证

| 模型 | 原始数据 RMSE | 清洗后 RMSE |
|------|:-----------:|:----------:|
| Ridge | 73.83 | 40.21 |
| SVR | 73.72 | 37.24 |
| Random Forest | 73.18 | 34.75 |
| Gradient Boosting | 74.15 | 35.71 |

## 9. 最终说明

`dataset_modeling_176.csv` 将作为后续模型训练、贝叶斯优化、残差修正、SHAP 分析和交叉验证的主数据集。

## 10. 输出文件清单

- `dataset_modeling_176.csv`: D:\AgentWorkspace\zhihu_final\data\processed\dataset_modeling_176.csv
- `dataset_full.csv`: D:\AgentWorkspace\zhihu_final\data\processed\dataset_full.csv
- `dataset_deduplicated.csv`: D:\AgentWorkspace\zhihu_final\data\processed\dataset_deduplicated.csv
- `dataset_winsorized.csv`: D:\AgentWorkspace\zhihu_final\data\processed\dataset_winsorized.csv
- `dataset_risk_labeled.csv`: D:\AgentWorkspace\zhihu_final\data\processed\dataset_risk_labeled.csv
- `iqr_outlier_summary.csv`: D:\AgentWorkspace\zhihu_final\outputs\tables\iqr_outlier_summary.csv
- `iqr_extreme_response_samples.csv`: D:\AgentWorkspace\zhihu_final\outputs\tables\iqr_extreme_response_samples.csv
- `manual_removed_samples.csv`: D:\AgentWorkspace\zhihu_final\outputs\tables\manual_removed_samples.csv
- `filtered_dataset_validation.csv`: D:\AgentWorkspace\zhihu_final\outputs\tables\filtered_dataset_validation.csv
- `raw_vs_filtered_summary.csv`: D:\AgentWorkspace\zhihu_final\outputs\tables\raw_vs_filtered_summary.csv
- `cleaning_sensitivity_validation.csv`: D:\AgentWorkspace\zhihu_final\outputs\tables\cleaning_sensitivity_validation.csv
- `02_impute_iqr_preprocess_report.md`: D:\AgentWorkspace\zhihu_final\outputs\reports\02_impute_iqr_preprocess_report.md
