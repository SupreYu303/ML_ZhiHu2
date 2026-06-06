# SHAP 模型解释与特征影响分析报告

## 1. 分析目的

SHAP (SHapley Additive exPlanations) 用于解释机器学习模型对各特征的贡献。本文基于基础 XGBoost 模型进行 SHAP 分析，揭示各输入特征对顶板位移预测结果的影响机制。

## 2. 解释对象

- 模型: XGBoost（final_base_xgboost.joblib）
- 选择原因: XGBoost 是最终基础预测模型，SHAP TreeExplainer 对树模型解释效果稳定
- 数据: 测试集 36 条样本

## 3. 全局特征重要性排序

| Rank | Feature | Mean ABS SHAP | Direction |
|------|---------|:-------------:|----------|
| 1 | depth | 12.9869 | 特征值增大整体倾向于提高预测顶板位移 |
| 2 | roof_strength | 8.7966 | 特征值增大整体倾向于提高预测顶板位移 |
| 3 | anchor_density | 7.3902 | 特征值增大整体倾向于降低预测顶板位移 |
| 4 | floor_strength | 6.6796 | 影响方向呈非线性或交互特征 |
| 5 | coal_strength | 6.0058 | 特征值增大整体倾向于提高预测顶板位移 |
| 6 | bolt_area | 4.9118 | 特征值增大整体倾向于降低预测顶板位移 |
| 7 | width | 4.4820 | 特征值增大整体倾向于降低预测顶板位移 |
| 8 | fracture_degree | 2.0805 | 特征值增大整体倾向于降低预测顶板位移 |

## 4. Top 5 重要特征解释

### 1. depth

- Mean |SHAP|: 12.9869
- 影响方向: 特征值增大整体倾向于提高预测顶板位移
- 特征均值: 468.1111

### 2. roof_strength

- Mean |SHAP|: 8.7966
- 影响方向: 特征值增大整体倾向于提高预测顶板位移
- 特征均值: 49.4053

### 3. anchor_density

- Mean |SHAP|: 7.3902
- 影响方向: 特征值增大整体倾向于降低预测顶板位移
- 特征均值: 1.9225

### 4. floor_strength

- Mean |SHAP|: 6.6796
- 影响方向: 影响方向呈非线性或交互特征
- 特征均值: 35.4303

### 5. coal_strength

- Mean |SHAP|: 6.0058
- 影响方向: 特征值增大整体倾向于提高预测顶板位移
- 特征均值: 14.2989

## 5. 局部样本解释

以下为测试集中预测误差最大的 5 个样本的 SHAP 局部解释：

| Sample | y_true | y_pred | Error | Top 1 Feat | Top 1 SHAP | Top 2 Feat | Top 2 SHAP |
|--------|--------|--------|-------|-----------|:----------:|-----------|:----------:|
| 14 | 69.0 | 23.8 | 45.2 | floor_strength | -17.412 | bolt_area | -7.880 |
| 17 | 142.0 | 109.7 | 32.3 | depth | 45.698 | coal_strength | 9.126 |
| 8 | 84.0 | 53.4 | 30.6 | depth | -7.925 | coal_strength | -7.191 |
| 7 | 28.5 | 56.4 | 28.0 | roof_strength | 9.597 | anchor_density | 8.318 |
| 32 | 75.0 | 101.5 | 26.5 | roof_strength | 21.196 | anchor_density | -12.421 |

## 6. SHAP 与相关性矩阵对比

相关性矩阵反映的是单变量线性关系，SHAP 反映的是模型内部非线性贡献。若二者排序不完全一致，说明顶板位移预测受到非线性和特征交互影响。

Pearson 相关性最高特征为 roof_strength，SHAP 重要性最高特征为 depth，二者不一致，进一步证实了变量间存在非线性耦合效应。

## 7. 结论

SHAP 分析结果表明，不同输入变量对顶板位移预测结果的贡献存在明显差异。与传统相关性分析不同，SHAP 能够反映非线性模型中各特征对单个样本预测结果的边际贡献。特征重要性排序进一步说明顶板位移受围岩强度、埋深、巷道宽度及支护参数等多因素耦合作用影响，为后续支护参数优化和工程风险识别提供了可解释依据。

## 8. 输出文件清单

- `shap_feature_importance.csv`
- `shap_local_explanations_top_error_samples.csv`
- `shap_summary_beeswarm.png`
- `shap_feature_importance_bar.png`
- `shap_dependence_top1.png`
- `shap_dependence_top2.png`
- `shap_dependence_top3.png`
- `shap_mean_abs_importance.png`
- `08_shap_analysis_report.md`
