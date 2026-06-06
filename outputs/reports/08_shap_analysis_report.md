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

## 5. 全局特征贡献度分析（SHAP 蜂群图）

SHAP 蜂群图从全局角度展示了各输入特征对 XGBoost 模型预测结果的贡献分布。图中每一个点代表一个样本在对应特征上的 SHAP 值，横坐标表示该特征对预测顶板位移的贡献方向和贡献大小。当 SHAP 值大于 0 时，说明该特征在该样本中推动模型输出更大的顶板位移预测值；当 SHAP 值小于 0 时，则说明该特征降低了模型预测结果。点的颜色表示特征取值大小，红色代表特征值较高，蓝色代表特征值较低。结果表明，depth、roof_strength、anchor_density、floor_strength 和 coal_strength是影响顶板位移预测的主要特征。其中，depth 的高取值样本整体表现出较明显的正向贡献，说明高埋深工况下模型倾向于预测更大的顶板位移；anchor_density 的高取值样本多表现为负向贡献，说明锚索密度提高有助于降低模型预测的顶板位移。roof_strength、floor_strength 和 coal_strength 的 SHAP 分布呈现一定正负交错特征，表明其影响机制并非单一线性关系，而可能受到围岩条件、埋深和支护参数之间耦合作用的影响。

## 6. 局部样本解释

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

## 7. 关键特征的非线性影响机制与边际效应分析

### 7.1 分析方法

为进一步揭示关键特征对顶板位移预测结果的作用机制，本文基于 SHAP dependence relationship 对 Top 5 关键特征进行非线性影响分析，并通过分位区间统计计算近似边际效应。边际效应用相邻特征区间平均 SHAP 值的变化量与平均特征值变化量之比表示，用于刻画特征变化对模型输出贡献的敏感程度。

### 7.2 depth 的非线性影响机制

- depth 是最重要特征。
- 埋深增加整体倾向于提高顶板位移预测值。
- 高埋深区间如果 SHAP 值升高，说明深部地应力对顶板变形具有增强作用。
- 若边际效应在某些区间更明显，说明这些埋深区间对位移预测更敏感。

**机制摘要**: 埋深是最重要影响因素。随着埋深增大，地应力水平通常升高，模型预测顶板位移整体增大；高埋深区间 SHAP 值明显为正，说明深部开采条件下顶板变形风险增强。

### 7.3 roof_strength 的非线性影响机制

- roof_strength 是重要影响因素。
- 如果 SHAP 显示 roof_strength 增大整体提高预测位移，不要简单解释为"强度越大越危险"。
- 应解释为模型捕捉到 roof_strength 与埋深、支护参数、围岩结构之间的耦合关系。
- 其作用具有非线性和工程耦合特征。

**机制摘要**: 顶板强度对位移预测具有重要影响。SHAP 显示 roof_strength 增大时预测位移可能升高，这可能不是单一强度因果作用，而反映了顶板强度与埋深、支护参数、地质条件之间的耦合关系。

### 7.4 anchor_density 的非线性影响机制

- anchor_density 增大整体倾向于降低预测位移。
- 说明锚索支护对控制顶板变形有积极作用。
- 如果高 anchor_density 区间边际效应减弱，可解释为支护参数加密存在边际收益递减。
- 这可以为支护参数优化提供依据。

**机制摘要**: 锚索密度增大整体倾向于降低预测顶板位移，说明锚索补强对顶板变形具有控制作用；若高密度区间边际效应减弱，可解释为支护加密存在边际收益递减。

### 7.5 floor_strength 的非线性影响机制

- floor_strength 对预测结果有贡献，但方向可能受到其他特征交互影响。
- 不宜做单因素因果解释。
- 应从围岩整体结构和工程耦合角度解释。

**机制摘要**: 底板强度表现出非线性或交互影响，说明其对顶板位移的作用可能通过围岩整体结构条件间接体现。

### 7.6 coal_strength 的非线性影响机制

- coal_strength 对预测结果有贡献，但方向可能受到其他特征交互影响。
- 不宜做单因素因果解释。
- 应从围岩整体结构和工程耦合角度解释。

**机制摘要**: 煤层强度对顶板位移存在一定贡献。若高煤层强度区间 SHAP 值升高，应谨慎解释为样本耦合效应，而不是简单认为煤层强度越高位移越大。

### 7.7 小结

- SHAP 非线性分析表明，关键特征对顶板位移预测的影响不是简单线性关系；
- depth、roof_strength、anchor_density 是影响模型输出的核心变量；
- anchor_density 的负向 SHAP 贡献说明支护增强能够降低预测位移；
- depth 的正向 SHAP 贡献说明深部工况更容易出现较大顶板变形；
- 关键特征存在区间敏感性和边际效应变化，说明支护优化应关注特征阈值区间。

## 8. 结论

SHAP 分析结果表明，不同输入变量对顶板位移预测结果的贡献存在明显差异。与传统相关性分析不同，SHAP 能够反映非线性模型中各特征对单个样本预测结果的边际贡献。特征重要性排序进一步说明顶板位移受围岩强度、埋深、巷道宽度及支护参数等多因素耦合作用影响，为后续支护参数优化和工程风险识别提供了可解释依据。

进一步的 SHAP 非线性影响与边际效应分析表明，depth、roof_strength 和 anchor_density 对模型输出存在明显区间敏感性，其中 depth 在高埋深区间对预测位移具有更强正向贡献，anchor_density 整体表现为负向贡献且可能存在支护边际收益递减现象。

## 9. 输出文件清单

- `shap_feature_importance.csv`
- `shap_local_explanations_top_error_samples.csv`
- `shap_marginal_effect_bins.csv`
- `shap_nonlinear_mechanism_summary.csv`
- `shap_summary_beeswarm.png`
- `shap_feature_importance_bar.png`
- `shap_dependence_top1.png`
- `shap_dependence_top2.png`
- `shap_dependence_top3.png`
- `shap_mean_abs_importance.png`
- `shap_nonlinear_effect_depth.png`
- `shap_nonlinear_effect_roof_strength.png`
- `shap_nonlinear_effect_anchor_density.png`
- `shap_nonlinear_effect_floor_strength.png`
- `shap_nonlinear_effect_coal_strength.png`
- `shap_marginal_effect_depth.png`
- `shap_marginal_effect_roof_strength.png`
- `shap_marginal_effect_anchor_density.png`
- `shap_marginal_effect_floor_strength.png`
- `shap_marginal_effect_coal_strength.png`
- `08_shap_analysis_report.md`
