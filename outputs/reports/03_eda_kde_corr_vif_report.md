# KDE、相关性矩阵与 VIF 分析报告

## 1. 主分析数据

- 主分析数据: `D:\AgentWorkspace\zhihu_final\data\processed\dataset_modeling_176.csv`
- 样本量: 176
- 输入特征数: 8
- 目标变量: roof_displacement

## 2. KDE 分布分析

| variable | count | min | max | mean | median | std | skew | kurt | distribution_type |
|----------|-------|-----|-----|------|--------|-----|------|------|------------------|
| coal_strength | 176 | 1.80 | 70.00 | 12.91 | 12.22 | 7.25 | 5.57 | 42.73 | 右偏；尖峰厚尾 |
| floor_strength | 176 | 8.10 | 112.80 | 36.86 | 32.48 | 18.57 | 1.47 | 2.86 | 右偏；适中 |
| roof_strength | 176 | 10.61 | 112.29 | 50.11 | 49.19 | 18.36 | 0.52 | 0.68 | 右偏；适中 |
| depth | 176 | 50.00 | 890.00 | 419.18 | 415.00 | 162.79 | 0.42 | 0.53 | 近似对称；适中 |
| width | 176 | 2.60 | 8.00 | 4.60 | 4.70 | 0.87 | 0.31 | 1.52 | 近似对称；适中 |
| bolt_area | 176 | 0.49 | 1.44 | 0.82 | 0.80 | 0.22 | 0.87 | 0.38 | 右偏；适中 |
| anchor_density | 176 | 0.33 | 6.25 | 1.84 | 1.67 | 1.10 | 1.42 | 2.70 | 右偏；适中 |
| roof_displacement | 176 | 2.70 | 179.00 | 58.14 | 52.00 | 42.80 | 0.58 | -0.48 | 右偏；适中 |

## 3. 清洗前后 roof_displacement 分布变化

- 原始 200 条数据 roof_displacement 最大值: 454.00 mm
- 清洗后 176 条数据 roof_displacement 最大值: 179.00 mm
- 原始标准差: 78.04 mm
- 清洗后标准差: 42.80 mm
- 清洗后目标变量分布更稳定，更适合作为主建模样本。

## 4. Pearson 相关性分析

各特征与 roof_displacement 的 Pearson 相关系数如下（按绝对值降序）：

| feature | pearson_r |
|---------|----------|
| roof_strength | 0.2091 |
| depth | 0.2049 |
| width | -0.1681 |
| fracture_degree | -0.1630 |
| anchor_density | -0.1543 |
| bolt_area | -0.1026 |
| floor_strength | 0.0834 |
| coal_strength | 0.0144 |

各特征与顶板位移的 Pearson 相关系数绝对值普遍偏低，说明线性相关性较弱。顶板位移受围岩强度、埋深、巷道宽度及支护参数等多因素非线性耦合作用影响。因此后续适合采用 XGBoost、SVR、GBDT 等非线性模型。

## 5. Spearman 相关性分析

各特征与 roof_displacement 的 Spearman 秩相关系数如下（按绝对值降序）：

| feature | spearman_r |
|---------|-----------|
| anchor_density | -0.2519 |
| roof_strength | 0.2158 |
| fracture_degree | -0.1915 |
| floor_strength | 0.1657 |
| width | -0.1530 |
| depth | 0.1456 |
| bolt_area | -0.1033 |
| coal_strength | 0.1009 |

Spearman 秩相关结果与 Pearson 一致，进一步证实了变量间的非线性关系。

## 6. 与 roof_displacement 的相关性排序

| feature | pearson_r | pearson_abs | spearman_r | spearman_abs |
|---------|----------|------------|-----------|------------|
| roof_strength | 0.2091 | 0.2091 | 0.2158 | 0.2158 |
| depth | 0.2049 | 0.2049 | 0.1456 | 0.1456 |
| width | -0.1681 | 0.1681 | -0.1530 | 0.1530 |
| fracture_degree | -0.1630 | 0.1630 | -0.1915 | 0.1915 |
| anchor_density | -0.1543 | 0.1543 | -0.2519 | 0.2519 |
| bolt_area | -0.1026 | 0.1026 | -0.1033 | 0.1033 |
| floor_strength | 0.0834 | 0.0834 | 0.1657 | 0.1657 |
| coal_strength | 0.0144 | 0.0144 | 0.1009 | 0.1009 |

## 7. VIF 多重共线性分析

| feature | VIF | collinearity_judgement |
|---------|-----|-----------------------|
| bolt_area | 1.6254 | 共线性较弱 |
| fracture_degree | 1.5132 | 共线性较弱 |
| floor_strength | 1.3747 | 共线性较弱 |
| roof_strength | 1.2767 | 共线性较弱 |
| anchor_density | 1.2472 | 共线性较弱 |
| depth | 1.2323 | 共线性较弱 |
| width | 1.2270 | 共线性较弱 |
| coal_strength | 1.1149 | 共线性较弱 |

所有 VIF 均小于 1.63，不存在严重多重共线性。8 个特征均可保留用于后续建模。

## 8. 数据版本分布对比

| dataset | sample_count | min | max | mean | median | std | skew | kurt |
|---------|-------------|-----|-----|------|--------|-----|------|------|
| dataset_full | 200 | 2.50 | 454.00 | 75.08 | 55.30 | 78.04 | 2.23 | 6.09 |
| dataset_deduplicated | 198 | 2.50 | 454.00 | 75.58 | 55.80 | 78.25 | 2.22 | 6.02 |
| dataset_winsorized | 200 | 2.50 | 454.00 | 75.08 | 55.30 | 78.04 | 2.23 | 6.09 |
| dataset_modeling_176 | 176 | 2.70 | 179.00 | 58.14 | 52.00 | 42.80 | 0.58 | -0.48 |

## 9. 后续建模建议

清洗后的顶板位移数据仍呈一定右偏分布，但极端响应值影响明显减弱。相关性分析表明，各输入特征与顶板位移之间的线性相关性整体较弱，说明顶板位移受围岩强度、埋深、巷道宽度及支护参数等多因素非线性耦合作用影响。VIF 分析结果表明各特征间不存在严重多重共线性，因此保留全部 8 个输入特征用于后续机器学习建模。

## 10. 输出文件清单

- `outputs/tables/kde_distribution_summary.csv`
- `outputs/tables/raw_vs_filtered_target_distribution.csv`
- `outputs/tables/pearson_corr_matrix.csv`
- `outputs/tables/spearman_corr_matrix.csv`
- `outputs/tables/target_correlation_ranking.csv`
- `outputs/tables/vif_summary.csv`
- `outputs/tables/dataset_versions_distribution_comparison.csv`
- `outputs/figures/kde_coal_strength.png`
- `outputs/figures/kde_floor_strength.png`
- `outputs/figures/kde_roof_strength.png`
- `outputs/figures/kde_depth.png`
- `outputs/figures/kde_width.png`
- `outputs/figures/kde_bolt_area.png`
- `outputs/figures/kde_anchor_density.png`
- `outputs/figures/kde_roof_displacement.png`
- `outputs/figures/raw_vs_filtered_roof_displacement_kde.png`
- `outputs/figures/raw_vs_filtered_roof_displacement_boxplot.png`
- `outputs/figures/pearson_corr_heatmap.png`
- `outputs/figures/spearman_corr_heatmap.png`

- `outputs/reports/03_eda_kde_corr_vif_report.md`
