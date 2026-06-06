# 煤巷顶板位移数据质量检查报告

## 1. 基本信息

- 原始文件: `data_raw.csv`
- 样本量: 200
- 字段数量: 10（标准化后）
- 原始字段: 序号, 围岩裂隙裂隙发育程度, 煤层强度, 底板强度, 顶板强度, 埋深, 毛宽, 锚杆支护面积, 锚索密度, 顶板位移
- 标准化字段: id, fracture_degree, coal_strength, floor_strength, roof_strength, depth, width, bolt_area, anchor_density, roof_displacement

## 2. 字段映射

| 原始字段 | 标准字段 |
|---------|--------|
| 序号 | id |
| 围岩裂隙裂隙发育程度 | fracture_degree |
| 煤层强度 | coal_strength |
| 底板强度 | floor_strength |
| 顶板强度 | roof_strength |
| 埋深 | depth |
| 毛宽 | width |
| 锚杆支护面积 | bolt_area |
| 锚索密度 | anchor_density |
| 顶板位移 | roof_displacement |

## 3. 缺失值统计

| 字段 | 缺失数量 | 缺失比例(%) |
|------|---------|-----------|
| id | 0 | 0.0 |
| fracture_degree | 0 | 0.0 |
| coal_strength | 0 | 0.0 |
| floor_strength | 0 | 0.0 |
| roof_strength | 0 | 0.0 |
| depth | 0 | 0.0 |
| width | 0 | 0.0 |
| bolt_area | 0 | 0.0 |
| anchor_density | 0 | 0.0 |
| roof_displacement | 0 | 0.0 |

## 4. 重复行检查

- 完整重复行（含 id）: 0
- 去掉 id 后重复行: 2

## 5. 字段类型

| 字段 | 类型 |
|------|------|
| id | 整数/字符串 |
| fracture_degree | 数值(float) |
| coal_strength | 数值(float) |
| floor_strength | 数值(float) |
| roof_strength | 数值(float) |
| depth | 数值(float) |
| width | 数值(float) |
| bolt_area | 数值(float) |
| anchor_density | 数值(float) |
| roof_displacement | 数值(float) |

## 6. 描述性统计

| variable | count | min | max | mean | median | std | skew | kurt |
|----------|-------|-----|-----|------|--------|-----|------|------|
| fracture_degree | 200 | 0.0000 | 1.0000 | 0.6250 | 1.0000 | 0.4853 | -0.5203 | -1.7468 |
| coal_strength | 200 | 1.8000 | 70.0000 | 12.7148 | 11.8550 | 6.8885 | 5.7782 | 46.8247 |
| floor_strength | 200 | 8.1000 | 112.8000 | 36.3486 | 32.2100 | 18.0418 | 1.4730 | 3.0008 |
| roof_strength | 200 | 10.6100 | 112.2900 | 49.9904 | 49.3100 | 18.0136 | 0.4642 | 0.6581 |
| depth | 200 | 50.0000 | 890.0000 | 426.8750 | 420.0000 | 167.0604 | 0.4138 | 0.3202 |
| width | 200 | 2.6000 | 8.5000 | 4.6258 | 4.7000 | 0.8919 | 0.5824 | 2.4718 |
| bolt_area | 200 | 0.4900 | 1.4400 | 0.8121 | 0.7600 | 0.2225 | 0.8577 | 0.3280 |
| anchor_density | 200 | 0.3300 | 6.2500 | 1.8845 | 1.6700 | 1.1334 | 1.4023 | 2.3667 |
| roof_displacement | 200 | 2.5000 | 454.0000 | 75.0844 | 55.3000 | 78.0420 | 2.2297 | 6.0884 |

## 7. roof_displacement 专项统计

- 最小值: 2.5000
- 最大值: 454.0000
- 均值: 75.0844
- 中位数: 55.3000
- 标准差: 78.0420
- 偏度: 2.2297
- 峰度: 6.0884
