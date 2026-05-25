# 建筑能耗聚类分析与短期负荷预测

基于深圳 20 栋建筑 2021 年全年逐小时能耗数据，结合建筑属性与典型气象年（EPW）天气信息，完成**日负荷曲线聚类**与**下一小时能耗预测**的完整机器学习流程。

## 项目简介

本项目包含两条主线：

1. **聚类分析**：将每栋建筑每天的 24 小时单位面积能耗曲线作为样本，经标准化与 KMeans 聚类（对比 k=4 / k=5，最终采用 k=4），识别典型用能模式，并分析工作日/周末、季节与建筑类型差异。
2. **负荷预测**：利用前 n 小时历史能耗、时间特征、天气变量与建筑属性，采用 Random Forest 预测当前小时总能耗，输出 RMSE、R² 及可视化结果。

## 项目结构

```text
├── final_assignment/          # 数据预处理
│   └── final_assignment/
│       ├── data/              # 原始建筑能耗、建筑信息、EPW 天气
│       ├── src/
│       │   └── 01_data_preprocessing.py
│       └── outputs/
│           ├── clean_hourly_energy.csv
│           └── daily_load_matrix.csv
│
├── clustering/                # KMeans 聚类（k=4 / k=5 对比，最终 k=4）
│   ├── clustering_K4_with_weekday_weekend/
│   └── clustering_K5_with_weekday_weekend/
│
└── prediction/                # Random Forest 短期预测
    ├── 03_prediction.py
    ├── prediction_metrics.csv
    └── figures/
```

## 主要结果

### 聚类（k = 4）

- 样本：每栋建筑每天 24 小时单位面积能耗曲线（24 维向量）
- 方法：标准化 → 肘部法则选 k → KMeans
- 输出：聚类中心曲线、工作日/周末占比、季节与建筑类型分布
- 结果目录：`clustering/clustering_K4_with_weekday_weekend/`

### 预测（Random Forest）

| 滞后窗口 | RMSE | R² |
|---------|------|-----|
| 6h | 68.35 | 0.9968 |
| 12h | 69.12 | 0.9968 |
| **24h** | **62.30** | **0.9974** |

- 特征：前 n 小时负荷滞后项、hour / month / is_weekend、温度 / 湿度 / 辐射 / 风速、建筑面积与类型
- 划分：按时间切割（2021-10-20 前训练，之后测试），20 栋建筑均参与
- 结果目录：`prediction/prediction_metrics.csv`、`prediction/figures/`

## 环境配置

```bash
pip install -r requirements.txt
```

依赖：pandas、numpy、scikit-learn、matplotlib

## 运行方式

```bash
# 1. 数据预处理
cd final_assignment/final_assignment/src
python 01_data_preprocessing.py

# 2. 聚类分析
cd ../../../clustering/clustering_K4_with_weekday_weekend
python clusteringt_K4_with_weekday_weekend.py

# 3. 负荷预测（需先完成步骤 1，生成 clean_hourly_energy.csv）
cd ../../../prediction
python 03_prediction.py
```

## 数据说明

| 数据 | 说明 |
|------|------|
| `building_1.csv` ~ `building_20.csv` | 20 栋建筑 2021 年逐小时分项能耗 |
| `building_info.csv` | 建筑类型、楼层、面积 |
| `*.epw` | 深圳典型气象年逐小时天气 |
| `clean_hourly_energy.csv` | 合并后的逐小时建模表（8760×20 行） |
| `daily_load_matrix.csv` | 聚类用每日 24 小时负荷矩阵 |

## License

MIT License — 仅供学习交流使用。
