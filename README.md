# EFA-GGAT
Open-source code of 《Gated Graph Attention Network Based on Feature Embedding and Feature Attention for Identifying Geochemical Anomalies: A Case Study of the Three Rivers Region in Southwest China》
# Geochemical-GNN-Mineral-Prospectivity

基于地球化学元素数据与图神经网络（GAT / GGAT / EFA-GGAT）的成矿潜力预测。

本仓库包含地球化学数据预处理脚本，以及三种图神经网络模型的训练与网格预测脚本。输入数据为包含采样点类型、经纬度和多元素含量的 Excel 表；输出为规则网格上的成矿潜力预测值，可用于 QGIS / ArcGIS 后续制图与分析。


---

## 项目简介

本项目面向地球化学成矿潜力预测，主要包括两部分：

1. **数据预处理**：读取原始地球化学 Excel，解析检测限、逗号小数等特殊数值格式，删除缺失率过高的元素列，中位数填补缺失值，并对主量元素做 CLR 变换、微量元素做 log10 变换，稀土元素保持原值。
2. **图神经网络预测**：将研究区划分为规则网格，聚合网格内样本特征与标签比例，对无样本网格使用普通克里金插值初始化特征，构建 KNN 图，使用 GAT、GGAT 或 EFA-GGAT 预测每个网格的成矿潜力。

仓库中的原始数据样例包含 `Au-Cu`、`Cu`、`Cu-Mo`、`Ag-Pb-Zn`、`Barren` 等类型。代码中将 `Barren` 映射为 0，其他类型映射为 1。

---

## 仓库结构

当前脚本可放在仓库根目录。建议后续整理为如下结构：

```text
.

├── Data_Preprocessing.py
├── Gat_Prediction.py
├── GGat_Prediction.py
├── EFA-GGat_Prediction.py
├── Original geochemical elements.xlsx
├── Preprocessed geochemical elements—sheet2.xlsx
├── README.md
