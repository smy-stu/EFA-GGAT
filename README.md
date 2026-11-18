# EFA-GGAT
Open-source code of 《Gated Graph Attention Network Based on Feature Embedding and Feature Attention for Identifying Geochemical Anomalies: A Case Study of the Three Rivers Region in Southwest China》
# Geochemical-Mineral-Prospectivity

Based on geochemical element data and graph neural networks (GAT / GGAT / EFA-GGAT) for mineralization potential prediction.

This repository contains scripts for geochemical data preprocessing, as well as scripts for training and grid prediction using three graph neural network models. The input data are Excel tables containing sample point types, longitude and latitude, and multi-element concentrations; the output is the mineralization potential prediction values on a regular grid, which can be used for subsequent mapping and analysis in QGIS / ArcGIS.

---

# Project Overview
This project focuses on geochemical mineralization potential prediction and mainly consists of two parts:

1.**Data preprocessing**: Read the raw geochemical Excel data, parse special numeric formats such as detection limits and comma decimal separators, remove element columns with excessively high missing rates, impute missing values with the median, apply CLR transformation to major elements and log10 transformation to trace elements, and retain the original values for rare earth elements.

2.**Graph neural network prediction**: Divide the study area into a regular grid, aggregate sample features and label proportions within each grid cell, initialize features for grid cells without samples using ordinary kriging interpolation, construct a KNN graph, and use GAT, GGAT, or EFA-GGAT to predict the mineralization potential of each grid cell.

The sample raw data in the repository include types such as Au-Cu, Cu, Cu-Mo, Ag-Pb-Zn, and Barren. In the code, Barren is mapped to 0 and all other types are mapped to 1.

Note: Due to differences in environments, the prediction results reproduced by the code may show minor differences, which fall within the acceptable error range.
