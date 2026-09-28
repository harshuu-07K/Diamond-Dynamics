# 💎 Diamond Dynamics: Price Prediction & Market Segmentation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-185a9d.svg)](https://xgboost.readthedocs.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning, Deep Learning (ANN), and Unsupervised Clustering project designed to optimize diamond pricing strategies and customer segmentation for luxury e-commerce and retail jewelers.

---

## 📌 Executive Summary & Problem Statement

The diamond market heavily depends on multiple qualitative and quantitative gemological attributes known as the **4Cs** (*Carat, Cut, Color, Clarity*) alongside physical dimension ratios (*x, y, z*, depth, and table) to determine value. 

### 🎯 Primary Objectives
1. **Dynamic Price Prediction (Regression)**: Predict diamond valuations in **Indian Rupees (INR ₹)** using multiple machine learning regressors and an **Artificial Neural Network (ANN)**, trained on log-transformed prices to achieve peak generalization ($R^2 > 0.98$).
2. **Customer & Market Segmentation (Clustering)**: Group diamonds into distinct, meaningful consumer market personas using **K-Means clustering** (evaluated with the Elbow Method & Silhouette score) and **Principal Component Analysis (PCA)** for 2D/3D visualization without price leakage.
3. **Interactive Streamlit Web Application**: An interactive, production-ready valuation platform enabling instant price estimation, cluster persona mapping, interactive EDA, and model benchmarks.

---

## 📊 Dataset Overview

- **Source:** Diamond Dataset
- **Total Records:** 53,940 instances
- **Features:** 10 core features:

| Column | Type | Description |
|---|---|---|
| `carat` | Numeric (Float) | Weight of the diamond (1 ct = 200 mg). Primary value determinant. |
| `cut` | Categorical (Ordinal) | Cut quality: *Fair < Good < Very Good < Premium < Ideal*. |
| `color` | Categorical (Ordinal) | Color grade: *J (worst) to D (best / colorless)*. |
| `clarity` | Categorical (Ordinal) | Blemish & inclusion grade: *I1 < SI2 < SI1 < VS2 < VS1 < VVS2 < VVS1 < IF*. |
| `depth` | Numeric (Float) | Total depth percentage: $\frac{z}{\text{mean}(x, y)} \times 100$. |
| `table` | Numeric (Float) | Width of top facet relative to widest point. |
| `price` | Numeric (Integer) | Diamond price in USD (converted to INR at 1 USD = 83 INR). |
| `x` | Numeric (Float) | Length of the diamond in millimeters (mm). |
| `y` | Numeric (Float) | Width of the diamond in millimeters (mm). |
| `z` | Numeric (Float) | Depth (height) of the diamond in millimeters (mm). |

---

## 🧹 Data Preprocessing & Cleaning

1. **Invalid Physical Dimension Filtering**: Diamonds cannot have zero or negative dimensions ($x, y, z \le 0$). 20 unphysical records were identified and dropped.
2. **Duplicate Handling**: 145 duplicate records were removed.
3. **Outlier & Skewness Treatment**:
   - Skewness measured with `.skew()`; target `price_inr` and `carat` exhibited heavy positive skewness ($\approx 1.62$ and $1.11$).
   - Evaluated outliers via **IQR (Interquartile Range)** method.
   - Filtered measurement anomalies ($y, z > 20$ mm or volume $> 1000$ $\text{mm}^3$).
   - Log transformation ($\log(1 + y)$) applied to `price_inr` to stabilize error variance and normalize target distribution for regression.

---

## ⚙️ Feature Engineering

| Feature | Formula / Logic | Business Rationale |
|---|---|---|
| `price_inr` | $\text{price} \times 83.0$ | Target variable in INR per project guidelines. |
| `volume` | $x \times y \times z$ | 3D volume approximation ($\text{mm}^3$) closely tied to raw carat mass. |
| `dimension_ratio` | $\frac{x + y}{2 \times z}$ | Proportional symmetry index. |
| `carat_category` | Light ($<0.5$), Medium ($0.5-1.5$), Heavy ($>1.5$) | Commercial weight tiering. |
| `price_per_carat_inr` | $\frac{\text{price\_inr}}{\text{carat}}$ | Unit density pricing metric. |
| `table_depth_ratio` | $\frac{\text{table}}{\text{depth}}$ | Facet-to-profile balance. |
| `surface_area_approx` | $2 \times (xy + yz + zx)$ | Facet area proxy. |

---

## 🤖 Regression Modeling & Evaluation Benchmark

All models were trained on 80% of data and evaluated on 20% unseen test data using **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, and **$R^2$ Score**:

| Model | MAE (INR ₹) | RMSE (INR ₹) | MSE | $R^2$ Score |
|---|:---:|:---:|:---:|:---:|
| **XGBoost Regressor** 🏆 | **₹ 22,137.42** | **₹ 43,672.85** | $1.907 \times 10^9$ | **0.9822** |
| **Random Forest Regressor** | ₹ 22,087.92 | ₹ 44,626.76 | $1.992 \times 10^9$ | 0.9814 |
| **Decision Tree Regressor** | ₹ 26,501.47 | ₹ 52,757.83 | $2.783 \times 10^9$ | 0.9740 |
| **Artificial Neural Network (ANN)** | ₹ 29,616.33 | ₹ 57,047.38 | $3.254 \times 10^9$ | 0.9696 |
| **K-Nearest Neighbors (KNN)** | ₹ 31,864.64 | ₹ 61,414.07 | $3.772 \times 10^9$ | 0.9648 |
| **Linear Regression** | ₹ 36,581.13 | ₹ 69,323.07 | $4.806 \times 10^9$ | 0.9552 |

### 🧠 ANN Architecture
- **Input:** 13 standardized features
- **Hidden Layers:**
  - Dense (128 units, ReLU) + BatchNormalization + Dropout (0.15)
  - Dense (64 units, ReLU) + BatchNormalization + Dropout (0.10)
  - Dense (32 units, ReLU)
- **Output:** Dense (1 unit, Linear) predicting $\log(1 + \text{price\_inr})$
- **Optimizer & Loss:** Adam ($lr = 0.003$), Huber Loss, EarlyStopping & ReduceLROnPlateau.

---

## 🏷️ Unsupervised Market Segmentation (Clustering)

To prevent data leakage, **price was excluded** during clustering. Segmentation was performed purely on physical dimensions and quality ratings using **K-Means**:

- **Optimal $k=4$** confirmed via **Elbow Method (Inertia inflection)** and **Silhouette Analysis**.
- **Dimensionality Reduction**: 2D PCA explains **67.53%** variance; 3D PCA explains **80.33%** variance.

### 👥 The 4 Market Personas

| Cluster ID | Market Segment Persona | Share (%) | Avg Carat | Avg Price (INR ₹) | Target Consumer / Use Case |
|:---:|---|:---:|:---:|:---:|---|
| **0** | **Mid-range Balanced Diamonds** | 30.33% | 0.94 ct | ₹ 3,85,825 | Mainstream commercial diamonds with good size-to-clarity balance. |
| **1** | **Affordable Small Diamonds** | 42.84% | 0.40 ct | ₹ 92,316 | Budget-conscious everyday wear, pavé settings, and fashion jewelry. |
| **2** | **Premium Heavy Diamonds** | 14.26% | 1.67 ct | ₹ 9,28,481 | High-end luxury solitaires, bridal flagships, and investment stones. |
| **3** | **High-Clarity Fine Solitaires** | 12.57% | 0.80 ct | ₹ 2,96,484 | Optical brilliance connoisseurs seeking top cut & clarity grades. |

---

## 🖥️ Streamlit Web Application Features

The interactive web dashboard (`app.py`) offers five core modules:

1. **💎 Valuation & Clustering Engine**:
   - Interactive inputs for Carat, Cut, Color, Clarity, Proportions, and Dimensions.
   - Real-time feature calculation (Volume, Dimension Ratio, Carat Class).
   - Instant Price Output in **INR (₹)**, **USD ($)**, and **Price per Carat**.
   - Automated Market Segment Classification with an interactive **PCA 2D projection showing your diamond's exact coordinate**.
2. **🏷️ Market Segmentation Explorer**:
   - Persona breakdown cards with average price, carat, and common cuts.
   - Interactive Elbow & Silhouette evaluation charts.
   - Interactive 3D PCA scatter plot showing cluster separation in space.
3. **📊 Exploratory Data Analytics (EDA)**:
   - Interactive histograms, boxplots across the 4Cs, and correlation matrix.
4. **🤖 AI Model Benchmark & Architecture**:
   - Performance leaderboard table (MAE, RMSE, MSE, $R^2$).
   - Feature importance bar chart from XGBoost.
   - Full ANN layer diagram and hyperparameters.
5. **📖 Gemology & Project Guide**:
   - 4Cs educational guide, mathematical formulas, and project metadata.

---

## 📂 Project Structure

```text
GUVI 3/
├── app.py                                          # Interactive Streamlit Web Application
├── train_models.py                                 # Automated Model Training & Pipeline Script
├── create_notebook.py                              # Jupyter Notebook generation utility
├── Diamond_Dynamics_Analysis_and_Modeling.ipynb    # Clean, fully commented Jupyter Notebook
├── requirements.txt                                # Python dependencies
├── README.md                                       # Comprehensive project documentation
├── diamonds.csv                                    # Original diamond dataset (53,940 rows)
├── diamonds_engineered.csv                         # Enriched & cleaned dataset with features
├── models/                                         # Serialized production models & artifacts
│   ├── best_regression_model.pkl                  # Trained XGBoost Regressor
│   ├── ann_model.keras                             # Trained TensorFlow ANN Regressor
│   ├── clustering_model.pkl                        # Trained K-Means Clusterer (k=4)
│   ├── pca_2d.pkl                                  # 2D PCA Transformer
│   ├── pca_3d.pkl                                  # 3D PCA Transformer
│   ├── scaler_regression.pkl                       # Regression feature scaler
│   ├── scaler_cluster.pkl                          # Clustering feature scaler
│   ├── regression_features.pkl                     # Regression feature list
│   ├── cluster_features.pkl                        # Cluster feature list
│   ├── category_mappings.json                      # Ordinal encoding definitions
│   ├── regression_metrics.json                     # Regression benchmark scores & weights
│   ├── clustering_metrics.json                     # Cluster profiles, inertia & silhouette
│   └── diamonds_app_sample.csv                     # Sample for rapid UI visualization
└── 💎Diamond Dynamics_ Price Prediction and...pdf  # Original Project Brief Specification
```

---

## 🚀 How to Run the Project Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Retrain All Models & Generate Pipeline
```bash
python train_models.py
```

### 3. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Open and Run the Jupyter Notebook
```bash
jupyter notebook Diamond_Dynamics_Analysis_and_Modeling.ipynb
```

---

## 👥 Authors & Acknowledgments

- **Course:** GUVI Geek Network | HCL Tech Capstone Project
- **Domain:** Data Science & Machine Learning (Luxury Goods & Retail Pricing)
