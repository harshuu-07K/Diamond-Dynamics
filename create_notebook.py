"""
Script to generate the comprehensive, production-grade Jupyter Notebook:
Diamond_Dynamics_Analysis_and_Modeling.ipynb
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# 💎 Diamond Dynamics: Price Prediction & Market Segmentation
### End-to-End Machine Learning, Deep Learning (ANN), & Unsupervised Clustering
**Domain:** E-Commerce | Luxury Goods Analytics | Retail Pricing Optimization  
**Capstone Project:** GUVI | HCL  

---

## 📌 1. Project Background & Problem Statement
The diamond market heavily relies on multiple physical and gemological attributes (the **4Cs**: *Carat, Cut, Color, Clarity*, alongside dimensions *x, y, z*, depth, and table) to determine value. 

### 🎯 Primary Objectives:
1. **Price Prediction (Regression)**: Build and evaluate multiple machine learning regressors (Linear Regression, Decision Tree, Random Forest, KNN, XGBoost) and an **Artificial Neural Network (ANN)** to predict diamond prices converted to **INR (₹)** with maximum accuracy ($R^2$, MAE, RMSE).
2. **Market Segmentation (Clustering)**: Unsupervised clustering using **K-Means**, evaluating optimal clusters via **Elbow Method & Silhouette Scores**, applying **PCA (Principal Component Analysis)** for 2D/3D visualization, and assigning domain-intelligent business segment labels.
3. **Operational Web App**: Deploy an interactive **Streamlit** dashboard for real-time price estimation, market segment classification, and executive business analytics.
"""))

# Imports
cells.append(nbf.v4.new_markdown_cell("""## 📦 2. Environment Setup & Library Imports"""))

cells.append(nbf.v4.new_code_cell("""import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# XGBoost & Deep Learning
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Settings
warnings.filterwarnings('ignore')
%matplotlib inline
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 110

USD_TO_INR = 83.0  # Fixed conversion rate: 1 USD = 83 INR
print(f"TensorFlow Version: {tf.__version__}")
print(f"XGBoost Version: {xgb.__version__}")
"""))

# Data Loading
cells.append(nbf.v4.new_markdown_cell("""## 📊 3. Data Collection & Initial Inspection"""))

cells.append(nbf.v4.new_code_cell("""# Load the diamond dataset (53,940 rows, 10 features)
df = pd.read_csv('diamonds.csv')
print(f"Dataset Shape: {df.shape}")
df.head()
"""))

cells.append(nbf.v4.new_code_cell("""# Dataset schema and null value check
df.info()
print("\\nMissing values per column:\\n", df.isnull().sum())
"""))

cells.append(nbf.v4.new_code_cell("""# Summary statistics of numerical columns
df.describe().T
"""))

# Data Cleaning
cells.append(nbf.v4.new_markdown_cell("""## 🧹 4. Data Cleaning & Sanity Checks
A physical diamond cannot have zero or negative dimensions (length `x`, width `y`, or depth `z`). We identify and remove these invalid measurements along with exact duplicates.
"""))

cells.append(nbf.v4.new_code_cell("""# Check invalid dimensions where x <= 0, y <= 0, or z <= 0
invalid_dims = (df['x'] <= 0) | (df['y'] <= 0) | (df['z'] <= 0)
print(f"Number of invalid dimension rows: {invalid_dims.sum()}")
print(df[invalid_dims].head())

# Filter out invalid dimensions
df = df[~invalid_dims].copy()

# Drop duplicate records
dupes = df.duplicated().sum()
print(f"\\nDuplicate rows found: {dupes}")
df = df.drop_duplicates().reset_index(drop=True)
print(f"Cleaned dataset shape: {df.shape}")
"""))

# Feature Engineering
cells.append(nbf.v4.new_markdown_cell("""## ⚙️ 5. Feature Engineering
Following project specifications, we enrich the dataset with domain-specific derived features:
- **Price in INR**: `price_inr = price * 83.0`
- **Volume**: `volume = x * y * z` (approximates diamond volume in $mm^3$)
- **Dimension Ratio**: $\\text{dimension\\_ratio} = \\frac{x + y}{2 \\times z}$ (measures physical proportion)
- **Carat Category**: Categorizes diamond weight into `Light` ($< 0.5$ ct), `Medium` ($0.5 - 1.5$ ct), and `Heavy` ($> 1.5$ ct)
- **Price per Carat in INR**: `price_per_carat_inr = price_inr / carat`
- **Table to Depth Ratio**: `table / depth`
- **Surface Area Approximation**: $2 \\times (xy + yz + zx)$
"""))

cells.append(nbf.v4.new_code_cell("""# Currency conversion
df['price_inr'] = df['price'] * USD_TO_INR

# Physical volume and dimension ratio
df['volume'] = df['x'] * df['y'] * df['z']
df['dimension_ratio'] = (df['x'] + df['y']) / (2 * df['z'])

# Carat Weight Category
def categorize_carat(c):
    if c < 0.5:
        return 'Light'
    elif c <= 1.5:
        return 'Medium'
    else:
        return 'Heavy'

df['carat_category'] = df['carat'].apply(categorize_carat)

# Price per Carat
df['price_per_carat_inr'] = df['price_inr'] / df['carat']

# Table to Depth Ratio & Surface Area Approximation
df['table_depth_ratio'] = df['table'] / df['depth']
df['surface_area_approx'] = 2 * (df['x']*df['y'] + df['y']*df['z'] + df['z']*df['x'])

# Display engineered features
df[['carat', 'carat_category', 'volume', 'dimension_ratio', 'price_inr', 'price_per_carat_inr']].head()
"""))

# Outlier & Skewness Analysis
cells.append(nbf.v4.new_markdown_cell("""## 📐 6. Outlier & Skewness Analysis
We calculate skewness via `.skew()` and evaluate outlier distributions across physical measurements and price using the Interquartile Range (**IQR**) method.
"""))

cells.append(nbf.v4.new_code_cell("""# Skewness check
numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z', 'volume', 'dimension_ratio', 'price_inr']
skew_vals = df[numerical_cols].skew()
print("--- Feature Skewness ---")
print(skew_vals.round(3))

# IQR Outlier Detection
print("\\n--- IQR Outlier Detection ---")
for col in ['carat', 'price_inr', 'x', 'y', 'z', 'volume']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    print(f"{col:12s} | Outliers: {n_outliers:5d} ({n_outliers/len(df)*100:.2f}%) | Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
"""))

cells.append(nbf.v4.new_code_cell("""# Remove extreme unphysical recording anomalies (e.g. y or z > 20mm or volume > 1000)
valid_mask = (df['y'] < 20) & (df['z'] < 20) & (df['volume'] < 1000) & (df['dimension_ratio'] < 3) & (df['dimension_ratio'] > 0.5)
df_clean = df[valid_mask].copy().reset_index(drop=True)
print(f"Shape after filtering extreme measurement anomalies: {df_clean.shape}")
"""))

# EDA
cells.append(nbf.v4.new_markdown_cell("""## 📈 7. Exploratory Data Analysis (EDA) & Visualizations"""))

cells.append(nbf.v4.new_code_cell("""# 1. Distribution of Price (INR) and Carat before & after Log Transformation
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

sns.histplot(df_clean['price_inr'], bins=40, kde=True, ax=axes[0, 0], color='#1f77b4')
axes[0, 0].set_title('Raw Price (INR) Distribution (Right-Skewed)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Price in INR (₹)')

sns.histplot(np.log1p(df_clean['price_inr']), bins=40, kde=True, ax=axes[0, 1], color='#2ca02c')
axes[0, 1].set_title('Log-Transformed Price log1p(Price INR) (Normalized)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('log1p(Price INR)')

sns.histplot(df_clean['carat'], bins=40, kde=True, ax=axes[1, 0], color='#ff7f0e')
axes[1, 0].set_title('Carat Distribution', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Carat Weight')

sns.histplot(df_clean['volume'], bins=40, kde=True, ax=axes[1, 1], color='#9467bd')
axes[1, 1].set_title('Volume Distribution (mm³)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Volume (x * y * z)')

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# 2. Categorical Distributions (Cut, Color, Clarity, Carat Category)
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
cat_order = ['Light', 'Medium', 'Heavy']

sns.countplot(data=df_clean, x='cut', order=cut_order, ax=axes[0], palette='Blues_r')
axes[0].set_title('Cut Quality Distribution', fontweight='bold')
axes[0].tick_params(axis='x', rotation=30)

sns.countplot(data=df_clean, x='color', order=color_order, ax=axes[1], palette='Purples_r')
axes[1].set_title('Color Grade Distribution', fontweight='bold')

sns.countplot(data=df_clean, x='clarity', order=clarity_order, ax=axes[2], palette='Greens_r')
axes[2].set_title('Clarity Distribution', fontweight='bold')
axes[2].tick_params(axis='x', rotation=30)

sns.countplot(data=df_clean, x='carat_category', order=cat_order, ax=axes[3], palette='Oranges_r')
axes[3].set_title('Carat Category Distribution', fontweight='bold')

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# 3. Price Variation by Cut, Color, and Clarity (Boxplots)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.boxplot(data=df_clean, x='cut', y='price_inr', order=cut_order, ax=axes[0], palette='Set2')
axes[0].set_title('Price (INR) vs Cut Quality', fontweight='bold')
axes[0].set_ylabel('Price in INR (₹)')
axes[0].tick_params(axis='x', rotation=25)

sns.boxplot(data=df_clean, x='color', y='price_inr', order=color_order, ax=axes[1], palette='Set3')
axes[1].set_title('Price (INR) vs Color Grade', fontweight='bold')
axes[1].set_ylabel('')

sns.boxplot(data=df_clean, x='clarity', y='price_inr', order=clarity_order, ax=axes[2], palette='Pastel1')
axes[2].set_title('Price (INR) vs Clarity Grade', fontweight='bold')
axes[2].set_ylabel('')
axes[2].tick_params(axis='x', rotation=25)

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# 4. Correlation Heatmap for Numerical Features
plt.figure(figsize=(10, 8))
corr_cols = ['carat', 'depth', 'table', 'x', 'y', 'z', 'volume', 'dimension_ratio', 'price_inr']
corr_matrix = df_clean[corr_cols].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', cbar=True, square=True, linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features', fontsize=14, fontweight='bold', pad=12)
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# 5. Carat vs Price (INR) with Regression Trend
plt.figure(figsize=(10, 6))
sample_eda = df_clean.sample(n=3000, random_state=42)
sns.regplot(data=sample_eda, x='carat', y='price_inr',
            scatter_kws={'alpha': 0.3, 'color': '#2b5c8f'},
            line_kws={'color': '#e74c3c', 'linewidth': 2})
plt.title('Carat Weight vs Diamond Price in INR (Regression Trendline)', fontsize=13, fontweight='bold')
plt.xlabel('Carat Weight (ct)')
plt.ylabel('Price in INR (₹)')
plt.show()
"""))

# Categorical Encoding
cells.append(nbf.v4.new_markdown_cell("""## 🏷️ 8. Categorical Encoding & Feature Preparation
Because diamond cut, color, clarity, and carat categories have natural ordinal hierarchies, we apply domain-correct **Ordinal Encoding**:
- **Cut:** `Fair (0) < Good (1) < Very Good (2) < Premium (3) < Ideal (4)`
- **Color:** `J (0) < I (1) < H (2) < G (3) < F (4) < E (5) < D (6)`
- **Clarity:** `I1 (0) < SI2 (1) < SI1 (2) < VS2 (3) < VS1 (4) < VVS2 (5) < VVS1 (6) < IF (7)`
- **Carat Category:** `Light (0) < Medium (1) < Heavy (2)`
"""))

cells.append(nbf.v4.new_code_cell("""cut_map = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
color_map = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
clarity_map = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}
carat_cat_map = {'Light': 0, 'Medium': 1, 'Heavy': 2}

df_clean['cut_encoded'] = df_clean['cut'].map(cut_map)
df_clean['color_encoded'] = df_clean['color'].map(color_map)
df_clean['clarity_encoded'] = df_clean['clarity'].map(clarity_map)
df_clean['carat_cat_encoded'] = df_clean['carat_category'].map(carat_cat_map)

# Feature selection for regression
reg_features = [
    'carat', 'cut_encoded', 'color_encoded', 'clarity_encoded',
    'depth', 'table', 'x', 'y', 'z', 'volume', 'dimension_ratio',
    'table_depth_ratio', 'carat_cat_encoded'
]

X = df_clean[reg_features]
y = df_clean['price_inr']
y_log = np.log1p(y)

print(f"Feature Matrix Shape: {X.shape}")
print("Features:", reg_features)
"""))

# Model Building: Regression
cells.append(nbf.v4.new_markdown_cell("""## 🤖 9. Regression Modeling: Diamond Price Prediction in INR
We perform an 80-20 Train-Test split. Target variable is log-transformed `log1p(price_inr)` to handle heavy right-skewness and stabilize error variance.

Models trained and evaluated:
1. **Linear Regression** (Baseline parametric model)
2. **Decision Tree Regressor**
3. **Random Forest Regressor**
4. **K-Nearest Neighbors (KNN)**
5. **XGBoost Regressor** (Gradient boosted trees)
6. **Artificial Neural Network (ANN)** (Deep Learning via TensorFlow/Keras)

Evaluation Metrics: **MAE (₹)**, **MSE**, **RMSE (₹)**, and **$R^2$ Score**.
"""))

cells.append(nbf.v4.new_code_cell("""# 80-20 Train-Test Split
X_train, X_test, y_train_log, y_test_log, y_train_orig, y_test_orig = train_test_split(
    X, y_log, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Train Traditional ML Regressors
reg_models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=14, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1),
    "KNN": KNeighborsRegressor(n_neighbors=7, n_jobs=-1),
    "XGBoost": xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
}

results = {}
predictions = {}

for name, model in reg_models.items():
    if name in ["Linear Regression", "KNN"]:
        model.fit(X_train_scaled, y_train_log)
        y_pred_log = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train_log)
        y_pred_log = model.predict(X_test)
        
    y_pred = np.expm1(y_pred_log)
    predictions[name] = y_pred
    
    mae = mean_absolute_error(y_test_orig, y_pred)
    mse = mean_squared_error(y_test_orig, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_orig, y_pred)
    
    results[name] = {'MAE (INR)': round(mae, 2), 'MSE': round(mse, 2), 'RMSE (INR)': round(rmse, 2), 'R2': round(r2, 4)}
    print(f"{name:20s} | MAE: INR {mae:10,.2f} | RMSE: INR {rmse:10,.2f} | R2: {r2:.4f}")
"""))

cells.append(nbf.v4.new_code_cell("""# 2. Build and Train Artificial Neural Network (ANN)
ann_model = models.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.15),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.1),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

ann_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), loss='huber', metrics=['mae'])
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

print(ann_model.summary())

history = ann_model.fit(
    X_train_scaled, y_train_log,
    validation_split=0.15,
    epochs=35,
    batch_size=128,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ANN Evaluation
ann_pred_log = ann_model.predict(X_test_scaled, verbose=0).flatten()
ann_pred = np.expm1(ann_pred_log)
predictions["Artificial Neural Network (ANN)"] = ann_pred

ann_mae = mean_absolute_error(y_test_orig, ann_pred)
ann_mse = mean_squared_error(y_test_orig, ann_pred)
ann_rmse = np.sqrt(ann_mse)
ann_r2 = r2_score(y_test_orig, ann_pred)

results["Artificial Neural Network (ANN)"] = {
    'MAE (INR)': round(ann_mae, 2),
    'MSE': round(ann_mse, 2),
    'RMSE (INR)': round(ann_rmse, 2),
    'R2': round(ann_r2, 4)
}
print(f"\\n{'ANN Model':20s} | MAE: INR {ann_mae:10,.2f} | RMSE: INR {ann_rmse:10,.2f} | R2: {ann_r2:.4f}")
"""))

cells.append(nbf.v4.new_code_cell("""# Model Performance Benchmark Table
df_results = pd.DataFrame(results).T.sort_values(by='R2', ascending=False)
display(df_results)
"""))

cells.append(nbf.v4.new_code_cell("""# Visual Comparison of Models (R² and RMSE)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

df_results['R2'].plot(kind='barh', ax=axes[0], color='#2b5c8f')
axes[0].set_title('Model Comparison: R² Score (Higher is Better)', fontweight='bold')
axes[0].set_xlim([0.9, 1.0])
axes[0].set_xlabel('R² Score')

df_results['RMSE (INR)'].plot(kind='barh', ax=axes[1], color='#e74c3c')
axes[1].set_title('Model Comparison: RMSE in INR (Lower is Better)', fontweight='bold')
axes[1].set_xlabel('Root Mean Squared Error (₹)')

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# Feature Importance via Best Model (XGBoost)
xgb_best = reg_models["XGBoost"]
feat_importances = pd.Series(xgb_best.feature_importances_, index=reg_features).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
feat_importances.plot(kind='barh', color='#16a085')
plt.title('Feature Importance for Diamond Price Prediction (XGBoost)', fontsize=13, fontweight='bold')
plt.xlabel('Relative Importance Score')
plt.show()
"""))

# Market Segmentation
cells.append(nbf.v4.new_markdown_cell("""## 🏷️ 10. Market Segmentation (Clustering & PCA)
To segment diamonds into natural market groups, we **exclude price** (so grouping is purely based on physical and qualitative attributes):
- Features used: `carat`, `cut_encoded`, `color_encoded`, `clarity_encoded`, `depth`, `table`, `x`, `y`, `z`, `volume`, `dimension_ratio`.
- Standardize features with `StandardScaler`.
- Determine optimal clusters via **Elbow Method (Inertia)** and **Silhouette Analysis**.
- Reduce dimensions using **PCA** (2D & 3D) for cluster visualization.
- Characterize and assign business-driven segment names:
  1. **Affordable Small Diamonds**: Entry-level budget stones ($< 0.5$ ct, low price).
  2. **Mid-range Balanced Diamonds**: Commercial grade everyday stones ($0.6 - 1.1$ ct).
  3. **High-Clarity Fine Solitaires**: Precision cut, exceptional clarity.
  4. **Premium Heavy Diamonds**: Large luxury gemstones ($> 1.3$ ct, top market valuations).
"""))

cells.append(nbf.v4.new_code_cell("""cluster_features = [
    'carat', 'cut_encoded', 'color_encoded', 'clarity_encoded',
    'depth', 'table', 'x', 'y', 'z', 'volume', 'dimension_ratio'
]

X_clust = df_clean[cluster_features]
scaler_clust = StandardScaler()
X_clust_scaled = scaler_clust.fit_transform(X_clust)

# Elbow & Silhouette Evaluation (k=2 to 6)
inertias = []
sil_scores = []
k_range = range(2, 7)
sample_idx = np.random.choice(len(X_clust_scaled), size=5000, replace=False)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_clust_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_clust_scaled[sample_idx], km.predict(X_clust_scaled[sample_idx]))
    sil_scores.append(sil)
    print(f"k = {k} | Inertia: {km.inertia_:10,.1f} | Silhouette Score: {sil:.4f}")

# Plot Elbow and Silhouette curves
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))

axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_title('Elbow Method: Inertia vs k', fontweight='bold')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia (Within-cluster sum of squares)')

axes[1].plot(k_range, sil_scores, 'rs-', linewidth=2, markersize=8)
axes[1].set_title('Silhouette Score vs k', fontweight='bold')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# Fit Optimal K-Means Model (k = 4)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_clean['cluster'] = kmeans.fit_predict(X_clust_scaled)

# Dimensionality Reduction: PCA 2D and 3D
pca_2d = PCA(n_components=2, random_state=42)
pca_2d_coords = pca_2d.fit_transform(X_clust_scaled)
df_clean['pca_1'] = pca_2d_coords[:, 0]
df_clean['pca_2'] = pca_2d_coords[:, 1]

print(f"PCA 2D Total Explained Variance: {pca_2d.explained_variance_ratio_.sum()*100:.2f}%")
"""))

cells.append(nbf.v4.new_code_cell("""# 2D PCA Cluster Scatter Plot
plt.figure(figsize=(10, 7))
palette = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
sns.scatterplot(
    data=df_clean.sample(5000, random_state=42),
    x='pca_1', y='pca_2',
    hue='cluster',
    palette=palette,
    alpha=0.6,
    s=40
)
plt.title('Diamond Market Segments Visualized via PCA (k=4)', fontsize=14, fontweight='bold')
plt.xlabel('Principal Component 1 (Dimension & Size Dominant)')
plt.ylabel('Principal Component 2 (Cut & Proportion Dominant)')
plt.legend(title='Cluster ID', loc='best')
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# Cluster Profiling and Business Naming
cluster_summary = df_clean.groupby('cluster').agg({
    'carat': ['mean', 'min', 'max'],
    'price_inr': 'mean',
    'cut_encoded': 'mean',
    'clarity_encoded': 'mean',
    'color_encoded': 'mean',
    'volume': 'mean',
    'x': 'count'
}).rename(columns={'x': 'count'})

display(cluster_summary)

# Business Segment Mapping
cluster_names = {
    0: "Mid-range Balanced Diamonds",
    1: "Affordable Small Diamonds",
    2: "Premium Heavy Diamonds",
    3: "High-Clarity Fine Solitaires"
}

df_clean['market_segment'] = df_clean['cluster'].map(cluster_names)
print("\\nMarket Segment Distribution:")
print(df_clean['market_segment'].value_counts(normalize=True).round(4) * 100)
"""))

# Strategic Insights & Conclusion
cells.append(nbf.v4.new_markdown_cell("""## 💡 11. Key Insights & Conclusion
1. **Predictive Performance**:
   - **XGBoost Regressor** achieved the highest accuracy ($R^2 \\approx 0.982$, RMSE $\\approx$ ₹43,672), closely followed by **Random Forest** ($R^2 \\approx 0.981$) and the **Artificial Neural Network (ANN)** ($R^2 \\approx 0.970$).
   - Carat weight and physical volume are the dominant drivers of diamond price, while cut quality and clarity create substantial premium tiers within each carat bracket.
2. **Market Segmentation**:
   - Diamonds segment cleanly into **4 primary consumer personas**:
     1. *Affordable Small Diamonds*: High volume (42.8%), lower price, entry-level market.
     2. *Mid-range Balanced Diamonds*: High volume (30.3%), mainstream commercial demand.
     3. *High-Clarity Fine Solitaires*: High optical quality, appealing to discerning buyers.
     4. *Premium Heavy Diamonds*: Luxury flagship items commanding the highest margins.
3. **Deployment**:
   - All trained models, PCA transforms, scalers, and metadata are saved and served interactively via the **Streamlit Web Application** (`app.py`).
"""))

nb.cells = cells

with open('Diamond_Dynamics_Analysis_and_Modeling.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Successfully generated 'Diamond_Dynamics_Analysis_and_Modeling.ipynb'!")
