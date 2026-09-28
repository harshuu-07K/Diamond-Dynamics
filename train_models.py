import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# Fix Windows cp1252 stdout encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Set random seed
np.random.seed(42)
tf.random.set_seed(42)

USD_TO_INR = 83.0  # Conversion rate: 1 USD = 83 INR
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

print("="*60)
print("[DIAMONDS] DIAMOND DYNAMICS: END-TO-END PIPELINE")
print("="*60)

# 1. LOAD DATA
df = pd.read_csv("diamonds.csv")
print(f"Original shape: {df.shape}")

# 2. DATA CLEANING
# Diamonds cannot have 0 or negative length, width, or depth (x, y, z)
invalid_mask = (df['x'] <= 0) | (df['y'] <= 0) | (df['z'] <= 0)
print(f"Invalid dimension rows (x, y, z <= 0): {invalid_mask.sum()}")
df = df[~invalid_mask].copy()

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")
df = df.drop_duplicates().reset_index(drop=True)
print(f"Cleaned shape: {df.shape}")

# 3. FEATURE ENGINEERING
# Convert price to INR
df['price_inr'] = df['price'] * USD_TO_INR

# Volume = x * y * z
df['volume'] = df['x'] * df['y'] * df['z']

# Dimension Ratio = (x + y) / (2 * z)
df['dimension_ratio'] = (df['x'] + df['y']) / (2 * df['z'])

# Carat Category: Light (<0.5), Medium (0.5-1.5), Heavy (>1.5)
def categorize_carat(c):
    if c < 0.5:
        return 'Light'
    elif c <= 1.5:
        return 'Medium'
    else:
        return 'Heavy'

df['carat_category'] = df['carat'].apply(categorize_carat)

# Price per Carat (in INR)
df['price_per_carat_inr'] = df['price_inr'] / df['carat']

# Table to Depth ratio
df['table_depth_ratio'] = df['table'] / df['depth']

# Surface Area approximation (2*(xy + yz + zx))
df['surface_area_approx'] = 2 * (df['x']*df['y'] + df['y']*df['z'] + df['z']*df['x'])

# Density proxy: carat / volume (carat is 0.2g, density = (carat*0.2)/volume in g/mm^3)
# To avoid division by zero or extreme outliers, handle volume > 0
df['density_proxy'] = (df['carat'] * 0.2) / (df['volume'] + 1e-5)

print("\n--- Feature Engineering Complete ---")
print("New columns added: price_inr, volume, dimension_ratio, carat_category, price_per_carat_inr, table_depth_ratio, surface_area_approx, density_proxy")

# 4. OUTLIER & SKEWNESS ANALYSIS
numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z', 'volume', 'dimension_ratio', 'price_inr']
skewness = df[numerical_cols].skew()
print("\n--- Feature Skewness ---")
print(skewness)

# Outlier bounds via IQR
outlier_info = {}
for col in ['carat', 'price_inr', 'x', 'y', 'z', 'volume']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    outlier_info[col] = {
        'Q1': float(Q1), 'Q3': float(Q3), 'IQR': float(IQR),
        'lower_bound': float(lower_bound), 'upper_bound': float(upper_bound),
        'outlier_count': int(n_outliers), 'outlier_pct': float(round(n_outliers / len(df) * 100, 2))
    }
print("\n--- Outlier Summary (IQR) ---")
for col, v in outlier_info.items():
    print(f"  {col}: {v['outlier_count']} outliers ({v['outlier_pct']}%)")

# Filter out extreme unphysical/measurement anomalies for robust modeling
# (e.g. y > 20mm or z > 20mm or volume > 1000mm3 or dimension_ratio > 3)
clean_filter = (df['y'] < 20) & (df['z'] < 20) & (df['volume'] < 1000) & (df['dimension_ratio'] < 3) & (df['dimension_ratio'] > 0.5)
df_model = df[clean_filter].copy().reset_index(drop=True)
print(f"Dataset after removing extreme physical anomalies: {df_model.shape}")

# Save the enriched dataset for EDA & App
df_model.to_csv("diamonds_engineered.csv", index=False)
print("Saved diamonds_engineered.csv")

# 5. ORDINAL ENCODING
cut_map = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
color_map = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
clarity_map = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}
carat_cat_map = {'Light': 0, 'Medium': 1, 'Heavy': 2}

df_model['cut_encoded'] = df_model['cut'].map(cut_map)
df_model['color_encoded'] = df_model['color'].map(color_map)
df_model['clarity_encoded'] = df_model['clarity'].map(clarity_map)
df_model['carat_cat_encoded'] = df_model['carat_category'].map(carat_cat_map)

# Feature sets
# For regression features:
reg_features = [
    'carat', 'cut_encoded', 'color_encoded', 'clarity_encoded',
    'depth', 'table', 'x', 'y', 'z', 'volume', 'dimension_ratio',
    'table_depth_ratio', 'carat_cat_encoded'
]

X_reg = df_model[reg_features]
# Log1p transformation on target (price_inr) to handle heavy right skewness
y_reg = df_model['price_inr']
y_reg_log = np.log1p(y_reg)

# Train/Test Split (80/20)
X_train, X_test, y_train_log, y_test_log, y_train_orig, y_test_orig = train_test_split(
    X_reg, y_reg_log, y_reg, test_size=0.2, random_state=42
)

# Feature Scaler
scaler_reg = StandardScaler()
X_train_scaled = scaler_reg.fit_transform(X_train)
X_test_scaled = scaler_reg.transform(X_test)

# 6. REGRESSION MODEL TRAINING & BENCHMARKING
print("\n" + "="*60)
print("[TRAINING] TRAINING REGRESSION MODELS")
print("="*60)

reg_models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=14, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1),
    "KNN": KNeighborsRegressor(n_neighbors=7, n_jobs=-1),
    "XGBoost": xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
}

results = {}
trained_models = {}

for name, model in reg_models.items():
    print(f"Training {name}...")
    if name in ["Linear Regression", "KNN"]:
        model.fit(X_train_scaled, y_train_log)
        y_pred_log = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train_log)
        y_pred_log = model.predict(X_test)
    
    y_pred = np.expm1(y_pred_log)
    
    mae = mean_absolute_error(y_test_orig, y_pred)
    mse = mean_squared_error(y_test_orig, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_orig, y_pred)
    
    results[name] = {
        'MAE': round(float(mae), 2),
        'MSE': round(float(mse), 2),
        'RMSE': round(float(rmse), 2),
        'R2': round(float(r2), 4)
    }
    trained_models[name] = model
    print(f"  -> MAE: INR {mae:,.2f} | RMSE: INR {rmse:,.2f} | R2: {r2:.4f}")

# 7. ARTIFICIAL NEURAL NETWORK (ANN) REGRESSION
print("\nTraining Artificial Neural Network (ANN)...")
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

history = ann_model.fit(
    X_train_scaled, y_train_log,
    validation_split=0.15,
    epochs=35,
    batch_size=128,
    callbacks=[early_stop, reduce_lr],
    verbose=0
)

ann_pred_log = ann_model.predict(X_test_scaled, verbose=0).flatten()
ann_pred = np.expm1(ann_pred_log)

ann_mae = mean_absolute_error(y_test_orig, ann_pred)
ann_mse = mean_squared_error(y_test_orig, ann_pred)
ann_rmse = np.sqrt(ann_mse)
ann_r2 = r2_score(y_test_orig, ann_pred)

results["Artificial Neural Network (ANN)"] = {
    'MAE': round(float(ann_mae), 2),
    'MSE': round(float(ann_mse), 2),
    'RMSE': round(float(ann_rmse), 2),
    'R2': round(float(ann_r2), 4)
}
print(f"  -> ANN MAE: INR {ann_mae:,.2f} | RMSE: INR {ann_rmse:,.2f} | R2: {ann_r2:.4f}")

# Compare and identify best model
best_model_name = max(results.items(), key=lambda x: x[1]['R2'])[0]
print(f"\n[BEST] Best Regression Model: {best_model_name} (R2 = {results[best_model_name]['R2']})")

# Feature importance from XGBoost
xgb_model = trained_models["XGBoost"]
feat_importances = pd.Series(xgb_model.feature_importances_, index=reg_features).sort_values(ascending=False)
feat_imp_dict = feat_importances.to_dict()

# Save regression artifacts
joblib.dump(trained_models["XGBoost"], os.path.join(MODELS_DIR, "best_regression_model.pkl"))
joblib.dump(scaler_reg, os.path.join(MODELS_DIR, "scaler_regression.pkl"))
joblib.dump(reg_features, os.path.join(MODELS_DIR, "regression_features.pkl"))
ann_model.save(os.path.join(MODELS_DIR, "ann_model.keras"))

with open(os.path.join(MODELS_DIR, "regression_metrics.json"), "w") as f:
    json.dump({
        'results': results,
        'best_model': best_model_name,
        'feature_importance': feat_imp_dict,
        'conversion_rate_usd_to_inr': USD_TO_INR
    }, f, indent=4)

print("Saved regression models and metrics!")

# 8. CLUSTERING & MARKET SEGMENTATION
print("\n" + "="*60)
print("[CLUSTERING] MARKET SEGMENTATION")
print("="*60)

# Unsupervised features: carat, cut, color, clarity, depth, table, x, y, z, volume, dimension_ratio (NO PRICE!)
cluster_features = [
    'carat', 'cut_encoded', 'color_encoded', 'clarity_encoded',
    'depth', 'table', 'x', 'y', 'z', 'volume', 'dimension_ratio'
]

X_cluster = df_model[cluster_features]
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

# Evaluate Elbow Method & Silhouette Scores for k = 2 to 6 (sample for silhouette to keep speed high)
elbow_inertias = {}
sample_indices = np.random.choice(len(X_cluster_scaled), size=5000, replace=False)
silhouette_scores = {}

for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cluster_scaled)
    elbow_inertias[k] = float(round(km.inertia_, 2))
    sil = silhouette_score(X_cluster_scaled[sample_indices], km.predict(X_cluster_scaled[sample_indices]))
    silhouette_scores[k] = float(round(sil, 4))
    print(f"  k={k} | Inertia: {km.inertia_:,.1f} | Silhouette Score: {sil:.4f}")

# Choose k=4 as optimal market segmentation:
# 1. Budget/Affordable Small Stones
# 2. Balanced Everyday Diamonds
# 3. High-Quality Fine Gemstones
# 4. Premium Heavy Luxury Solitaires
optimal_k = 4
best_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = best_kmeans.fit_predict(X_cluster_scaled)
df_model['cluster'] = cluster_labels

# Dimensionality Reduction with PCA (2D and 3D)
pca_2d = PCA(n_components=2, random_state=42)
pca_2d_coords = pca_2d.fit_transform(X_cluster_scaled)
df_model['pca_1'] = pca_2d_coords[:, 0]
df_model['pca_2'] = pca_2d_coords[:, 1]

pca_3d = PCA(n_components=3, random_state=42)
pca_3d_coords = pca_3d.fit_transform(X_cluster_scaled)
df_model['pca_3d_1'] = pca_3d_coords[:, 0]
df_model['pca_3d_2'] = pca_3d_coords[:, 1]
df_model['pca_3d_3'] = pca_3d_coords[:, 2]

pca_variance = {
    '2D_explained_variance_ratio': [float(round(v, 4)) for v in pca_2d.explained_variance_ratio_],
    '2D_total_variance': float(round(pca_2d.explained_variance_ratio_.sum(), 4)),
    '3D_explained_variance_ratio': [float(round(v, 4)) for v in pca_3d.explained_variance_ratio_],
    '3D_total_variance': float(round(pca_3d.explained_variance_ratio_.sum(), 4))
}
print(f"PCA 2D Total Variance Explained: {pca_variance['2D_total_variance']*100:.2f}%")
print(f"PCA 3D Total Variance Explained: {pca_variance['3D_total_variance']*100:.2f}%")

# Cluster Profiling & Domain-Intelligent Naming
cluster_profiles = {}
for c in range(optimal_k):
    sub = df_model[df_model['cluster'] == c]
    mean_carat = sub['carat'].mean()
    mean_price = sub['price_inr'].mean()
    mean_cut = sub['cut_encoded'].mean()
    mean_color = sub['color_encoded'].mean()
    mean_clarity = sub['clarity_encoded'].mean()
    mean_volume = sub['volume'].mean()
    count = len(sub)
    pct = (count / len(df_model)) * 100
    
    cluster_profiles[c] = {
        'count': int(count),
        'percentage': float(round(pct, 2)),
        'mean_carat': float(round(mean_carat, 3)),
        'mean_price_inr': float(round(mean_price, 2)),
        'mean_price_usd': float(round(mean_price / USD_TO_INR, 2)),
        'mean_volume': float(round(mean_volume, 2)),
        'cut_score': float(round(mean_cut, 2)),
        'color_score': float(round(mean_color, 2)),
        'clarity_score': float(round(mean_clarity, 2)),
        'most_common_cut': sub['cut'].mode()[0],
        'most_common_color': sub['color'].mode()[0],
        'most_common_clarity': sub['clarity'].mode()[0]
    }

# Rank clusters by mean_carat / mean_price to assign meaningful business names
sorted_clusters_by_carat = sorted(cluster_profiles.keys(), key=lambda c: cluster_profiles[c]['mean_carat'])

# Naming schema following project requirements:
# - Affordable Small Diamonds (lowest carat & price)
# - Mid-range Balanced Diamonds (medium carat, standard grade)
# - High-Clarity Fine Solitaires (moderate carat, superior cut/clarity)
# - Premium Heavy Diamonds (highest carat & luxury price)
name_mapping = {}
descriptions = {}

if optimal_k == 4:
    c_lowest = sorted_clusters_by_carat[0]
    c_highest = sorted_clusters_by_carat[3]
    mid_clusters = [sorted_clusters_by_carat[1], sorted_clusters_by_carat[2]]
    # Distinguish mid-clusters by clarity/cut
    if cluster_profiles[mid_clusters[0]]['clarity_score'] > cluster_profiles[mid_clusters[1]]['clarity_score']:
        c_fine = mid_clusters[0]
        c_balanced = mid_clusters[1]
    else:
        c_fine = mid_clusters[1]
        c_balanced = mid_clusters[0]
        
    name_mapping[c_lowest] = "Affordable Small Diamonds"
    descriptions[c_lowest] = "Entry-level, lightweight stones (<0.6 ct) suitable for daily jewelry and budget-conscious buyers."
    
    name_mapping[c_balanced] = "Mid-range Balanced Diamonds"
    descriptions[c_balanced] = "Popular commercial-grade diamonds (0.6 - 1.1 ct) offering good balance of size and affordability."
    
    name_mapping[c_fine] = "High-Clarity Fine Solitaires"
    descriptions[c_fine] = "Precision-cut, high-clarity diamonds with excellent optical symmetry and exceptional brilliance."
    
    name_mapping[c_highest] = "Premium Heavy Diamonds"
    descriptions[c_highest] = "Substantial, high-carat (>1.3 ct) investment-grade diamonds commanding top luxury market valuations."

for c, prof in cluster_profiles.items():
    prof['segment_name'] = name_mapping.get(c, f"Segment {c}")
    prof['description'] = descriptions.get(c, "")
    print(f"\nCluster {c}: {prof['segment_name']}")
    print(f"  Count: {prof['count']} ({prof['percentage']}%)")
    print(f"  Avg Carat: {prof['mean_carat']} | Avg Price: INR {prof['mean_price_inr']:,.2f} ($ {prof['mean_price_usd']:,.2f})")
    print(f"  Avg Cut: {prof['most_common_cut']} | Color: {prof['most_common_color']} | Clarity: {prof['most_common_clarity']}")

# Add segment names to dataframe
df_model['segment_name'] = df_model['cluster'].map(lambda c: cluster_profiles[c]['segment_name'])

# Save clustering artifacts
joblib.dump(best_kmeans, os.path.join(MODELS_DIR, "clustering_model.pkl"))
joblib.dump(scaler_cluster, os.path.join(MODELS_DIR, "scaler_cluster.pkl"))
joblib.dump(pca_2d, os.path.join(MODELS_DIR, "pca_2d.pkl"))
joblib.dump(pca_3d, os.path.join(MODELS_DIR, "pca_3d.pkl"))
joblib.dump(cluster_features, os.path.join(MODELS_DIR, "cluster_features.pkl"))

with open(os.path.join(MODELS_DIR, "clustering_metrics.json"), "w") as f:
    json.dump({
        'optimal_k': optimal_k,
        'elbow_inertias': elbow_inertias,
        'silhouette_scores': silhouette_scores,
        'pca_variance': pca_variance,
        'cluster_profiles': cluster_profiles
    }, f, indent=4)

# Save processed sample for rapid interactive app loading
df_sample = df_model.sample(n=min(5000, len(df_model)), random_state=42).copy()
df_sample.to_csv(os.path.join(MODELS_DIR, "diamonds_app_sample.csv"), index=False)
print("\nSaved all clustering models, PCA transforms, and app sample data!")

# Save categorical mappings
with open(os.path.join(MODELS_DIR, "category_mappings.json"), "w") as f:
    json.dump({
        'cut_map': cut_map,
        'color_map': color_map,
        'clarity_map': clarity_map,
        'carat_cat_map': carat_cat_map,
        'cut_order': list(cut_map.keys()),
        'color_order': list(color_map.keys()),
        'clarity_order': list(clarity_map.keys()),
        'carat_cat_order': list(carat_cat_map.keys())
    }, f, indent=4)

print("\nPipeline execution complete! All artifacts saved in 'models/' directory.")
