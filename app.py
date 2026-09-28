"""
Diamond Dynamics: Price Prediction & Market Segmentation
Interactive Streamlit Web Application
GUVI | HCL Capstone Project
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Diamond Dynamics | Valuation & Market Segmentation",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN LUXURY CSS STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;0,700;1,600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Gradient Banner */
    .hero-banner {
        background: radial-gradient(circle at 10% 20%, rgba(26, 42, 75, 0.95) 0%, rgba(13, 21, 38, 0.98) 90%),
                    linear-gradient(135deg, #0e1e38 0%, #1e3a5f 50%, #0b1528 100%);
        border: 1px solid rgba(82, 168, 255, 0.2);
        border-radius: 18px;
        padding: 32px 36px;
        margin-bottom: 24px;
        box-shadow: 0 12px 36px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #ffffff 30%, #a5d8ff 70%, #d0ebff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        max-width: 800px;
        line-height: 1.5;
    }
    
    /* Luxury Cards */
    .metric-card {
        background: rgba(18, 26, 43, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
    }
    
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
        margin-bottom: 4px;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #38bdf8;
        font-feature-settings: "tnum";
    }
    
    .metric-sub {
        font-size: 0.85rem;
        color: #cbd5e1;
        margin-top: 4px;
    }
    
    /* Result Showcase Banner */
    .price-showcase {
        background: linear-gradient(135deg, rgba(14, 116, 144, 0.25) 0%, rgba(30, 58, 138, 0.35) 100%);
        border: 1.5px solid rgba(56, 189, 248, 0.4);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-top: 15px;
        box-shadow: 0 10px 30px -5px rgba(14, 116, 144, 0.3);
    }
    
    .price-number {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        font-weight: 800;
        color: #38bdf8;
        margin: 6px 0;
        text-shadow: 0 0 25px rgba(56, 189, 248, 0.4);
    }
    
    .cluster-tag {
        display: inline-block;
        padding: 6px 16px;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: #ffffff;
        font-weight: 600;
        font-size: 0.95rem;
        border-radius: 9999px;
        letter-spacing: 0.3px;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* Sidebar styling */
    .sidebar-brand {
        padding: 12px 0 20px 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD ARTIFACTS WITH CACHING ---
MODELS_DIR = "models"

@st.cache_resource
def load_models():
    reg_model = joblib.load(os.path.join(MODELS_DIR, "best_regression_model.pkl"))
    scaler_reg = joblib.load(os.path.join(MODELS_DIR, "scaler_regression.pkl"))
    reg_features = joblib.load(os.path.join(MODELS_DIR, "regression_features.pkl"))
    
    cluster_model = joblib.load(os.path.join(MODELS_DIR, "clustering_model.pkl"))
    scaler_cluster = joblib.load(os.path.join(MODELS_DIR, "scaler_cluster.pkl"))
    pca_2d = joblib.load(os.path.join(MODELS_DIR, "pca_2d.pkl"))
    pca_3d = joblib.load(os.path.join(MODELS_DIR, "pca_3d.pkl"))
    cluster_features = joblib.load(os.path.join(MODELS_DIR, "cluster_features.pkl"))
    
    return {
        'reg_model': reg_model,
        'scaler_reg': scaler_reg,
        'reg_features': reg_features,
        'cluster_model': cluster_model,
        'scaler_cluster': scaler_cluster,
        'pca_2d': pca_2d,
        'pca_3d': pca_3d,
        'cluster_features': cluster_features
    }

@st.cache_data
def load_metadata():
    with open(os.path.join(MODELS_DIR, "category_mappings.json")) as f:
        mappings = json.load(f)
    with open(os.path.join(MODELS_DIR, "regression_metrics.json")) as f:
        reg_metrics = json.load(f)
    with open(os.path.join(MODELS_DIR, "clustering_metrics.json")) as f:
        cluster_metrics = json.load(f)
    sample_df = pd.read_csv(os.path.join(MODELS_DIR, "diamonds_app_sample.csv"))
    
    return mappings, reg_metrics, cluster_metrics, sample_df

try:
    artifacts = load_models()
    mappings, reg_metrics, cluster_metrics, sample_df = load_metadata()
    USD_TO_INR = reg_metrics.get('conversion_rate_usd_to_inr', 83.0)
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please ensure train_models.py has completed successfully.")
    st.stop()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2 style="font-family:'Playfair Display',serif; margin:0; color:#38bdf8;">💎 Diamond Dynamics</h2>
        <p style="font-size:0.85rem; color:#94a3b8; margin:2px 0 0 0;">Price Prediction & Market Segmentation</p>
    </div>
    """, unsafe_allow_html=True)
    
    nav_option = st.radio(
        "Navigate Dashboard",
        [
            "💎 Valuation & Clustering Engine",
            "🏷️ Market Segmentation Explorer",
            "📊 Exploratory Data Analytics (EDA)",
            "🤖 AI Model Benchmark & Architecture",
            "📖 Gemology & Project Guide"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 💱 Currency Exchange Rate")
    st.markdown(f"**1 USD ($) = {USD_TO_INR:.1f} INR (₹)**")
    st.caption("Exchange rate fixed per project specifications for consistent real-time conversion.")
    
    st.markdown("---")
    st.markdown("### 🏆 Production AI Model")
    st.markdown(f"**Regression:** `{reg_metrics['best_model']}` ($R^2 = {reg_metrics['results'][reg_metrics['best_model']]['R2']:.4f}$)")
    st.markdown(f"**Clustering:** `K-Means (k={cluster_metrics['optimal_k']})`")
    st.markdown(f"**PCA Variance:** `{cluster_metrics['pca_variance']['2D_total_variance']*100:.1f}% (2D) | {cluster_metrics['pca_variance']['3D_total_variance']*100:.1f}% (3D)`")


# ==============================================================================
# TAB 1: VALUATION & CLUSTERING ENGINE
# ==============================================================================
if nav_option == "💎 Valuation & Clustering Engine":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Diamond Valuation & Market Segment Predictor</div>
        <div class="hero-subtitle">
            Configure diamond 4C parameters and physical dimensions to predict the market valuation in <b>INR (₹)</b> 
            using the best-performing ensemble regressor (XGBoost) and identify the corresponding target market cluster segment.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_input, col_result = st.columns([1.1, 1.2], gap="large")
    
    with col_input:
        st.subheader("⚙️ Diamond Specifications (Inputs)")
        
        # Carat Weight
        carat = st.number_input(
            "Carat Weight (ct)",
            min_value=0.10,
            max_value=5.00,
            value=0.70,
            step=0.01,
            help="Weight of the diamond in carats (1 carat = 200 milligrams)."
        )
        
        # Cut, Color, Clarity
        c1, c2, c3 = st.columns(3)
        with c1:
            cut = st.selectbox(
                "Cut Quality",
                options=mappings['cut_order'],
                index=4,  # Ideal
                help="Proportion and finish rating: Fair, Good, Very Good, Premium, Ideal."
            )
        with c2:
            color = st.selectbox(
                "Color Grade",
                options=mappings['color_order'],
                index=4,  # F
                help="Colorless D to faint light yellow/brown J."
            )
        with c3:
            clarity = st.selectbox(
                "Clarity Grade",
                options=mappings['clarity_order'],
                index=4,  # VS1
                help="Inclusion purity from Flawless (IF) to Included (I1)."
            )
        
        # Dimensions x, y, z
        st.markdown("#### 📏 Physical Dimensions (mm)")
        d1, d2, d3 = st.columns(3)
        with d1:
            x = st.number_input("Length x (mm)", min_value=1.0, max_value=15.0, value=5.65, step=0.05)
        with d2:
            y = st.number_input("Width y (mm)", min_value=1.0, max_value=15.0, value=5.68, step=0.05)
        with d3:
            z = st.number_input("Depth z (mm)", min_value=1.0, max_value=12.0, value=3.52, step=0.05)
            
        # Depth % and Table %
        st.markdown("#### 📐 Proportions (%)")
        p1, p2 = st.columns(2)
        with p1:
            depth = st.slider("Total Depth %", min_value=50.0, max_value=75.0, value=61.8, step=0.1)
        with p2:
            table = st.slider("Table Width %", min_value=45.0, max_value=75.0, value=57.0, step=0.5)
            
        # Automated Derived Features Preview
        volume = x * y * z
        dim_ratio = (x + y) / (2.0 * z) if z > 0 else 1.0
        table_depth = table / depth if depth > 0 else 1.0
        
        if carat < 0.5:
            carat_cat = "Light"
        elif carat <= 1.5:
            carat_cat = "Medium"
        else:
            carat_cat = "Heavy"
            
        st.markdown("#### 🔍 Real-Time Derived Feature Pipeline")
        f1, f2, f3 = st.columns(3)
        f1.metric("Volume (mm³)", f"{volume:.2f}")
        f2.metric("Dimension Ratio", f"{dim_ratio:.3f}")
        f3.metric("Carat Class", carat_cat)
        
        predict_btn = st.button("🔮 Calculate Valuation & Predict Market Segment", type="primary", use_container_width=True)

    with col_result:
        st.subheader("💎 Valuation & Market Segment Output")
        
        # Prepare feature vector for Regression
        cut_enc = mappings['cut_map'][cut]
        color_enc = mappings['color_map'][color]
        clarity_enc = mappings['clarity_map'][clarity]
        carat_cat_enc = mappings['carat_cat_map'][carat_cat]
        
        reg_input = pd.DataFrame([{
            'carat': carat,
            'cut_encoded': cut_enc,
            'color_encoded': color_enc,
            'clarity_encoded': clarity_enc,
            'depth': depth,
            'table': table,
            'x': x,
            'y': y,
            'z': z,
            'volume': volume,
            'dimension_ratio': dim_ratio,
            'table_depth_ratio': table_depth,
            'carat_cat_encoded': carat_cat_enc
        }])[artifacts['reg_features']]
        
        # Predict Price
        pred_log = artifacts['reg_model'].predict(reg_input)[0]
        pred_price_inr = np.expm1(pred_log)
        pred_price_usd = pred_price_inr / USD_TO_INR
        price_per_carat_inr = pred_price_inr / carat
        price_per_carat_usd = pred_price_usd / carat
        
        # Cluster Prediction
        cluster_input = pd.DataFrame([{
            'carat': carat,
            'cut_encoded': cut_enc,
            'color_encoded': color_enc,
            'clarity_encoded': clarity_enc,
            'depth': depth,
            'table': table,
            'x': x,
            'y': y,
            'z': z,
            'volume': volume,
            'dimension_ratio': dim_ratio
        }])[artifacts['cluster_features']]
        
        cluster_input_scaled = artifacts['scaler_cluster'].transform(cluster_input)
        assigned_cluster = int(artifacts['cluster_model'].predict(cluster_input_scaled)[0])
        cluster_info = cluster_metrics['cluster_profiles'][str(assigned_cluster)]
        
        # Transform into PCA space
        pca_point_2d = artifacts['pca_2d'].transform(cluster_input_scaled)[0]
        
        # Display Price Showcase
        st.markdown(f"""
        <div class="price-showcase">
            <span class="metric-label" style="color:#7dd3fc;">Estimated Diamond Valuation</span>
            <div class="price-number">₹ {pred_price_inr:,.0f}</div>
            <div style="font-size:1.15rem; color:#e0f2fe; margin-bottom:12px;">
                <b>$ {pred_price_usd:,.2f} USD</b> &nbsp;•&nbsp; <b>₹ {price_per_carat_inr:,.0f} / ct</b>
            </div>
            <div style="margin-top:10px;">
                <span class="cluster-tag">🏷️ {cluster_info['segment_name']} (Cluster {assigned_cluster})</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        
        # Cluster Description Box
        st.info(f"**Target Market Segment:** {cluster_info['segment_name']}\n\n{cluster_info['description']}\n\n"
                f"*Market Share:* **{cluster_info['percentage']}%** | *Segment Avg Carat:* **{cluster_info['mean_carat']} ct** | "
                f"*Segment Avg Price:* **₹ {cluster_info['mean_price_inr']:,.0f}**")
        
        # Visual Position in PCA Space
        st.markdown("#### 🗺️ Market Cluster Position (PCA 2D Mapping)")
        
        pca_sample = sample_df.sample(n=min(1200, len(sample_df)), random_state=42)
        fig_pca = px.scatter(
            pca_sample,
            x='pca_1',
            y='pca_2',
            color='segment_name',
            opacity=0.45,
            labels={'pca_1': 'PC 1 (Size & Volume)', 'pca_2': 'PC 2 (Cut & Proportion)', 'segment_name': 'Segment'},
            title="Diamond Position in Market Segment Space",
            color_discrete_sequence=['#38bdf8', '#818cf8', '#f43f5e', '#10b981']
        )
        
        # Add User's Diamond Point
        fig_pca.add_trace(go.Scatter(
            x=[pca_point_2d[0]],
            y=[pca_point_2d[1]],
            mode='markers+text',
            marker=dict(symbol='star', size=18, color='#fbbf24', line=dict(color='#ffffff', width=2)),
            text=["⭐ This Diamond"],
            textposition="top center",
            name="Your Diamond"
        ))
        
        fig_pca.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pca, use_container_width=True)


# ==============================================================================
# TAB 2: MARKET SEGMENTATION EXPLORER
# ==============================================================================
elif nav_option == "🏷️ Market Segmentation Explorer":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Market Segmentation & Clustering Analytics</div>
        <div class="hero-subtitle">
            Unsupervised machine learning via <b>K-Means (k=4)</b> segments diamonds into four actionable market personas.
            Evaluated using the <b>Elbow Method (Inertia)</b> and <b>Silhouette Analysis</b>, with <b>PCA</b> dimensionality reduction.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 4 Cluster Profile Cards
    st.subheader("👥 The 4 Diamond Market Personas")
    p_cols = st.columns(4)
    
    profiles = cluster_metrics['cluster_profiles']
    icons = ["⚖️", "✨", "👑", "💎"]
    colors = ["#38bdf8", "#10b981", "#fbbf24", "#f43f5e"]
    
    for i, c_idx in enumerate(sorted(profiles.keys())):
        p = profiles[c_idx]
        with p_cols[i]:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid {colors[i]};">
                <div style="font-size:1.5rem; margin-bottom:4px;">{icons[i]}</div>
                <div class="metric-label">Cluster {c_idx}</div>
                <div style="font-size:1.15rem; font-weight:700; color:#f1f5f9; margin-bottom:8px;">{p['segment_name']}</div>
                <p style="font-size:0.83rem; color:#94a3b8; height:50px;">{p['description']}</p>
                <hr style="border-color:rgba(148,163,184,0.15); margin:8px 0;">
                <div style="font-size:0.85rem; color:#cbd5e1;"><b>Market Share:</b> {p['percentage']}%</div>
                <div style="font-size:0.85rem; color:#cbd5e1;"><b>Avg Carat:</b> {p['mean_carat']} ct</div>
                <div style="font-size:0.85rem; color:#cbd5e1;"><b>Avg Price:</b> ₹ {p['mean_price_inr']:,.0f}</div>
                <div style="font-size:0.85rem; color:#cbd5e1;"><b>Common Cut:</b> {p['most_common_cut']}</div>
            </div>
            """, unsafe_allow_html=True)
            
    st.write("")
    
    # Elbow & Silhouette Curves + 3D PCA
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("📉 Elbow Method & Silhouette Scores")
        elbow_data = cluster_metrics['elbow_inertias']
        sil_data = cluster_metrics['silhouette_scores']
        
        k_vals = [int(k) for k in elbow_data.keys()]
        inertias = [elbow_data[str(k)] for k in k_vals]
        sils = [sil_data[str(k)] for k in k_vals]
        
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(
            x=k_vals, y=inertias, name="Inertia (Elbow Curve)",
            mode='lines+markers', line=dict(color='#38bdf8', width=3)
        ))
        fig_eval.update_layout(
            title="K-Means Inertia vs Number of Clusters (k)",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Inertia",
            template="plotly_dark",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_eval, use_container_width=True)
        
    with col_b:
        st.subheader("🎯 Silhouette Score Evaluation")
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Bar(
            x=k_vals, y=sils, name="Silhouette Score",
            marker_color=['#94a3b8' if k != 4 else '#10b981' for k in k_vals]
        ))
        fig_sil.update_layout(
            title="Silhouette Score Across Cluster Numbers (k=4 selected)",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Silhouette Score",
            template="plotly_dark",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_sil, use_container_width=True)
        
    # Interactive 3D PCA Cluster Space
    st.subheader("🌌 3D Interactive PCA Cluster Projection")
    st.caption("Principal Component Analysis reduces 11 dimensions to 3 principal components explaining 80.3% of the total dataset variance.")
    
    pca_sample = sample_df.sample(n=min(2500, len(sample_df)), random_state=42)
    fig_3d = px.scatter_3d(
        pca_sample,
        x='pca_3d_1',
        y='pca_3d_2',
        z='pca_3d_3',
        color='segment_name',
        opacity=0.6,
        size_max=5,
        title="3D Diamond Market Clusters",
        labels={'pca_3d_1': 'PC 1 (Dimension)', 'pca_3d_2': 'PC 2 (Cut/Depth)', 'pca_3d_3': 'PC 3 (Clarity/Color)'},
        color_discrete_sequence=['#38bdf8', '#10b981', '#fbbf24', '#f43f5e']
    )
    fig_3d.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    st.plotly_chart(fig_3d, use_container_width=True)


# ==============================================================================
# TAB 3: EXPLORATORY DATA ANALYTICS (EDA)
# ==============================================================================
elif nav_option == "📊 Exploratory Data Analytics (EDA)":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Exploratory Data Analysis & Visualizations</div>
        <div class="hero-subtitle">
            Comprehensive statistical exploration of <b>53,940 diamonds</b>, analyzing distribution skews, 
            price correlations, and the pricing dynamics across cut, color, and clarity.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Overview metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Diamonds Analyzed", f"{len(sample_df)*10.75:,.0f} (Full: 53,940)")
    m2.metric("Median Price in INR", "₹ 1,99,283", "+83x USD")
    m3.metric("Average Carat Weight", "0.798 ct", "Range: 0.2 - 5.0")
    m4.metric("Clean Data Retention", "99.7%", "20 invalid zero-dim filtered")
    
    st.write("")
    
    eda_tab1, eda_tab2, eda_tab3 = st.tabs(["📊 Price & Carat Distributions", "💎 4Cs Price Impact", "🔥 Correlation Matrix"])
    
    with eda_tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig_hist = px.histogram(
                sample_df,
                x='price_inr',
                nbins=50,
                marginal="box",
                title="Diamond Price in INR (₹) Distribution (Heavily Right-Skewed)",
                color_discrete_sequence=['#38bdf8']
            )
            fig_hist.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with c2:
            fig_carat = px.scatter(
                sample_df,
                x='carat',
                y='price_inr',
                color='cut',
                opacity=0.5,
                trendline="ols",
                title="Carat vs Price (INR) with Regression Trend",
                color_discrete_sequence=px.colors.sequential.Tealgrn
            )
            fig_carat.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_carat, use_container_width=True)
            
    with eda_tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig_box_cut = px.box(
                sample_df,
                x='cut',
                y='price_inr',
                category_orders={'cut': mappings['cut_order']},
                color='cut',
                title="Price Distribution Across Cut Qualities",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_box_cut.update_layout(template="plotly_dark", height=380, showlegend=False)
            st.plotly_chart(fig_box_cut, use_container_width=True)
            
        with c2:
            fig_box_clarity = px.box(
                sample_df,
                x='clarity',
                y='price_inr',
                category_orders={'clarity': mappings['clarity_order']},
                color='clarity',
                title="Price Distribution Across Clarity Grades",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_box_clarity.update_layout(template="plotly_dark", height=380, showlegend=False)
            st.plotly_chart(fig_box_clarity, use_container_width=True)
            
    with eda_tab3:
        corr_cols = ['carat', 'depth', 'table', 'x', 'y', 'z', 'volume', 'dimension_ratio', 'price_inr']
        corr_matrix = sample_df[corr_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.3f',
            color_continuous_scale='RdBu_r',
            title="Correlation Heatmap of Physical & Valuation Features",
            aspect="auto"
        )
        fig_corr.update_layout(template="plotly_dark", height=480)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.info("💡 **Insight:** Carat, Volume, and dimensions x, y, z exhibit near-perfect correlation with Price in INR (~0.90 to 0.92), confirming that physical mass and space displacement are the primary price determinants.")


# ==============================================================================
# TAB 4: MODEL BENCHMARK & ARCHITECTURE
# ==============================================================================
elif nav_option == "🤖 AI Model Benchmark & Architecture":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Model Performance & Machine Learning Benchmark</div>
        <div class="hero-subtitle">
            Evaluating 5 Regression ML algorithms and an <b>Artificial Neural Network (ANN)</b> trained on log-transformed prices.
            All models were evaluated using <b>MAE (₹)</b>, <b>MSE</b>, <b>RMSE (₹)</b>, and <b>R² Score</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("🏆 Model Evaluation Leaderboard")
    
    results = reg_metrics['results']
    results_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model Name'})
    results_df = results_df.sort_values(by='R2', ascending=False).reset_index(drop=True)
    
    # Styled Table
    styled_df = results_df.copy()
    styled_df['MAE (INR)'] = styled_df['MAE'].apply(lambda v: f"₹ {v:,.2f}")
    styled_df['RMSE (INR)'] = styled_df['RMSE'].apply(lambda v: f"₹ {v:,.2f}")
    styled_df['MSE'] = styled_df['MSE'].apply(lambda v: f"{v:,.0f}")
    styled_df['R² Score'] = styled_df['R2'].apply(lambda v: f"{v:.4f}")
    
    st.dataframe(
        styled_df[['Model Name', 'R² Score', 'MAE (INR)', 'RMSE (INR)', 'MSE']],
        use_container_width=True,
        hide_index=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 R² Score Comparison (Higher is Better)")
        fig_r2 = px.bar(
            results_df,
            x='R2',
            y='Model Name',
            orientation='h',
            color='R2',
            color_continuous_scale='Teal',
            range_x=[0.94, 1.0],
            text='R2'
        )
        fig_r2.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig_r2.update_layout(template="plotly_dark", height=350, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_r2, use_container_width=True)
        
    with col2:
        st.subheader("📉 Root Mean Squared Error (RMSE in ₹)")
        fig_rmse = px.bar(
            results_df,
            x='RMSE',
            y='Model Name',
            orientation='h',
            color='RMSE',
            color_continuous_scale='Reds_r',
            text='RMSE'
        )
        fig_rmse.update_traces(texttemplate='₹ %{text:,.0f}', textposition='outside')
        fig_rmse.update_layout(template="plotly_dark", height=350, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_rmse, use_container_width=True)
        
    # Feature Importance Plot
    st.subheader("🌲 Feature Importance (from XGBoost Ensemble)")
    feat_imp = pd.DataFrame(list(reg_metrics['feature_importance'].items()), columns=['Feature', 'Importance'])
    feat_imp = feat_imp.sort_values(by='Importance', ascending=True)
    
    fig_imp = px.bar(
        feat_imp,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Viridis',
        title="Predictive Feature Weights"
    )
    fig_imp.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_imp, use_container_width=True)
    
    # Deep Learning Architecture
    st.subheader("🧠 Deep Learning ANN Architecture")
    st.markdown("""
    ```
    Input Layer: 13 Standardized Engineered Features
         │
         ▼
    Dense(128, activation='relu') + BatchNormalization() + Dropout(0.15)
         │
         ▼
    Dense(64, activation='relu') + BatchNormalization() + Dropout(0.10)
         │
         ▼
    Dense(32, activation='relu')
         │
         ▼
    Output Layer: Dense(1, linear) -> Predicts log1p(Price in INR) -> np.expm1()
    
    Optimizer: Adam (lr=0.003) | Loss: Huber Loss | Callbacks: EarlyStopping, ReduceLROnPlateau
    ```
    """)


# ==============================================================================
# TAB 5: GEMOLOGY & PROJECT GUIDE
# ==============================================================================
elif nav_option == "📖 Gemology & Project Guide":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Diamond Gemology & Project Architecture Guide</div>
        <div class="hero-subtitle">
            Overview of the 4Cs diamond grading standards, mathematical formulas used in feature engineering, 
            and project background.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        ### 💎 The 4Cs of Diamond Quality
        - **Carat (Weight):** Measures diamond mass. $1 \\text{ carat} = 200 \\text{ mg}$. The single strongest driver of raw value.
        - **Cut (Proportion & Finish):** Rated *Fair, Good, Very Good, Premium, Ideal*. Dictates how light interacts with facets to create fire and brilliance.
        - **Color (Grading):** Evaluated from **D** (completely colorless, rarest) to **J** (noticeable yellow/brown tint).
        - **Clarity (Purity):** Evaluates internal inclusions and surface blemishes under 10x magnification, from **IF** (Internally Flawless) down to **I1** (Included).
        """)
        
    with c2:
        st.markdown("""
        ### 📐 Feature Engineering Formulas
        - **Price in INR:** $\\text{Price}_{\\text{INR}} = \\text{Price}_{\\text{USD}} \\times 83.0$
        - **Volume:** $\\text{Volume} = x \\times y \\times z \\text{ mm}^3$
        - **Dimension Ratio:** $\\frac{x + y}{2 \\times z}$ (symmetry factor)
        - **Carat Class:**
          - *Light:* $< 0.5$ ct
          - *Medium:* $0.5 - 1.5$ ct
          - *Heavy:* $> 1.5$ ct
        - **Price per Carat:** $\\frac{\\text{Price}_{\\text{INR}}}{\\text{Carat}}$
        """)
        
    st.markdown("---")
    st.markdown("""
    ### 🏢 About This Project
    - **Course & Organization:** GUVI Geek Network | HCL Tech Capstone Project
    - **Title:** Diamond Dynamics: Price Prediction & Market Segmentation
    - **Domain:** E-Commerce, Luxury Retail Analytics, Dynamic Pricing Optimization
    - **Models:** Linear Regression, Decision Tree, Random Forest, KNN, XGBoost, TensorFlow ANN, K-Means Clustering, PCA.
    """)

# Footer
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.82rem; margin-top: 50px; padding: 20px 0; border-top: 1px solid rgba(148, 163, 184, 0.1);">
    💎 <b>Diamond Dynamics</b> • GUVI | HCL Capstone Project • End-to-End ML, ANN & Market Segmentation
</div>
""", unsafe_allow_html=True)
