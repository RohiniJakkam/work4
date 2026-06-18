
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
 
st.set_page_config(
    page_title="Patient Segmentation System",
    page_icon="🏥",
    layout="wide"
)
 
@st.cache_data
def load_data():
    df = pd.read_csv("patient_segments2.csv")
    return df
 
@st.cache_resource
def train_models(df):
    features = [c for c in df.columns if c not in ["diagnosis", "Cluster"]]
    X = df[features]
    y = df["diagnosis"]
 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
 
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
 
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
 
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
 
    return scaler, kmeans, rf, pca, X_scaled, X_pca, X_test, y_test, features
 
df = load_data()
scaler, kmeans, rf, pca, X_scaled, X_pca, X_test, y_test, features = train_models(df)
 
st.title(" Patient Segmentation System")
st.subheader("Breast Cancer Patient Segmentation — K-Means Clustering + Random Forest")
st.markdown("---")
 
tab1, tab2, tab3 = st.tabs([" Overview & Visuals", "🔮 Real-Time Prediction", "📋 Model Evaluation"])
 
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", len(df))
    with col2:
        st.metric("Features Used", len(features))
    with col3:
        malignant = int((df["diagnosis"] == 1).sum())
        st.metric("Malignant Cases", malignant)
    with col4:
        benign = int((df["diagnosis"] == 0).sum())
        st.metric("Benign Cases", benign)
 
    st.markdown("---")
    st.header(" Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.write("Dataset Shape:", df.shape)
 
    st.markdown("---")
    st.header(" Elbow Method")
    st.image("elbow.png", use_container_width=True)
    st.markdown("Used to determine the optimal number of clusters for K-Means. The bend at k=2 confirms the choice.")
 
    st.markdown("---")
    st.header(" PCA Cluster Visualization")
 
    fig_pca, ax_pca = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3", "#FF5722"]
    cluster_labels = kmeans.predict(X_scaled)
    for i, color in enumerate(colors):
        mask = cluster_labels == i
        ax_pca.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=f"Cluster {i}", alpha=0.6, s=30)
    ax_pca.set_xlabel("PCA Component 1")
    ax_pca.set_ylabel("PCA Component 2")
    ax_pca.set_title("Patient Clusters in PCA Space")
    ax_pca.legend()
    fig_pca.tight_layout()
    st.pyplot(fig_pca)
    st.markdown("PCA reduces the 30 features into 2 components to visualize how the two clusters separate in space.")
 
    st.markdown("---")
    st.header(" Feature Importance")
 
    importances = rf.feature_importances_
    feat_df = pd.DataFrame({"Feature": features, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=False).head(10)
 
    fig_fi, ax_fi = plt.subplots(figsize=(8, 5))
    colors_bar = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(feat_df)))
    bars = ax_fi.barh(feat_df["Feature"], feat_df["Importance"], color=colors_bar[::-1])
    ax_fi.set_xlabel("Importance Score")
    ax_fi.set_title("Top 10 Feature Importances (Random Forest)")
    ax_fi.invert_yaxis()
    fig_fi.tight_layout()
    st.pyplot(fig_fi)
 
with tab2:
    st.header("🔮 Real-Time Patient Prediction")
    st.markdown("Drag the sliders to adjust measurements. Results update instantly.")
 
    st.markdown("---")
 
    top5 = ["concave points_mean", "perimeter_mean", "area_mean", "concavity_mean", "radius_mean"]
    feature_stats = {f: {"min": df[f].min(), "max": df[f].max(), "mean": df[f].mean()} for f in features}
 
    st.subheader("🎚️ Quick Mode — Top 5 Features")
    st.markdown("Slide to adjust the most important features. Remaining features use dataset mean.")
 
    quick_vals = {}
    for feat in top5:
        fmin = float(feature_stats[feat]["min"])
        fmax = float(feature_stats[feat]["max"])
        fmean = float(feature_stats[feat]["mean"])
        step = float((fmax - fmin) / 200)
        col_label, col_slider, col_val = st.columns([2, 6, 1])
        with col_label:
            st.markdown(f"<div style='padding-top:28px; font-weight:600; font-size:0.85rem'>{feat}</div>", unsafe_allow_html=True)
        with col_slider:
            quick_vals[feat] = st.slider(
                feat, min_value=fmin, max_value=fmax,
                value=fmean, step=step,
                key=f"quick_{feat}", label_visibility="collapsed"
            )
        with col_val:
            st.markdown(f"<div style='padding-top:28px; font-size:0.85rem; color:#888'>{quick_vals[feat]:.2f}</div>", unsafe_allow_html=True)
 
    with st.expander("⚙️ Advanced — Set All 30 Features"):
        st.markdown("Slide to fine-tune every feature. Values default to dataset means.")
        advanced_vals = {}
        for feat in features:
            fmin = float(feature_stats[feat]["min"])
            fmax = float(feature_stats[feat]["max"])
            fmean = float(feature_stats[feat]["mean"])
            step = float((fmax - fmin) / 200)
            col_label, col_slider, col_val = st.columns([2, 6, 1])
            with col_label:
                st.markdown(f"<div style='padding-top:28px; font-weight:600; font-size:0.8rem'>{feat}</div>", unsafe_allow_html=True)
            with col_slider:
                advanced_vals[feat] = st.slider(
                    feat, min_value=fmin, max_value=fmax,
                    value=fmean, step=step,
                    key=f"adv_{feat}", label_visibility="collapsed"
                )
            with col_val:
                st.markdown(f"<div style='padding-top:28px; font-size:0.8rem; color:#888'>{advanced_vals[feat]:.2f}</div>", unsafe_allow_html=True)
 
    input_data = {}
    for feat in features:
        if feat in quick_vals:
            input_data[feat] = quick_vals[feat]
        elif feat in advanced_vals:
            input_data[feat] = advanced_vals[feat]
        else:
            input_data[feat] = float(feature_stats[feat]["mean"])
 
    input_array = np.array([[input_data[f] for f in features]])
    input_scaled = scaler.transform(input_array)
 
    diagnosis_pred = rf.predict(input_scaled)[0]
    diagnosis_proba = rf.predict_proba(input_scaled)[0]
    cluster_pred = kmeans.predict(input_scaled)[0]
    input_pca = pca.transform(input_scaled)
 
    st.markdown("---")
    st.subheader(" Prediction Results")
 
    r1, r2, r3 = st.columns(3)
 
    with r1:
        label = "🔴 Malignant" if diagnosis_pred == 1 else "🟢 Benign"
        color = "#ff4b4b" if diagnosis_pred == 1 else "#00c853"
        st.markdown(
            f"""
            <div style='background:{color}22; border:2px solid {color}; border-radius:10px; padding:20px; text-align:center'>
                <h3 style='color:{color}; margin:0'>Diagnosis</h3>
                <h1 style='color:{color}; margin:5px 0'>{label}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
 
    with r2:
        cluster_color = "#9c27b0" if cluster_pred == 1 else "#1976d2"
        cluster_icon = "🔵" if cluster_pred == 0 else "🟣"
        cluster_desc = "High-risk group" if cluster_pred == 1 else "Low-risk group"
        st.markdown(
            f"""
            <div style='background:{cluster_color}22; border:2px solid {cluster_color}; border-radius:10px; padding:20px; text-align:center'>
                <h3 style='color:{cluster_color}; margin:0'>K-Means Cluster</h3>
                <div style='font-size:3rem; margin:8px 0'>{cluster_icon}</div>
                <h1 style='color:{cluster_color}; margin:0; font-size:2.5rem'>Cluster {cluster_pred}</h1>
                <p style='color:{cluster_color}; margin:6px 0 0 0; font-size:0.9rem; font-weight:600'>{cluster_desc}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
 
    with r3:
        conf_malignant = diagnosis_proba[1] * 100
        conf_benign = diagnosis_proba[0] * 100
        st.markdown(
            f"""
            <div style='background:#ff980022; border:2px solid #ff9800; border-radius:10px; padding:20px; text-align:center'>
                <h3 style='color:#ff9800; margin:0'>Confidence</h3>
                <h2 style='color:#ff9800; margin:5px 0'>Malignant: {conf_malignant:.1f}%</h2>
                <h2 style='color:#ff9800; margin:0'>Benign: {conf_benign:.1f}%</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    fig_res, axes = plt.subplots(1, 2, figsize=(12, 4))
 
    axes[0].barh(["Benign", "Malignant"], [diagnosis_proba[0], diagnosis_proba[1]],
                 color=["#00c853", "#ff4b4b"])
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Probability")
    axes[0].set_title("Prediction Probability")
    for i, v in enumerate([diagnosis_proba[0], diagnosis_proba[1]]):
        axes[0].text(v + 0.01, i, f"{v*100:.1f}%", va="center")
 
    colors_pca = ["#2196F3", "#FF5722"]
    cluster_labels = kmeans.predict(X_scaled)
    for i, color in enumerate(colors_pca):
        mask = cluster_labels == i
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=f"Cluster {i}", alpha=0.4, s=20)
    axes[1].scatter(input_pca[0, 0], input_pca[0, 1], c="yellow", marker="*", s=300,
                    zorder=5, edgecolors="black", linewidths=1, label="This Patient")
    axes[1].set_xlabel("PCA 1")
    axes[1].set_ylabel("PCA 2")
    axes[1].set_title("Patient Position in Cluster Space")
    axes[1].legend()
 
    fig_res.tight_layout()
    st.pyplot(fig_res)
 
    st.markdown("---")
    st.subheader(" Input Feature Summary")
    input_df = pd.DataFrame(input_data, index=["Your Input"]).T
    input_df.columns = ["Value"]
    input_df["Dataset Mean"] = [feature_stats[f]["mean"] for f in input_df.index]
    input_df["vs Mean"] = input_df["Value"] - input_df["Dataset Mean"]
    st.dataframe(input_df.style.format("{:.4f}").background_gradient(subset=["vs Mean"], cmap="RdYlGn_r"), use_container_width=True)
 
with tab3:
    st.header("📋 Model Evaluation")
 
    y_pred = rf.predict(X_test)
 
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"], ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title("Random Forest Confusion Matrix")
    fig_cm.tight_layout()
    st.pyplot(fig_cm)
 
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=["Benign", "Malignant"], output_dict=True)
    report_df = pd.DataFrame(report).T
    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
 
    st.subheader("K-Means Silhouette Score")
    sil = silhouette_score(X_scaled, kmeans.labels_)
    st.metric("Silhouette Score", f"{sil:.4f}")
    st.markdown("A silhouette score closer to 1.0 means the clusters are well separated and compact.")
 
    st.markdown("---")
    st.subheader(" Top 5 Features")
    st.table(pd.DataFrame({"Rank": range(1, 6), "Feature": ["concave_points_mean", "perimeter_mean", "area_mean", "concavity_mean", "radius_mean"]}))
 
st.markdown("---")
st.caption("Patient Segmentation System | Machine Learning in Healthcare | Breast Cancer Diagnosis")
