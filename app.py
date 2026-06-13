import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cancer Gene Expression Classifier",
    page_icon="🧬",
    layout="wide"
)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf = joblib.load("random_forest_model.pkl")
    le = joblib.load("label_encoder.pkl")
    return rf, le

@st.cache_data
def load_reference():
    df = pd.read_parquet("tcga_5cancer_matrix.parquet")
    return df

rf, le = load_models()
df_ref = load_reference()
feature_names = [c for c in df_ref.columns if c != "cancer_type"]

# ── Cancer info ───────────────────────────────────────────────────────────────
cancer_info = {
    "BRCA": {
        "name": "Breast Invasive Carcinoma",
        "color": "#E8593C",
        "description": "Most common cancer in women worldwide. Key biomarkers: GATA3, HNF1B.",
        "emoji": "🎀"
    },
    "LUAD": {
        "name": "Lung Adenocarcinoma",
        "color": "#3B8BD4",
        "description": "Most common type of lung cancer. Key biomarkers: NKX2-1, NAPSA, SFTPA1.",
        "emoji": "🫁"
    },
    "PRAD": {
        "name": "Prostate Adenocarcinoma",
        "color": "#1D9E75",
        "description": "Most common cancer in men. Key biomarkers: KLK3 (PSA), KLK4, ACP3.",
        "emoji": "🔵"
    },
    "GBM": {
        "name": "Glioblastoma Multiforme",
        "color": "#EF9F27",
        "description": "Most aggressive brain tumor. Key biomarkers: FEZ1, brain-specific genes.",
        "emoji": "🧠"
    },
    "COAD": {
        "name": "Colon Adenocarcinoma",
        "color": "#7F77DD",
        "description": "Common gastrointestinal cancer. Key biomarkers: CDX1, gut-specific genes.",
        "emoji": "🟣"
    }
}

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧬 Cancer Gene Expression Classifier")
st.markdown("**AI-powered cancer type prediction from RNA-seq gene expression data**")
st.markdown("Trained on **2,811 real patient samples** from The Cancer Genome Atlas (TCGA) · Accuracy: **99.3%**")

st.divider()

# ── Sidebar info ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About This App")
    st.markdown("""
    This app uses a **Random Forest classifier** trained on TCGA RNA-seq data
    to predict cancer type from gene expression profiles.
    
    **Model Details:**
    - Algorithm: Random Forest (200 trees)
    - Features: 5,000 most variable genes
    - Training samples: 2,248 patients
    - Test accuracy: 99.3%
    
    **Cancer Types:**
    """)
    for code, info in cancer_info.items():
        st.markdown(f"{info['emoji']} **{code}** — {info['name']}")

    st.divider()
    st.markdown("**Built with:**")
    st.markdown("Python · scikit-learn · SHAP · Streamlit")
    st.markdown("Data: TCGA via GDC Portal")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔬 Predict from File",
    "🎲 Try Random Sample",
    "📊 Dataset Overview"
])

# ══ Tab 1 — Upload CSV ════════════════════════════════════════════════════════
with tab1:
    st.header("Upload Gene Expression CSV")
    st.markdown("""
    Upload a CSV file where:
    - Each **row** is one patient sample
    - Each **column** is a gene (Ensembl ID)
    - Values are **log1p normalized** RNA-seq counts
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        try:
            user_df = pd.read_csv(uploaded_file, index_col=0)
            st.success(f"Loaded {user_df.shape[0]} samples, {user_df.shape[1]} genes")

            # Align with training features
            common_genes = [g for g in feature_names if g in user_df.columns]
            st.info(f"Matched {len(common_genes)} out of {len(feature_names)} training genes")

            if len(common_genes) < 100:
                st.error("Too few matching genes. Please use TCGA STAR counts data.")
            else:
                X_user = np.zeros((user_df.shape[0], len(feature_names)), dtype=np.float32)
                for i, gene in enumerate(feature_names):
                    if gene in user_df.columns:
                        X_user[:, i] = user_df[gene].values

                probs = rf.predict_proba(X_user)
                preds = rf.predict(X_user)
                pred_labels = le.inverse_transform(preds)

                st.subheader("Predictions")
                results = pd.DataFrame({
                    "Sample": user_df.index,
                    "Predicted Cancer": pred_labels,
                    "Confidence": [f"{probs[i].max()*100:.1f}%" for i in range(len(preds))]
                })
                st.dataframe(results, use_container_width=True)

        except Exception as e:
            st.error(f"Error reading file: {e}")

# ══ Tab 2 — Random Sample ════════════════════════════════════════════════════
with tab2:
    st.header("Try a Random Patient Sample")
    st.markdown("Pick a cancer type and we'll grab a real patient from the TCGA dataset and predict it.")

    col1, col2 = st.columns([1, 2])

    with col1:
        selected_cancer = st.selectbox(
            "Select cancer type to sample from:",
            options=list(cancer_info.keys()),
            format_func=lambda x: f"{cancer_info[x]['emoji']} {x} — {cancer_info[x]['name']}"
        )

        if st.button("🎲 Run Prediction", type="primary"):
            # Sample a random patient of selected type
            cancer_df = df_ref[df_ref["cancer_type"] == selected_cancer]
            sample = cancer_df.sample(1, random_state=np.random.randint(0, 10000))
            X_sample = sample.drop(columns=["cancer_type"]).values.astype(np.float32)

            # Predict
            probs = rf.predict_proba(X_sample)[0]
            pred_idx = np.argmax(probs)
            pred_label = le.classes_[pred_idx]
            confidence = probs[pred_idx] * 100
            correct = pred_label == selected_cancer

            with col2:
                # Result card
                info = cancer_info[pred_label]
                if correct:
                    st.success(f"✅ Correct Prediction!")
                else:
                    st.error(f"❌ Incorrect Prediction")

                st.markdown(f"### {info['emoji']} Predicted: **{pred_label}**")
                st.markdown(f"**{info['name']}**")
                st.markdown(f"_{info['description']}_")
                st.markdown(f"**Confidence: {confidence:.1f}%**")
                st.markdown(f"**True label: {selected_cancer}**")

                # Probability bar chart
                fig, ax = plt.subplots(figsize=(6, 3))
                colors = [cancer_info[c]["color"] for c in le.classes_]
                bars = ax.barh(le.classes_, probs * 100,
                               color=colors, edgecolor="none", alpha=0.85)
                ax.set_xlabel("Probability (%)")
                ax.set_title("Prediction Probabilities")
                ax.set_xlim(0, 105)
                for bar, prob in zip(bars, probs):
                    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                            f"{prob*100:.1f}%", va="center", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

# ══ Tab 3 — Dataset Overview ══════════════════════════════════════════════════
with tab3:
    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples", "2,811")
    col2.metric("Genes (Features)", "5,000")
    col3.metric("Cancer Types", "5")
    col4.metric("Model Accuracy", "99.3%")

    st.subheader("Sample Distribution")
    counts = df_ref["cancer_type"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 3))
    colors = [cancer_info[c]["color"] for c in counts.index]
    bars = ax.bar(counts.index, counts.values,
                  color=colors, edgecolor="none", width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 10, str(val),
                ha="center", fontsize=11)
    ax.set_ylabel("Number of Samples")
    ax.set_title("Patients per Cancer Type")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Cancer Type Details")
    for code, info in cancer_info.items():
        count = counts.get(code, 0)
        with st.expander(f"{info['emoji']} {code} — {info['name']} ({count} samples)"):
            st.markdown(info["description"])
