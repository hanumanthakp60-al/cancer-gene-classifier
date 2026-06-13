import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet("tcga_5cancer_matrix.parquet")
X = df.drop(columns=["cancer_type"])
y = df["cancer_type"].values
le = LabelEncoder()
y_encoded = le.fit_transform(y)
feature_names = X.columns.tolist()
X = X.values.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ── 2. Retrain Random Forest ──────────────────────────────────────────────────
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=20,
    class_weight="balanced", random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
print("Done!")

# ── 3. SHAP values ────────────────────────────────────────────────────────────
print("\nCalculating SHAP values (5-10 minutes)...")
np.random.seed(42)
sample_idx = np.random.choice(len(X_test), 200, replace=False)
X_sample = X_test[sample_idx]

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_sample)
# shape: (samples, genes, classes) = (200, 5000, 5)
shap_array = np.array(shap_values)
print(f"SHAP array shape: {shap_array.shape}")

# ── 4. Plot 1 — Top 20 genes overall ─────────────────────────────────────────
print("\nPlotting top 20 genes overall...")
# mean over samples (axis=0) then mean over classes (axis=-1) → (5000,)
mean_shap_per_gene = np.abs(shap_array).mean(axis=0).mean(axis=-1)
print(f"mean_shap_per_gene shape: {mean_shap_per_gene.shape}")

top20_idx = np.argsort(mean_shap_per_gene)[-20:][::-1]
top20_genes = [feature_names[i] for i in top20_idx]
top20_values = mean_shap_per_gene[top20_idx]

plt.figure(figsize=(10, 7))
plt.barh(range(20), top20_values[::-1], color="#3B8BD4", edgecolor="none", alpha=0.85)
plt.yticks(range(20), top20_genes[::-1], fontsize=10)
plt.xlabel("Mean |SHAP Value|")
plt.title("Top 20 Most Important Genes — All Cancer Types")
plt.tight_layout()
plt.savefig("plot8_shap_top20_overall.png", dpi=150)
plt.close()
print("Saved: plot8_shap_top20_overall.png")

# ── 5. Plot 2 — Top 10 genes per cancer type ─────────────────────────────────
print("Plotting top 10 genes per cancer type...")
cancer_colors = {
    "BRCA": "#E8593C", "COAD": "#7F77DD",
    "GBM": "#EF9F27", "LUAD": "#3B8BD4", "PRAD": "#1D9E75"
}

fig, axes = plt.subplots(1, 5, figsize=(22, 6))
for i, cancer in enumerate(le.classes_):
    # shap_array[:, :, i] → (200, 5000) for class i
    sv = np.abs(shap_array[:, :, i]).mean(axis=0)  # (5000,)
    print(f"  {cancer} sv shape: {sv.shape}")

    top10_idx = np.argsort(sv)[-10:][::-1]
    top10_genes = [feature_names[j] for j in top10_idx]
    top10_vals = sv[top10_idx]  # (10,)

    ax = axes[i]
    ax.barh(range(10), top10_vals[::-1],
            color=cancer_colors[cancer], edgecolor="none", alpha=0.85)
    ax.set_yticks(range(10))
    ax.set_yticklabels(top10_genes[::-1], fontsize=8)
    ax.set_title(cancer, fontsize=13, fontweight="bold",
                 color=cancer_colors[cancer])
    ax.set_xlabel("Mean |SHAP|", fontsize=8)

plt.suptitle("Top 10 Biomarker Genes per Cancer Type", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("plot9_shap_per_cancer.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: plot9_shap_per_cancer.png")

# ── 6. Plot 3 — Beeswarm for BRCA ────────────────────────────────────────────
print("Plotting SHAP beeswarm for BRCA...")
plt.figure(figsize=(10, 8))
# BRCA is class index 0
shap.summary_plot(
    shap_array[:, :, 0],
    X_sample,
    feature_names=feature_names,
    max_display=20,
    show=False,
    plot_type="dot"
)
plt.title("BRCA — SHAP Beeswarm (Top 20 Genes)", fontsize=13)
plt.tight_layout()
plt.savefig("plot10_shap_beeswarm_brca.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: plot10_shap_beeswarm_brca.png")

# ── 7. Print top 5 genes per cancer type ─────────────────────────────────────
print("\n── Top 5 Biomarker Genes per Cancer Type ─────────────────")
for i, cancer in enumerate(le.classes_):
    sv = np.abs(shap_array[:, :, i]).mean(axis=0)  # (5000,)
    top5_idx = np.argsort(sv)[-5:][::-1]
    print(f"\n{cancer}:")
    for rank, idx in enumerate(top5_idx, 1):
        print(f"  {rank}. {feature_names[idx]}  (SHAP: {sv[idx]:.4f})")

print("\n── SHAP Analysis Complete! ───────────────────────────────")
print("Plots saved: plot8, plot9, plot10")
