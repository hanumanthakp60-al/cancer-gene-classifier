import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from umap import UMAP
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load the matrix ────────────────────────────────────────────────────────
print("Loading matrix...")
df = pd.read_parquet("tcga_5cancer_matrix.parquet")
print(f"Shape: {df.shape}")
print(f"\nCancer type distribution:")
print(df["cancer_type"].value_counts())

# Separate features and labels
X = df.drop(columns=["cancer_type"]).values.astype(np.float32)
y = df["cancer_type"].values

# Color map for 5 cancer types
colors = {"BRCA": "#E8593C", "LUAD": "#3B8BD4", 
          "PRAD": "#1D9E75", "GBM": "#EF9F27", "COAD": "#7F77DD"}
color_list = [colors[c] for c in y]

print(f"\nFeature matrix shape: {X.shape}")

# ── 2. Plot 1 — Expression distribution ──────────────────────────────────────
print("\nPlotting expression distribution...")
plt.figure(figsize=(10, 4))
sample_genes = X[:, :100].flatten()  # sample of values
plt.hist(sample_genes, bins=80, color="#3B8BD4", alpha=0.7, edgecolor="none")
plt.xlabel("Log1p Expression Value")
plt.ylabel("Frequency")
plt.title("Gene Expression Distribution After Log Normalization")
plt.tight_layout()
plt.savefig("plot1_expression_distribution.png", dpi=150)
plt.close()
print("Saved: plot1_expression_distribution.png")

# ── 3. Plot 2 — Cancer type sample counts ────────────────────────────────────
print("Plotting class distribution...")
plt.figure(figsize=(8, 4))
counts = df["cancer_type"].value_counts()
bars = plt.bar(counts.index, counts.values, 
               color=[colors[c] for c in counts.index], 
               edgecolor="none", width=0.6)
for bar, val in zip(bars, counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
             str(val), ha="center", va="bottom", fontsize=11)
plt.xlabel("Cancer Type")
plt.ylabel("Number of Samples")
plt.title("Sample Count per Cancer Type")
plt.tight_layout()
plt.savefig("plot2_class_distribution.png", dpi=150)
plt.close()
print("Saved: plot2_class_distribution.png")

# ── 4. Plot 3 — PCA ───────────────────────────────────────────────────────────
print("\nRunning PCA (this is fast)...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
variance_explained = pca.explained_variance_ratio_ * 100

plt.figure(figsize=(9, 7))
for cancer, color in colors.items():
    mask = y == cancer
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                c=color, label=cancer, alpha=0.6, s=15, edgecolors="none")
plt.xlabel(f"PC1 ({variance_explained[0]:.1f}% variance)")
plt.ylabel(f"PC2 ({variance_explained[1]:.1f}% variance)")
plt.title("PCA — 5 Cancer Types")
plt.legend(markerscale=2, framealpha=0.9)
plt.tight_layout()
plt.savefig("plot3_pca.png", dpi=150)
plt.close()
print("Saved: plot3_pca.png")

# ── 5. Plot 4 — UMAP ──────────────────────────────────────────────────────────
print("\nRunning UMAP (takes 3-5 minutes)...")
umap = UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
X_umap = umap.fit_transform(X)

plt.figure(figsize=(9, 7))
for cancer, color in colors.items():
    mask = y == cancer
    plt.scatter(X_umap[mask, 0], X_umap[mask, 1], 
                c=color, label=cancer, alpha=0.6, s=15, edgecolors="none")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP — 5 Cancer Types")
plt.legend(markerscale=2, framealpha=0.9)
plt.tight_layout()
plt.savefig("plot4_umap.png", dpi=150)
plt.close()
print("Saved: plot4_umap.png")

# ── 6. Basic stats ────────────────────────────────────────────────────────────
print("\n── Data Summary ──────────────────────────────────────────")
print(f"Total samples      : {X.shape[0]}")
print(f"Genes (features)   : {X.shape[1]}")
print(f"Cancer types       : {len(np.unique(y))}")
print(f"Null values        : {df.isnull().sum().sum()}")
print(f"Min expression     : {X.min():.3f}")
print(f"Max expression     : {X.max():.3f}")
print(f"Mean expression    : {X.mean():.3f}")
print("──────────────────────────────────────────────────────────")
print("\nEDA complete! Check the 4 PNG plots in your folder.")
