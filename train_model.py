import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                              accuracy_score)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet("tcga_5cancer_matrix.parquet")

X = df.drop(columns=["cancer_type"]).values.astype(np.float32)
y = df["cancer_type"].values

# Encode labels to numbers
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Classes: {le.classes_}")
print(f"Shape: {X.shape}")

# ── 2. Train/Test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\nTrain samples: {len(X_train)}")
print(f"Test samples : {len(X_test)}")

# ── 3. Helper — plot confusion matrix ────────────────────────────────────────
def plot_confusion(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"{title} — Confusion Matrix (%)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")

# ── 4. Model 1 — Random Forest ────────────────────────────────────────────────
print("\n── Random Forest ─────────────────────────────────────────")
print("Training... (3-5 minutes)")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
print(f"Accuracy: {rf_acc*100:.2f}%")
print(classification_report(y_test, rf_preds, target_names=le.classes_))
plot_confusion(y_test, rf_preds, "Random Forest", "plot5_rf_confusion.png")

# ── 5. Model 2 — XGBoost ─────────────────────────────────────────────────────
print("\n── XGBoost ───────────────────────────────────────────────")
print("Training... (5-8 minutes)")

# Compute class weights for XGBoost
from collections import Counter
counter = Counter(y_train)
total = len(y_train)
sample_weights = np.array([total / (len(counter) * counter[c]) for c in y_train])

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.3,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)
xgb.fit(X_train, y_train, sample_weight=sample_weights,
        eval_set=[(X_test, y_test)], verbose=50)
xgb_preds = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)
print(f"Accuracy: {xgb_acc*100:.2f}%")
print(classification_report(y_test, xgb_preds, target_names=le.classes_))
plot_confusion(y_test, xgb_preds, "XGBoost", "plot6_xgb_confusion.png")

# ── 6. Model comparison bar chart ─────────────────────────────────────────────
print("\n── Model Comparison ──────────────────────────────────────")
models = ["Random Forest", "XGBoost"]
accuracies = [rf_acc * 100, xgb_acc * 100]

plt.figure(figsize=(7, 4))
bars = plt.bar(models, accuracies,
               color=["#3B8BD4", "#E8593C"],
               width=0.4, edgecolor="none")
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() - 3,
             f"{acc:.1f}%", ha="center", va="top",
             color="white", fontsize=13, fontweight="bold")
plt.ylim(80, 102)
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.savefig("plot7_model_comparison.png", dpi=150)
plt.close()
print("Saved: plot7_model_comparison.png")

print(f"\n{'='*50}")
print(f"Random Forest : {rf_acc*100:.2f}%")
print(f"XGBoost       : {xgb_acc*100:.2f}%")
print(f"{'='*50}")
print("\nPhase 3 (classical ML) complete!")
print("Next: PyTorch CNN + SHAP explainability")
