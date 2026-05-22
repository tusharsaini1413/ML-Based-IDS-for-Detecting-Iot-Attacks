"""
02_visualize.py
Loads saved artifacts from results/ and generates 6 figures in figures/.
No model training — purely visualization. Fast (~1 min).
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                              precision_score, recall_score,
                              f1_score, accuracy_score)

sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'figure.dpi': 120,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight'
})

# Models to include (skip DNN if not trained)
ALL_MODELS = ["XGB_SMOTE", "RF_SMOTE", "RF_RUS", "SVM", "XGB_PCA", "XGB_RFFS", "DNN"]
LABEL_MAP = {
    "XGB_SMOTE": "XGBoost+SMOTE",
    "RF_SMOTE":  "RF+SMOTE",
    "RF_RUS":    "RF+RUS",
    "SVM":       "SVM",
    "XGB_PCA":   "XGBoost+PCA",
    "XGB_RFFS":  "XGBoost+RF-FS",
    "DNN":       "DNN"
}

# Filter to only models that have predictions saved
MODELS = [m for m in ALL_MODELS
          if os.path.exists(f"results/{m}_ypred.npy")]
print(f"Models found: {MODELS}\n")

# =============================================================
# FIG 1 — Class distribution before/after balancing
# =============================================================
print("→ Fig 1: Class distribution")
with open("results/class_distribution.json") as f:
    cd = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
panels = [("before", "Before Balancing"),
          ("after_smote", "After SMOTE"),
          ("after_rus", "After RUS")]
for ax, (key, title) in zip(axes, panels):
    d = cd[key]
    labels = ["Normal (0)", "Attack (1)"]
    vals = [d.get("0", 0), d.get("1", 0)]
    bars = ax.bar(labels, vals, color=['#2ecc71', '#e74c3c'])
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel("Sample count")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}",
                ha='center', va='bottom', fontsize=10)
plt.suptitle("Class Distribution: Imbalance Handling Comparison",
             fontweight='bold', y=1.04)
plt.savefig("figures/01_class_distribution.png")
plt.close()

# =============================================================
# FIG 2 — Multi-metric grouped bar chart
# =============================================================
print("→ Fig 2: Multi-metric comparison")
rows = []
for m in MODELS:
    yt = np.load(f"results/{m}_ytest.npy")
    yp = np.load(f"results/{m}_ypred.npy")
    rows.append({
        "Model":     LABEL_MAP[m],
        "Accuracy":  accuracy_score(yt, yp),
        "Precision": precision_score(yt, yp, zero_division=0),
        "Recall":    recall_score(yt, yp, zero_division=0),
        "F1":        f1_score(yt, yp, zero_division=0)
    })
metrics_df = pd.DataFrame(rows)
metrics_df.to_csv("results/comparison_table.csv", index=False)
print(metrics_df.to_string(index=False))

melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
plt.figure(figsize=(12, 6))
sns.barplot(data=melted, x="Model", y="Score", hue="Metric", palette="viridis")
plt.title("Model Performance Comparison Across All Metrics", fontweight='bold')
ymin = max(0.5, melted['Score'].min() - 0.05)
plt.ylim(ymin, 1.0)
plt.xticks(rotation=20, ha='right')
plt.legend(loc='lower right')
plt.savefig("figures/02_model_comparison_bars.png")
plt.close()

# =============================================================
# FIG 3 — Confusion matrices (top 3 by F1)
# =============================================================
print("\n→ Fig 3: Confusion matrices (top 3)")
top3_labels = metrics_df.nlargest(3, "F1")["Model"].tolist()
inv_label = {v: k for k, v in LABEL_MAP.items()}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, model_label in zip(axes, top3_labels):
    m = inv_label[model_label]
    yt = np.load(f"results/{m}_ytest.npy")
    yp = np.load(f"results/{m}_ypred.npy")
    cm = confusion_matrix(yt, yp)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'], cbar=False)
    ax.set_title(model_label, fontweight='bold')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.suptitle("Confusion Matrices — Top 3 Models by F1",
             fontweight='bold', y=1.05)
plt.savefig("figures/03_confusion_matrices.png")
plt.close()

# =============================================================
# FIG 4 — ROC curves overlay
# =============================================================
print("→ Fig 4: ROC curves")
plt.figure(figsize=(8, 7))
colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))
for m, c in zip(MODELS, colors):
    yt = np.load(f"results/{m}_ytest.npy")
    yprob = np.load(f"results/{m}_yprob.npy")
    fpr, tpr, _ = roc_curve(yt, yprob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, color=c,
             label=f"{LABEL_MAP[m]} (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — All Models", fontweight='bold')
plt.legend(loc='lower right', fontsize=9)
plt.savefig("figures/04_roc_curves.png")
plt.close()

# =============================================================
# FIG 5 — 5-fold CV box plot
# =============================================================
print("→ Fig 5: 5-Fold CV box plot")
cv = np.load("results/cv_scores.npy")
plt.figure(figsize=(6, 6))
plt.boxplot([cv], labels=["XGBoost + RF-FS"], widths=0.4, patch_artist=True,
            boxprops=dict(facecolor='#3498db', alpha=0.6),
            medianprops=dict(color='red', linewidth=2))
plt.scatter([1] * len(cv), cv, color='red', zorder=3, s=60,
            label='Per-fold accuracy')
plt.title(f"5-Fold Cross-Validation\nMean = {cv.mean():.4f},  Std = {cv.std():.4f}",
          fontweight='bold')
plt.ylabel("Accuracy")
plt.ylim(cv.min() - 0.005, cv.max() + 0.005)
plt.legend()
plt.savefig("figures/05_cv_boxplot.png")
plt.close()

# =============================================================
# FIG 6 — Feature importance (top 20)
# =============================================================
print("→ Fig 6: Feature importance")
fi = pd.read_csv("results/feature_importance.csv").head(20)
plt.figure(figsize=(8, 8))
sns.barplot(data=fi, y="feature", x="importance",
            palette="rocket_r", legend=False, hue="feature")
plt.title("Top 20 Features by Random Forest Importance", fontweight='bold')
plt.xlabel("Importance Score")
plt.ylabel("")
plt.savefig("figures/06_feature_importance.png")
plt.close()

# =============================================================
# SUMMARY
# =============================================================
print("\n" + "=" * 60)
print("All 6 figures saved to figures/")
print("=" * 60)
print(f"  01_class_distribution.png")
print(f"  02_model_comparison_bars.png")
print(f"  03_confusion_matrices.png")
print(f"  04_roc_curves.png")
print(f"  05_cv_boxplot.png")
print(f"  06_feature_importance.png")
print(f"\nResults table: results/comparison_table.csv")
print("\nNext: python 03_shap.py")
