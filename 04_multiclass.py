"""
04_multiclass.py
Multi-class (per-attack-type) classification + hierarchical pipeline evaluation.
This is your strongest new contribution beyond binary DoS detection.

Generates:
  - figures/10_multiclass_confusion.png
  - figures/11_per_class_f1.png
  - results/multiclass_report.csv
  - results/hierarchical_summary.txt
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)

plt.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight'
})
sns.set_style("whitegrid")

# =============================================================
# LOAD ARTIFACTS
# =============================================================
print("Loading data...")
X_train_smote = np.load("results/X_train_smote.npy")
y_multi_train = np.load("results/y_multi_train.npy")
X_test_s      = np.load("results/X_test_s.npy")
y_multi_test  = np.load("results/y_multi_test.npy")
y_test_binary = np.load("results/y_test.npy")
le = joblib.load("models/label_encoder_multi.pkl")

class_names = list(le.classes_)
n_classes = len(class_names)
print(f"  Classes ({n_classes}): {class_names}")

if n_classes < 3:
    print("\nWARNING: Only {} class(es) found in multi-class label.".format(n_classes))
    print("Multi-class analysis is most meaningful with 3+ classes.")
    print("Continuing anyway, but results may look identical to binary.")

# =============================================================
# TRAIN MULTI-CLASS MODEL
# =============================================================
print("\nTraining multi-class XGBoost (max 30K samples for speed)...")
N = min(30000, len(X_train_smote))

mc_model = XGBClassifier(
    n_estimators=300, max_depth=8, learning_rate=0.1,
    use_label_encoder=False, eval_metric='mlogloss',
    random_state=42, n_jobs=-1, objective='multi:softprob',
    num_class=n_classes
)

# Subsample preserving class balance approximately
idx = np.random.RandomState(42).choice(len(X_train_smote), N, replace=False)
mc_model.fit(X_train_smote[idx], y_multi_train[idx])

# Predict
y_mc_pred = mc_model.predict(X_test_s)

acc = accuracy_score(y_multi_test, y_mc_pred)
f1_macro = f1_score(y_multi_test, y_mc_pred, average='macro')
f1_weighted = f1_score(y_multi_test, y_mc_pred, average='weighted')
print(f"\n  Multi-class accuracy : {acc:.4f}")
print(f"  Macro F1             : {f1_macro:.4f}")
print(f"  Weighted F1          : {f1_weighted:.4f}")

joblib.dump(mc_model, "models/XGB_multiclass.pkl")

# =============================================================
# PER-CLASS REPORT
# =============================================================
print("\nGenerating per-class report...")
present_classes = sorted(np.unique(np.concatenate([y_multi_test, y_mc_pred])))
present_names = [class_names[i] for i in present_classes]

report = classification_report(
    y_multi_test, y_mc_pred,
    labels=present_classes,
    target_names=present_names,
    output_dict=True, zero_division=0
)
report_df = pd.DataFrame(report).T.round(4)
report_df.to_csv("results/multiclass_report.csv")
print(report_df)

# =============================================================
# FIG 10 — Multi-class confusion matrix (normalized)
# =============================================================
print("\n→ Fig 10: Multi-class confusion matrix")
cm = confusion_matrix(y_multi_test, y_mc_pred, labels=present_classes)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

fig_size = max(8, min(14, n_classes * 0.8))
plt.figure(figsize=(fig_size + 2, fig_size))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=present_names, yticklabels=present_names,
            cbar=True, annot_kws={'size': 9})
plt.title("Multi-Class Confusion Matrix (Normalized)\nPer-Attack-Type Detection",
          fontweight='bold')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.savefig("figures/10_multiclass_confusion.png")
plt.close()

# =============================================================
# FIG 11 — Per-class F1 bar chart
# =============================================================
print("→ Fig 11: Per-class F1 bar chart")
class_f1 = {}
for cls in present_names:
    if cls in report:
        class_f1[cls] = report[cls]['f1-score']

items = sorted(class_f1.items(), key=lambda x: x[1], reverse=True)
classes, scores = zip(*items)

plt.figure(figsize=(max(8, len(classes) * 0.8), 6))
colors = sns.color_palette("viridis", len(classes))
bars = plt.bar(range(len(classes)), scores, color=colors)
plt.title("F1-Score by Attack Type", fontweight='bold')
plt.ylabel("F1-Score")
plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
plt.ylim(0, 1.05)
for i, (b, s) in enumerate(zip(bars, scores)):
    plt.text(b.get_x() + b.get_width() / 2, s + 0.01,
             f"{s:.2f}", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig("figures/11_per_class_f1.png")
plt.close()

# =============================================================
# HIERARCHICAL PIPELINE EVALUATION
# =============================================================
print("\n→ Evaluating hierarchical (Binary → Multi-class) pipeline...")

xgb_binary = joblib.load("models/XGB_RFFS.pkl")
rf_selector = joblib.load("models/rf_selector.pkl")

# Stage 1: binary classification
stage1_pred = xgb_binary.predict(rf_selector.transform(X_test_s))

# Stage 2: only on samples Stage 1 flagged as attack
flagged_idx = np.where(stage1_pred == 1)[0]
print(f"  Stage 1 flagged {len(flagged_idx)} / {len(X_test_s)} as attacks")

if len(flagged_idx) > 0:
    stage2_pred = mc_model.predict(X_test_s[flagged_idx])
else:
    stage2_pred = np.array([])

# Build final prediction: "Normal" if Stage 1 said benign, else Stage 2 label
final_pred_codes = np.full(len(y_multi_test), -1, dtype=int)
# Identify the 'Normal' code (handles edge case where Normal isn't in classes)
normal_code = None
for cls in class_names:
    if cls.lower() == 'normal':
        normal_code = le.transform([cls])[0]
        break

if normal_code is None:
    normal_code = 0  # fallback

final_pred_codes[stage1_pred == 0] = normal_code
final_pred_codes[flagged_idx] = stage2_pred

hier_acc = (final_pred_codes == y_multi_test).mean()
flat_acc = acc

print(f"  Flat multi-class accuracy        : {flat_acc:.4f}")
print(f"  Hierarchical pipeline accuracy   : {hier_acc:.4f}")

with open("results/hierarchical_summary.txt", "w") as f:
    f.write("=" * 60 + "\n")
    f.write("HIERARCHICAL PIPELINE EVALUATION\n")
    f.write("=" * 60 + "\n\n")
    f.write("Architecture: Binary Classifier → Multi-class Classifier\n")
    f.write("Stage 1: XGBoost + RF-FS (binary attack/normal)\n")
    f.write("Stage 2: XGBoost multi-class (per-attack-type)\n\n")
    f.write(f"Flat multi-class accuracy      : {flat_acc:.4f}\n")
    f.write(f"Hierarchical pipeline accuracy : {hier_acc:.4f}\n")
    f.write(f"Stage-1 binary detection rate  : "
            f"{(stage1_pred == 1).sum() / (y_test_binary == 1).sum():.4f}\n")
    f.write(f"\nMacro F1 (multi-class)         : {f1_macro:.4f}\n")
    f.write(f"Weighted F1 (multi-class)      : {f1_weighted:.4f}\n")

# =============================================================
# SUMMARY
# =============================================================
print("\n" + "=" * 60)
print("Multi-class analysis complete")
print("=" * 60)
print("  figures/10_multiclass_confusion.png")
print("  figures/11_per_class_f1.png")
print("  results/multiclass_report.csv")
print("  results/hierarchical_summary.txt")
print("\nALL DONE. Total artifacts:")
print(f"  → figures/   ({len(os.listdir('figures'))} PNG files)")
print(f"  → results/   ({len(os.listdir('results'))} files)")
print(f"  → models/    ({len(os.listdir('models'))} saved models)")
print("\nReady to build the PPT.")
