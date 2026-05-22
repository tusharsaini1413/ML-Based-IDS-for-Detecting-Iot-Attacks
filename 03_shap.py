"""
03_shap.py
SHAP-based explainability on the best model (XGBoost + RF-FS).
Generates 3 figures: summary, bar (global), force (per-sample).

This is your "explainable AI" contribution for the end-sem panel.
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

try:
    import shap
except ImportError:
    print("ERROR: shap not installed. Run: pip install shap")
    raise

plt.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight'
})

# =============================================================
# LOAD ARTIFACTS
# =============================================================
print("Loading model and data...")
model = joblib.load("models/XGB_RFFS.pkl")
rf_selector = joblib.load("models/rf_selector.pkl")
feat_names_full = joblib.load("models/feature_names.pkl")

X_test = np.load("results/X_test_s.npy")
y_test = np.load("results/y_test.npy")

X_test_sel = rf_selector.transform(X_test)
selected_mask = rf_selector.get_support()
selected_feature_names = [n for n, k in zip(feat_names_full, selected_mask) if k]

print(f"  Test set        : {X_test.shape}")
print(f"  After selector  : {X_test_sel.shape}")
print(f"  Selected feats  : {len(selected_feature_names)}")

# Subsample for SHAP (full set is overkill and slow)
SAMPLE_N = min(500, len(X_test_sel))
np.random.seed(42)
idx = np.random.choice(len(X_test_sel), SAMPLE_N, replace=False)
sample = X_test_sel[idx]
sample_y = y_test[idx]

# =============================================================
# COMPUTE SHAP VALUES
# =============================================================
print(f"\nComputing SHAP values on {SAMPLE_N} samples (~1-2 min)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)
print(f"  SHAP values shape: {np.array(shap_values).shape}")

# =============================================================
# FIG 7 — SHAP Summary (beeswarm)
# =============================================================
print("\n→ Fig 7: SHAP summary plot")
plt.figure()
shap.summary_plot(shap_values, sample,
                  feature_names=selected_feature_names,
                  show=False, max_display=15)
plt.title("SHAP Summary — Feature Impact on Attack Detection",
          fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("figures/07_shap_summary.png", bbox_inches='tight')
plt.close()

# =============================================================
# FIG 8 — SHAP Bar (mean absolute)
# =============================================================
print("→ Fig 8: SHAP bar plot")
plt.figure()
shap.summary_plot(shap_values, sample,
                  feature_names=selected_feature_names,
                  plot_type="bar", show=False, max_display=15)
plt.title("Mean |SHAP Value| — Global Feature Importance",
          fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("figures/08_shap_bar.png", bbox_inches='tight')
plt.close()

# =============================================================
# FIG 9 — SHAP Force plot (one attack sample explained)
# =============================================================
print("→ Fig 9: SHAP force plot")
attack_indices = np.where(sample_y == 1)[0]
if len(attack_indices) == 0:
    print("  WARNING: no attack samples found in subset; using sample 0")
    idx_to_explain = 0
else:
    idx_to_explain = int(attack_indices[0])

plt.figure(figsize=(16, 3.5))
shap.force_plot(explainer.expected_value, shap_values[idx_to_explain],
                sample[idx_to_explain],
                feature_names=selected_feature_names,
                matplotlib=True, show=False)
plt.title(f"Why was sample #{idx_to_explain} classified as ATTACK?",
          fontweight='bold', pad=15)
plt.savefig("figures/09_shap_force.png", bbox_inches='tight')
plt.close()

# =============================================================
# SAVE SHAP RANKINGS
# =============================================================
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_rank = pd.DataFrame({
    "feature": selected_feature_names,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)
shap_rank.to_csv("results/shap_ranking.csv", index=False)

print("\nTop 10 features by mean |SHAP|:")
print(shap_rank.head(10).to_string(index=False))

print("\n" + "=" * 60)
print("SHAP explainability complete")
print("=" * 60)
print("  figures/07_shap_summary.png")
print("  figures/08_shap_bar.png")
print("  figures/09_shap_force.png")
print("  results/shap_ranking.csv")
print("\nNext: python 04_multiclass.py")
