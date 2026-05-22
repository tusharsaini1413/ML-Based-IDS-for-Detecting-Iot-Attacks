"""
05_fix_and_rerun.py
Diagnostic + fix for two issues:
  1. Suspicious 100% accuracy due to leaky features (drops them)
  2. Multi-class IndexError (proper data alignment)

This RE-RUNS preprocessing and training cleanly. After this finishes,
also re-run: python 02_visualize.py, python 03_shap.py
"""
import os
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier

RANDOM_STATE = 42
SAMPLE_SIZE = 100000
np.random.seed(RANDOM_STATE)
sns.set_style("whitegrid")
plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 200, 'savefig.bbox': 'tight'})


def section(t):
    print(f"\n{'=' * 60}\n  {t}\n{'=' * 60}")


# =============================================================
# STEP 1 — LEAKAGE DIAGNOSTIC
# =============================================================
section("STEP 1: DIAGNOSING FEATURE LEAKAGE")

df = pd.read_csv("edge_iiot.csv", low_memory=False)
df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
print(f"Loaded {df.shape}")

# Check the suspected leaky features
suspects = ['dns.qry.name.len', 'mqtt.conack.flags', 'mqtt.protoname',
            'mqtt.msg_decoded_as', 'http.request.method', 'http.referer']

print("\nFor each suspect feature, check separability between Normal vs Attack:")
print("(If 'Normal' values are all identical, that feature is leaking the label)\n")

leaky_features = []
for col in suspects:
    if col not in df.columns:
        continue
    # Compute uniqueness for each class
    try:
        normal_vals = df.loc[df['Attack_label'] == 0, col]
        attack_vals = df.loc[df['Attack_label'] == 1, col]
        n_unique_normal = normal_vals.nunique()
        n_unique_attack = attack_vals.nunique()
        normal_mode = normal_vals.mode().iloc[0] if len(normal_vals) else None
        normal_mode_pct = (normal_vals == normal_mode).mean() * 100 if len(normal_vals) else 0
        print(f"  {col:30s} | Normal: {n_unique_normal:>4d} unique, "
              f"mode={normal_mode} ({normal_mode_pct:.1f}%) | "
              f"Attack: {n_unique_attack:>4d} unique")
        # Flag as leaky if Normal class has near-constant value
        if normal_mode_pct > 95 and n_unique_attack > 5:
            leaky_features.append(col)
    except Exception as e:
        print(f"  {col:30s} | (error: {e})")

print(f"\nDetected leaky features: {leaky_features}")

# Drop both standard drops and leaky features
DROP_STANDARD = [
    'frame.time', 'ip.src_host', 'ip.dst_host',
    'arp.src.proto_ipv4', 'arp.dst.proto_ipv4',
    'http.file_data', 'http.request.full_uri',
    'icmp.transmit_timestamp', 'http.request.uri.query',
    'tcp.options', 'tcp.payload', 'tcp.srcport', 'tcp.dstport',
    'udp.port', 'mqtt.msg', 'mqtt.topic'
]
DROP_LEAKY = ["dns.qry.name.len", "http.request.method", "http.referer", "mqtt.msg_decoded_as", "mqtt.conack.flags", "mqtt.protoname"]  # discovered above
print(f"\nDropping standard non-features: {len(DROP_STANDARD)} columns")
print(f"Dropping detected leaky features: {DROP_LEAKY}")

# =============================================================
# STEP 2 — CLEAN PREPROCESSING (no leaky features)
# =============================================================
section("STEP 2: PREPROCESSING (LEAKAGE-FREE)")

df = df.drop(columns=[c for c in (DROP_STANDARD + DROP_LEAKY) if c in df.columns],
             errors='ignore')
df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
print(f"After cleaning: {df.shape}")

y_binary = df['Attack_label'].astype(int)
y_multi_raw = df['Attack_type'].astype(str)

le_multi = LabelEncoder()
y_multi_encoded = le_multi.fit_transform(y_multi_raw)
joblib.dump(le_multi, "models/label_encoder_multi.pkl")

X = df.drop(columns=['Attack_label', 'Attack_type'])
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
X = X.astype(float)
print(f"Feature matrix: {X.shape}")

# Split
X_train, X_test, y_train, y_test, y_multi_train, y_multi_test = train_test_split(
    X, y_binary, y_multi_encoded, test_size=0.2,
    stratify=y_binary, random_state=RANDOM_STATE)

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(list(X.columns), "models/feature_names.pkl")

# Save BOTH original and SMOTE'd training data for downstream scripts
np.save("results/X_train_s.npy", X_train_s)
np.save("results/y_train.npy", y_train.values)

# SMOTE
X_train_smote, y_train_smote = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train_s, y_train)
X_train_rus, y_train_rus = RandomUnderSampler(random_state=RANDOM_STATE).fit_resample(X_train_s, y_train)
print(f"After SMOTE: {X_train_smote.shape}")

# Save class distribution
cd = {"before": dict(pd.Series(y_train).value_counts()),
      "after_smote": dict(pd.Series(y_train_smote).value_counts()),
      "after_rus": dict(pd.Series(y_train_rus).value_counts())}
with open("results/class_distribution.json", "w") as f:
    json.dump({k: {str(kk): int(vv) for kk, vv in v.items()}
               for k, v in cd.items()}, f, indent=2)

# =============================================================
# STEP 3 — FEATURE SELECTION
# =============================================================
section("STEP 3: FEATURE SELECTION")

pca = PCA(n_components=min(20, X_train_smote.shape[1]), random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_smote)
X_test_pca = pca.transform(X_test_s)
joblib.dump(pca, "models/pca.pkl")
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")

rf_base = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf_base.fit(X_train_smote, y_train_smote)
rf_selector = SelectFromModel(rf_base, threshold='median', prefit=True)
X_train_rffs = rf_selector.transform(X_train_smote)
X_test_rffs = rf_selector.transform(X_test_s)
joblib.dump(rf_selector, "models/rf_selector.pkl")
selected = [n for n, k in zip(X.columns, rf_selector.get_support()) if k]
print(f"RF selected {len(selected)} of {X.shape[1]} features")

pd.DataFrame({
    "feature": X.columns,
    "importance": rf_base.feature_importances_
}).sort_values("importance", ascending=False).to_csv("results/feature_importance.csv", index=False)

# =============================================================
# STEP 4 — TRAIN ALL BINARY MODELS
# =============================================================
section("STEP 4: TRAIN BINARY MODELS")


def train(name, model, Xtr, ytr, Xte, yte):
    t = time.time()
    model.fit(Xtr, ytr)
    yp = model.predict(Xte)
    try:
        yprob = model.predict_proba(Xte)[:, 1]
    except Exception:
        yprob = yp.astype(float)
    m = {"accuracy": float(accuracy_score(yte, yp)),
         "precision": float(precision_score(yte, yp, zero_division=0)),
         "recall": float(recall_score(yte, yp, zero_division=0)),
         "f1": float(f1_score(yte, yp, zero_division=0))}
    np.save(f"results/{name}_ypred.npy", yp)
    np.save(f"results/{name}_yprob.npy", yprob)
    np.save(f"results/{name}_ytest.npy", yte)
    joblib.dump(model, f"models/{name}.pkl")
    print(f"  {name:12s} acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  ({time.time()-t:.1f}s)")
    return m


all_metrics = {}
all_metrics["XGB_SMOTE"] = train("XGB_SMOTE",
    XGBClassifier(n_estimators=200, max_depth=6, eval_metric='logloss',
                  random_state=RANDOM_STATE, n_jobs=-1),
    X_train_smote, y_train_smote, X_test_s, y_test)

all_metrics["RF_SMOTE"] = train("RF_SMOTE",
    RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    X_train_smote, y_train_smote, X_test_s, y_test)

all_metrics["RF_RUS"] = train("RF_RUS",
    RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    X_train_rus, y_train_rus, X_test_s, y_test)

all_metrics["SVM"] = train("SVM",
    SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
    X_train_smote[:10000], y_train_smote[:10000], X_test_s, y_test)

all_metrics["XGB_PCA"] = train("XGB_PCA",
    XGBClassifier(n_estimators=200, max_depth=6, eval_metric='logloss',
                  random_state=RANDOM_STATE, n_jobs=-1),
    X_train_pca, y_train_smote, X_test_pca, y_test)

all_metrics["XGB_RFFS"] = train("XGB_RFFS",
    XGBClassifier(n_estimators=200, max_depth=6, eval_metric='logloss',
                  random_state=RANDOM_STATE, n_jobs=-1),
    X_train_rffs, y_train_smote, X_test_rffs, y_test)

# =============================================================
# STEP 5 — 5-FOLD CV
# =============================================================
section("STEP 5: 5-FOLD CROSS VALIDATION")

X_all = X.values; y_all = y_binary.values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = []

for fold, (tr, te) in enumerate(skf.split(X_all, y_all), 1):
    s = StandardScaler()
    Xtr_s = s.fit_transform(X_all[tr])
    Xte_s = s.transform(X_all[te])
    Xtr_b, ytr_b = SMOTE(random_state=RANDOM_STATE).fit_resample(Xtr_s, y_all[tr])
    rfb = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rfb.fit(Xtr_b, ytr_b)
    sel = SelectFromModel(rfb, threshold='median', prefit=True)
    clf = XGBClassifier(n_estimators=200, max_depth=6, eval_metric='logloss',
                        random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(sel.transform(Xtr_b), ytr_b)
    acc = accuracy_score(y_all[te], clf.predict(sel.transform(Xte_s)))
    cv_scores.append(acc)
    print(f"  Fold {fold}: {acc:.4f}")

print(f"\n  Mean: {np.mean(cv_scores):.4f}  Std: {np.std(cv_scores):.4f}")
np.save("results/cv_scores.npy", np.array(cv_scores))

# =============================================================
# STEP 6 — MULTI-CLASS (THE FIX)
# =============================================================
section("STEP 6: MULTI-CLASS CLASSIFICATION")

n_classes = len(le_multi.classes_)
print(f"Classes ({n_classes}): {list(le_multi.classes_)}")

# Use ORIGINAL (non-SMOTE'd) X_train_s — aligns with y_multi_train
N = min(50000, len(X_train_s))
idx_mc = np.random.RandomState(42).choice(len(X_train_s), N, replace=False)

mc_model = XGBClassifier(
    n_estimators=300, max_depth=8, learning_rate=0.1,
    eval_metric='mlogloss', random_state=42, n_jobs=-1,
    objective='multi:softprob', num_class=n_classes)
mc_model.fit(X_train_s[idx_mc], y_multi_train[idx_mc])

y_mc_pred = mc_model.predict(X_test_s)
mc_acc = accuracy_score(y_multi_test, y_mc_pred)
mc_f1_macro = f1_score(y_multi_test, y_mc_pred, average='macro')
mc_f1_weighted = f1_score(y_multi_test, y_mc_pred, average='weighted')

print(f"\n  Multi-class accuracy : {mc_acc:.4f}")
print(f"  Macro F1             : {mc_f1_macro:.4f}")
print(f"  Weighted F1          : {mc_f1_weighted:.4f}")
joblib.dump(mc_model, "models/XGB_multiclass.pkl")

# Per-class report
present = sorted(np.unique(np.concatenate([y_multi_test, y_mc_pred])))
present_names = [le_multi.classes_[i] for i in present]
report = classification_report(y_multi_test, y_mc_pred,
                                labels=present, target_names=present_names,
                                output_dict=True, zero_division=0)
pd.DataFrame(report).T.round(4).to_csv("results/multiclass_report.csv")

# Figure 10: multi-class confusion matrix
cm = confusion_matrix(y_multi_test, y_mc_pred, labels=present)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
plt.figure(figsize=(12, 10))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=present_names, yticklabels=present_names,
            cbar=True, annot_kws={'size': 8})
plt.title(f"Multi-Class Confusion Matrix (Normalized)\n"
          f"15-Class Attack Detection — Accuracy: {mc_acc:.4f}", fontweight='bold')
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("figures/10_multiclass_confusion.png")
plt.close()

# Figure 11: per-class F1
class_f1 = {cls: report[cls]['f1-score'] for cls in present_names if cls in report}
items = sorted(class_f1.items(), key=lambda x: x[1], reverse=True)
classes, scores = zip(*items)
plt.figure(figsize=(12, 6))
colors = sns.color_palette("viridis", len(classes))
bars = plt.bar(range(len(classes)), scores, color=colors)
plt.title(f"F1-Score by Attack Type (15 Classes)", fontweight='bold')
plt.ylabel("F1-Score")
plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
plt.ylim(0, 1.05)
for b, s in zip(bars, scores):
    plt.text(b.get_x() + b.get_width()/2, s + 0.01, f"{s:.2f}",
             ha='center', fontsize=9)
plt.tight_layout()
plt.savefig("figures/11_per_class_f1.png")
plt.close()

# =============================================================
# STEP 7 — HIERARCHICAL PIPELINE
# =============================================================
section("STEP 7: HIERARCHICAL PIPELINE")

xgb_bin = joblib.load("models/XGB_RFFS.pkl")
stage1 = xgb_bin.predict(rf_selector.transform(X_test_s))
flagged = np.where(stage1 == 1)[0]
stage2 = mc_model.predict(X_test_s[flagged]) if len(flagged) else np.array([])

normal_code = None
for cls in le_multi.classes_:
    if cls.lower() == 'normal':
        normal_code = le_multi.transform([cls])[0]
        break
if normal_code is None:
    normal_code = 0

final = np.full(len(y_multi_test), normal_code, dtype=int)
final[flagged] = stage2
hier_acc = (final == y_multi_test).mean()

with open("results/hierarchical_summary.txt", "w") as f:
    f.write(f"Flat multi-class accuracy      : {mc_acc:.4f}\n")
    f.write(f"Hierarchical pipeline accuracy : {hier_acc:.4f}\n")
    f.write(f"Macro F1                       : {mc_f1_macro:.4f}\n")
    f.write(f"Weighted F1                    : {mc_f1_weighted:.4f}\n")

print(f"  Flat multi-class: {mc_acc:.4f}")
print(f"  Hierarchical:     {hier_acc:.4f}")

# =============================================================
# SAVE EVERYTHING
# =============================================================
with open("results/all_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)
np.save("results/X_test_s.npy", X_test_s)
np.save("results/y_test.npy", y_test.values)
np.save("results/y_multi_test.npy", y_multi_test)
np.save("results/X_train_smote.npy", X_train_smote)
np.save("results/y_train_smote.npy", y_train_smote)
np.save("results/y_multi_train.npy", y_multi_train)

section("DONE")
print("Next steps:")
print("  1. python 02_visualize.py    (regenerate fig 1-6 with new results)")
print("  2. python 03_shap.py         (regenerate SHAP plots)")
print("  3. Check figures/ folder")
