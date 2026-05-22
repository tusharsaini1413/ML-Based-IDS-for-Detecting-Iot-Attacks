"""
01_train_save.py
Master training script. Trains 7 model variants, runs 5-fold CV,
saves all artifacts (models, predictions, probabilities, metrics) to disk.

Run ONCE. All downstream scripts read from saved artifacts.
"""
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier

# =============================================================
# CONFIG — edit only this section if you need to
# =============================================================
CSV_PATH        = "edge_iiot.csv"
SAMPLE_SIZE     = 100000          # reduce if memory issues
RANDOM_STATE    = 42
TEST_SIZE       = 0.2
PCA_COMPONENTS  = 20
SVM_SUBSET      = 10000          # SVM is slow; train on subset

BINARY_LABEL_COL = 'Attack_label'   # binary 0/1 column
MULTI_LABEL_COL  = 'Attack_type'    # multi-class attack-name column
# =============================================================

np.random.seed(RANDOM_STATE)

# TensorFlow (optional — DNN won't train if it fails)
TF_AVAILABLE = True
try:
    import tensorflow as tf
    tf.random.set_seed(RANDOM_STATE)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
except Exception as e:
    print(f"WARNING: TensorFlow unavailable ({e}). DNN will be skipped.")
    TF_AVAILABLE = False


def section(title):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def safe_metrics(y_true, y_pred):
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0))
    }


# =============================================================
# 1. LOAD + SAMPLE
# =============================================================
section("1. LOADING DATASET")
t0 = time.time()
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"{CSV_PATH} not found. Place edge_iiot.csv in this folder.")

df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"Full dataset shape: {df.shape}")

if len(df) > SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Sampled to: {df.shape}")
print(f"Time: {time.time()-t0:.1f}s")

# =============================================================
# 2. AUTO-DETECT LABEL COLUMNS IF MISSING
# =============================================================
section("2. RESOLVING LABEL COLUMNS")

available = list(df.columns)

if BINARY_LABEL_COL not in available:
    candidates = [c for c in available if 'label' in c.lower()]
    if candidates:
        BINARY_LABEL_COL = candidates[0]
        print(f"  Auto-selected binary column: {BINARY_LABEL_COL}")
    else:
        raise ValueError(
            f"Binary label column not found. Available: {available[-10:]}")

if MULTI_LABEL_COL not in available:
    candidates = [c for c in available
                  if 'type' in c.lower() or 'class' in c.lower() or 'category' in c.lower()]
    candidates = [c for c in candidates if c != BINARY_LABEL_COL]
    if candidates:
        MULTI_LABEL_COL = candidates[0]
        print(f"  Auto-selected multi-class column: {MULTI_LABEL_COL}")
    else:
        # Fall back: derive a single-value multi-class from binary
        print("  No multi-class column found. Using binary as multi-class proxy.")
        MULTI_LABEL_COL = BINARY_LABEL_COL

print(f"  Binary column      : {BINARY_LABEL_COL}")
print(f"  Multi-class column : {MULTI_LABEL_COL}")

# =============================================================
# 3. CLEAN
# =============================================================
section("3. CLEANING")

# Common columns to drop (high cardinality, identifiers, raw payloads)
DROP_PATTERNS = [
    'frame.time', 'ip.src_host', 'ip.dst_host',
    'arp.src.proto_ipv4', 'arp.dst.proto_ipv4',
    'http.file_data', 'http.request.full_uri',
    'icmp.transmit_timestamp', 'http.request.uri.query',
    'tcp.options', 'tcp.payload', 'tcp.srcport', 'tcp.dstport',
    'udp.port', 'mqtt.msg', 'mqtt.topic'
]
drop_cols = [c for c in DROP_PATTERNS if c in df.columns]
df = df.drop(columns=drop_cols, errors='ignore')
print(f"  Dropped columns: {drop_cols}")

# Drop NaN and duplicate rows
before = len(df)
df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
print(f"  Rows: {before} → {len(df)} (removed NaN/duplicates)")
print(f"  Shape after cleaning: {df.shape}")

# =============================================================
# 4. SEPARATE LABELS, ENCODE FEATURES
# =============================================================
section("4. PREPARING FEATURES AND LABELS")

y_binary = df[BINARY_LABEL_COL].astype(int)
y_multi_raw = df[MULTI_LABEL_COL].astype(str)

le_multi = LabelEncoder()
y_multi_encoded = le_multi.fit_transform(y_multi_raw)
joblib.dump(le_multi, "models/label_encoder_multi.pkl")

# Drop both label cols from features
X = df.drop(columns=[BINARY_LABEL_COL, MULTI_LABEL_COL], errors='ignore')

# Encode any remaining categorical (object) columns
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
X = X.astype(float)

print(f"  Feature matrix shape : {X.shape}")
print(f"  Binary distribution  :\n{y_binary.value_counts().to_string()}")
print(f"  Multi-class classes  : {len(le_multi.classes_)}")
print(f"  Top classes          : {pd.Series(y_multi_raw).value_counts().head(5).to_dict()}")

# =============================================================
# 5. SPLIT + SCALE
# =============================================================
section("5. SPLIT AND SCALE")

X_train, X_test, y_train, y_test, y_multi_train, y_multi_test = train_test_split(
    X, y_binary, y_multi_encoded,
    test_size=TEST_SIZE, stratify=y_binary, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(list(X.columns), "models/feature_names.pkl")
print(f"  Train: {X_train_s.shape}  Test: {X_test_s.shape}")

# =============================================================
# 6. IMBALANCE HANDLING
# =============================================================
section("6. CLASS BALANCING")

smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train_s, y_train)
print(f"  After SMOTE: {X_train_smote.shape}, "
      f"classes: {dict(pd.Series(y_train_smote).value_counts())}")

rus = RandomUnderSampler(random_state=RANDOM_STATE)
X_train_rus, y_train_rus = rus.fit_resample(X_train_s, y_train)
print(f"  After RUS  : {X_train_rus.shape}, "
      f"classes: {dict(pd.Series(y_train_rus).value_counts())}")

# Save class distribution for visualization
class_dist = {
    "before":      dict(pd.Series(y_train).value_counts()),
    "after_smote": dict(pd.Series(y_train_smote).value_counts()),
    "after_rus":   dict(pd.Series(y_train_rus).value_counts())
}
with open("results/class_distribution.json", "w") as f:
    json.dump({k: {str(kk): int(vv) for kk, vv in v.items()}
               for k, v in class_dist.items()}, f, indent=2)

# =============================================================
# 7. FEATURE SELECTION VARIANTS
# =============================================================
section("7. FEATURE SELECTION")

print("  → PCA (20 components)...")
pca = PCA(n_components=min(PCA_COMPONENTS, X_train_smote.shape[1]),
          random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_smote)
X_test_pca  = pca.transform(X_test_s)
joblib.dump(pca, "models/pca.pkl")
print(f"     explained variance: {pca.explained_variance_ratio_.sum():.4f}")

print("  → RF-based feature selection...")
rf_base = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf_base.fit(X_train_smote, y_train_smote)
rf_selector = SelectFromModel(rf_base, threshold='median', prefit=True)
X_train_rffs = rf_selector.transform(X_train_smote)
X_test_rffs  = rf_selector.transform(X_test_s)
selected = [n for n, k in zip(X.columns, rf_selector.get_support()) if k]
print(f"     selected {len(selected)} of {X.shape[1]} features")
joblib.dump(rf_selector, "models/rf_selector.pkl")

# Save feature importances for plotting
fi_df = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_base.feature_importances_
}).sort_values("importance", ascending=False)
fi_df.to_csv("results/feature_importance.csv", index=False)


# =============================================================
# 8. TRAIN ALL MODELS
# =============================================================
section("8. TRAINING MODELS")


def train_and_save(name, model, X_tr, y_tr, X_te, y_te, is_keras=False):
    t = time.time()
    print(f"\n  → {name} ...")
    if is_keras:
        model.fit(X_tr, y_tr, epochs=10, batch_size=128,
                  verbose=0, validation_split=0.1)
        y_prob = model.predict(X_te, verbose=0).ravel()
        y_pred = (y_prob > 0.5).astype(int)
    else:
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        try:
            y_prob = model.predict_proba(X_te)[:, 1]
        except Exception:
            try:
                y_prob = model.decision_function(X_te)
            except Exception:
                y_prob = y_pred.astype(float)

    m = safe_metrics(y_te, y_pred)
    np.save(f"results/{name}_ypred.npy", y_pred)
    np.save(f"results/{name}_yprob.npy", y_prob)
    np.save(f"results/{name}_ytest.npy", y_te)
    if not is_keras:
        joblib.dump(model, f"models/{name}.pkl")
    print(f"     acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  ({time.time()-t:.1f}s)")
    return m


all_metrics = {}

all_metrics["XGB_SMOTE"] = train_and_save(
    "XGB_SMOTE",
    XGBClassifier(n_estimators=200, max_depth=6, use_label_encoder=False,
                  eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1),
    X_train_smote, y_train_smote, X_test_s, y_test)

all_metrics["RF_SMOTE"] = train_and_save(
    "RF_SMOTE",
    RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    X_train_smote, y_train_smote, X_test_s, y_test)

all_metrics["RF_RUS"] = train_and_save(
    "RF_RUS",
    RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    X_train_rus, y_train_rus, X_test_s, y_test)

print(f"\n  → SVM (training on {SVM_SUBSET}-sample subset for speed)...")
svm_size = min(SVM_SUBSET, len(X_train_smote))
all_metrics["SVM"] = train_and_save(
    "SVM",
    SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
    X_train_smote[:svm_size], y_train_smote[:svm_size], X_test_s, y_test)

all_metrics["XGB_PCA"] = train_and_save(
    "XGB_PCA",
    XGBClassifier(n_estimators=200, max_depth=6, use_label_encoder=False,
                  eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1),
    X_train_pca, y_train_smote, X_test_pca, y_test)

all_metrics["XGB_RFFS"] = train_and_save(
    "XGB_RFFS",
    XGBClassifier(n_estimators=200, max_depth=6, use_label_encoder=False,
                  eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1),
    X_train_rffs, y_train_smote, X_test_rffs, y_test)

if TF_AVAILABLE:
    dnn = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_smote.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    dnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    all_metrics["DNN"] = train_and_save(
        "DNN", dnn, X_train_smote, y_train_smote, X_test_s, y_test, is_keras=True)
    dnn.save("models/DNN.keras")
else:
    print("\n  → DNN skipped (TensorFlow unavailable)")


# =============================================================
# 9. 5-FOLD CROSS VALIDATION ON BEST PIPELINE
# =============================================================
section("9. 5-FOLD CROSS VALIDATION (XGBoost + RF-FS)")

X_all = X.values
y_all = y_binary.values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = []

for fold, (tr, te) in enumerate(skf.split(X_all, y_all), 1):
    s = StandardScaler()
    X_tr_s = s.fit_transform(X_all[tr])
    X_te_s = s.transform(X_all[te])

    X_tr_b, y_tr_b = SMOTE(random_state=RANDOM_STATE).fit_resample(X_tr_s, y_all[tr])

    rf_b = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf_b.fit(X_tr_b, y_tr_b)
    sel = SelectFromModel(rf_b, threshold='median', prefit=True)

    clf = XGBClassifier(n_estimators=200, max_depth=6, use_label_encoder=False,
                        eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(sel.transform(X_tr_b), y_tr_b)
    acc = accuracy_score(y_all[te], clf.predict(sel.transform(X_te_s)))
    cv_scores.append(acc)
    print(f"  Fold {fold}: {acc:.4f}")

print(f"\n  Mean: {np.mean(cv_scores):.4f}    Std: {np.std(cv_scores):.4f}")

# =============================================================
# 10. SAVE ARTIFACTS
# =============================================================
section("10. SAVING ARTIFACTS")

with open("results/all_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)
np.save("results/cv_scores.npy", np.array(cv_scores))

# Save test data + multi-class labels for downstream scripts
np.save("results/X_test_s.npy", X_test_s)
np.save("results/y_test.npy", y_test.values)
np.save("results/y_multi_test.npy", y_multi_test)
np.save("results/X_train_smote.npy", X_train_smote)
np.save("results/y_train_smote.npy", y_train_smote)
np.save("results/y_multi_train.npy", y_multi_train)

print(f"\n  ✔ Models saved   → models/")
print(f"  ✔ Predictions    → results/*_ypred.npy")
print(f"  ✔ Probabilities  → results/*_yprob.npy")
print(f"  ✔ Metrics        → results/all_metrics.json")
print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
print("\nNext: python 02_visualize.py")
