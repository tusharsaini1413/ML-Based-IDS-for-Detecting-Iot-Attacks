"""
00_check_data.py
Run this FIRST to verify your CSV column names.
Edit BINARY_LABEL_COL and MULTI_LABEL_COL in 01_train_save.py if they differ.
"""
import pandas as pd
import sys

CSV_PATH = "edge_iiot.csv"

try:
    df = pd.read_csv(CSV_PATH, low_memory=False, nrows=5000)
except FileNotFoundError:
    print(f"ERROR: {CSV_PATH} not found in current folder.")
    print("Place your dataset here and re-run.")
    sys.exit(1)

print(f"Dataset: {CSV_PATH}")
print(f"Sample shape: {df.shape}")
print(f"Total columns: {df.shape[1]}")
print()
print("=" * 70)
print("ALL COLUMNS:")
print("=" * 70)
for i, col in enumerate(df.columns):
    print(f"  {i+1:3d}. {col}")

print()
print("=" * 70)
print("LABEL COLUMN CANDIDATES (likely contain target labels):")
print("=" * 70)
keywords = ['label', 'attack', 'class', 'type', 'target']
candidates = [c for c in df.columns if any(k in c.lower() for k in keywords)]
for col in candidates:
    unique_vals = df[col].unique()
    n_unique = len(unique_vals)
    sample = list(unique_vals[:10])
    print(f"  → {col}")
    print(f"     unique values: {n_unique}")
    print(f"     sample: {sample}")
    print()

print("=" * 70)
print("NEXT STEPS:")
print("=" * 70)
print("Open 01_train_save.py and confirm/edit these two lines (~line 38-39):")
print()
print("    BINARY_LABEL_COL = 'Attack_label'    # values 0/1")
print("    MULTI_LABEL_COL  = 'Attack_type'     # values like 'Normal', 'DDoS_UDP', etc.")
print()
print("If you only have ONE label column, the script will auto-derive the other.")
print("=" * 70)
