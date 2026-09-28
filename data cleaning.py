"""
data_cleaning.py
Loads all 8 CIC-IDS2017 CSV files, cleans and encodes them,
saves a single cleaned CSV for downstream scripts.
Run once — takes ~5-10 minutes.
"""

import pandas as pd
import numpy as np
import os
import glob

# ── PATHS — CHANGE THESE TO MATCH YOUR MACHINE ────────────────────────────────
RAW_PATH = r"C:\Users\Damilola Egbadon\Documents\CIC Network Dataset\MachineLearningCSV\CSV"
OUT_PATH = r"C:\Users\Damilola Egbadon\Documents\CIC Network Dataset"
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_FILE = os.path.join(OUT_PATH, "cleaned_dataset.csv")

# ── LABEL ENCODING ────────────────────────────────────────────────────────────
LABEL_MAP = {
    'BENIGN':                    0,
    'Bot':                       1,
    'DDoS':                      2,
    'DoS GoldenEye':             3,
    'DoS Hulk':                  4,
    'DoS Slowhttptest':          5,
    'DoS slowloris':             6,
    'FTP-Patator':               7,
    'Heartbleed':                8,
    'Infiltration':              9,
    'PortScan':                  10,
    'SSH-Patator':               11,
    'Web Attack \x96 Brute Force':  12,
    'Web Attack - Brute Force':  12,
    'Web Attack \x96 Sql Injection': 13,
    'Web Attack - Sql Injection': 13,
    'Web Attack \x96 XSS':       14,
    'Web Attack - XSS':          14,
}

# ── LOAD ALL CSV FILES ────────────────────────────────────────────────────────
csv_files = glob.glob(os.path.join(RAW_PATH, "*.csv"))
print(f"Found {len(csv_files)} CSV files:")
for f in csv_files:
    print(f"  {os.path.basename(f)}")

dfs = []
for f in csv_files:
    print(f"\nLoading {os.path.basename(f)}...")
    df = pd.read_csv(f, encoding='utf-8', low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"  Shape: {df.shape}")
    dfs.append(df)

print("\nConcatenating all files...")
data = pd.concat(dfs, ignore_index=True)
print(f"Combined shape: {data.shape}")

# ── CLEAN ─────────────────────────────────────────────────────────────────────
print("\nCleaning data...")

# Strip column names
data.columns = data.columns.str.strip()

# Replace infinite values
n_inf = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
print(f"  Infinite values found: {n_inf}")
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill missing values with column median
n_missing = data.isnull().sum().sum()
print(f"  Missing values found: {n_missing}")
for col in data.select_dtypes(include=[np.number]).columns:
    if data[col].isnull().any():
        data[col].fillna(data[col].median(), inplace=True)

# Remove duplicates
n_before = len(data)
data.drop_duplicates(inplace=True)
n_after = len(data)
print(f"  Duplicates removed: {n_before - n_after:,}")
print(f"  Rows remaining: {n_after:,}")

# ── LABEL ENCODING ────────────────────────────────────────────────────────────
label_col = ' Label' if ' Label' in data.columns else 'Label'
print(f"\nLabel column: '{label_col}'")
print(f"Unique labels found:")
for lbl in sorted(data[label_col].unique()):
    print(f"  '{lbl}'  →  {data[label_col].value_counts()[lbl]:,} rows")

data[label_col] = data[label_col].str.strip()
data['Label_Encoded'] = data[label_col].map(LABEL_MAP)

unmapped = data['Label_Encoded'].isnull().sum()
if unmapped > 0:
    print(f"WARNING: {unmapped} rows could not be mapped. Check label names.")
    print(data[data['Label_Encoded'].isnull()][label_col].unique())
else:
    print("All labels mapped successfully.")

data.drop(columns=[label_col], inplace=True)

# ── SAVE ─────────────────────────────────────────────────────────────────────
print(f"\nSaving cleaned dataset to: {OUTPUT_FILE}")
data.to_csv(OUTPUT_FILE, index=False)
print(f"Done. Final shape: {data.shape}")
print(f"Class distribution:")
for code, name in sorted({v:k for k,v in LABEL_MAP.items() if k not in ['Web Attack \\x96 Brute Force','Web Attack \\x96 Sql Injection','Web Attack \\x96 XSS']}.items()):
    count = (data['Label_Encoded'] == code).sum()
    if count > 0:
        print(f"  {name:<35}: {count:>9,}")
print("\nNext step: run feature_selection.py")
