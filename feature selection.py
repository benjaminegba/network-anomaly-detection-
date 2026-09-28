"""
feature_selection.py
Loads cleaned dataset, applies 70/20/10 stratified split,
selects top 20 features by Random Forest importance,
applies Min-Max normalisation, saves all arrays.
Run once after data_cleaning.py.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

# ── PATHS — CHANGE THESE TO MATCH YOUR MACHINE ────────────────────────────────
BASE     = r"C:\Users\Damilola Egbadon\Documents\CIC Network Dataset"
CLEAN    = os.path.join(BASE, "cleaned_dataset.csv")
OUT      = os.path.join(BASE, "processed")
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT, exist_ok=True)

# ── LOAD ─────────────────────────────────────────────────────────────────────
print("Loading cleaned dataset...")
data = pd.read_csv(CLEAN, low_memory=False)
print(f"Shape: {data.shape}")

X = data.drop(columns=['Label_Encoded']).select_dtypes(include=[np.number])
y = data['Label_Encoded'].values.astype(int)
print(f"Features: {X.shape[1]}  |  Samples: {len(y)}")

# ── TRAIN / TEST / VAL SPLIT (70/20/10) ──────────────────────────────────────
print("\nSplitting 70/20/10 stratified...")
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.10/0.80,   # 10% of total = 12.5% of remaining 80%
    random_state=42, stratify=y_train_full)

print(f"  Train : {X_train.shape[0]:,}")
print(f"  Test  : {X_test.shape[0]:,}")
print(f"  Val   : {X_val.shape[0]:,}")

# ── FEATURE SELECTION ─────────────────────────────────────────────────────────
print("\nSelecting top 20 features via Random Forest importance (10% sample)...")
sample_size = int(len(X_train) * 0.10)
idx_sample  = np.random.RandomState(42).choice(len(X_train), sample_size, replace=False)
X_sample    = X_train.iloc[idx_sample]
y_sample    = y_train[idx_sample]

rf_sel = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf_sel.fit(X_sample, y_sample)

importances = pd.Series(rf_sel.feature_importances_, index=X_train.columns)
top20       = importances.nlargest(20).index.tolist()
print("Top 20 features selected:")
for i, f in enumerate(top20, 1):
    print(f"  {i:>2}. {f}")

# ── APPLY FEATURE SELECTION ───────────────────────────────────────────────────
X_train = X_train[top20].values
X_test  = X_test[top20].values
X_val   = X_val[top20].values

# ── MIN-MAX NORMALISATION ─────────────────────────────────────────────────────
print("\nApplying Min-Max normalisation...")
scaler  = MinMaxScaler()
X_train = scaler.fit_transform(X_train)   # fit on train only
X_test  = scaler.transform(X_test)
X_val   = scaler.transform(X_val)

# ── SAVE ─────────────────────────────────────────────────────────────────────
print("\nSaving arrays and metadata...")
np.save(os.path.join(OUT, "X_train.npy"), X_train)
np.save(os.path.join(OUT, "X_test.npy"),  X_test)
np.save(os.path.join(OUT, "X_val.npy"),   X_val)
np.save(os.path.join(OUT, "y_train.npy"), y_train)
np.save(os.path.join(OUT, "y_test.npy"),  y_test)
np.save(os.path.join(OUT, "y_val.npy"),   y_val)

with open(os.path.join(OUT, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(OUT, "selected_features.pkl"), "wb") as f:
    pickle.dump(top20, f)

print(f"Saved to: {OUT}")
print(f"  X_train: {X_train.shape}")
print(f"  X_test : {X_test.shape}")
print(f"  X_val  : {X_val.shape}")
print("\nNext step: run isolation_forest.py")
