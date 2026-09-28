"""
random_forest.py
Trains a Random Forest classifier on all labelled attack flows from the
training partition. Evaluates on attack flows in the test partition.
Saves model, per-class metrics, and confusion matrix.
"""

import numpy as np
import pickle
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE = r"C:\Users\Damilola Egbadon\Documents\CIC Network Dataset\processed"

# ── LABEL MAP ─────────────────────────────────────────────────────────────────
LABEL_MAP = {
    1:  'Bot',
    2:  'DDoS',
    3:  'DoS GoldenEye',
    4:  'DoS Hulk',
    5:  'DoS Slowhttptest',
    6:  'DoS slowloris',
    7:  'FTP-Patator',
    8:  'Heartbleed',
    9:  'Infiltration',
    10: 'PortScan',
    11: 'SSH-Patator',
    12: 'Web Attack BruteForce',
    13: 'Web Attack SQL Injection',
    14: 'Web Attack XSS'
}

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading data...")
X_train = np.load(os.path.join(BASE, "X_train.npy"))
y_train = np.load(os.path.join(BASE, "y_train.npy"))
X_test  = np.load(os.path.join(BASE, "X_test.npy"))
y_test  = np.load(os.path.join(BASE, "y_test.npy"))

print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")

# ── EXTRACT ATTACK FLOWS ONLY ─────────────────────────────────────────────────
# RF trains on ALL labelled attack flows in training set (ground truth labels)
# This is the supervised stage: it learns to distinguish attack types
train_attack_mask = y_train != 0
X_rf_train = X_train[train_attack_mask]
y_rf_train = y_train[train_attack_mask]

# Test: only evaluate on flows the IF flagged as anomalous
# We load IF threshold and model to replicate the pipeline
print("\nLoading Isolation Forest threshold...")
with open(os.path.join(BASE, "if_threshold.pkl"), "rb") as f:
    if_threshold = pickle.load(f)
with open(os.path.join(BASE, "isolation_forest_model.pkl"), "rb") as f:
    if_model = pickle.load(f)

# Score test set with IF
print("Scoring test set with Isolation Forest...")
scores_test = if_model.decision_function(X_test)
if_flagged  = scores_test < if_threshold   # True = IF said anomaly

# RF test set = flows IF flagged AND are true attacks
y_test_bin   = (y_test != 0).astype(int)
test_flagged_mask = if_flagged & (y_test != 0)   # true attack flows IF caught
X_rf_test = X_test[test_flagged_mask]
y_rf_test = y_test[test_flagged_mask]

print(f"\nRF training set : {X_rf_train.shape[0]:,} attack flows "
      f"({len(np.unique(y_rf_train))} classes)")
print(f"RF test set     : {X_rf_test.shape[0]:,} attack flows (IF-flagged)")

print("\nClass distribution — RF training set:")
unique, counts = np.unique(y_rf_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {LABEL_MAP.get(int(u), str(u)):<30}: {c:>8,}")

# ── TRAIN RANDOM FOREST ───────────────────────────────────────────────────────
print("\nTraining Random Forest (200 estimators)...")
t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced',   # handles minority classes (Heartbleed, etc.)
    min_samples_leaf=2
)
rf.fit(X_rf_train, y_rf_train)
elapsed = time.time() - t0
print(f"Training complete in {elapsed:.1f}s")

# Save model
model_path = os.path.join(BASE, "random_forest_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(rf, f)
print(f"Model saved to: {model_path}")

# ── EVALUATE ON IF-FLAGGED TEST FLOWS ────────────────────────────────────────
print("\nEvaluating on IF-flagged attack flows in test set...")
y_pred = rf.predict(X_rf_test)

acc = accuracy_score(y_rf_test, y_pred)
f1w = f1_score(y_rf_test, y_pred, average='weighted', zero_division=0)
f1m = f1_score(y_rf_test, y_pred, average='macro',    zero_division=0)

present_classes  = sorted(np.unique(y_rf_test))
present_names    = [LABEL_MAP[c] for c in present_classes]

print(f"\n{'='*60}")
print(f"RANDOM FOREST — TEST SET EVALUATION (IF-flagged flows)")
print(f"{'='*60}")
print(f"Flows evaluated      : {len(y_rf_test):,}")
print(f"Overall Accuracy     : {acc*100:.2f}%")
print(f"Weighted F1-Score    : {f1w:.4f}")
print(f"Macro F1-Score       : {f1m:.4f}")
print()
print(classification_report(
    y_rf_test, y_pred,
    labels=present_classes,
    target_names=present_names,
    digits=4,
    zero_division=0
))

# ── FULL PIPELINE METRICS (IF + RF combined) ──────────────────────────────────
# For flows IF flagged: RF classifies
# For flows IF missed:  counted as false negatives
# BENIGN flows IF flagged incorrectly: false alarms

print(f"\n{'='*60}")
print(f"DUAL-MODEL PIPELINE — END-TO-END METRICS (full test set)")
print(f"{'='*60}")

total       = len(y_test)
total_atk   = int((y_test != 0).sum())
total_norm  = int((y_test == 0).sum())

# IF stage
if_tp = int((if_flagged & (y_test != 0)).sum())   # attacks IF caught
if_fp = int((if_flagged & (y_test == 0)).sum())   # benign IF flagged wrong
if_fn = int((~if_flagged & (y_test != 0)).sum())  # attacks IF missed
if_tn = int((~if_flagged & (y_test == 0)).sum())  # benign correctly passed

# RF stage: of IF-caught attacks, how many does RF classify correctly?
rf_correct = int((y_pred == y_rf_test).sum())
rf_total   = len(y_rf_test)

# Pipeline detection rate = attacks IF caught AND RF correctly classifies
pipeline_correct_detections = rf_correct
pipeline_total_attacks      = total_atk

detection_rate  = pipeline_correct_detections / pipeline_total_attacks
false_alarm_rate = if_fp / total_norm
precision_pipe  = rf_correct / (rf_total + if_fp) if (rf_total + if_fp) > 0 else 0

print(f"Total test flows     : {total:,}")
print(f"  Attack flows       : {total_atk:,}")
print(f"  Benign flows       : {total_norm:,}")
print()
print(f"Isolation Forest stage:")
print(f"  Attacks caught (TP): {if_tp:,}  ({if_tp/total_atk*100:.1f}% of all attacks)")
print(f"  Attacks missed (FN): {if_fn:,}")
print(f"  False alarms   (FP): {if_fp:,}")
print()
print(f"Random Forest stage (on IF-flagged attacks):")
print(f"  Correctly classified: {rf_correct:,} / {rf_total:,}  ({rf_correct/rf_total*100:.2f}%)")
print()
print(f"Combined Pipeline:")
print(f"  Detection Rate     : {detection_rate*100:.2f}%")
print(f"  False Alarm Rate   : {false_alarm_rate*100:.2f}%")
print(f"  Pipeline Precision : {precision_pipe*100:.2f}%")

# ── SAVE PIPELINE METRICS ─────────────────────────────────────────────────────
metrics = {
    'rf_accuracy':         acc,
    'rf_f1_weighted':      f1w,
    'rf_f1_macro':         f1m,
    'if_detection_rate':   if_tp / total_atk,
    'if_false_alarm_rate': if_fp / total_norm,
    'pipeline_detection_rate':  detection_rate,
    'pipeline_false_alarm_rate': false_alarm_rate,
    'pipeline_precision':  precision_pipe,
    'total_test':          total,
    'total_attacks':       total_atk,
    'if_tp': if_tp, 'if_fp': if_fp, 'if_fn': if_fn, 'if_tn': if_tn,
}
metrics_path = os.path.join(BASE, "pipeline_metrics.pkl")
with open(metrics_path, "wb") as f:
    pickle.dump(metrics, f)
print(f"\nPipeline metrics saved to: {metrics_path}")

# ── PER-CLASS F1 BAR CHART ────────────────────────────────────────────────────
report = classification_report(
    y_rf_test, y_pred,
    labels=present_classes,
    target_names=present_names,
    output_dict=True,
    zero_division=0
)
class_f1s   = [report[n]['f1-score'] for n in present_names]
class_supp  = [report[n]['support']  for n in present_names]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(present_names, class_f1s, color='#2E75B6', edgecolor='white', height=0.6)
ax.set_xlabel('F1-Score', fontsize=12)
ax.set_title('Random Forest — Per-Class F1-Score on Test Set', fontsize=13, fontweight='bold')
ax.set_xlim(0, 1.05)
ax.axvline(f1w, color='#c0392b', linestyle='--', linewidth=1.5,
           label=f'Weighted avg F1 = {f1w:.4f}')
for bar, f1, supp in zip(bars, class_f1s, class_supp):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{f1:.3f}  (n={supp:,})', va='center', fontsize=9.5)
ax.legend(fontsize=11)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
chart_path = os.path.join(BASE, "rf_per_class_f1.png")
plt.savefig(chart_path, dpi=150)
plt.close()
print(f"Per-class F1 chart saved to: {chart_path}")

# ── CONFUSION MATRIX ─────────────────────────────────────────────────────────
cm = confusion_matrix(y_rf_test, y_pred, labels=present_classes)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(present_names)))
ax.set_yticks(range(len(present_names)))
ax.set_xticklabels(present_names, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(present_names, fontsize=9)
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('Actual', fontsize=11)
ax.set_title('Random Forest — Confusion Matrix (IF-flagged test flows)', fontsize=12, fontweight='bold')
thresh_cm = cm.max() / 2.0
for i in range(len(present_classes)):
    for j in range(len(present_classes)):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=8,
                color='white' if cm[i, j] > thresh_cm else 'black')
plt.tight_layout()
cm_path = os.path.join(BASE, "rf_confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"Confusion matrix saved to: {cm_path}")

print("\n" + "="*60)
print("Next step: run rag_pipeline.py")
print("="*60)
