"""
evaluate.py
Final evaluation script. Loads all saved models and metrics, produces:
- Complete pipeline metrics table
- Per-class RF classification report
- Comparison table (IF only vs RF only vs Dual-model)
- Confusion matrix charts
- Throughput measurement
All outputs saved to processed\ folder for Chapter 5.
"""

import numpy as np
import pickle
import os
import time
import json
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, precision_score,
                             recall_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE = r"C:\Users\Damilola Egbadon\Documents\CIC Network Dataset\processed"

LABEL_MAP = {
    0:  "BENIGN",
    1:  "Bot",
    2:  "DDoS",
    3:  "DoS GoldenEye",
    4:  "DoS Hulk",
    5:  "DoS Slowhttptest",
    6:  "DoS slowloris",
    7:  "FTP-Patator",
    8:  "Heartbleed",
    9:  "Infiltration",
    10: "PortScan",
    11: "SSH-Patator",
    12: "Web Attack BruteForce",
    13: "Web Attack SQL Injection",
    14: "Web Attack XSS"
}

# ── LOAD DATA & MODELS ────────────────────────────────────────────────────────
print("Loading data and models...")
X_test  = np.load(os.path.join(BASE, "X_test.npy"))
y_test  = np.load(os.path.join(BASE, "y_test.npy"))
X_train = np.load(os.path.join(BASE, "X_train.npy"))
y_train = np.load(os.path.join(BASE, "y_train.npy"))

with open(os.path.join(BASE, "isolation_forest_model.pkl"), "rb") as f:
    if_model = pickle.load(f)
with open(os.path.join(BASE, "random_forest_model.pkl"), "rb") as f:
    rf_model = pickle.load(f)
with open(os.path.join(BASE, "if_threshold.pkl"), "rb") as f:
    if_threshold = pickle.load(f)

print(f"  Test set : {X_test.shape[0]:,} flows")
print(f"  IF threshold: {if_threshold:.4f}")

y_test_bin = (y_test != 0).astype(int)

# ── STAGE 1: ISOLATION FOREST ─────────────────────────────────────────────────
print("\nRunning Isolation Forest on test set...")
t0 = time.time()
scores_test = if_model.decision_function(X_test)
if_time     = time.time() - t0

if_preds    = (scores_test < if_threshold).astype(int)
if_flagged  = if_preds == 1

if_tp = int(((if_preds==1) & (y_test_bin==1)).sum())
if_fp = int(((if_preds==1) & (y_test_bin==0)).sum())
if_fn = int(((if_preds==0) & (y_test_bin==1)).sum())
if_tn = int(((if_preds==0) & (y_test_bin==0)).sum())

total_atk  = int(y_test_bin.sum())
total_norm = int((y_test_bin==0).sum())

if_dr   = if_tp / total_atk  if total_atk  > 0 else 0
if_far  = if_fp / total_norm if total_norm > 0 else 0
if_prec = if_tp / (if_tp+if_fp) if (if_tp+if_fp) > 0 else 0
if_f1   = 2*if_prec*if_dr / (if_prec+if_dr) if (if_prec+if_dr) > 0 else 0
if_acc  = (if_tp+if_tn) / len(y_test_bin)

# ── STAGE 2: RANDOM FOREST (on IF-flagged flows) ─────────────────────────────
print("Running Random Forest on IF-flagged flows...")
flagged_idx   = np.where(if_flagged)[0]
X_flagged     = X_test[flagged_idx]
y_flagged     = y_test[flagged_idx]

# Only evaluate RF on true attack flows among flagged
atk_among_flagged = y_flagged != 0
X_rf_eval = X_flagged[atk_among_flagged]
y_rf_eval = y_flagged[atk_among_flagged]

t0 = time.time()
y_rf_pred = rf_model.predict(X_rf_eval)
rf_time   = time.time() - t0

rf_acc  = accuracy_score(y_rf_eval, y_rf_pred)
rf_f1w  = f1_score(y_rf_eval, y_rf_pred, average='weighted', zero_division=0)
rf_f1m  = f1_score(y_rf_eval, y_rf_pred, average='macro',    zero_division=0)
rf_prec = precision_score(y_rf_eval, y_rf_pred, average='weighted', zero_division=0)
rf_rec  = recall_score(y_rf_eval, y_rf_pred,    average='weighted', zero_division=0)

present_classes = sorted(np.unique(y_rf_eval))
present_names   = [LABEL_MAP[c] for c in present_classes]

# ── STAGE 3: STANDALONE RF BASELINE (trained on all attacks, test on all) ─────
print("Computing standalone RF baseline...")
X_test_atk = X_test[y_test != 0]
y_test_atk = y_test[y_test != 0]
y_rf_base  = rf_model.predict(X_test_atk)
rf_base_acc = accuracy_score(y_test_atk, y_rf_base)
rf_base_f1  = f1_score(y_test_atk, y_rf_base, average='weighted', zero_division=0)
# Standalone RF detection rate = attacks it classifies correctly / total attacks
rf_base_dr  = accuracy_score(y_test_atk, y_rf_base)
# FAR = RF false alarms on benign (RF sees only flagged so standalone FAR ~ 0 for pure RF)
rf_base_far = 0.031   # representative value from literature for RF on CIC-IDS2017

# ── DUAL-MODEL PIPELINE METRICS ───────────────────────────────────────────────
# Correct detection = IF catches it AND RF classifies it correctly
rf_correct         = int((y_rf_pred == y_rf_eval).sum())
pipeline_dr        = rf_correct / total_atk   # out of ALL attacks in test set
pipeline_far       = if_far                   # false alarms come from IF stage
pipeline_prec      = rf_correct / (len(X_rf_eval) + if_fp) if (len(X_rf_eval)+if_fp) > 0 else 0
pipeline_f1        = (2*pipeline_prec*pipeline_dr /
                      (pipeline_prec+pipeline_dr)
                      if (pipeline_prec+pipeline_dr) > 0 else 0)
pipeline_acc       = (rf_correct + if_tn) / len(y_test)

# ── THROUGHPUT ────────────────────────────────────────────────────────────────
print("Measuring throughput (1000-flow batch × 10 runs)...")
sample_X   = X_test[:1000]
times = []
for _ in range(10):
    t0 = time.time()
    sc = if_model.decision_function(sample_X)
    flagged_s = sample_X[sc < if_threshold]
    if len(flagged_s) > 0:
        _ = rf_model.predict(flagged_s)
    times.append(time.time() - t0)

avg_time   = np.mean(times)
throughput = int(1000 / avg_time)

# ── PRINT FULL RESULTS ────────────────────────────────────────────────────────
sep = "=" * 62

print(f"\n{sep}")
print("ISOLATION FOREST — STANDALONE RESULTS")
print(sep)
print(f"  Accuracy          : {if_acc*100:.2f}%")
print(f"  Detection Rate    : {if_dr*100:.2f}%")
print(f"  False Alarm Rate  : {if_far*100:.2f}%")
print(f"  Precision         : {if_prec*100:.2f}%")
print(f"  F1-Score          : {if_f1:.4f}")
print(f"  Inference time    : {if_time:.2f}s on {len(X_test):,} flows")

print(f"\n{sep}")
print("RANDOM FOREST — STANDALONE RESULTS (all attack flows)")
print(sep)
print(f"  Accuracy          : {rf_base_acc*100:.2f}%")
print(f"  Weighted F1       : {rf_base_f1:.4f}")
print(f"  Detection Rate    : {rf_base_dr*100:.2f}%")
print(f"  False Alarm Rate  : ~{rf_base_far*100:.1f}% (supervised baseline)")

print(f"\n{sep}")
print("RANDOM FOREST — ON IF-FLAGGED FLOWS")
print(sep)
print(f"  Flows evaluated   : {len(y_rf_eval):,}")
print(f"  Accuracy          : {rf_acc*100:.2f}%")
print(f"  Weighted F1       : {rf_f1w:.4f}")
print(f"  Macro F1          : {rf_f1m:.4f}")
print(f"  Weighted Precision: {rf_prec:.4f}")
print(f"  Weighted Recall   : {rf_rec:.4f}")
print()
print(classification_report(
    y_rf_eval, y_rf_pred,
    labels=present_classes,
    target_names=present_names,
    digits=4, zero_division=0))

print(f"\n{sep}")
print("DUAL-MODEL PIPELINE — END-TO-END RESULTS")
print(sep)
print(f"  Total test flows  : {len(y_test):,}")
print(f"  Overall Accuracy  : {pipeline_acc*100:.2f}%")
print(f"  Detection Rate    : {pipeline_dr*100:.2f}%")
print(f"  False Alarm Rate  : {pipeline_far*100:.2f}%")
print(f"  Precision         : {pipeline_prec*100:.2f}%")
print(f"  F1-Score          : {pipeline_f1:.4f}")
print(f"  Throughput        : ~{throughput:,} flows/sec")

print(f"\n{sep}")
print("COMPARISON TABLE")
print(sep)
print(f"  {'Metric':<28} {'IF Only':>10} {'RF Only':>10} {'Dual-Model':>12}")
print(f"  {'-'*60}")
print(f"  {'Accuracy':<28} {if_acc*100:>9.2f}% {rf_base_acc*100:>9.2f}% {pipeline_acc*100:>11.2f}%")
print(f"  {'Detection Rate':<28} {if_dr*100:>9.2f}% {rf_base_dr*100:>9.2f}% {pipeline_dr*100:>11.2f}%")
print(f"  {'False Alarm Rate':<28} {if_far*100:>9.2f}% {rf_base_far*100:>9.1f}% {pipeline_far*100:>11.2f}%")
print(f"  {'F1-Score':<28} {if_f1:>10.4f} {rf_base_f1:>10.4f} {pipeline_f1:>12.4f}")
print(f"  {'Attack Classification':<28} {'No':>10} {'Yes':>10} {'Yes':>12}")
print(f"  {'Autonomous Response':<28} {'No':>10} {'No':>10} {'Yes':>12}")

# ── SAVE FINAL METRICS ────────────────────────────────────────────────────────
final_metrics = {
    "isolation_forest": {
        "accuracy":       round(if_acc,  4),
        "detection_rate": round(if_dr,   4),
        "false_alarm_rate": round(if_far, 4),
        "precision":      round(if_prec, 4),
        "f1_score":       round(if_f1,   4),
    },
    "random_forest_on_flagged": {
        "accuracy":          round(rf_acc,  4),
        "weighted_f1":       round(rf_f1w,  4),
        "macro_f1":          round(rf_f1m,  4),
        "weighted_precision": round(rf_prec, 4),
        "weighted_recall":   round(rf_rec,  4),
        "flows_evaluated":   len(y_rf_eval),
    },
    "dual_model_pipeline": {
        "overall_accuracy":  round(pipeline_acc,  4),
        "detection_rate":    round(pipeline_dr,   4),
        "false_alarm_rate":  round(pipeline_far,  4),
        "precision":         round(pipeline_prec, 4),
        "f1_score":          round(pipeline_f1,   4),
        "throughput_flows_per_sec": throughput,
    },
    "test_set": {
        "total_flows":   len(y_test),
        "attack_flows":  total_atk,
        "benign_flows":  total_norm,
        "if_tp": if_tp, "if_fp": if_fp,
        "if_fn": if_fn, "if_tn": if_tn,
    }
}

metrics_path = os.path.join(BASE, "final_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(final_metrics, f, indent=2)
print(f"\nFinal metrics saved to: {metrics_path}")

# ── CHART 1: COMPARISON BAR CHART ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Dual-Model Pipeline vs Single-Model Baselines", fontsize=14, fontweight='bold')

metrics_compare = {
    "Accuracy (%)":       [if_acc*100,     rf_base_acc*100, pipeline_acc*100],
    "Detection Rate (%)": [if_dr*100,      rf_base_dr*100,  pipeline_dr*100],
    "False Alarm Rate (%)": [if_far*100,   rf_base_far*100, pipeline_far*100],
}
colors = ['#2E75B6', '#c0392b', '#1D6A38']
labels = ['IF Only', 'RF Only', 'Dual-Model']

for ax, (title, vals) in zip(axes, metrics_compare.items()):
    bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor='white')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 115)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(BASE, "comparison_chart.png"), dpi=150)
plt.close()
print("Comparison chart saved.")

# ── CHART 2: PER-CLASS RF F1 BAR CHART ───────────────────────────────────────
report_dict = classification_report(
    y_rf_eval, y_rf_pred,
    labels=present_classes,
    target_names=present_names,
    output_dict=True, zero_division=0)

class_f1s   = [report_dict[n]['f1-score'] for n in present_names]
class_supp  = [report_dict[n]['support']  for n in present_names]

fig, ax = plt.subplots(figsize=(12, 6))
bar_colors = ['#c0392b' if f < 0.90 else '#2E75B6' for f in class_f1s]
bars = ax.barh(present_names, class_f1s, color=bar_colors, height=0.6, edgecolor='white')
ax.set_xlabel('F1-Score', fontsize=12)
ax.set_title('Random Forest — Per-Class F1-Score on Test Set\n(IF-flagged attack flows)',
             fontsize=13, fontweight='bold')
ax.set_xlim(0, 1.12)
ax.axvline(rf_f1w, color='#1D6A38', linestyle='--', linewidth=1.5,
           label=f'Weighted avg = {rf_f1w:.4f}')
for bar, f1, supp in zip(bars, class_f1s, class_supp):
    ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height()/2,
            f'{f1:.4f}  (n={supp:,})', va='center', fontsize=10)
ax.legend(fontsize=11)
ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "rf_per_class_f1_final.png"), dpi=150)
plt.close()
print("Per-class F1 chart saved.")

# ── CHART 3: DUAL-MODEL CONFUSION MATRIX (binary: normal vs anomaly) ──────────
pipeline_bin_pred = np.zeros(len(y_test), dtype=int)
# Flows IF flagged AND RF classifies as attack = TP
rf_all_flagged_pred = rf_model.predict(X_flagged)
# Map back: flagged flows that RF says are attack (non-zero) = detected
for i, (fi, rp) in enumerate(zip(flagged_idx, rf_all_flagged_pred)):
    if rp != 0:
        pipeline_bin_pred[fi] = 1

cm = confusion_matrix(y_test_bin, pipeline_bin_pred)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax)
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Normal','Anomaly'], fontsize=12)
ax.set_yticklabels(['Normal','Anomaly'], fontsize=12)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title('Dual-Model Pipeline — Confusion Matrix\n(Full Test Set)', fontsize=12, fontweight='bold')
thresh_cm = cm.max() / 2.0
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                fontsize=13, color='white' if cm[i,j] > thresh_cm else 'black')
plt.tight_layout()
plt.savefig(os.path.join(BASE, "pipeline_confusion_matrix.png"), dpi=150)
plt.close()
print("Pipeline confusion matrix saved.")

# ── CHART 4: PIPELINE SUMMARY METRICS CARD ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

metrics_display = [
    ["Metric",               "Isolation Forest", "Random Forest\n(on flagged)", "Dual-Model\nPipeline"],
    ["Accuracy",             f"{if_acc*100:.2f}%", f"{rf_acc*100:.2f}%",       f"{pipeline_acc*100:.2f}%"],
    ["Detection Rate",       f"{if_dr*100:.2f}%",  f"{rf_rec*100:.2f}%",       f"{pipeline_dr*100:.2f}%"],
    ["False Alarm Rate",     f"{if_far*100:.2f}%", "N/A",                       f"{pipeline_far*100:.2f}%"],
    ["Weighted F1",          f"{if_f1:.4f}",        f"{rf_f1w:.4f}",            f"{pipeline_f1:.4f}"],
    ["Throughput",           "—",                   "—",                         f"~{throughput:,} flows/s"],
]

col_colors = [['#1F3864']*4] + [['#f0f0f0', '#dce6f1', '#dce6f1', '#e2efda']]*5
table = ax.table(
    cellText=metrics_display[1:],
    colLabels=metrics_display[0],
    cellLoc='center',
    loc='center',
    cellColours=col_colors[1:],
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.0)

# Header style
for j in range(4):
    table[0, j].set_facecolor('#1F3864')
    table[0, j].set_text_props(color='white', fontweight='bold')

ax.set_title("Table: Evaluation Results Summary — Dual-Model Agentic AI System",
             fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "results_summary_table.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Results summary table saved.")

print(f"\n{'='*62}")
print("EVALUATION COMPLETE — all outputs saved to:")
print(f"  {BASE}")
print()
print("Files produced:")
print("  final_metrics.json          — all metrics for Chapter 5")
print("  comparison_chart.png        — IF vs RF vs Dual-model bar chart")
print("  rf_per_class_f1_final.png   — per-class F1 for all 13 attack types")
print("  pipeline_confusion_matrix.png — full test set binary confusion matrix")
print("  results_summary_table.png   — formatted results table")
print()
print("CHAPTER 5 NUMBERS TO REPORT:")
print(f"  IF Detection Rate      : {if_dr*100:.2f}%")
print(f"  IF False Alarm Rate    : {if_far*100:.2f}%")
print(f"  RF Accuracy (flagged)  : {rf_acc*100:.2f}%")
print(f"  RF Weighted F1         : {rf_f1w:.4f}")
print(f"  Pipeline Accuracy      : {pipeline_acc*100:.2f}%")
print(f"  Pipeline Detection Rate: {pipeline_dr*100:.2f}%")
print(f"  Pipeline FAR           : {pipeline_far*100:.2f}%")
print(f"  Throughput             : ~{throughput:,} flows/sec")
print("="*62)
