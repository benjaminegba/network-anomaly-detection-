"""
isolation_forest_v4.py
Loads the already-trained IF model and sweeps a much wider threshold range,
targeting maximum detection rate (recall) with acceptable false alarm rate.
No retraining needed — just re-threshold.
"""

import numpy as np
import pickle
import os
from sklearn.metrics import classification_report, f1_score, recall_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = r"C:\Users\Damilola Egbadon\Documents\CIC Network Dataset\processed"

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("Loading model and data...")
with open(os.path.join(BASE, "isolation_forest_model.pkl"), "rb") as f:
    clf = pickle.load(f)

X_val  = np.load(os.path.join(BASE, "X_val.npy"))
y_val  = np.load(os.path.join(BASE, "y_val.npy"))
X_test = np.load(os.path.join(BASE, "X_test.npy"))
y_test = np.load(os.path.join(BASE, "y_test.npy"))
X_train = np.load(os.path.join(BASE, "X_train.npy"))
y_train = np.load(os.path.join(BASE, "y_train.npy"))

y_val_bin  = (y_val  != 0).astype(int)
y_test_bin = (y_test != 0).astype(int)

# ── SCORE ─────────────────────────────────────────────────────────────────────
print("Computing scores...")
scores_val  = clf.decision_function(X_val)
scores_test = clf.decision_function(X_test)

print(f"\nVal score stats:")
print(f"  min={scores_val.min():.4f}  max={scores_val.max():.4f}  "
      f"mean={scores_val.mean():.4f}  median={np.median(scores_val):.4f}")

# Attack scores vs normal scores
atk_scores  = scores_val[y_val_bin == 1]
norm_scores = scores_val[y_val_bin == 0]
print(f"  Attack score mean  : {atk_scores.mean():.4f}")
print(f"  Normal score mean  : {norm_scores.mean():.4f}")

# ── WIDE SWEEP ────────────────────────────────────────────────────────────────
# Sweep every percentile 1..99 of the FULL val score distribution
# (not just the anomaly region) to find where detection rate jumps
print("\nSweeping ALL percentiles 1-99...")

thresholds = np.percentile(scores_val, np.arange(1, 100, 0.5))

results = []
for t in thresholds:
    preds = (scores_val < t).astype(int)
    n_flagged = preds.sum()
    tp = int(((preds==1) & (y_val_bin==1)).sum())
    fp = int(((preds==1) & (y_val_bin==0)).sum())
    fn = int(((preds==0) & (y_val_bin==1)).sum())
    tn = int(((preds==0) & (y_val_bin==0)).sum())
    total_atk  = int(y_val_bin.sum())
    total_norm = int((y_val_bin==0).sum())
    dr  = tp / total_atk  if total_atk  > 0 else 0
    far = fp / total_norm if total_norm > 0 else 0
    prec = tp / (tp+fp)   if (tp+fp)   > 0 else 0
    f1   = 2*prec*dr/(prec+dr) if (prec+dr) > 0 else 0
    results.append({'t': t, 'dr': dr, 'far': far, 'f1': f1,
                    'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn})

# Print detection rate landscape
print("\nDetection rate vs False Alarm Rate at key percentiles:")
print(f"{'Pct':>5}  {'Threshold':>10}  {'DetRate':>8}  {'FAR':>7}  {'F1':>7}  {'Flagged':>9}")
for i, pct in enumerate(np.arange(1, 100, 0.5)):
    r = results[i]
    if pct % 5 == 0 or pct in [1, 2, 3]:
        n = int(r['tp'] + r['fp'])
        print(f"{pct:>5.1f}  {r['t']:>10.4f}  "
              f"{r['dr']*100:>7.1f}%  {r['far']*100:>6.2f}%  "
              f"{r['f1']:>7.4f}  {n:>9,}")

# ── PICK BEST THRESHOLD ───────────────────────────────────────────────────────
# Strategy: maximise detection rate subject to FAR <= 15%
# (RF will clean up FPs at the next stage)
best = None
best_dr = 0
for r in results:
    if r['far'] <= 0.15 and r['dr'] > best_dr:
        best_dr = r['dr']
        best = r

if best is None:
    # Relax FAR constraint
    best = max(results, key=lambda r: r['dr'])

print(f"\n{'='*55}")
print(f"SELECTED THRESHOLD  : {best['t']:.4f}")
print(f"Val Detection Rate  : {best['dr']*100:.2f}%")
print(f"Val False Alarm Rate: {best['far']*100:.2f}%")
print(f"Val F1              : {best['f1']:.4f}")
print(f"{'='*55}")

# ── APPLY TO TEST SET ─────────────────────────────────────────────────────────
preds_test = (scores_test < best['t']).astype(int)
y_bin = y_test_bin
tp = int(((preds_test==1)&(y_bin==1)).sum())
fp = int(((preds_test==1)&(y_bin==0)).sum())
fn = int(((preds_test==0)&(y_bin==1)).sum())
tn = int(((preds_test==0)&(y_bin==0)).sum())
total_atk  = int(y_bin.sum())
total_norm = int((y_bin==0).sum())
dr_test  = tp/total_atk  if total_atk  > 0 else 0
far_test = fp/total_norm if total_norm > 0 else 0
prec     = tp/(tp+fp)    if (tp+fp)    > 0 else 0
f1_test  = 2*prec*dr_test/(prec+dr_test) if (prec+dr_test) > 0 else 0

print(f"\n{'='*55}")
print(f"ISOLATION FOREST — TEST SET (new threshold)")
print(f"{'='*55}")
print(f"Detection Rate (Recall): {dr_test*100:.2f}%")
print(f"False Alarm Rate       : {far_test*100:.2f}%")
print(f"Precision              : {prec*100:.2f}%")
print(f"F1-Score               : {f1_test:.4f}")
print(f"Flows flagged anomalous: {preds_test.sum():,} / {len(preds_test):,}")
print()
print(classification_report(y_bin, preds_test,
      target_names=["Normal","Anomaly"], digits=4))

# ── SAVE NEW THRESHOLD ────────────────────────────────────────────────────────
thresh_path = os.path.join(BASE, "if_threshold.pkl")
with open(thresh_path, "wb") as f:
    pickle.dump(best['t'], f)
print(f"New threshold saved to: {thresh_path}")

# ── SAVE ATTACK FLOWS FOR RF ──────────────────────────────────────────────────
scores_train = clf.decision_function(X_train)
train_flagged = scores_train < best['t']
y_train_bin   = (y_train != 0).astype(int)
attack_mask   = train_flagged & (y_train_bin == 1)

X_train_attacks = X_train[attack_mask]
y_train_attacks = y_train[attack_mask]
np.save(os.path.join(BASE, "X_train_attacks.npy"), X_train_attacks)
np.save(os.path.join(BASE, "y_train_attacks.npy"), y_train_attacks)

print(f"\nAttack flows for RF:")
print(f"  X_train_attacks: {X_train_attacks.shape}")
print(f"  y_train_attacks: {y_train_attacks.shape}")

# ── PLOT DR vs FAR CURVE ──────────────────────────────────────────────────────
dr_vals  = [r['dr']  for r in results]
far_vals = [r['far'] for r in results]
f1_vals  = [r['f1']  for r in results]
t_vals   = [r['t']   for r in results]

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(t_vals, [d*100 for d in dr_vals],  color='#2E75B6', lw=2, label='Detection Rate %')
ax.plot(t_vals, [f*100 for f in far_vals], color='#c0392b', lw=2,
        linestyle='--', label='False Alarm Rate %')
ax.plot(t_vals, [v*100 for v in f1_vals],  color='#1D6A38', lw=1.5,
        linestyle=':', label='F1 × 100')
ax.axvline(best['t'], color='#f59e0b', lw=2, linestyle='-.',
           label=f"Selected threshold = {best['t']:.4f}")
ax.set_xlabel('Anomaly Score Threshold', fontsize=12)
ax.set_ylabel('%', fontsize=12)
ax.set_title('IF Threshold Sweep — Detection Rate vs False Alarm Rate', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "if_threshold_curve_v4.png"), dpi=150)
plt.close()
print("Threshold curve saved.")
print("\nNow re-run random_forest.py to retrain RF on the new attack flows.")
