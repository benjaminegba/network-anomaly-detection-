"""
agent.py
LangChain ReAct agent powered by Llama-3 8B (via Ollama) for autonomous
incident report generation. Loads the FAISS knowledge base built by
rag_pipeline.py and processes a sample of detected attack flows from the
test set, producing structured 5-field incident reports saved as JSON.
"""

import os
import json
import pickle
import numpy as np
from datetime import datetime, timezone

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE   = r"C:\Users\Damilola Egbadon\Documents\CIC Network Dataset\processed"
KB_DIR = os.path.join(BASE, "knowledge_base")
OUT    = os.path.join(BASE, "incident_reports")
os.makedirs(OUT, exist_ok=True)

# ── LABEL MAP ─────────────────────────────────────────────────────────────────
LABEL_MAP = {
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

SEVERITY_MAP = {
    "Heartbleed":               "CRITICAL",
    "Infiltration":             "CRITICAL",
    "Web Attack SQL Injection": "HIGH",
    "Web Attack BruteForce":    "HIGH",
    "Web Attack XSS":           "HIGH",
    "DDoS":                     "HIGH",
    "DoS GoldenEye":            "HIGH",
    "DoS Hulk":                 "HIGH",
    "DoS Slowhttptest":         "MEDIUM",
    "DoS slowloris":            "MEDIUM",
    "FTP-Patator":              "MEDIUM",
    "SSH-Patator":              "MEDIUM",
    "Bot":                      "MEDIUM",
    "PortScan":                 "LOW",
}

# ── LOAD FAISS KB ─────────────────────────────────────────────────────────────
print("Loading FAISS knowledge base...")
import faiss
from sentence_transformers import SentenceTransformer

index = faiss.read_index(os.path.join(KB_DIR, "faiss_index.bin"))
with open(os.path.join(KB_DIR, "chunks.pkl"),   "rb") as f:
    all_chunks = pickle.load(f)
with open(os.path.join(KB_DIR, "metadata.pkl"), "rb") as f:
    all_metadata = pickle.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
print(f"KB loaded: {index.ntotal} vectors")

# ── RAG RETRIEVAL FUNCTION ────────────────────────────────────────────────────
def retrieve_context(attack_label: str, k: int = 3) -> str:
    """Retrieve top-k relevant chunks for the given attack label."""
    query = f"network attack {attack_label} indicators response mitigation"
    q_vec = embedder.encode(
        [query], normalize_embeddings=True
    ).astype(np.float32)
    D, I = index.search(q_vec, k=k)
    chunks = []
    for idx, score in zip(I[0], D[0]):
        if idx >= 0:
            chunks.append(
                f"[Source: {all_metadata[idx]['attack_type']} | "
                f"Score: {score:.3f}]\n{all_chunks[idx]}"
            )
    return "\n\n".join(chunks)

# ── LOAD LANGCHAIN + OLLAMA ───────────────────────────────────────────────────
print("Initialising LangChain + Ollama (Llama-3)...")
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3",
    temperature=0.1,
    num_predict=600,
)

# ── TOOLS ─────────────────────────────────────────────────────────────────────
def retrieve_threat_intel(attack_label: str) -> str:
    """Retrieve threat intelligence for the specified attack type."""
    return retrieve_context(attack_label, k=3)



# ── DIRECT RAG + LLM PIPELINE ─────────────────────────────────────────────────
# We use a direct LLM call with RAG context rather than full ReAct loop
# to avoid formatting issues with Llama-3 tool-call parsing.

def generate_incident_report(attack_label: str, flow_idx: int) -> dict:
    """
    Generate a structured incident report for a detected attack flow.
    Uses RAG retrieval + direct LLM call for robustness on local hardware.
    """
    severity  = SEVERITY_MAP.get(attack_label, "MEDIUM")
    context   = retrieve_context(attack_label, k=3)
    timestamp = datetime.now(timezone.utc).isoformat()

    prompt_text = f"""You are a cybersecurity incident response AI.

A network anomaly detection system has flagged a flow as: {attack_label}
Severity level: {severity}

Relevant threat intelligence retrieved from the knowledge base:
---
{context}
---

Based on the above, produce a structured incident report as a valid JSON object
with exactly these 5 fields and no other text:

{{
  "threat_classification": "{attack_label}",
  "severity_score": "{severity}",
  "affected_network_segment": "<which part of the network is at risk>",
  "immediate_response": "<2-3 immediate actions the SOC team should take now>",
  "long_term_mitigation": "<2-3 strategic controls to prevent recurrence>"
}}

Respond with only the JSON object. No explanation, no markdown, no preamble."""

    response = llm.invoke(prompt_text)
    raw_text = response.content.strip()

    # Parse JSON from response
    try:
        # Strip markdown code fences if present
        clean = raw_text
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        report = json.loads(clean.strip())
    except json.JSONDecodeError:
        # Fallback: structured report from context without LLM JSON parse
        report = {
            "threat_classification": attack_label,
            "severity_score":        severity,
            "affected_network_segment": "Network perimeter / internal hosts",
            "immediate_response":    f"Isolate affected hosts. Block source IPs. "
                                     f"Capture traffic for forensic analysis.",
            "long_term_mitigation":  f"Deploy IDS signatures for {attack_label}. "
                                     f"Review firewall rules. Patch vulnerable services.",
            "note": "JSON parse fallback — LLM response stored in raw_llm_output",
            "raw_llm_output": raw_text[:500]
        }

    report["flow_index"]  = flow_idx
    report["timestamp"]   = timestamp
    report["pipeline"]    = "IsolationForest → RandomForest → RAG → LangChain/Llama3"

    return report

# ── LOAD TEST SET AND SELECT SAMPLE FLOWS ────────────────────────────────────
print("\nLoading test set for sample flow processing...")
X_test = np.load(os.path.join(BASE, "X_test.npy"))
y_test = np.load(os.path.join(BASE, "y_test.npy"))

with open(os.path.join(BASE, "if_threshold.pkl"), "rb") as f:
    if_threshold = pickle.load(f)
with open(os.path.join(BASE, "isolation_forest_model.pkl"), "rb") as f:
    if_model = pickle.load(f)
with open(os.path.join(BASE, "random_forest_model.pkl"), "rb") as f:
    rf_model = pickle.load(f)

# Score with IF
print("Running IF + RF pipeline on test set sample...")
scores_test  = if_model.decision_function(X_test)
if_flagged   = scores_test < if_threshold
flagged_idx  = np.where(if_flagged)[0]

# Pick one example per attack class present in flagged flows
y_flagged    = y_test[flagged_idx]
sample_flows = {}
for idx, label in zip(flagged_idx, y_flagged):
    if label != 0 and label not in sample_flows:
        sample_flows[label] = int(idx)
    if len(sample_flows) == 14:
        break

print(f"Selected {len(sample_flows)} sample flows (one per attack class)")

# RF classify
sample_indices = list(sample_flows.values())
X_sample       = X_test[sample_indices]
y_pred_labels  = rf_model.predict(X_sample)

print("\nRF predictions on sample flows:")
for i, (true_label, pred_label) in enumerate(
        zip(sample_flows.keys(), y_pred_labels)):
    true_name = LABEL_MAP.get(int(true_label), str(true_label))
    pred_name = LABEL_MAP.get(int(pred_label), str(pred_label))
    match     = "✓" if true_label == pred_label else "✗"
    print(f"  {match}  True: {true_name:<30}  Pred: {pred_name}")

# ── GENERATE INCIDENT REPORTS ─────────────────────────────────────────────────
print(f"\nGenerating incident reports via Llama-3 (this will take a few minutes)...")
print("=" * 60)

all_reports = []
for i, (true_label, flow_idx) in enumerate(sample_flows.items()):
    attack_name = LABEL_MAP.get(int(true_label), str(true_label))
    print(f"\n[{i+1}/{len(sample_flows)}] Generating report for: {attack_name}")

    report = generate_incident_report(attack_name, flow_idx)
    all_reports.append(report)

    # Print summary
    print(f"  Threat      : {report.get('threat_classification', 'N/A')}")
    print(f"  Severity    : {report.get('severity_score', 'N/A')}")
    print(f"  Segment     : {report.get('affected_network_segment', 'N/A')[:70]}")
    print(f"  Immediate   : {report.get('immediate_response', 'N/A')[:70]}...")
    print(f"  Long-term   : {report.get('long_term_mitigation', 'N/A')[:70]}...")

    # Save individual report
    fname = f"report_{i+1:02d}_{attack_name.replace(' ','_')}.json"
    with open(os.path.join(OUT, fname), "w") as f:
        json.dump(report, f, indent=2)

# Save all reports together
all_path = os.path.join(OUT, "all_incident_reports.json")
with open(all_path, "w") as f:
    json.dump(all_reports, f, indent=2)

print(f"\n{'='*60}")
print(f"All {len(all_reports)} incident reports saved to: {OUT}")
print(f"Combined report file: {all_path}")
print(f"\nNext step: run evaluate.py")
print("="*60)
