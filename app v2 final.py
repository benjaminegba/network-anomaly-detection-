"""
app_v2.py  —  Web dashboard for the Dual-Model Agentic AI System
Run with: python app_v2.py
Then open http://localhost:5000 in your browser.
"""

from flask import Flask, render_template_string, request, jsonify
import numpy as np
import pickle
import os
import json
from datetime import datetime, timezone

app = Flask(__name__)

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE   = r"C:\Users\Damilola Egbadon\Documents\CIC Network Dataset\processed"
KB_DIR = os.path.join(BASE, "knowledge_base")

LABEL_MAP = {
    0:"BENIGN", 1:"Bot", 2:"DDoS", 3:"DoS GoldenEye", 4:"DoS Hulk",
    5:"DoS Slowhttptest", 6:"DoS slowloris", 7:"FTP-Patator", 8:"Heartbleed",
    9:"Infiltration", 10:"PortScan", 11:"SSH-Patator",
    12:"Web Attack BruteForce", 13:"Web Attack SQL Injection", 14:"Web Attack XSS"
}
SEVERITY_MAP = {
    "Heartbleed":"CRITICAL","Infiltration":"CRITICAL",
    "Web Attack SQL Injection":"HIGH","Web Attack BruteForce":"HIGH",
    "Web Attack XSS":"HIGH","DDoS":"HIGH","DoS GoldenEye":"HIGH","DoS Hulk":"HIGH",
    "DoS Slowhttptest":"MEDIUM","DoS slowloris":"MEDIUM",
    "FTP-Patator":"MEDIUM","SSH-Patator":"MEDIUM","Bot":"MEDIUM",
    "PortScan":"LOW","BENIGN":"NONE"
}
FEATURE_NAMES = [
    "Packet Length Std","Avg Bwd Segment Size","Packet Length Variance",
    "Max Packet Length","Subflow Fwd Bytes","Bwd Packet Length Std",
    "Bwd Packet Length Max","Average Packet Size","Total Length of Bwd Packets",
    "Subflow Bwd Bytes","Total Length of Fwd Packets","Packet Length Mean",
    "Fwd Packet Length Max","Avg Fwd Segment Size","Bwd Packet Length Mean",
    "Fwd IAT Max","PSH Flag Count","Subflow Fwd Packets",
    "Fwd Header Length","Fwd IAT Std"
]

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
print("Loading models...")
with open(os.path.join(BASE,"isolation_forest_model.pkl"),"rb") as f: if_model=pickle.load(f)
with open(os.path.join(BASE,"if_threshold.pkl"),"rb") as f:           if_threshold=pickle.load(f)
with open(os.path.join(BASE,"random_forest_model.pkl"),"rb") as f:    rf_model=pickle.load(f)

import faiss
from sentence_transformers import SentenceTransformer
faiss_index = faiss.read_index(os.path.join(KB_DIR,"faiss_index.bin"))
with open(os.path.join(KB_DIR,"chunks.pkl"),"rb") as f:   all_chunks=pickle.load(f)
with open(os.path.join(KB_DIR,"metadata.pkl"),"rb") as f: all_metadata=pickle.load(f)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

X_test = np.load(os.path.join(BASE,"X_test.npy"))
y_test = np.load(os.path.join(BASE,"y_test.npy"))

from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3", temperature=0.1, num_predict=500)
print("All models loaded.")

# ── HELPERS ───────────────────────────────────────────────────────────────────
def retrieve_context(attack_label, k=3):
    q = f"network attack {attack_label} indicators response mitigation"
    qv = embedder.encode([q], normalize_embeddings=True).astype(np.float32)
    D,I = faiss_index.search(qv, k)
    return [{"text":all_chunks[i],"source":all_metadata[i]["attack_type"],"score":round(float(d),4)}
            for i,d in zip(I[0],D[0]) if i>=0]

def extract_json(raw_text):
    """Robustly pull a JSON object out of an LLM response, even with
    leading/trailing prose or partial markdown fences."""
    import re
    text = raw_text.strip()

    # 1. Straight parse (best case: clean JSON, nothing else)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Strip ```json ... ``` or ``` ... ``` fences if present
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 3. Brace-counting: find first '{' and its matching '}'
    #    (handles prose before/after the JSON object)
    start = text.find('{')
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    # 4. Last resort: greedy regex for the outermost {...}
    greedy_match = re.search(r"\{.*\}", text, re.DOTALL)
    if greedy_match:
        try:
            return json.loads(greedy_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def generate_report(attack_label, severity, chunks):
    ctx = "\n\n".join(c["text"] for c in chunks)
    prompt = f"""You are a cybersecurity incident response AI.

Detected attack: {attack_label}
Severity: {severity}

Threat intelligence:
---
{ctx}
---

Return ONLY a JSON object with exactly these 5 fields, no other text:
{{
  "threat_classification": "{attack_label}",
  "severity_score": "{severity}",
  "affected_network_segment": "<network component at risk>",
  "immediate_response": "<2-3 immediate SOC actions>",
  "long_term_mitigation": "<2-3 strategic controls>"
}}"""
    resp   = llm.invoke(prompt)
    raw    = resp.content.strip()
    parsed = extract_json(raw)
    if parsed is not None:
        return parsed

    print(f"[generate_report] Failed to parse LLM JSON output. Raw response:\n{raw}")
    return {
        "threat_classification": attack_label,
        "severity_score": severity,
        "affected_network_segment": "Network perimeter / internal hosts",
        "immediate_response": f"Isolate affected hosts. Block source IPs. Capture traffic.",
        "long_term_mitigation": f"Deploy IDS signatures for {attack_label}. Review firewall rules.",
        "_fallback_used": True
    }

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dual-Model Agentic AI — Network Anomaly Detection</title>
<style>
:root{
  --navy:#0D1B3E;--blue:#1F4E79;--lblue:#2E75B6;--teal:#0d6e5e;
  --red:#c0392b;--green:#1D6A38;--gold:#C9A84C;--amber:#f59e0b;
  --purple:#5c35a0;--gray:#f4f6f9;--border:#dde3ea;--text:#1a1a2e;--muted:#6b7280;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--gray);color:var(--text);}

header{
  background:var(--navy);padding:0 2rem;
  display:flex;align-items:center;justify-content:space-between;
  height:64px;border-bottom:3px solid var(--gold);
}
header h1{font-size:1rem;font-weight:700;color:#fff;letter-spacing:.04em;}
.header-tag{font-size:.72rem;color:var(--gold);letter-spacing:.08em;text-transform:uppercase;}
.status-dot{
  width:8px;height:8px;border-radius:50%;background:#22c55e;
  display:inline-block;margin-right:6px;animation:pulse 2s infinite;
}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

.main{display:grid;grid-template-columns:300px 1fr;gap:1.5rem;padding:1.5rem;max-width:1400px;margin:0 auto;}

.card{background:#fff;border-radius:10px;border:1px solid var(--border);overflow:hidden;margin-bottom:1rem;}
.card-header{
  padding:.75rem 1rem;background:var(--navy);color:#fff;
  font-size:.78rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
}
.card-body{padding:1rem;}

.btn{
  width:100%;padding:.75rem 1rem;border:none;border-radius:7px;
  font-size:.95rem;font-weight:700;cursor:pointer;
  transition:opacity .2s,transform .1s;letter-spacing:.02em;margin-bottom:.5rem;
}
.btn:hover{opacity:.88;}
.btn:active{transform:scale(.98);}
.btn:disabled{opacity:.4;cursor:not-allowed;}
.btn-primary{background:var(--lblue);color:#fff;}
.btn-run{background:var(--teal);color:#fff;}

.feature-grid{display:grid;grid-template-columns:1fr auto;gap:.25rem .5rem;font-size:.78rem;}
.feat-name{color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:.15rem 0;}
.feat-val{font-weight:700;font-family:'Courier New',monospace;text-align:right;padding:.15rem 0;color:var(--text);}

/* pipeline stages */
.stage{border-left:4px solid var(--border);padding-left:1rem;margin-bottom:1.2rem;transition:border-color .3s;}
.stage.active{border-color:var(--amber);}
.stage.done-ok{border-color:var(--green);}
.stage.done-skip{border-color:#cbd5e1;}

.stage-header{display:flex;align-items:center;gap:.75rem;margin-bottom:.5rem;}
.stage-num{
  width:30px;height:30px;border-radius:50%;background:var(--navy);color:#fff;
  font-size:.82rem;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0;
}
.stage-title{font-size:.92rem;font-weight:700;color:var(--navy);}
.stage-sub{font-size:.74rem;color:var(--muted);}
.stage-badge{
  margin-left:auto;padding:.2rem .7rem;border-radius:99px;
  font-size:.72rem;font-weight:700;letter-spacing:.04em;white-space:nowrap;
}
.bw{background:#f1f5f9;color:var(--muted);}
.br{background:#fef3c7;color:#92400e;}
.ba{background:#fee2e2;color:var(--red);}
.bn{background:#dcfce7;color:var(--green);}
.bc{background:#fee2e2;color:var(--red);}
.bh{background:#ffedd5;color:#c2410c;}
.bm{background:#fef9c3;color:#a16207;}
.bl{background:#f0fdf4;color:var(--green);}

.stage-content{font-size:.82rem;color:var(--muted);line-height:1.6;}
.metrics-row{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.4rem;}
.mpill{padding:.25rem .6rem;border-radius:6px;font-size:.78rem;font-weight:600;background:var(--gray);color:var(--navy);border:1px solid var(--border);}

.score-track{height:8px;border-radius:99px;background:var(--border);overflow:hidden;margin-top:.4rem;}
.score-fill{height:100%;border-radius:99px;transition:width .6s ease;}

.rag-chunk{background:var(--gray);border-radius:6px;padding:.6rem .75rem;font-size:.78rem;border-left:3px solid var(--purple);margin-bottom:.4rem;}
.rag-src{font-weight:700;color:var(--purple);font-size:.7rem;text-transform:uppercase;letter-spacing:.04em;margin-bottom:.2rem;}

.report-grid{display:grid;grid-template-columns:1fr 1fr;gap:.75rem;margin-top:.5rem;}
.rf{grid-column:span 1;}
.rff{grid-column:span 2;}
.rl{font-size:.7rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:.2rem;}
.rv{font-size:.85rem;color:var(--text);line-height:1.5;background:var(--gray);padding:.5rem .75rem;border-radius:6px;border:1px solid var(--border);}

/* ground truth reveal */
.reveal{
  border-radius:8px;padding:1rem;margin-top:.75rem;
  border:2px solid var(--border);
}
.reveal.correct{border-color:var(--green);background:#f0fdf4;}
.reveal.wrong{border-color:var(--amber);background:#fffbeb;}
.reveal.benign{border-color:var(--blue);background:#eff6ff;}
.reveal-title{font-weight:700;font-size:.9rem;margin-bottom:.4rem;}

.spinner{
  display:inline-block;width:13px;height:13px;
  border:2px solid rgba(0,0,0,.1);border-top-color:var(--lblue);
  border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle;margin-right:5px;
}
@keyframes spin{to{transform:rotate(360deg)}}

.err{background:#fee2e2;border:1px solid #fca5a5;border-radius:6px;padding:.75rem;color:var(--red);font-size:.82rem;}

.history-list{max-height:180px;overflow-y:auto;}
.hist-item{display:flex;align-items:center;gap:.5rem;padding:.4rem 0;border-bottom:1px solid var(--border);font-size:.8rem;}
.hdot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.htime{color:var(--muted);font-size:.72rem;margin-left:auto;}

.verdict{
  text-align:center;padding:1.2rem;border-radius:10px;margin-top:.5rem;
  font-size:1.1rem;font-weight:700;
}
.verdict.correct{background:#dcfce7;color:var(--green);}
.verdict.wrong{background:#fef9c3;color:#92400e;}
.verdict.benign{background:#dbeafe;color:var(--blue);}
</style>
</head>
<body>

<header>
  <div>
    <div class="header-tag">FUTA · IFT/22/9207 · Benjamin-Egbadon Osesimeokhian</div>
    <h1>Dual-Model Agentic AI System — Live Network Anomaly Detection</h1>
  </div>
  <div style="font-size:.82rem;color:#9ca3af">
    <span class="status-dot"></span>System Online &nbsp;|&nbsp; IF + RF + FAISS RAG + Llama-3 8B
  </div>
</header>

<div class="main">

  <!-- LEFT -->
  <div>
    <div class="card">
      <div class="card-header">⚡ Flow Input</div>
      <div class="card-body">
        <p style="font-size:.82rem;color:var(--muted);margin-bottom:.75rem;line-height:1.5">
          Click <b>Load Random Flow</b> to pull an unknown network flow from the test set.
          The system will analyse it — you won't know the answer until the pipeline finishes.
        </p>
        <button class="btn btn-primary" id="loadBtn" onclick="loadFlow()">
          🎲 Load Random Flow
        </button>
        <button class="btn btn-run" id="runBtn" onclick="runPipeline()" disabled>
          ▶ Run Full Pipeline
        </button>
      </div>
    </div>

    <div class="card" id="featureCard" style="display:none">
      <div class="card-header">📊 Flow Features (Top 20)</div>
      <div class="card-body">
        <p style="font-size:.75rem;color:var(--muted);margin-bottom:.6rem">
          These 20 values are all the system sees. No label. No hint.
        </p>
        <div class="feature-grid" id="featureGrid"></div>
        <div style="margin-top:.5rem;font-size:.74rem;color:var(--border);font-style:italic" id="flowIdx"></div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">🕓 History</div>
      <div class="card-body">
        <div class="history-list" id="histList">
          <div data-empty style="font-size:.8rem;color:var(--muted)">No flows analysed yet.</div>
        </div>
      </div>
    </div>
  </div>

  <!-- RIGHT -->
  <div>
    <div class="card">
      <div class="card-header">🔬 Pipeline Execution</div>
      <div class="card-body">

        <div class="stage" id="s1">
          <div class="stage-header">
            <div class="stage-num">1</div>
            <div><div class="stage-title">Isolation Forest</div>
            <div class="stage-sub">Unsupervised anomaly detection · 200 estimators · contamination=0.1688</div></div>
            <span class="stage-badge bw" id="b1">Waiting</span>
          </div>
          <div class="stage-content" id="c1">Load a flow to begin.</div>
        </div>

        <div style="height:1px;background:var(--border);margin:.8rem 0"></div>

        <div class="stage" id="s2">
          <div class="stage-header">
            <div class="stage-num">2</div>
            <div><div class="stage-title">Random Forest Classifier</div>
            <div class="stage-sub">Supervised multi-class · 200 estimators · 14 attack types</div></div>
            <span class="stage-badge bw" id="b2">Waiting</span>
          </div>
          <div class="stage-content" id="c2">Activated only if IF flags an anomaly.</div>
        </div>

        <div style="height:1px;background:var(--border);margin:.8rem 0"></div>

        <div class="stage" id="s3">
          <div class="stage-header">
            <div class="stage-num">3</div>
            <div><div class="stage-title">RAG Pipeline</div>
            <div class="stage-sub">FAISS · all-MiniLM-L6-v2 · cosine similarity · k=3</div></div>
            <span class="stage-badge bw" id="b3">Waiting</span>
          </div>
          <div class="stage-content" id="c3">Retrieves relevant threat intelligence.</div>
        </div>

        <div style="height:1px;background:var(--border);margin:.8rem 0"></div>

        <div class="stage" id="s4">
          <div class="stage-header">
            <div class="stage-num">4</div>
            <div><div class="stage-title">LangChain ReAct Agent</div>
            <div class="stage-sub">Llama-3 8B · Q4_K_M · fully offline · structured report</div></div>
            <span class="stage-badge bw" id="b4">Waiting</span>
          </div>
          <div class="stage-content" id="c4">Generates incident report autonomously.</div>
        </div>

        <!-- GROUND TRUTH REVEAL -->
        <div id="revealBox" style="display:none"></div>

      </div>
    </div>
  </div>
</div>

<script>
let currentFlow = null;
let currentStage = 0;

function badge(id,txt,cls){const b=document.getElementById('b'+id);b.textContent=txt;b.className='stage-badge '+cls;}
function stage(id,cls){document.getElementById('s'+id).className='stage '+cls;}
function content(id,html){document.getElementById('c'+id).innerHTML=html;}
function sevCls(s){return{CRITICAL:'bc',HIGH:'bh',MEDIUM:'bm',LOW:'bl',NONE:'bn'}[s]||'bw';}

async function loadFlow(){
  document.getElementById('loadBtn').disabled=true;
  document.getElementById('loadBtn').innerHTML='<span class="spinner"></span>Loading...';
  document.getElementById('revealBox').style.display='none';
  [1,2,3,4].forEach(i=>{
    badge(i,'Waiting','bw'); stage(i,'');
    content(i,i===1?'Click Run Pipeline to analyse this flow.':
               i===2?'Activated only if IF flags an anomaly.':
               i===3?'Retrieves relevant threat intelligence.':
               'Generates incident report autonomously.');
  });

  try{
    const r=await fetch('/api/random_flow',{method:'POST'});
    const d=await r.json();
    if(d.error){throw new Error(d.error);}
    currentFlow=d;

    const g=document.getElementById('featureGrid');
    g.innerHTML='';
    d.features.forEach((v,i)=>{
      g.innerHTML+=`<div class="feat-name" title="${d.feature_names[i]}">${d.feature_names[i]}</div>
                    <div class="feat-val">${v.toFixed(4)}</div>`;
    });
    document.getElementById('flowIdx').textContent='Flow index: '+d.flow_index+' · True label hidden until pipeline completes';
    document.getElementById('featureCard').style.display='block';
    document.getElementById('runBtn').disabled=false;
  }catch(e){
    alert('Error: '+e.message);
  }
  document.getElementById('loadBtn').disabled=false;
  document.getElementById('loadBtn').innerHTML='🎲 Load Random Flow';
}

async function runPipeline(){
  if(!currentFlow)return;
  document.getElementById('runBtn').disabled=true;
  document.getElementById('runBtn').innerHTML='<span class="spinner"></span>Analysing...';
  document.getElementById('revealBox').style.display='none';

  try{
    // ── STAGE 1: IF ────────────────────────────────────────────────
    currentStage=1;
    stage(1,'active'); badge(1,'Running…','br');
    content(1,'<span class="spinner"></span>Isolation Forest scoring flow...');

    const r1=await fetch('/api/run_if',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({flow:currentFlow.features})});
    const d1=await r1.json();
    if(d1.error) throw new Error(d1.error);

    const scoreNorm=Math.max(0,Math.min(1,(d1.score-(-0.30))/(0.17-(-0.30))));
    const anomPct=Math.round((1-scoreNorm)*100);
    const isAnom=d1.is_anomalous;

    stage(1,isAnom?'done-ok':'done-ok');
    badge(1,isAnom?'ANOMALY':'NORMAL',isAnom?'ba':'bn');
    content(1,`
      <div class="metrics-row">
        <span class="mpill">IF Score: ${d1.score.toFixed(4)}</span>
        <span class="mpill">Threshold: ${d1.threshold.toFixed(4)}</span>
        <span class="mpill">Anomaly likelihood: ${anomPct}%</span>
      </div>
      <div style="font-size:.75rem;color:var(--muted);margin:.3rem 0">Anomaly likelihood</div>
      <div class="score-track">
        <div class="score-fill" style="width:${anomPct}%;background:${anomPct>60?'var(--red)':anomPct>35?'var(--amber)':'var(--green)'}"></div>
      </div>
      <div style="margin-top:.5rem;font-size:.82rem">
        ${isAnom
          ? '⚠️ Scored below threshold — <b>flagged as anomalous</b>. Passing to Random Forest classifier.'
          : '✅ Scored above threshold — <b>classified as normal traffic</b>. Pipeline stops here.'}
      </div>`);

    if(!isAnom){
      [2,3,4].forEach(i=>{stage(i,'done-skip');badge(i,'Skipped','bw');
        content(i,'Not activated — flow was normal.');});
      showReveal('BENIGN', null, currentFlow.true_label_name);
      addHistory('BENIGN', 'NONE', currentFlow.true_label_name);
      done(); return;
    }

    // ── STAGE 2: RF ────────────────────────────────────────────────
    currentStage=2;
    stage(2,'active'); badge(2,'Running…','br');
    content(2,'<span class="spinner"></span>Random Forest classifying attack type...');

    const r2=await fetch('/api/run_rf',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({flow:currentFlow.features})});
    const d2=await r2.json();
    if(d2.error) throw new Error(d2.error);

    stage(2,'done-ok'); badge(2,d2.predicted_label,'ba');
    const probs=d2.top_probabilities.map(([l,p])=>`<span class="mpill">${l}: ${(p*100).toFixed(1)}%</span>`).join('');
    content(2,`
      <div style="margin-bottom:.4rem">
        <b>Predicted Attack Type:</b>
        <span style="font-size:1.05rem;font-weight:700;color:var(--red);margin-left:.5rem">${d2.predicted_label}</span>
      </div>
      <div class="metrics-row">${probs}</div>
      <div style="margin-top:.4rem;font-size:.8rem;color:var(--muted)">
        Confidence: ${(d2.confidence*100).toFixed(2)}% across 200 decision trees.
      </div>`);

    // ── STAGE 3: RAG ───────────────────────────────────────────────
    currentStage=3;
    stage(3,'active'); badge(3,'Retrieving…','br');
    content(3,'<span class="spinner"></span>Querying FAISS knowledge base...');

    const r3=await fetch('/api/run_rag',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({attack_label:d2.predicted_label})});
    const d3=await r3.json();
    if(d3.error) throw new Error(d3.error);

    stage(3,'done-ok'); badge(3,'Retrieved','bn');
    const chunks=d3.chunks.map(c=>`
      <div class="rag-chunk">
        <div class="rag-src">${c.source} · score: ${c.score}</div>
        <div>${c.text.substring(0,180)}…</div>
      </div>`).join('');
    content(3,`
      <div style="font-size:.82rem;margin-bottom:.4rem">
        Retrieved <b>${d3.chunks.length}</b> chunks for: <b>${d2.predicted_label}</b>
      </div>${chunks}`);

    // ── STAGE 4: AGENT ─────────────────────────────────────────────
    currentStage=4;
    stage(4,'active'); badge(4,'Generating…','br');
    content(4,'<span class="spinner"></span>Llama-3 8B generating incident report — please wait (~20s)...');

    const r4=await fetch('/api/run_agent',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({attack_label:d2.predicted_label,chunks:d3.chunks})});
    const d4=await r4.json();
    if(d4.error) throw new Error(d4.error);

    const sev=d4.report.severity_score||'MEDIUM';
    stage(4,'done-ok'); badge(4,sev,sevCls(sev));
    content(4,`
      <div class="report-grid">
        <div class="rf"><div class="rl">Threat Classification</div>
          <div class="rv" style="font-weight:700;color:var(--red)">${d4.report.threat_classification||'—'}</div></div>
        <div class="rf"><div class="rl">Severity Score</div>
          <div class="rv" style="font-weight:700">${sev}</div></div>
        <div class="rff"><div class="rl">Affected Network Segment</div>
          <div class="rv">${d4.report.affected_network_segment||'—'}</div></div>
        <div class="rff"><div class="rl">Immediate Response</div>
          <div class="rv">${d4.report.immediate_response||'—'}</div></div>
        <div class="rff"><div class="rl">Long-Term Mitigation</div>
          <div class="rv">${d4.report.long_term_mitigation||'—'}</div></div>
      </div>
      <div style="margin-top:.6rem;font-size:.74rem;color:var(--muted)">
        Generated ${d4.timestamp} · IF → RF → FAISS RAG → Llama-3 8B
      </div>`);

    showReveal(d2.predicted_label, sev, currentFlow.true_label_name);
    addHistory(d2.predicted_label, sev, currentFlow.true_label_name);

  }catch(e){
    const failedStage = currentStage || 1;
    stage(failedStage,''); badge(failedStage,'Error','ba');
    content(failedStage,`<div class="err">Error: ${e.message}</div>`);
    console.error(e);
  }
  done();
}

function showReveal(predicted, severity, trueLabel){
  const box=document.getElementById('revealBox');
  box.style.display='block';

  const isCorrect = predicted===trueLabel;
  const isBenign  = predicted==='BENIGN';

  let cls, icon, title, body;
  if(isBenign && trueLabel==='BENIGN'){
    cls='benign'; icon='✅'; title='Correct — Benign Traffic';
    body=`The system correctly identified this as normal (BENIGN) traffic. No attack present.`;
  } else if(isBenign && trueLabel!=='BENIGN'){
    cls='wrong'; icon='⚠️'; title='Missed Detection';
    body=`The system flagged this as normal but the true label was <b>${trueLabel}</b>. This is a false negative — a known limitation of Isolation Forest on this dataset.`;
  } else if(isCorrect){
    cls='correct'; icon='✅'; title='Correct Detection & Classification';
    body=`System predicted: <b>${predicted}</b> · True label: <b>${trueLabel}</b><br>
    The dual-model pipeline correctly detected and classified this attack.`;
  } else {
    cls='wrong'; icon='⚠️'; title='Misclassification';
    body=`System predicted: <b>${predicted}</b> · True label: <b>${trueLabel}</b><br>
    The attack was detected but misclassified. This occurs with minority classes.`;
  }

  box.innerHTML=`
    <div class="reveal ${cls}">
      <div class="reveal-title">${icon} Ground Truth Reveal</div>
      <div style="font-size:.85rem;line-height:1.6">${body}</div>
    </div>`;
}

function addHistory(predicted, severity, trueLabel){
  const list=document.getElementById('histList');
  if(list.querySelector('[data-empty]')) list.innerHTML='';
  const correct=predicted===trueLabel;
  const color=correct?'var(--green)':'var(--amber)';
  const sevColors={CRITICAL:'#c0392b',HIGH:'#ea580c',MEDIUM:'#ca8a04',LOW:'#16a34a',NONE:'#2563eb'};
  const t=new Date().toLocaleTimeString();
  list.insertAdjacentHTML('afterbegin',`
    <div class="hist-item">
      <div class="hdot" style="background:${sevColors[severity]||'#555'}"></div>
      <span>${predicted}</span>
      <span style="font-size:.7rem;color:${correct?'var(--green)':'var(--amber)'};font-weight:600">
        ${correct?'✓':'~'} ${trueLabel}
      </span>
      <span class="htime">${t}</span>
    </div>`);
}

function done(){
  document.getElementById('runBtn').disabled=false;
  document.getElementById('runBtn').innerHTML='▶ Run Full Pipeline';
}
</script>
</body>
</html>"""

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/random_flow', methods=['POST'])
def api_random_flow():
    try:
        idx        = int(np.random.randint(0, len(X_test)))
        flow       = X_test[idx]
        true_label = int(y_test[idx])
        return jsonify({
            'features':        flow.tolist(),
            'feature_names':   FEATURE_NAMES,
            'true_label':      true_label,
            'true_label_name': LABEL_MAP.get(true_label, str(true_label)),
            'flow_index':      idx
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/run_if', methods=['POST'])
def api_run_if():
    try:
        flow  = np.array(request.json['flow']).reshape(1,-1)
        score = float(if_model.decision_function(flow)[0])
        return jsonify({'score':score,'threshold':float(if_threshold),'is_anomalous':bool(score<if_threshold)})
    except Exception as e:
        return jsonify({'error':str(e)}), 500

@app.route('/api/run_rf', methods=['POST'])
def api_run_rf():
    try:
        flow    = np.array(request.json['flow']).reshape(1,-1)
        pred    = int(rf_model.predict(flow)[0])
        proba   = rf_model.predict_proba(flow)[0]
        classes = rf_model.classes_
        top     = np.argsort(proba)[::-1][:3]
        return jsonify({
            'predicted_class':   pred,
            'predicted_label':   LABEL_MAP.get(pred,str(pred)),
            'confidence':        float(proba[np.where(classes==pred)[0][0]]),
            'top_probabilities': [(LABEL_MAP.get(int(classes[i]),str(classes[i])),float(proba[i])) for i in top]
        })
    except Exception as e:
        return jsonify({'error':str(e)}), 500

@app.route('/api/run_rag', methods=['POST'])
def api_run_rag():
    try:
        chunks = retrieve_context(request.json['attack_label'], k=3)
        return jsonify({'chunks':chunks})
    except Exception as e:
        return jsonify({'error':str(e)}), 500

@app.route('/api/run_agent', methods=['POST'])
def api_run_agent():
    try:
        attack = request.json['attack_label']
        chunks = request.json['chunks']
        sev    = SEVERITY_MAP.get(attack,'MEDIUM')
        report = generate_report(attack, sev, chunks)
        return jsonify({
            'report':    report,
            'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        })
    except Exception as e:
        return jsonify({'error':str(e)}), 500

if __name__=='__main__':
    print("\n"+"="*50)
    print("Dashboard: http://localhost:5000")
    print("Ctrl+C to stop.")
    print("="*50+"\n")
    app.run(debug=False, port=5000)
