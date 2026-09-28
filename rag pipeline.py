"""
rag_pipeline.py
Builds a FAISS vector knowledge base from threat intelligence documents
covering all 14 CIC-IDS2017 attack categories.
Run once. Saves the FAISS index and chunk metadata for agent.py to load.
"""

import os
import json
import pickle
import numpy as np

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE     = r"C:\Users\Damilola Egbadon\Documents\CIC Network Dataset\processed"
KB_DIR   = os.path.join(BASE, "knowledge_base")
os.makedirs(KB_DIR, exist_ok=True)

# ── THREAT INTEL DOCUMENTS ────────────────────────────────────────────────────
# One document per attack category. Each contains description, indicators,
# and recommended response — used by the RAG pipeline at inference time.

THREAT_DOCS = {
    "Bot": """
Bot malware establishes persistent command-and-control (C2) communication channels
between infected hosts and attacker-controlled servers. Infected machines (bots) form
botnets used for spam distribution, DDoS amplification, credential harvesting, and
cryptocurrency mining. Network indicators include periodic outbound beaconing to
external IPs on non-standard ports, encrypted C2 traffic disguised as HTTP/HTTPS,
low-and-slow data exfiltration, and unusual DNS queries to dynamically generated
domains (DGA). Packet length patterns are typically short and regular due to heartbeat
traffic. Detection relies on behavioural baselining — identifying hosts with abnormal
outbound connection frequency, unexpected geographic destinations, or traffic at
unusual hours. Immediate response: isolate the infected host, block C2 IP ranges at
the perimeter firewall, capture network traffic for forensic analysis, and perform
full malware scan. Long-term: implement DNS sinkholing, deploy endpoint detection
and response (EDR), and enforce application whitelisting.
""",

    "DDoS": """
Distributed Denial of Service (DDoS) attacks overwhelm target systems by flooding
them with traffic from multiple sources simultaneously, exhausting network bandwidth,
CPU, or memory resources. Common variants include UDP floods, TCP SYN floods, HTTP
floods, and amplification attacks using DNS or NTP reflection. CIC-IDS2017 DDoS
flows are characterised by extremely high packet rates, small packet sizes, single
destination IP with many source IPs, and very short inter-arrival times. The attack
source IPs are typically spoofed or distributed across botnets. Network anomaly
indicators include sudden spikes in inbound traffic volume exceeding 10x baseline,
high proportion of incomplete TCP handshakes (SYN without ACK), and uniform packet
sizes suggesting automation. Immediate response: activate upstream scrubbing service
or null-route the targeted IP, apply rate limiting and access control lists (ACLs)
at border routers, contact ISP for upstream filtering. Long-term: deploy dedicated
DDoS mitigation appliance, implement anycast routing, and establish DDoS response
runbooks and ISP escalation procedures.
""",

    "DoS GoldenEye": """
DoS GoldenEye is a Layer 7 HTTP Denial of Service tool targeting web servers by
sending large numbers of HTTP GET and POST requests with randomised headers and
keep-alive connections to exhaust web server thread pools and connection tables.
Unlike volumetric DDoS, GoldenEye operates at relatively low packet rates, making
it harder to detect via traffic volume thresholds alone. Network flow indicators
include high numbers of long-duration HTTP connections from few source IPs, large
forward packet lengths indicating full HTTP requests, high PSH flag counts, and
low backward packet counts (server unable to respond). The attack targets Apache,
IIS, and Nginx servers directly. Immediate response: block offending source IPs at
the web application firewall (WAF), implement connection rate limiting per source IP,
enable CAPTCHA challenges for HTTP endpoints, restart affected web services.
Long-term: deploy a WAF with Layer 7 DDoS protection, implement connection timeout
policies, use a CDN with built-in DDoS protection.
""",

    "DoS Hulk": """
DoS Hulk is a Python-based HTTP flooding tool that generates unique randomised URLs
and HTTP headers for each request to bypass caching layers and overwhelm web servers.
It creates a high volume of GET requests with randomised User-Agent strings and
Referer headers, making signature-based detection difficult. Network indicators in
CIC-IDS2017 include high average packet sizes (full HTTP headers), high subflow
forward byte counts, very high flow rates from limited source IPs, and elevated
average forward segment sizes. The randomisation of headers means each request looks
different at the application layer. Immediate response: implement IP-based rate
limiting at the load balancer, deploy WAF rules detecting randomised header patterns,
temporarily serve cached static responses, and block source IP ranges. Long-term:
implement request throttling and challenge-response mechanisms, deploy HTTP/2 push
protection, and configure auto-scaling to absorb legitimate traffic spikes.
""",

    "DoS Slowhttptest": """
Slow HTTP attacks (Slowloris, Slow POST, Slow Read) exploit the HTTP protocol by
opening many connections to a web server and sending requests as slowly as possible
to keep connections open without triggering timeout mechanisms. Slowhttptest is a
tool implementing multiple slow attack variants. Network flow indicators include
extremely long flow durations, very low packet rates, small packet sizes, high
connection counts from few source IPs, and minimal backward traffic (server waiting
for client to complete request). The attack exhausts server connection pools without
generating high traffic volumes, making it stealthy. In CIC-IDS2017, flows show
very low average packet length with anomalously long inter-arrival times. Immediate
response: configure web server minimum data rate requirements (e.g. Apache
RequestReadTimeout), set aggressive connection timeout values, block slow-sending
IPs. Long-term: deploy a reverse proxy with connection management, implement
connection state tracking at the firewall.
""",

    "DoS slowloris": """
Slowloris is a low-and-slow HTTP DoS attack tool that opens multiple connections
to a web server and keeps them alive indefinitely by sending partial HTTP request
headers slowly, never completing the request. This exhausts the maximum concurrent
connection limit of the server without consuming significant bandwidth. Network
indicators include many long-lived TCP connections from the same source IP, very
infrequent packet arrival (one header fragment every 15-20 seconds), small packet
sizes, high PSH flag counts, and minimal server response traffic. The attack is
particularly effective against Apache web servers with default configurations.
Flow duration in CIC-IDS2017 Slowloris samples is unusually long compared to
legitimate web traffic. Immediate response: enable mod_reqtimeout in Apache,
configure server maximum connection limits per IP, implement connection rate
limiting. Long-term: migrate to event-driven web servers (Nginx, Node.js) which
are inherently resistant to Slowloris, deploy reverse proxy with connection pooling.
""",

    "FTP-Patator": """
FTP-Patator is a brute-force credential attack tool targeting FTP servers by
systematically attempting username and password combinations from wordlists.
The attack generates high volumes of FTP authentication attempts, with each
failed attempt resulting in a short connection with standard FTP error responses.
Network indicators include high frequency of short FTP connections (port 21) from
single source IPs, repeated failed authentication responses (FTP 530 error code),
uniform inter-arrival times (automated tool pacing), small and consistent packet
sizes, and high connection-to-transfer ratio (many logins, few successful data
transfers). CIC-IDS2017 FTP-Patator flows show distinctive subflow patterns with
high packet counts and low byte counts per flow. Immediate response: block source
IP after N failed attempts (fail2ban), enforce account lockout policies, disable
FTP and migrate to SFTP. Long-term: implement multi-factor authentication, replace
FTP with SFTP or FTPS, deploy honeypot FTP accounts to detect scanning activity.
""",

    "Heartbleed": """
Heartbleed (CVE-2014-0160) is a critical vulnerability in the OpenSSL cryptographic
library affecting the TLS/DTLS heartbeat extension. An attacker sends a malformed
heartbeat request with a payload length larger than the actual payload, causing the
server to return up to 64KB of arbitrary server memory contents per request, leaking
private keys, session tokens, passwords, and other sensitive data without leaving
any server-side log entries. Network indicators include TLS heartbeat requests on
port 443 with anomalous payload length fields, repeated short TLS connections
targeting the same server, and slightly larger-than-expected TLS response packets.
The attack is particularly dangerous as it leaves no trace in application logs.
Immediate response: immediately patch OpenSSL to version 1.0.1g or later, revoke
and reissue all SSL/TLS certificates, invalidate all active session tokens, force
password resets for all users. Long-term: implement TLS certificate transparency
monitoring, deploy intrusion detection signatures for CVE-2014-0160 patterns,
maintain a current vulnerability management programme.
""",

    "Infiltration": """
Network infiltration attacks involve adversaries who have already gained initial
access to a network and are conducting lateral movement, reconnaissance, or data
exfiltration. In CIC-IDS2017, infiltration flows represent internal network scanning,
exploitation of Dropbox vulnerabilities, and backdoor communication. Network
indicators include unusual internal-to-internal scanning traffic, connections to
cloud storage services from unexpected internal hosts, anomalous outbound data
volumes during off-hours, new listening ports appearing on internal hosts, and
internal hosts communicating with external IPs never seen before. These flows are
difficult to detect because they blend with legitimate internal traffic. Immediate
response: isolate suspected compromised hosts, capture full packet data for forensic
analysis, review authentication logs for lateral movement indicators, reset
credentials for affected accounts. Long-term: implement network segmentation and
zero-trust architecture, deploy user and entity behaviour analytics (UEBA), enforce
least-privilege access controls.
""",

    "PortScan": """
Port scanning is a reconnaissance technique used by attackers to discover open
network ports and identify running services on target hosts as a precursor to
exploitation. Common scan types include TCP SYN scan (half-open), TCP Connect scan,
UDP scan, and service version detection. Network flow indicators include high numbers
of connections from single source IP to single destination IP on many different
destination ports, very short flow durations (closed ports reject immediately), high
ratio of RST packets to SYN packets, and low byte counts per flow. In CIC-IDS2017,
PortScan flows show distinctive patterns of many short flows with consistent source
IP. Nmap is the most commonly used port scanning tool. Immediate response: block
scanning source IP at perimeter firewall, alert security operations, review firewall
rules to ensure unnecessary ports are closed. Long-term: deploy port scan detection
signatures in IDS/IPS, implement network access control (NAC), conduct regular
internal port scanning to identify unauthorised services.
""",

    "SSH-Patator": """
SSH-Patator is a brute-force credential attack tool targeting SSH servers (port 22)
by attempting large numbers of username and password combinations from wordlists.
Similar to FTP-Patator in methodology but targeting the SSH protocol. Network
indicators include high-frequency SSH connection attempts from single source IPs,
repeated SSH authentication failures in server logs, uniform inter-arrival times
between connection attempts (automated pacing), and short flow durations with small
packet counts (failed authentication terminates quickly). CIC-IDS2017 SSH-Patator
flows show high packet counts with low byte counts and distinctive subflow patterns.
The tool can also perform distributed attacks using multiple source IPs. Immediate
response: block source IP after failed authentication threshold (fail2ban), disable
password authentication and enforce SSH key-based authentication, change SSH port
from default 22 if not already done. Long-term: implement SSH certificate-based
authentication, deploy a privileged access management (PAM) solution, monitor for
credential stuffing from breach databases.
""",

    "Web Attack BruteForce": """
Web application brute-force attacks target HTTP-based authentication mechanisms by
systematically attempting username and password combinations against login forms,
basic authentication endpoints, or API authentication. Tools include Hydra, Burp
Suite Intruder, and custom scripts. Network indicators include high volume of HTTP
POST requests to login endpoints from few source IPs, consistent request timing
(automation signature), identical request sizes per attempt, server returning
HTTP 401 or 302 redirect responses repeatedly, and minimal successful response
content (failed login page). In CIC-IDS2017, web brute-force flows show high
forward packet counts, elevated PSH flag counts, and distinctive request/response
size patterns. Immediate response: implement account lockout after N failed attempts,
block offending IP at WAF, enable CAPTCHA on login endpoints, alert on authentication
anomalies. Long-term: enforce multi-factor authentication (MFA), deploy adaptive
authentication, implement CAPTCHA on all public-facing authentication endpoints.
""",

    "Web Attack SQL Injection": """
SQL injection attacks exploit vulnerabilities in web application database query
construction by inserting malicious SQL code into user-supplied input fields,
potentially allowing unauthorised data access, modification, or deletion. Attack
patterns include classic SQLi (WHERE clause manipulation), blind SQLi (boolean-based,
time-based), error-based SQLi, and UNION-based extraction. Network indicators include
HTTP requests containing SQL keywords (SELECT, UNION, DROP, INSERT, OR, AND) in
URL parameters or POST body, abnormally long URL query strings, unusual characters
(single quotes, semicolons, comment sequences --), server error responses (HTTP 500),
and unusually large response bodies indicating data extraction. In CIC-IDS2017, SQL
injection flows show distinctive forward packet length patterns. Immediate response:
block malicious requests at WAF using SQL injection signatures, patch the vulnerable
application input validation, audit database access logs for unauthorised queries.
Long-term: implement parameterised queries and prepared statements throughout the
application, deploy a WAF with OWASP ModSecurity ruleset, conduct regular penetration
testing and code review.
""",

    "Web Attack XSS": """
Cross-Site Scripting (XSS) attacks inject malicious client-side scripts (typically
JavaScript) into web pages viewed by other users. Reflected XSS delivers the payload
through a URL parameter; stored XSS persists the payload in the server database;
DOM-based XSS manipulates client-side JavaScript. Network indicators include HTTP
requests containing JavaScript tags (<script>, onerror=, onload=), HTML encoding of
script characters in URL parameters (%3Cscript%3E), unusual URL lengths, and HTTP
responses containing injected script content in the body. In CIC-IDS2017, XSS flows
show characteristic HTTP request patterns with script-related character sequences.
Successful XSS can lead to session hijacking, credential theft, defacement, and
drive-by malware distribution. Immediate response: identify and remove stored XSS
payloads from the database, invalidate active user sessions, block attack source IP
at WAF. Long-term: implement Content Security Policy (CSP) headers, enforce output
encoding in all web application templates, deploy WAF with XSS detection rules,
conduct regular security code reviews focusing on input validation and output encoding.
"""
}

# ── CHUNK DOCUMENTS ───────────────────────────────────────────────────────────
def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

print("Chunking threat intelligence documents...")
all_chunks = []
all_metadata = []

for attack_type, doc_text in THREAT_DOCS.items():
    chunks = chunk_text(doc_text.strip(), chunk_size=120, overlap=20)
    for j, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        all_metadata.append({
            "attack_type": attack_type,
            "chunk_id":    j,
            "source":      f"ThreatIntel_{attack_type}"
        })

print(f"Total chunks: {len(all_chunks)} across {len(THREAT_DOCS)} attack categories")

# ── EMBED WITH SENTENCE TRANSFORMER ──────────────────────────────────────────
print("\nLoading sentence transformer model (all-MiniLM-L6-v2)...")
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding chunks (this may take 1-2 minutes)...")
embeddings = embedder.encode(
    all_chunks,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True   # needed for cosine similarity via inner product
)
print(f"Embeddings shape: {embeddings.shape}")   # (N, 384)

# ── BUILD FAISS INDEX ─────────────────────────────────────────────────────────
print("\nBuilding FAISS index...")
import faiss

dim = embeddings.shape[1]   # 384
index = faiss.IndexFlatIP(dim)   # Inner product = cosine similarity (normalised)
index.add(embeddings.astype(np.float32))
print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

# ── SAVE ─────────────────────────────────────────────────────────────────────
index_path    = os.path.join(KB_DIR, "faiss_index.bin")
chunks_path   = os.path.join(KB_DIR, "chunks.pkl")
metadata_path = os.path.join(KB_DIR, "metadata.pkl")
embedder_path = os.path.join(KB_DIR, "embedder_name.txt")

faiss.write_index(index, index_path)
with open(chunks_path,   "wb") as f: pickle.dump(all_chunks,    f)
with open(metadata_path, "wb") as f: pickle.dump(all_metadata,  f)
with open(embedder_path, "w")  as f: f.write("all-MiniLM-L6-v2")

print(f"\nSaved:")
print(f"  FAISS index  : {index_path}")
print(f"  Chunks       : {chunks_path}  ({len(all_chunks)} entries)")
print(f"  Metadata     : {metadata_path}")

# ── QUICK RETRIEVAL TEST ──────────────────────────────────────────────────────
print("\n--- Retrieval Test ---")
test_queries = [
    ("DDoS",              "high packet rate flood overwhelming server bandwidth"),
    ("Web Attack XSS",    "malicious javascript injected into web page"),
    ("SSH-Patator",       "brute force authentication attempts on SSH port 22"),
    ("Heartbleed",        "openssl memory leak vulnerability TLS heartbeat"),
]

for attack, query in test_queries:
    q_vec = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    D, I  = index.search(q_vec, k=3)
    top_hit = all_metadata[I[0][0]]["attack_type"]
    score   = D[0][0]
    status  = "✓" if top_hit == attack else "✗"
    print(f"  {status} Query: '{query[:45]}...'")
    print(f"    Expected: {attack:<30} Got: {top_hit}  (score={score:.4f})")

print("\n" + "="*55)
print("RAG pipeline ready.")
print("Next step: run agent.py")
print("="*55)
