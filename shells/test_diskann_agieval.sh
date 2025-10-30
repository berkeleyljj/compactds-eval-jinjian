#!/usr/bin/env bash
set -euo pipefail

# Quick DiskANN retrieval test on AGI Eval English queries against Massive-Serve
# It samples N queries from AGI Eval English datasets and POSTs them in one batch.
#
# Usage:
#   ./test_diskann_agieval.sh
#   API_URL=http://128.208.4.44:30888/search N=32 K=10 L=2000 W=4 THREADS=128 ./test_diskann_agieval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="/home/ubuntu/jinjian/eval/olmes:/home/ubuntu/jinjian/eval:${PYTHONPATH:-}"

API_URL="${API_URL:-http://128.208.4.44:30888/search}"
N="${N:-100}"            # total queries to send
K="${K:-10}"             # n_docs
L="${L:-2000}"          # DiskANN L
W="${W:-4}"             # DiskANN W (beam)
THREADS="${THREADS:-128}"  # DiskANN search threads

# Location of AGI Eval English data (from oe_eval dependency)
AGI_DIR="/home/ubuntu/jinjian/eval/olmes/oe_eval/dependencies/AGIEval/data/v1"
export AGI_DIR N K L W THREADS

if [[ ! -d "$AGI_DIR" ]]; then
  echo "AGI Eval data directory not found: $AGI_DIR" >&2
  exit 1
fi

# Build a JSON body on stdin with sampled queries
python - <<'PY' | curl -sS -X POST "$API_URL" \
  -H 'Content-Type: application/json' \
  -d @- | { command -v jq >/dev/null 2>&1 && jq '. | {results: .results|keys, counts: (.results.passages|if type=="array" then [.[0]|length] else null end)}' || cat; }
import json, os, random

agi_dir = os.environ.get("AGI_DIR")
N = int(os.environ.get("N", "16"))
K = int(os.environ.get("K", "10"))
L = int(os.environ.get("L", "2000"))
W = int(os.environ.get("W", "4"))
THREADS = int(os.environ.get("THREADS", "128"))

datasets = [
    "lsat-ar",
    "lsat-lr",
    "lsat-rc",
    "logiqa-en",
    "sat-math",
    "sat-en",
    "aqua-rat",
    "gaokao-english",
]

def load_jsonl(path):
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def build_query(doc):
    parts = []
    passage = doc.get("passage")
    if passage:
        parts.append("Passage: " + passage)
    parts.append("Question: " + doc.get("question", ""))
    options = doc.get("options") or []
    labels = "ABCDEFGH"
    for i, opt in enumerate(options[:len(labels)]):
        parts.append(f" ({labels[i]}) {opt}")
    return "\n".join(parts)

# Distribute samples across datasets
per = max(1, N // len(datasets))
queries = []
for name in datasets:
    path = os.path.join(agi_dir, f"{name}.jsonl")
    try:
        items = list(load_jsonl(path))
        random.seed(2025)
        sample = items[:per] if len(items) <= per else random.sample(items, per)
        for d in sample:
            queries.append(build_query(d))
    except Exception:
        continue

# Cap to N total
queries = queries[:N]

payload = {
    "backend": "diskann",
    "queries": queries,
    "n_docs": K,
    "diskann_L": L,
    "diskann_W": W,
    "diskann_threads": THREADS,
}
print(json.dumps(payload))
PY

echo "[DiskANN Test] Sent $N AGI Eval queries • n_docs=$K • L=$L • W=$W • threads=$THREADS → $API_URL"


