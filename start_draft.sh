#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start_draft.sh — Start the speculative decoding DRAFT model server (v2.0)
#
# Runs the smallest available model on port 8081 so the main verifier
# (port 8080) can use it for speculative decoding.
#
# Once both servers are running, enable speculative decoding by adding:
#   LLAMA_DRAFT_MODEL_URL=http://localhost:8081
# to your .env file and restarting the SynapseOS API.
#
# Recommended draft models (download one, ~1–4 GB):
#   TinyLlama 1.1B (fastest, best for draft):
#     curl -L https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf \
#          -o models/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf
#
#   Phi-3-mini 3.8B (higher acceptance rate, still fast):
#     curl -L https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct-Q4_K_M.gguf \
#          -o models/Phi-3-mini-4k-instruct-Q4_K_M.gguf
#
#   Llama-3.2-1B (good balance):
#     curl -L https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q5_K_M.gguf \
#          -o models/Llama-3.2-1B-Instruct-Q5_K_M.gguf
#
# Usage:
#   ./start_draft.sh
#   ./start_draft.sh --port 8082
# ─────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${SCRIPT_DIR}/bin/llama-b9279"
SERVER="${BIN_DIR}/llama-server"
PORT=8081
# Draft model uses fewer threads — leave most CPU for the verifier
N_THREADS=4
N_CTX=2048    # smaller context saves RAM; draft only needs K+1 tokens at a time
N_BATCH=128

# ── GPU layers for draft model ────────────────────────────────────────────────
N_GPU_LAYERS=0
if [ -n "$LLAMA_DRAFT_GPU_LAYERS" ]; then
    N_GPU_LAYERS="$LLAMA_DRAFT_GPU_LAYERS"
elif command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
    if [ "${GPU_COUNT:-0}" -gt 0 ]; then
        N_GPU_LAYERS=99
    fi
fi

# ── Draft model auto-discovery ────────────────────────────────────────────────
# Priority: smallest/fastest models first for lowest draft latency
DRAFT_PRIORITY_LIST=(
    "smollm" "tinyllama" "tiny-llama"
    "llama-3.2-1b" "llama-3.2-1"
    "phi-3-mini" "phi-3.5-mini"
    "llama-3.2-3b" "llama-3.2-3"
    "qwen2.5-3b" "stablelm-3b"
    "phi-3-medium"
)

MODEL=""
MODEL_REASON=""
MODELS_DIR="${SCRIPT_DIR}/models"

if [ -n "$LLAMA_DRAFT_MODEL_PATH" ] && [ -f "$LLAMA_DRAFT_MODEL_PATH" ]; then
    MODEL="$LLAMA_DRAFT_MODEL_PATH"
    MODEL_REASON="[env override: LLAMA_DRAFT_MODEL_PATH]"
else
    for fragment in "${DRAFT_PRIORITY_LIST[@]}"; do
        found=$(find "$MODELS_DIR" -maxdepth 1 -iname "*${fragment}*.gguf" 2>/dev/null | head -1)
        if [ -n "$found" ]; then
            MODEL="$found"
            MODEL_REASON="[draft priority: ${fragment}] $(basename "$found")"
            break
        fi
    done

    # Last resort: smallest .gguf (by file size) — opposite of verifier's largest
    if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
        smallest=$(find "$MODELS_DIR" -maxdepth 1 -name "*.gguf" 2>/dev/null \
            | xargs -I{} ls -s {} 2>/dev/null \
            | sort -n | head -1 | awk '{print $2}')
        if [ -n "$smallest" ] && [ -f "$smallest" ]; then
            MODEL="$smallest"
            MODEL_REASON="[fallback: smallest file] $(basename "$smallest")"
        fi
    fi
fi

# ── Validate ──────────────────────────────────────────────────────────────────
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ERROR: No draft model found in ${MODELS_DIR}/"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  Download a small draft model (choose one):"
    echo ""
    echo "  TinyLlama 1.1B — fastest, ~700 MB:"
    echo "    curl -L https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf \\"
    echo "         -o models/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"
    echo ""
    echo "  Llama-3.2-1B — good quality, ~1 GB:"
    echo "    curl -L https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q5_K_M.gguf \\"
    echo "         -o models/Llama-3.2-1B-Instruct-Q5_K_M.gguf"
    echo ""
    echo "  Phi-3-mini 3.8B — highest acceptance rate, ~2.2 GB:"
    echo "    curl -L https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct-Q4_K_M.gguf \\"
    echo "         -o models/Phi-3-mini-4k-instruct-Q4_K_M.gguf"
    echo ""
    echo "  Then add to .env: LLAMA_DRAFT_MODEL_URL=http://localhost:8081"
    echo ""
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SynapseOS — Draft Model Server  (v2.0 speculative decoding)"
echo "  Model   : $(basename "$MODEL")"
echo "  Select  : ${MODEL_REASON}"
echo "  Port    : $PORT"
echo "  Threads : $N_THREADS  (leaves CPU headroom for verifier)"
echo "  Context : $N_CTX tokens"
echo "  GPU     : layers=${N_GPU_LAYERS}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Enable in .env: LLAMA_DRAFT_MODEL_URL=http://localhost:${PORT}"
echo "  Health  : http://localhost:${PORT}/health"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

export LD_LIBRARY_PATH="${BIN_DIR}:${LD_LIBRARY_PATH}"

exec "${SERVER}" \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --threads "$N_THREADS" \
    --ctx-size "$N_CTX" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --batch-size "$N_BATCH" \
    --chat-template chatml \
    "$@"
