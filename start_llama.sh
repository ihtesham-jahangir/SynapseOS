#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start_llama.sh — Start llama.cpp HTTP server (official binary)
#
# Exposes an OpenAI-compatible API:
#   POST /v1/chat/completions
#   POST /completion
#   GET  /health
#
# Model is auto-selected from the models/ directory using ModelSelector
# priority rules. Override with: LLAMA_MODEL_PATH=/path/to/model.gguf ./start_llama.sh
#
# Usage:
#   ./start_llama.sh
#   ./start_llama.sh --port 8080
# ─────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${SCRIPT_DIR}/bin/llama-b9279"
SERVER="${BIN_DIR}/llama-server"
PORT=8080
N_THREADS=4          # physical cores only — SMT hurts matrix multiply on i5-8350U
N_CTX=2048           # context window (matches context_token_budget + max_tokens headroom)
N_BATCH=512
# Parallel inference slots (v2.1): 2 slots allows concurrent requests without thrashing.
# Must match LLAMA_N_PARALLEL in .env so InferenceBatcher knows the slot count.
N_PARALLEL="${LLAMA_N_PARALLEL:-2}"

# ── Auto-extract bundled binary ───────────────────────────────────────────────
# The repo ships a pre-built Ubuntu x86-64 binary as a tar.gz (~14 MB).
# Extract it on first run so users never need a manual step.
if [ ! -f "${SERVER}" ]; then
    TARBALL="${SCRIPT_DIR}/bin/llama-b9279-ubuntu-x64.tar.gz"
    if [ -f "${TARBALL}" ]; then
        echo "  Extracting bundled llama-server binary (first run)..."
        (cd "${SCRIPT_DIR}/bin" && tar xzf llama-b9279-ubuntu-x64.tar.gz)
        chmod +x "${SERVER}"
        echo "  Done — binary ready at ${SERVER}"
    fi
fi

# ── GPU auto-detection ────────────────────────────────────────────────────────
# Priority: LLAMA_GPU_LAYERS env var > NVIDIA detection > Vulkan detection > CPU
# NOTE: Intel iGPU requires Vulkan-enabled llama.cpp binary. Only auto-enable
#       GPU layers for NVIDIA (nvidia-smi) or if LLAMA_GPU_LAYERS is set explicitly.
N_GPU_LAYERS=0
GPU_MODE="CPU only"

if [ -n "$LLAMA_GPU_LAYERS" ]; then
    N_GPU_LAYERS="$LLAMA_GPU_LAYERS"
    GPU_MODE="env override (LLAMA_GPU_LAYERS=${LLAMA_GPU_LAYERS})"
elif command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
    if [ "${GPU_COUNT:-0}" -gt 0 ]; then
        N_GPU_LAYERS=99
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
        GPU_MODE="NVIDIA GPU — ${GPU_NAME} (${GPU_VRAM})"
    fi
elif ldd "${SERVER}" 2>/dev/null | grep -q libvulkan && command -v vulkaninfo &>/dev/null; then
    # Binary was compiled with Vulkan — safe to enable GPU offload for Intel/AMD iGPU
    N_GPU_LAYERS=20
    GPU_MODE="Vulkan iGPU (set LLAMA_GPU_LAYERS=0 to force CPU)"
fi

# ── Model auto-discovery ──────────────────────────────────────────────────────
# Honor explicit override first (env var or LLAMA_MODEL_PATH from .env)
if [ -n "$LLAMA_MODEL_PATH" ] && [ -f "$LLAMA_MODEL_PATH" ]; then
    MODEL="$LLAMA_MODEL_PATH"
    MODEL_REASON="[env override]"
else
    # Try Python ModelSelector (best-quality selection with priority ranking)
    if command -v python3 &>/dev/null; then
        SELECTED=$(python3 -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
try:
    from orchestrator.config.model_selector import select_model
    path, reason = select_model('${SCRIPT_DIR}/models')
    if path:
        print(path)
        import sys; print(reason, file=sys.stderr)
except Exception as e:
    import sys; print(f'ModelSelector unavailable: {e}', file=sys.stderr)
" 2>/tmp/model_selector_reason)
        MODEL_REASON=$(cat /tmp/model_selector_reason 2>/dev/null || echo "")

        if [ -n "$SELECTED" ] && [ -f "$SELECTED" ]; then
            MODEL="$SELECTED"
        fi
    fi

    # Bash fallback: scan models/ directory in priority order
    if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
        MODEL_REASON="[bash fallback — scanning models/]"
        MODELS_DIR="${SCRIPT_DIR}/models"

        # Priority substrings (first match wins — mirrors MODEL_PRIORITY in model_selector.py)
        PRIORITY_LIST=(
            "qwen2.5-7b" "qwen2.5-coder-7b"
            "llama-3.1-8b" "llama-3.2-8b" "llama-3-8b"
            "mistral-7b" "mistral-nemo"
            "gemma-2-9b" "gemma-2-2b"
            "deepseek-r1-7b" "deepseek-coder-6.7b"
            "llama-3.2-3b" "llama-3.2-3"
            "phi-3.5-mini" "phi-3-mini" "phi-3-medium"
            "qwen2.5-3b" "stablelm-3b" "openhermes"
            "llama-3.2-1b" "tinyllama" "smollm"
        )

        for fragment in "${PRIORITY_LIST[@]}"; do
            found=$(find "$MODELS_DIR" -maxdepth 1 -iname "*${fragment}*.gguf" 2>/dev/null | head -1)
            if [ -n "$found" ]; then
                MODEL="$found"
                MODEL_REASON="[bash priority: ${fragment}] $(basename "$found")"
                break
            fi
        done

        # Last resort: largest .gguf in models/
        if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
            largest=$(find "$MODELS_DIR" -maxdepth 1 -name "*.gguf" 2>/dev/null \
                | xargs -I{} ls -s {} 2>/dev/null \
                | sort -rn | head -1 | awk '{print $2}')
            if [ -n "$largest" ] && [ -f "$largest" ]; then
                MODEL="$largest"
                MODEL_REASON="[fallback: largest file] $(basename "$largest")"
            fi
        fi
    fi
fi

# ── Validate ──────────────────────────────────────────────────────────────────
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo ""
    echo "ERROR: No .gguf model found in ${SCRIPT_DIR}/models/"
    echo ""
    echo "Download one with (example — Llama-3.2-3B, ~2.2 GB):"
    echo "  curl -L https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q5_K_M.gguf \\"
    echo "       -o models/Llama-3.2-3B-Instruct-Q5_K_M.gguf"
    echo ""
    echo "Or set LLAMA_MODEL_PATH=/path/to/model.gguf in your environment."
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SynapseOS — llama.cpp Server  (build b9279, v2.1)"
echo "  Model    : $(basename "$MODEL")"
echo "  Select   : ${MODEL_REASON}"
echo "  Port     : $PORT"
echo "  Threads  : $N_THREADS"
echo "  Context  : $N_CTX tokens"
echo "  Parallel : $N_PARALLEL slots  (set LLAMA_N_PARALLEL to change)"
echo "  GPU      : ${GPU_MODE} (layers=${N_GPU_LAYERS})"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Chat     : http://localhost:$PORT/v1/chat/completions"
echo "  Health   : http://localhost:$PORT/health"
echo "  Docs     : http://localhost:$PORT"
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
    --parallel "$N_PARALLEL" \
    --chat-template chatml \
    "$@"
