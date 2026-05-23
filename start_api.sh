#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start_api.sh — Start the SynapseOS FastAPI orchestration server
#
# Requires llama.cpp server to be running first:
#   ./start_llama.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source "${SCRIPT_DIR}/venv/bin/activate"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SynapseOS — Orchestration API"
echo "  API  : http://localhost:8000"
echo "  Docs : http://localhost:8000/docs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exec uvicorn orchestrator.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    --workers 1
