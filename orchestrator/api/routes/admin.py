"""
Admin API – session management, cache control, benchmark endpoints.

DELETE /v1/admin/sessions/{session_id}  – clear a session's memory
POST   /v1/admin/benchmark              – run a benchmark suite
GET    /v1/admin/cache/stats            – KV cache statistics
DELETE /v1/admin/cache                  – clear KV cache
POST   /v1/admin/classify               – test intent classifier
"""
from __future__ import annotations

import time
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from orchestrator.core.types import Intent
from orchestrator.router.intent_classifier import HybridIntentClassifier
from orchestrator.memory.memory_manager import MemoryManager
from orchestrator.cache.kv_manager import KVCacheManager
from orchestrator.cache.response_cache import ResponseCache
from orchestrator.utils.audit_log import AuditLog
from orchestrator.api.dependencies import get_container, get_memory_manager, get_kv_cache, get_audit_log
from orchestrator.config.settings import get_settings
from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _require_admin_key(key: str = Security(_api_key_header)) -> None:
    """Admin endpoints always require the API key, regardless of API_AUTH_ENABLED."""
    cfg = get_settings().api
    if key != cfg.key:
        raise HTTPException(status_code=403, detail="Invalid or missing admin API key")


router = APIRouter(
    prefix="/v1/admin",
    tags=["admin"],
    dependencies=[Depends(_require_admin_key)],
)


class ClassifyRequest(BaseModel):
    text: str


class BenchmarkRequest(BaseModel):
    prompts: List[str]
    iterations: int = 3


class BenchmarkResult(BaseModel):
    prompt: str
    avg_latency_ms: float
    intent: str
    confidence: float


@router.get("/sessions")
async def list_sessions(
    memory: MemoryManager = Depends(get_memory_manager),
) -> dict:
    """List all sessions currently active in L1 memory with turn/token counts."""
    sessions = memory.list_sessions()
    return {"sessions": sessions, "total": len(sessions)}


@router.delete("/sessions/{session_id}")
async def clear_session(
    session_id: str,
    memory: MemoryManager = Depends(get_memory_manager),
    audit: AuditLog = Depends(get_audit_log),
) -> dict:
    """Clear all memory tiers for a specific session."""
    await memory.clear_session(session_id)
    import asyncio as _asyncio
    _asyncio.create_task(audit.record("clear_session", detail=session_id))
    return {"status": "cleared", "session_id": session_id}


@router.post("/classify", response_model=Dict)
async def classify_intent(
    request: ClassifyRequest,
) -> Dict:
    """Test the intent classifier against a text string."""
    container = get_container()
    intent: Intent = await container.intent_classifier.classify(request.text)
    from orchestrator.router.routing_engine import RoutingEngine
    decision = RoutingEngine().route(intent)
    return {
        "text": request.text[:100],
        "intent": intent.intent_type.value,
        "confidence": round(intent.confidence, 3),
        "classification_stage": intent.classification_stage,
        "subtype": intent.subtype,
        "requires_expert": intent.requires_expert,
        "requires_rag": intent.requires_rag,
        "secondary_intents": [
            {"intent_type": s.intent_type.value, "confidence": round(s.confidence, 3)}
            for s in intent.secondary_intents
        ],
        "routing": {
            "experts": [e.value for e in decision.active_experts],
            "run_rag": decision.run_rag,
            "priority": decision.priority,
            "temperature": decision.generation_override.temperature,
            "max_tokens": decision.generation_override.max_tokens,
        },
    }


@router.post("/benchmark", response_model=List[BenchmarkResult])
async def run_benchmark(
    request: BenchmarkRequest,
) -> List[BenchmarkResult]:
    """
    Run lightweight benchmark: classify + route (without LLM inference).
    Useful for measuring orchestration overhead without GPU.
    """
    container = get_container()
    results: List[BenchmarkResult] = []

    for prompt in request.prompts:
        latencies = []
        last_intent = None

        for _ in range(request.iterations):
            t0 = time.perf_counter()
            intent = await container.intent_classifier.classify(prompt)
            _ = container.routing_engine.route(intent)
            latencies.append((time.perf_counter() - t0) * 1000)
            last_intent = intent

        results.append(
            BenchmarkResult(
                prompt=prompt[:80],
                avg_latency_ms=sum(latencies) / len(latencies),
                intent=last_intent.intent_type.value,
                confidence=round(last_intent.confidence, 3),
            )
        )

    return results


@router.get("/cache/stats")
async def get_cache_stats(
    kv_cache: KVCacheManager = Depends(get_kv_cache),
) -> dict:
    return kv_cache.get_stats()


@router.delete("/cache")
async def clear_kv_cache(
    kv_cache: KVCacheManager = Depends(get_kv_cache),
    audit: AuditLog = Depends(get_audit_log),
) -> dict:
    kv_cache.clear()
    import asyncio as _asyncio
    _asyncio.create_task(audit.record("clear_kv_cache"))
    return {"status": "cleared"}


@router.get("/cache/response/stats")
async def get_response_cache_stats() -> dict:
    """Return hit/miss statistics for the response-level LRU cache."""
    container = get_container()
    return container.response_cache.stats()


@router.delete("/cache/response")
async def clear_response_cache() -> dict:
    """Evict all entries from the response cache."""
    container = get_container()
    await container.response_cache.clear()
    return {"status": "cleared"}


@router.get("/audit")
async def get_audit_log_entries(
    limit: int = 50,
    offset: int = 0,
    audit: AuditLog = Depends(get_audit_log),
) -> dict:
    """Return recent admin audit log entries (newest first)."""
    entries = await audit.list(limit=limit, offset=offset)
    return {"entries": entries, "limit": limit, "offset": offset}


@router.get("/sessions/{session_id}/history")
async def export_session_history(
    session_id: str,
    memory: MemoryManager = Depends(get_memory_manager),
) -> dict:
    """
    Export the full conversation history for a session.
    Returns L1 turns + L2 summaries as a JSON-serialisable structure
    suitable for archiving or re-importing.
    """
    from orchestrator.core.types import MessageRole
    turns = await memory.get_conversation(session_id)
    summaries = await memory._l2.get_all_for_session(session_id)
    return {
        "session_id": session_id,
        "turns": [
            {"role": m.role.value, "content": m.content,
             "timestamp": m.timestamp.isoformat() if m.timestamp else None}
            for m in turns
        ],
        "summaries": [
            {"id": s.id, "content": s.content,
             "importance": s.importance_score,
             "timestamp": s.timestamp.isoformat() if s.timestamp else None}
            for s in summaries
        ],
    }


@router.post("/sessions/{session_id}/history")
async def import_session_history(
    session_id: str,
    body: dict,
    memory: MemoryManager = Depends(get_memory_manager),
    audit: AuditLog = Depends(get_audit_log),
) -> dict:
    """
    Import conversation turns into a session's L1 memory.
    Useful for restoring a session from an export or migrating between instances.
    Each item in ``turns`` must have ``role`` and ``content`` fields.
    """
    from orchestrator.core.types import Message, MessageRole
    turns = body.get("turns", [])
    imported = 0
    for turn in turns:
        try:
            role = MessageRole(turn.get("role", "user"))
        except ValueError:
            role = MessageRole.USER
        msg = Message(role=role, content=str(turn.get("content", "")))
        await memory._l1.push_turn(session_id, msg)
        imported += 1
    import asyncio as _asyncio
    _asyncio.create_task(audit.record("import_history", detail=f"{session_id} ({imported} turns)"))
    return {"session_id": session_id, "imported_turns": imported}


@router.post("/model/reload")
async def reload_model(
    audit: AuditLog = Depends(get_audit_log),
) -> dict:
    """
    Re-initialise the LlamaClient connection and verify the backend is reachable.
    Use after restarting the llama.cpp server with a new model without restarting
    the API process.  Does NOT hot-swap model weights on a running server.
    """
    container = get_container()
    await container.llama_client.close()
    # Re-open the underlying httpx client (LlamaClient is lazy — next call reopens it)
    reachable = await container.llama_client.health_check()
    import asyncio as _asyncio
    _asyncio.create_task(audit.record("model_reload", detail=f"reachable={reachable}"))
    return {
        "status": "reloaded",
        "backend_reachable": reachable,
        "server_url": container.llama_client._base_url,
    }


@router.post("/backup")
async def create_backup(
    audit: AuditLog = Depends(get_audit_log),
) -> dict:
    """
    Create a timestamped backup of the SQLite database.
    The backup is written to the same directory as the main DB with a
    ``_backup_YYYYMMDD_HHMMSS`` suffix.  FAISS indices are in-memory and
    rebuilt from the vector data on restart, so only SQLite needs backup.
    """
    import aiosqlite
    import shutil
    from datetime import datetime
    from pathlib import Path
    from orchestrator.config.settings import get_settings

    cfg = get_settings()
    src = Path(cfg.storage.sqlite_path)
    if not src.exists():
        raise HTTPException(status_code=404, detail="Database file not found")

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dst = src.parent / f"{src.stem}_backup_{ts}{src.suffix}"

    try:
        # Use SQLite online backup API via aiosqlite for a consistent snapshot
        async with aiosqlite.connect(str(src)) as src_db:
            async with aiosqlite.connect(str(dst)) as dst_db:
                await src_db.backup(dst_db)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Backup failed: {exc}") from exc

    import asyncio as _asyncio
    _asyncio.create_task(audit.record("backup", detail=str(dst)))
    return {
        "status": "ok",
        "backup_path": str(dst),
        "size_bytes": dst.stat().st_size,
    }


@router.post("/metrics/reset")
async def reset_metrics(
    audit: AuditLog = Depends(get_audit_log),
) -> dict:
    """
    Reset in-process Prometheus counters and histograms to zero.
    Useful after a canary deployment where metric label names have changed.
    Warning: this only resets the in-process registry — Prometheus server
    retains historical data from prior scrapes.
    """
    from prometheus_client import REGISTRY
    import asyncio as _asyncio
    collectors = list(REGISTRY._names_to_collectors.values())
    reset_count = 0
    for collector in collectors:
        if hasattr(collector, "_metrics"):
            collector._metrics.clear()
            reset_count += 1
    _asyncio.create_task(audit.record("metrics_reset", detail=f"{reset_count} collectors cleared"))
    return {"status": "reset", "collectors_cleared": reset_count}


@router.get("/experts")
async def list_experts() -> dict:
    container = get_container()
    return {"experts": container.expert_manager.available_experts()}


@router.get("/memory/stats")
async def memory_stats(
    memory: MemoryManager = Depends(get_memory_manager),
) -> dict:
    """
    Unified memory tier statistics.

    Returns per-tier item counts and token budgets so operators can gauge
    memory pressure without digging into raw SQLite or FAISS files.
    """
    container = get_container()

    # L1: in-process data — zero I/O
    l1_sessions = container.l1.active_sessions()
    l1_tokens = sum(container.l1.session_token_count(s) for s in l1_sessions)
    l1_idle = {
        s: round(container.l1.session_idle_seconds(s) or 0, 1)
        for s in l1_sessions
    }

    # L2: SQLite row count
    try:
        db = await container.l2._get_db()
        async with db.execute("SELECT COUNT(*) FROM conversation_summaries") as cur:
            l2_count = (await cur.fetchone())[0]
    except Exception:
        l2_count = -1

    # L3: FAISS vector count (in-process, no I/O)
    l3_count = container.l3._index._index.ntotal if (
        container.l3._index._index is not None
    ) else 0

    # L4: SQLite row count
    try:
        db4 = await container.l4._get_db()
        async with db4.execute("SELECT COUNT(*) FROM knowledge_base") as cur:
            l4_count = (await cur.fetchone())[0]
    except Exception:
        l4_count = -1

    return {
        "l1": {
            "active_sessions": len(l1_sessions),
            "total_tokens": l1_tokens,
            "ttl_seconds": container.l1._ttl,
            "session_idle_seconds": l1_idle,
        },
        "l2": {
            "total_summaries": l2_count,
        },
        "l3": {
            "total_vectors": l3_count,
            "cleared_sessions": len(container.l3._cleared_sessions),
        },
        "l4": {
            "total_entries": l4_count,
            "categories": await container.l4.list_categories(),
        },
    }
