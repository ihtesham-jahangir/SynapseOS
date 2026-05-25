<div align="center">

<img src="logo_main.png" alt="SynapseOS Logo" width="350" />

# SynapseOS

### AI Operating System — Private, On-Device, Orchestrated Intelligence

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776ab?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![SynapseOS](https://img.shields.io/badge/SynapseOS-v3.5-blueviolet?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij48cGF0aCBmaWxsPSJ3aGl0ZSIgZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6bTAgMThjLTQuNDEgMC04LTMuNTktOC04czMuNTktOCA4LTggOCAzLjU5IDggOC0zLjU5IDgtOCA4eiIvPjwvc3ZnPg==&logoColor=white)](https://github.com/ihtesham-jahangir/SynapseOS)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-b9279-ff6b35?logo=meta&logoColor=white)](https://github.com/ggerganov/llama.cpp)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-700%2B%20passing-brightgreen)](orchestrator/tests/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ed?logo=docker&logoColor=white)](docker-compose.yml)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/ihtesham-jahangir/SynapseOS)

**Run a full AI inference stack — locally, privately, without any cloud API.**

[Quick Start](#quick-start) · [Architecture](#architecture) · [API Reference](#api-reference) · [Multi-Agent System](#multi-agent-system-v30) · [Experts](#expert-system-v31--v32) · [Performance](#performance-tuning) · [Configuration](#configuration) · [Monitoring](#monitoring)

</div>

---

## The Problem

Every major AI product is built on the same broken assumption: **one model does everything**.

That assumption fails in practice:

- A single LLM wastes compute on simple queries that need a fraction of the resources.
- Context windows fill up — long conversations lose coherence after a few turns.
- There is no routing, no specialization, no persistent memory — just raw token prediction.
- Deploying on-premise or on the edge means no paid API, which means starting from scratch.
- Complex multi-step tasks — "research X, then write code, then summarize" — cannot be handled by a single serial inference call.

**The answer is not a bigger model. It is smarter orchestration.**

---

## What is SynapseOS?

SynapseOS is an **AI Operating System** — a multi-model orchestration runtime that sits above the model layer. It classifies intent across 15 domains, routes requests through single-agent or multi-agent execution paths, fuses knowledge from memory and documents, activates domain-specific experts, and returns a grounded, contextually-aware response.

```
                          User Request
                               │
                    ┌──────────▼──────────┐
                    │   MultiAgentEngine   │  complexity detection · cache check
                    └──────────┬──────────┘
                               │
               ┌───────────────┴───────────────┐
               │ simple query                   │ complex query
               ▼                               ▼
   ┌───────────────────────┐      ┌──────────────────────────┐
   │   Intent Classifier   │      │   TaskPlanner (LLM)       │
   │  15 domains · keyword │      │   JSON DAG decomposition  │
   │  + semantic + subtype │      └──────────────┬───────────┘
   └───────────┬───────────┘                     │
               │                        ┌────────▼────────┐
       ┌───────┴───────┐                │   AgentPool      │
       ▼               ▼                │  topological DAG │
  Memory + RAG    Expert Router         │  parallel waves  │
  L1·L2·L3·L4   11 domain experts      └────────┬────────┘
       │               │               Research·Coder·Reasoner
       └───────┬───────┘               Summarizer·General
               ▼                                │
      Adaptive Fusion Engine                    ▼
      score · dedup · rank            Synthesized Answer
               │
               ▼
    Semantic Response Cache       SharedMemoryBus (pub/sub)
    exact match · cosine ≥ 0.90   SSE progress streaming
               │
               ▼
         llama.cpp Runtime
      CPU optimized (4 threads)
      Vulkan iGPU / NVIDIA GPU
               │
               ▼
   Streaming Response (SSE / REST / WebSocket)
```

---

## Key Features

### Core Engine (v1.0–v1.3)
- **Intent-aware routing** — 15 intent classes across 3 tiers (original 9 + 6 new v3.5 domains), each mapped to the optimal execution path
- **Four-tier memory** — L1 hot cache → L2 session summaries → L3 FAISS vector store → L4 persistent knowledge base
- **Adaptive context fusion** — scores, deduplicates, and ranks context from all active paths; fills the token budget with maximum information density
- **Streaming inference** — SSE token streaming and WebSocket with TTFT measurement
- **Response cache** — LRU + TTL cache (BLAKE2b keys) for deterministic intents; skip logic for personal/contextual queries
- **API key authentication** — `X-API-Key` header; public paths always open
- **Prometheus + Grafana** — 14-panel auto-provisioned dashboard; 24+ metrics
- **Document deletion** — soft-delete + FAISS index rebuild; BM25 kept in sync
- **GPU auto-detection** — `nvidia-smi` detection + Vulkan iGPU support; falls back gracefully to CPU

### Advanced RAG (v1.2)
- **Hybrid search** — BM25 keyword + FAISS semantic combined via Reciprocal Rank Fusion; catches exact matches that pure vector search misses
- **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` scores query-document pairs jointly
- **Semantic chunking** — splits at topic boundaries using embedding cosine similarity, not fixed character counts

### Performance (v2.0–v2.1)
- **Speculative decoding** — draft model pre-generates tokens, verifier accepts/rejects; 2–3× CPU speedup when a small draft model is available
- **SQLite WAL mode** — write-ahead logging on L2 and L4 caches for concurrent read safety
- **Async compression queue** — single-worker asyncio queue for L1→L2 summarization; per-session dedup, backpressure, ordered shutdown
- **Inference batcher** — semaphore gate capping concurrent llama.cpp requests to the server's `--parallel` slot count
- **Circuit breaker** — three-state (CLOSED → OPEN → HALF_OPEN) with configurable failure threshold and recovery timeout
- **429/503 retry** — tenacity exponential back-off for transient server-busy responses

### Multi-Agent System (v3.0)
- **LLM task planner** — decomposes complex queries into a JSON dependency DAG; falls back to a single task on parse error
- **Topological DAG executor** — waves of parallel sub-tasks respecting dependency order; cascade-failure propagation
- **Five specialized agents** — Research, Coder, Reasoner, Summarizer, General; each with role-tuned system prompts and temperature
- **SharedMemoryBus** — asyncio pub/sub with topic history, blocking `wait_for()`, and subscriber queues
- **SSE progress streaming** — `GET /v1/agents/tasks/{id}/stream` replays completed subtasks then subscribes for live events
- **SQLite task persistence** — completed `TaskGraph` objects survive restarts; in-process LRU registry (O(1)) + SQLite fallback
- **Automatic complexity routing** — `POST /v1/chat` auto-routes complex queries through multi-agent; simple queries bypass decomposition

### Extended Intent Domains (v3.1)
- **6 new intent classes** — `writing`, `security`, `planning`, `education`, `creative`, `data_analysis` (15 total)
- **Multi-intent detection** — every response carries `secondary_intents` with confidence scores; used for multi-expert routing
- **Subtype extraction** — granular subtypes within intents (e.g. `coding.python`, `security.injection`, `writing.email`)
- **Per-intent adaptive compute** — RoutingEngine sets domain-specific temperature and max_tokens per intent:
  - Security: temperature=0.10, priority=3 (highest)
  - Creative: temperature=0.90
  - Coding: max_tokens ×1.5
  - Retrieval: max_tokens ×0.7
- **Context-aware classification** — short queries use keyword fast-path; ambiguous queries escalate to semantic embedding

### Expert System (v3.2)
- **11 domain experts** — Code, Math, Translation, Summarization, Reasoning + Writing, Security, Planning, Education, Creative, DataAnalysis
- **External prompt files** — all system prompts live in `orchestrator/prompts/*.txt`; edit without touching Python
- **Multi-expert routing** — secondary intent triggers a second expert simultaneously (e.g. `security` + `data_analysis` query activates both)
- **Few-shot examples** — each expert carries domain examples embedded in the system prompt
- **Dynamic few-shot selection** — BGE embeddings select the most relevant examples for each incoming query

### Request Tracing & Structured Errors (v3.3)
- **Request-ID middleware** — every request gets a UUID; propagated as `X-Request-ID` response header
- **Structured error format** — all 4xx/5xx responses return `{"error": {"code": "...", "message": "...", "request_id": "..."}}`
- **Enhanced admin classify** — `/v1/admin/classify` returns full routing decision: classification_stage, subtype, secondary_intents, per-intent temperature, max_tokens, active experts, run_rag flag

### Semantic Cache & Inference Optimization (v3.5)
- **Two-tier response cache** — Tier 1: exact BLAKE2b hash match (O(1), <1ms); Tier 2: BGE cosine similarity ≥ 0.90 (~186ms); rephrased queries hit cache without re-running inference
- **All 11 deterministic intents cached** — writing, security, planning, education, creative, data_analysis added to cacheable set
- **Optimized inference defaults** — `max_tokens=256` (was 512), `context_token_budget=1400` (was 3200), `L1_MAX_TOKENS=512` (was 2048), `L2_MAX_TOKENS=1024` (was 4096)
- **Optimized llama.cpp flags** — `--threads 4` (physical cores only, avoids SMT contention), `--ctx-size 2048`, `--parallel 2`
- **Vulkan iGPU support** — `start_llama.sh` downloads the Vulkan-enabled binary; safe detection prevents false-positive GPU offload on non-Vulkan builds
- **expert_used propagation** — `GenerationPlan.expert_used` correctly populated and returned in every `ChatResponse`
- **Agent enabled-flag check** — `MultiAgentEngine` respects `AGENT_ENABLED=false` across all entry points
- **Real intent in agent responses** — agent pipeline now classifies and returns the actual query intent instead of hardcoding `REASONING`

---

## Architecture

### Directory Structure

```
SynapseOS/
├── orchestrator/
│   ├── api/
│   │   ├── app.py                  # FastAPI app factory, lifespan, middleware stack
│   │   ├── dependencies.py         # DI container — all singletons wired here
│   │   ├── middleware/
│   │   │   ├── auth.py             # APIKeyMiddleware
│   │   │   ├── rate_limit.py       # Token-bucket rate limiter
│   │   │   ├── request_id.py       # RequestIDMiddleware — UUID per request (v3.3)
│   │   │   └── logging_middleware.py
│   │   └── routes/
│   │       ├── chat.py             # POST /v1/chat, /v1/chat/stream, WS /v1/ws/{id}
│   │       ├── agents.py           # POST /v1/agents/tasks, GET /tasks/{id}, GET /tasks/{id}/stream
│   │       ├── documents.py        # POST/DELETE /v1/documents, /v1/knowledge
│   │       ├── admin.py            # classify (full routing info), benchmark, cache, experts, stats
│   │       └── health.py           # /health (8 components), /health/live, /health/ready
│   ├── agents/                     # v3.0 multi-agent system
│   │   ├── task_types.py           # SubTask, TaskGraph, AgentRole, TaskStatus, BusMessage
│   │   ├── memory_bus.py           # SharedMemoryBus — asyncio pub/sub
│   │   ├── base_agent.py           # BaseAgent ABC — timing, error capture, bus publish
│   │   ├── specialized_agents.py   # ResearchAgent, CoderAgent, ReasonerAgent, SummarizerAgent, GeneralAgent
│   │   ├── task_planner.py         # TaskPlanner — LLM JSON DAG decomposition
│   │   ├── agent_pool.py           # AgentPool — topological DAG executor, cascade failure
│   │   ├── multi_agent_engine.py   # MultiAgentEngine — routing, real intent classification, synthesis
│   │   └── task_store.py           # TaskStore — SQLite WAL persistence
│   ├── config/
│   │   ├── settings.py             # Pydantic-settings, all env vars, MultiAgentSettings
│   │   └── model_selector.py       # Priority-ranked .gguf auto-discovery
│   ├── core/
│   │   ├── types.py                # All domain types (Intent, FusedContext, GenerationPlan, ...)
│   │   ├── base.py                 # Abstract base classes
│   │   └── exceptions.py           # Typed exception hierarchy
│   ├── router/
│   │   ├── intent_classifier.py    # 15-domain hybrid classifier — keyword + semantic + subtype + multi-intent
│   │   └── routing_engine.py       # Per-intent temperature, max_tokens, priority, multi-expert routing
│   ├── memory/
│   │   ├── memory_manager.py       # MemoryManager — orchestrates L1–L4 + compression queue
│   │   ├── l1_cache.py             # In-process session turn cache (512 token limit)
│   │   ├── l2_cache.py             # SQLite session summary store (WAL, 1024 token limit)
│   │   ├── l3_cache.py             # FAISS semantic memory (facts/entities)
│   │   ├── l4_cache.py             # SQLite persistent knowledge base (WAL)
│   │   ├── compressor.py           # ContextCompressor — LLM-based turn summarization
│   │   └── compression_queue.py    # Async single-worker queue for L1→L2 compression
│   ├── rag/
│   │   ├── pipeline.py             # RAGPipeline — ingest, retrieve, delete
│   │   ├── chunker.py              # RecursiveTextChunker + SemanticChunker
│   │   ├── embedder.py             # BGEEmbedder (bge-small-en-v1.5, 384-dim)
│   │   ├── indexer.py              # FAISSIndex — add, search, delete, rebuild
│   │   ├── bm25_index.py           # BM25Plus keyword index with JSON persistence
│   │   ├── retriever.py            # SemanticRetriever + HybridRetriever (RRF)
│   │   └── reranker.py             # EmbeddingReranker + CrossEncoderReranker
│   ├── fusion/
│   │   └── fusion_engine.py        # AdaptiveFusionEngine — score · dedup · rank · fill budget
│   ├── experts/
│   │   ├── expert_manager.py       # ExpertManager — lazy-loaded plugin registry, multi-expert routing
│   │   ├── writing_expert.py       # WritingExpert — loads from prompts/expert_writing.txt (v3.2)
│   │   ├── security_expert.py      # SecurityExpert — low temperature, threat-focused (v3.2)
│   │   ├── planning_expert.py      # PlanningExpert — structured plans, milestones (v3.2)
│   │   ├── education_expert.py     # EducationExpert — Socratic method, scaffolded explanations (v3.2)
│   │   ├── creative_expert.py      # CreativeExpert — high temperature, originality focus (v3.2)
│   │   └── data_expert.py          # DataAnalysisExpert — statistical reasoning, visualizations (v3.2)
│   ├── prompts/                    # External system prompt files (v3.2)
│   │   ├── task_planner.txt        # TaskPlanner LLM decomposition prompt
│   │   ├── expert_writing.txt
│   │   ├── expert_security.txt
│   │   ├── expert_planning.txt
│   │   ├── expert_education.txt
│   │   ├── expert_creative.txt
│   │   └── expert_data_analysis.txt
│   ├── runtime/
│   │   ├── inference_engine.py     # InferenceEngine + GenerationPlan (expert_used propagation fixed)
│   │   ├── llama_client.py         # LlamaClient — circuit breaker, 429/503 retry, EOS stripping
│   │   ├── speculative.py          # SpeculativeDecoder — draft + verify + bonus token
│   │   ├── request_batcher.py      # InferenceBatcher — semaphore gate, Prometheus metrics
│   │   └── adaptive_compute.py     # AdaptiveComputeController — per-intent temperature/tokens
│   ├── cache/
│   │   ├── response_cache.py       # Two-tier cache: BLAKE2b exact + BGE semantic similarity (v3.5)
│   │   └── kv_manager.py           # KVCacheManager — token tracking
│   ├── streaming/
│   │   ├── token_streamer.py       # TokenStreamer — SSE events, TTFT, callbacks
│   │   └── websocket_manager.py    # WebSocketManager — session multiplexing
│   ├── utils/
│   │   ├── metrics.py              # 24+ Prometheus metrics incl. v3.0 agent metrics
│   │   ├── logging_utils.py        # structlog configuration
│   │   ├── audit_log.py            # Audit logging for admin actions
│   │   ├── circuit_breaker.py      # Shared circuit breaker implementation
│   │   └── async_utils.py          # gather_with_fallback, AsyncLRUCache
│   └── tests/                      # 700+ tests — unit + integration
│       ├── test_api.py
│       ├── test_v30.py             # 34 tests — multi-agent system
│       ├── test_v31.py             # New intent domains, keyword scoring, multi-intent
│       ├── test_v32.py             # New experts, routing, adaptive temperature
│       ├── test_v33.py             # Request-ID middleware, structured errors
│       ├── test_v34.py             # Admin classify full routing, expert_used propagation
│       ├── test_v35.py             # 99 tests — semantic cache, intent count, routing
│       └── test_prompts.py         # 116 tests — prompt file validation, encoding, content
├── monitoring/
│   ├── prometheus.yml
│   ├── prometheus-local.yml
│   └── grafana/
│       ├── provisioning/           # Auto-provisioned datasource + dashboard
│       └── dashboards/
│           └── synapseos.json      # 14-panel Grafana dashboard
├── models/                         # GGUF model files (gitignored)
├── bin/                            # llama.cpp binary (build b9279, CPU + Vulkan)
├── docker-compose.yml
├── Dockerfile
├── start_llama.sh                  # Optimized launcher: threads=4, ctx=2048, parallel=2, Vulkan-aware
├── start_draft.sh                  # Draft model launcher for speculative decoding
├── start_api.sh                    # SynapseOS API launcher
└── .env                            # All configuration
```

### Request Lifecycle

```
POST /v1/chat
  │
  ├─ RequestIDMiddleware          — attach UUID, propagate as X-Request-ID header
  ├─ RateLimitMiddleware          — token bucket per IP
  ├─ APIKeyMiddleware             — key validation (if enabled)
  ├─ RequestLoggingMiddleware     — structured request/response log
  │
  ▼
MultiAgentEngine.process_request()
  │
  ├─ ResponseCache.get()          ─── Tier 1: BLAKE2b exact match (<1ms)
  │                               ─── Tier 2: BGE cosine similarity ≥ 0.90 (~186ms)
  │
  ├─ cfg.agent.enabled check      — skip decomposition if AGENT_ENABLED=false
  ├─ TaskPlanner.is_complex()     — keyword + length heuristic (>200 chars)
  │
  ├─ [complex path] ──────────────────────────────────────────────────────┐
  │   TaskPlanner.plan()          — LLM → JSON DAG                        │
  │   AgentPool.execute()         — topological waves, 180s timeout/task  │
  │     ├─ ResearchAgent / CoderAgent / ReasonerAgent / …                 │
  │     ├─ Cascade-fail on dep failure                                    │
  │     └─ SharedMemoryBus.publish("progress.{task_id}")                  │
  │   IntentClassifier.classify() — real intent for agent response        │
  │   MultiAgentEngine._synthesize() — prefer SUMMARIZER, else concat     │
  │   ResponseCache.set()         — both exact + semantic embedding stored │
  │   TaskStore.save()            — async SQLite WAL persistence           │
  │   return ChatResponse(metadata={"multi_agent": True, "task_id": …})  │
  │                                                                        │
  └─ [simple path] ──────────────────────────────────────────────────────┐
      InferenceEngine.build_generation_plan()                             │
        ├─ IntentClassifier       — keyword fast-path → semantic fallback │
        │   returns: intent, subtype, secondary_intents, confidence       │
        ├─ RoutingEngine          — per-intent temperature, max_tokens,   │
        │   priority, multi-expert list, run_rag flag                     │
        ├─ [parallel fan-out]                                             │
        │    ├─ MemoryManager.retrieve_all()  — L1+L2+L3+L4              │
        │    ├─ RAGPipeline.retrieve()        — BM25+FAISS → reranker    │
        │    └─ ExpertManager.get_guidance()  — primary + secondary expert│
        ├─ AdaptiveFusionEngine   — score + deduplicate + rank            │
        │   token budget: 1400 tokens (was 3200)                         │
        └─ AdaptiveComputeController — per-intent temperature clamp       │
      InferenceEngine.generate() / stream()                               │
        ├─ InferenceBatcher       — semaphore gate (n_parallel=2 slots)   │
        ├─ SpeculativeDecoder     — draft+verify (if configured)          │
        └─ LlamaClient.chat()     — circuit breaker + 429/503 retry       │
      ResponseCache.set()        — store exact hash + BGE embedding       │
      MemoryManager.record_turn() — write L1, async L2 compression        │
      ChatResponse(expert_used=plan.expert_used, …)  — correctly filled  │
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- A `.gguf` model file in `./models/`
- llama.cpp binary (included in `./bin/llama-b9279/`)

### Local (recommended)

```bash
# 1. Clone and install
git clone https://github.com/ihtesham-jahangir/SynapseOS
cd SynapseOS
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure (defaults work out of the box)
cp .env.example .env

# 3. Start llama.cpp server (optimized: 4 threads, ctx=2048, 2 parallel slots)
./start_llama.sh

# 4. Start SynapseOS API
./start_api.sh
```

**API live at `http://localhost:8000` · Docs at `http://localhost:8000/docs`**

### Verify

```bash
curl http://localhost:8000/health/live
# {"status":"alive"}

curl http://localhost:8000/health
# {"status":"healthy","components":[llama_server, embedding_model, faiss_index,
#   websocket_manager, compression_queue, agent_pool, llama_circuit_breaker, embedding_circuit_breaker]}
```

### Monitoring (Prometheus + Grafana)

```bash
# Prometheus
docker run -d --name synapseos-prometheus --network host \
  -v $(pwd)/monitoring/prometheus-local.yml:/etc/prometheus/prometheus.yml:ro \
  prom/prometheus:v2.52.0

# Grafana
docker run -d --name synapseos-grafana --network host \
  -v $(pwd)/monitoring/grafana/provisioning:/etc/grafana/provisioning:ro \
  -v $(pwd)/monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro \
  -e GF_SECURITY_ADMIN_PASSWORD=admin \
  grafana/grafana:10.4.3
```

**Grafana at `http://localhost:3000` · login: `admin` / `admin`**

---

## API Reference

### Chat

```bash
# Standard response — complex queries auto-routed to multi-agent
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: changeme-in-production" \
  -d '{
    "session_id": "my-session",
    "messages": [{"role": "user", "content": "Explain async/await in Python."}]
  }'
```

**Response (simple query):**
```json
{
  "session_id": "my-session",
  "content": "...",
  "intent": "coding",
  "expert_used": "code",
  "memory_levels_used": ["l1_hot", "l2_summary", "l3_vector", "l4_knowledge"],
  "rag_chunks_used": 2,
  "tokens_generated": 148,
  "total_tokens": 412,
  "time_to_first_token_ms": 6120.4,
  "total_time_ms": 21890.3,
  "metadata": {
    "multi_agent": false,
    "intent": "coding",
    "subtype": "python",
    "classification_stage": "semantic",
    "secondary_intents": [{"intent": "education", "confidence": 0.61}]
  }
}
```

**Response (complex query — auto multi-agent):**
```json
{
  "session_id": "my-session",
  "content": "Based on the analysis...",
  "intent": "coding",
  "total_time_ms": 498040.0,
  "metadata": {
    "multi_agent": true,
    "task_id": "28784cdb-9def-41c9-abd2-2826afa7c0f0",
    "sub_tasks": 4,
    "sub_task_roles": ["research", "coder", "reasoner", "summarizer"],
    "sub_task_statuses": {"t1": "done", "t2": "done", "t3": "done", "t4": "done"}
  }
}
```

**Response headers:**
```
X-Request-ID: a3f2c1d8    ← unique per request, usable for log correlation
```

### Streaming (SSE)

```bash
curl -N -X POST http://localhost:8000/v1/chat/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: changeme-in-production" \
  -d '{"session_id": "s1", "messages": [{"role": "user", "content": "Write a merge sort."}]}'
```

```
data: {"event_type": "token", "session_id": "s1", "content": "Here", "done": false}
data: {"event_type": "token", "session_id": "s1", "content": " is", "done": false}
...
data: {"event_type": "done", "session_id": "s1", "content": "",
       "done": true, "token_count": 82, "time_to_first_token_ms": 5980}
```

### Intent Classification (Admin)

```bash
curl -X POST http://localhost:8000/v1/admin/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Explain SQL injection and how to prevent it"}'
```

```json
{
  "intent": "security",
  "confidence": 0.94,
  "classification_stage": "keyword",
  "subtype": "injection",
  "requires_expert": true,
  "requires_rag": false,
  "secondary_intents": [
    {"intent_type": "education", "confidence": 0.67}
  ],
  "routing": {
    "experts": ["security", "education"],
    "run_rag": false,
    "priority": 3,
    "temperature": 0.1,
    "max_tokens": 256
  }
}
```

### WebSocket

```javascript
const ws = new WebSocket("ws://localhost:8000/v1/ws/my-session");
ws.send(JSON.stringify({role: "user", content: "Hello"}));
ws.onmessage = (e) => {
  const event = JSON.parse(e.data);
  if (event.event_type === "token") process.stdout.write(event.content);
};
```

---

## Multi-Agent System (v3.0)

The multi-agent engine automatically activates for queries containing complexity signals (`first … then`, `compare`, `analyze`, `step by step`, or queries longer than 200 characters).

### Submit a Task

```bash
curl -X POST http://localhost:8000/v1/agents/tasks \
  -H "Content-Type: application/json" \
  -H "X-API-Key: changeme-in-production" \
  -d '{
    "session_id": "research-session",
    "query": "Design a distributed caching system: compare Redis vs Memcached, implement LRU eviction in Python, explain consistent hashing, and recommend an architecture for 1M daily users",
    "max_subtasks": 5
  }'
```

**Response:**
```json
{
  "task_id": "28784cdb-9def-41c9-abd2-2826afa7c0f0",
  "session_id": "research-session",
  "status": "done",
  "sub_tasks": [
    {"id": "t1", "role": "research",   "status": "done", "dependencies": [],           "elapsed_ms": 120000},
    {"id": "t2", "role": "coder",      "status": "done", "dependencies": ["t1"],       "elapsed_ms": 98000},
    {"id": "t3", "role": "reasoner",   "status": "done", "dependencies": ["t1", "t2"], "elapsed_ms": 110000},
    {"id": "t4", "role": "summarizer", "status": "done", "dependencies": ["t1","t2","t3"], "elapsed_ms": 85000}
  ],
  "final_answer": "Based on the analysis, I recommend...",
  "total_elapsed_ms": 498040
}
```

### Stream Progress (SSE)

```bash
curl -N http://localhost:8000/v1/agents/tasks/28784cdb-.../stream
```

```
data: {"type": "subtask_done", "subtask_id": "t1", "role": "research", "elapsed_ms": 120000}
data: {"type": "subtask_done", "subtask_id": "t2", "role": "coder",    "elapsed_ms": 98000}
data: {"type": "complete", "status": "done", "final_answer": "...", "total_elapsed_ms": 498040}
```

### How Multi-Agent Routing Works

```
Query: "Compare X and Y, analyze trade-offs, implement in Python, then summarize"
  │
  ├─ is_complex() → True  (keywords: "compare", "analyze", "implement", "summarize")
  ├─ AGENT_ENABLED check → True
  │
  ├─ TaskPlanner.plan() → JSON DAG:
  │     t1: research X        (deps: [])
  │     t2: research Y        (deps: [])
  │     t3: reasoner X vs Y   (deps: [t1, t2])
  │     t4: coder Python impl (deps: [t3])
  │     t5: summarizer        (deps: [t1,t2,t3,t4])
  │
  ├─ AgentPool.execute():
  │     Wave 1: t1, t2 concurrently (no deps)
  │     Wave 2: t3 (t1+t2 done)
  │     Wave 3: t4 (t3 done)
  │     Wave 4: t5 (t4 done)
  │
  ├─ IntentClassifier.classify(query) → real intent (not hardcoded REASONING)
  ├─ _synthesize() → returns t5 (SUMMARIZER wins)
  └─ ResponseCache.set() → next identical query returns in <100ms (semantic cache)
```

### Agent Roles

| Role | Focus | Temperature | Max tokens |
|------|-------|-------------|------------|
| **Research** | Information gathering and synthesis | 0.3 | 300 |
| **Coder** | Clean, correct, runnable code | 0.2 | 350 |
| **Reasoner** | Step-by-step logical analysis | 0.4 | 300 |
| **Summarizer** | Coherent synthesis of all partial results | 0.5 | 350 |
| **General** | Fallback for unclassified sub-tasks | 0.7 | 300 |

---

## Expert System (v3.1 + v3.2)

Experts inject domain-optimized system prompts into the inference context before generation. Each expert loads its prompt from `orchestrator/prompts/`, enabling prompt engineering without code changes.

### 11 Available Experts

| Expert | Intent | Temperature | Focus |
|--------|--------|-------------|-------|
| **Code** | `coding` | 0.20 | Correct, tested, idiomatic code |
| **Math** | `math` | 0.10 | Step-by-step derivations, LaTeX |
| **Translation** | `translation` | 0.30 | Fluency, cultural nuance |
| **Summarization** | `summarization` | 0.40 | Concise, information-dense |
| **Reasoning** | `reasoning` | 0.50 | Structured arguments |
| **Writing** | `writing` | 0.60 | Clarity, tone, audience-aware |
| **Security** | `security` | 0.10 | Threat modeling, OWASP, CVEs |
| **Planning** | `planning` | 0.45 | Milestones, dependencies, risks |
| **Education** | `education` | 0.55 | Socratic method, scaffolding |
| **Creative** | `creative` | 0.90 | Originality, ideation, imagination |
| **Data Analysis** | `data_analysis` | 0.35 | Statistical reasoning, chart recommendations |

### Multi-Expert Routing

When a query carries strong secondary intents, two experts activate simultaneously:

```
Query: "Write a security audit report for our API endpoints"
  │
  ├─ primary intent: security  → SecurityExpert  (temperature=0.10)
  ├─ secondary intent: writing → WritingExpert   (confidence=0.72)
  │
  └─ Both expert prompts merged into context before generation
```

### Editing Expert Prompts

```bash
# No code changes needed — just edit the .txt file
vim orchestrator/prompts/expert_security.txt

# Restart API to pick up changes
./start_api.sh
```

---

## Performance Tuning

### Benchmark Results (i5-8350U, CPU-only, swap cleared)

| Configuration | tok/s | Notes |
|---|---|---|
| Baseline (threads=8, ctx=4096, swap full) | 1.2 | Default out-of-box |
| threads=4, ctx=2048 | 1.9 | Physical cores only, smaller KV cache |
| + Swap flush (`sudo swapoff -a && swapon -a`) | **3.5** | Eliminates model paging |
| Cache hit (exact match) | — | **<1ms** |
| Cache hit (rephrased query, semantic) | — | **~186ms** |
| Repeat complex multi-agent query | — | **<100ms** |

### Response Cache Tiers

```
Query arrives
    │
    ├─ Tier 1: BLAKE2b hash lookup         O(1)    <1ms     exact same wording
    │
    ├─ Tier 2: BGE embedding similarity    O(n)    ~186ms   cosine ≥ 0.90
    │          "write a python function to reverse a string"
    │          "Python function that reverses a string"  → CACHE HIT
    │
    └─ Miss → full inference pipeline (26s–8min depending on complexity)
```

### Optimal llama.cpp Settings

```bash
# CPU-only (i5-8350U or similar laptop CPU)
bin/llama-b9279/llama-server \
  --model models/your-model.gguf \
  --threads 4 \          # physical cores only — SMT hurts matrix multiply
  --ctx-size 2048 \      # fits context_token_budget(1400) + max_tokens(256) + overhead
  --n-gpu-layers 0 \     # CPU is faster than Intel iGPU on shared-memory bus
  --batch-size 512 \
  --parallel 2 \         # 2 concurrent inference slots
  --chat-template chatml

# NVIDIA GPU (any discrete GPU with VRAM ≥ 4GB)
LLAMA_GPU_LAYERS=99 ./start_llama.sh   # full offload — 10-50x faster

# Intel iGPU with Vulkan binary (modest gain, ~1.5x on prefill)
LLAMA_GPU_LAYERS=20 ./start_llama.sh   # partial offload
```

### Swap Management (Linux)

```bash
# Flush stale swap — reclaims RAM for the model
# Kill llama server first to make room, then:
sudo swapoff -a && sudo swapon -a
free -h   # verify Swap used drops near 0
```

### Speculative Decoding (2–3× speedup)

```bash
# Start draft model on port 8081
./start_draft.sh   # uses TinyLlama-1.1B by default

# Enable in .env
LLAMA_DRAFT_MODEL_URL=http://localhost:8081
LLAMA_DRAFT_K_TOKENS=5
```

---

## Document Management (RAG)

```bash
# Ingest a document
curl -X POST http://localhost:8000/v1/documents \
  -H "Content-Type: application/json" \
  -H "X-API-Key: changeme-in-production" \
  -d '{
    "content": "Your document text here...",
    "source": "manual.txt",
    "metadata": {"category": "product"}
  }'

# Upload a file
curl -X POST http://localhost:8000/v1/documents/upload \
  -F "file=@document.txt"

# Delete a document (rebuilds FAISS + BM25 index)
curl -X DELETE http://localhost:8000/v1/documents/manual-v1 \
  -H "X-API-Key: changeme-in-production"

# Add a persistent fact (L4 knowledge base — highest retrieval priority)
curl -X POST http://localhost:8000/v1/documents/knowledge \
  -H "Content-Type: application/json" \
  -H "X-API-Key: changeme-in-production" \
  -d '{
    "content": "Our product pricing starts at $49/month.",
    "category": "pricing",
    "keywords": "price cost subscription",
    "priority": 1.0
  }'
```

---

## Full Endpoint Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/chat` | Chat — auto-routes simple/complex; semantic cache check |
| `POST` | `/v1/chat/stream` | Chat with SSE token streaming |
| `WS`   | `/v1/ws/{session_id}` | WebSocket bidirectional chat |
| `POST` | `/v1/agents/tasks` | Submit a multi-agent task (blocking) |
| `GET`  | `/v1/agents/tasks/{id}` | Poll task status / result |
| `GET`  | `/v1/agents/tasks/{id}/stream` | SSE stream of sub-task progress |
| `POST` | `/v1/documents` | Ingest text document into RAG |
| `POST` | `/v1/documents/upload` | Ingest file upload into RAG |
| `GET`  | `/v1/documents/stats` | RAG index statistics |
| `DELETE` | `/v1/documents/{doc_id}` | Delete document from RAG + BM25 |
| `POST` | `/v1/documents/knowledge` | Add fact to L4 knowledge base |
| `GET`  | `/v1/documents/knowledge/categories` | List knowledge categories |
| `GET`  | `/v1/stats` | Runtime statistics (all subsystems) |
| `POST` | `/v1/admin/classify` | Classify intent + full routing decision |
| `POST` | `/v1/admin/benchmark` | Run latency benchmark |
| `GET`  | `/v1/admin/cache/stats` | Response cache stats (hits, semantic hits, embeddings) |
| `DELETE` | `/v1/admin/cache` | Flush response cache |
| `GET`  | `/v1/admin/experts` | List available experts |
| `DELETE` | `/v1/admin/sessions/{id}` | Clear session memory |
| `GET`  | `/health` | Full component health check (8 components) |
| `GET`  | `/health/live` | Kubernetes liveness probe |
| `GET`  | `/health/ready` | Kubernetes readiness probe |
| `GET`  | `/metrics` | Prometheus metrics |
| `GET`  | `/docs` | OpenAPI interactive docs |

---

## Configuration

```env
# ── llama.cpp ────────────────────────────────────────────────────────────────
LLAMA_SERVER_URL=http://localhost:8080
LLAMA_CONTEXT_SIZE=2048         # reduced from 4096 — fits token budget + output headroom
LLAMA_THREADS=4                 # physical cores only (SMT hurts LLM matrix multiply)
LLAMA_GPU_LAYERS=0              # 0=CPU, 20=Vulkan iGPU, 99=NVIDIA full offload
LLAMA_TIMEOUT=120
LLAMA_N_PARALLEL=2              # concurrent inference slots — match --parallel in start_llama.sh

# ── Speculative decoding (optional — 2-3x speedup) ───────────────────────────
# LLAMA_DRAFT_MODEL_URL=http://localhost:8081
# LLAMA_DRAFT_K_TOKENS=5

# ── Embedding model ──────────────────────────────────────────────────────────
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DEVICE=cpu

# ── Memory tiers ─────────────────────────────────────────────────────────────
L1_MAX_TURNS=20
L1_MAX_TOKENS=512               # reduced from 2048 — limits context injected into prompt
L2_MAX_TOKENS=1024              # reduced from 4096
L3_SIMILARITY_THRESHOLD=0.72
L4_MAX_RESULTS=5
SQLITE_PATH=data/synapseos.db

# ── Context fusion ───────────────────────────────────────────────────────────
CONTEXT_TOKEN_BUDGET=1400       # reduced from 3200 — fits in 2048 ctx with 256 output headroom
FUSION_SEMANTIC_WEIGHT=0.45
FUSION_RECENCY_WEIGHT=0.25
FUSION_PRIORITY_WEIGHT=0.20
FUSION_IMPORTANCE_WEIGHT=0.10

# ── Generation defaults ──────────────────────────────────────────────────────
DEFAULT_MAX_TOKENS=256          # reduced from 512 — halves generation time for short answers
DEFAULT_TEMPERATURE=0.7

# ── RAG ──────────────────────────────────────────────────────────────────────
RAG_TOP_K=6
RAG_RERANK_TOP_N=3
RAG_USE_HYBRID_SEARCH=false     # enable for exact ID/name match queries
RAG_USE_CROSS_ENCODER=false     # enable for higher reranking accuracy (~85MB model)
RAG_USE_SEMANTIC_CHUNKING=false # enable for topic-boundary document splits

# ── Multi-agent system ───────────────────────────────────────────────────────
AGENT_ENABLED=true              # set false to disable decomposition entirely
AGENT_MAX_SUBTASKS=5
AGENT_MAX_PARALLEL=4
AGENT_TASK_TIMEOUT_S=180

# ── API ──────────────────────────────────────────────────────────────────────
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=changeme-in-production
API_AUTH_ENABLED=false
API_RATE_LIMIT_REQUESTS=60
API_RATE_LIMIT_WINDOW=60
```

### Model Selection

`start_llama.sh` auto-discovers the best `.gguf` from `./models/` by priority:

```
1. Qwen2.5-7B / Qwen2.5-Coder-7B
2. Llama-3.1/3.2-8B
3. Mistral-7B / Mistral-Nemo
4. Gemma-2-9B / Gemma-2-2B
5. DeepSeek-R1-7B
6. Llama-3.2-3B  (default bundled model)
7. Phi-3.5-mini / Phi-3-medium
8. TinyLlama / SmolLM          (used as draft model for speculative decoding)
9. Largest .gguf               (fallback)
```

Override: `LLAMA_MODEL_PATH=/path/to/model.gguf ./start_llama.sh`

---

## Monitoring

SynapseOS exposes Prometheus metrics at `/metrics`. The Grafana dashboard provides 14 panels:

| Panel | What it shows |
|-------|---------------|
| Request Rate | Requests/s by intent (15 domains) |
| Error Rate | Errors/s |
| Active Sessions | Current open sessions |
| Cache Hit Rate | Response cache — exact + semantic hits |
| Latency p50/p95/p99 | End-to-end latency distribution |
| Time to First Token | TTFT p50/p95 |
| Token Throughput | Tokens/second |
| Memory Cache Hits | L1/L2/L3/L4 hits by tier |
| RAG Chunks Retrieved | Chunks/s |
| Intent Distribution | Request mix by intent (1h) |
| Latency by Intent | p95 latency per intent |
| Agent Task Rate | Multi-agent tasks by role and status |
| Agent Pool Active | Concurrent sub-tasks in flight |
| Bus Messages | SharedMemoryBus message rate |

Key Prometheus metrics:

```
# Core
synapseos_requests_total{intent, status}
synapseos_request_latency_seconds_bucket{intent, le}
synapseos_time_to_first_token_seconds_bucket{le}
synapseos_tokens_per_second
synapseos_active_sessions
synapseos_memory_cache_hits_total{level}
synapseos_rag_chunks_retrieved_total
synapseos_llama_errors_total
synapseos_compression_queue_depth

# v3.0 Multi-agent
synapseos_agent_tasks_total{role, status}
synapseos_agent_task_duration_ms_bucket{role, le}
synapseos_agent_pool_active
synapseos_bus_messages_total{topic}
```

---

## Live Test Results

All results from local CPU-only run (Llama-3.2-3B-Instruct-Q5_K_M, i5-8350U, 4 threads, swap flushed):

| Test | Result |
|------|--------|
| Simple query → InferenceEngine (not decomposed) | PASS |
| Complex query → multi-agent auto-routing (4–5 sub-tasks) | PASS |
| Distributed cache design (research+coder+reasoner+summarizer) | PASS |
| Security API analysis (5 agents, 10.7 min) | PASS |
| ML churn prediction (5 agents, 8.7 min) | PASS |
| Dependency chain (t2 runs after t1 done) | PASS |
| Cascade failure (A→B→C, A fails → B+C skipped) | PASS |
| Task poll `GET /v1/agents/tasks/{id}` | PASS |
| SSE replay for completed task | PASS |
| SQLite persistence survives restart | PASS |
| Complex query repeat: 498,040ms → **<100ms** (semantic cache) | PASS |
| Rephrased query: new wording → **186ms** (semantic similarity hit) | PASS |
| Response cache hit (coding intent) → 26,000ms → **<1ms** | PASS |
| `expert_used` correctly populated in every ChatResponse | PASS |
| Security query → `expert_used=security`, temp=0.10 | PASS |
| Writing query → `expert_used=writing` | PASS |
| Coding query → `expert_used=code` | PASS |
| `/v1/admin/classify` returns classification_stage + subtype + routing | PASS |
| Multi-agent response carries real intent (not always REASONING) | PASS |
| `AGENT_ENABLED=false` disables decomposition cleanly | PASS |
| `X-Request-ID` header present on every response | PASS |
| Structured error `{"error": {"code", "message", "request_id"}}` | PASS |
| SSE token streaming | PASS |
| WebSocket bidirectional | PASS |
| Multi-turn memory (name recalled across turns) | PASS |
| Circuit breaker state in `/health` | PASS |
| 8 health components all reported | PASS |
| Prometheus metrics (24+ counters/histograms) | PASS |
| All 6 expert prompt files load and pass encoding check | PASS |
| test_prompts.py: 116 prompt validation tests | PASS |

---

## Testing

```bash
source venv/bin/activate

# Full suite (700+ tests)
pytest orchestrator/tests/ -v

# By area
pytest orchestrator/tests/test_v30.py -v       # multi-agent system (34 tests)
pytest orchestrator/tests/test_v31.py -v       # new intent domains, multi-intent
pytest orchestrator/tests/test_v32.py -v       # new experts, adaptive routing
pytest orchestrator/tests/test_v33.py -v       # request-ID middleware, structured errors
pytest orchestrator/tests/test_v34.py -v       # admin classify, expert_used fix
pytest orchestrator/tests/test_v35.py -v       # semantic cache, intent count (99 tests)
pytest orchestrator/tests/test_prompts.py -v   # prompt file validation (116 tests)

# With coverage
pytest orchestrator/tests/ --cov=orchestrator --cov-report=term-missing
```

| Test file | What it covers |
|-----------|----------------|
| `test_api.py` | API endpoints — chat, documents, admin, health |
| `test_v30.py` | SharedMemoryBus, TaskPlanner, AgentPool, MultiAgentEngine (34 tests) |
| `test_v31.py` | 15 intent types, keyword scoring, multi-intent detection, subtype extraction |
| `test_v32.py` | 6 new experts, adaptive temperature, multi-expert routing, few-shot selection |
| `test_v33.py` | RequestIDMiddleware, structured error format, `X-Request-ID` header |
| `test_v34.py` | Admin classify full routing info, `expert_used` propagation, agent intent fix |
| `test_v35.py` | Semantic cache (two-tier), BLAKE2b + cosine similarity, intent set (99 tests) |
| `test_prompts.py` | Prompt file existence, UTF-8, min length, domain keywords (116 tests) |
| `test_improvements.py` | Circuit breaker, BLAKE2b keys, temperature clamp, cascade failure |
| `test_tier2.py` | BM25, hybrid retriever, cross-encoder, semantic chunking |
| `test_tier3.py` | Auth middleware, FAISS delete/rebuild, doc deletion |

---

## Roadmap

| Version | Status | Milestone |
|---------|--------|-----------|
| **v1.0** | ✅ Done | FastAPI core · intent routing · 4-tier memory · RAG · llama.cpp |
| **v1.1** | ✅ Done | Expert plugin system — code, math, translation, summarization, reasoning |
| **v1.2** | ✅ Done | Hybrid RAG (BM25+FAISS) · cross-encoder reranking · semantic chunking |
| **v1.3** | ✅ Done | API auth · Prometheus + Grafana · document deletion · GPU auto-detect |
| **v2.0** | ✅ Done | Speculative decoding · draft+verify · bonus token · 2–3× CPU speedup |
| **v2.1** | ✅ Done | SQLite WAL · InferenceBatcher · async CompressionQueue · circuit breaker · 429/503 retry |
| **v3.0** | ✅ Done | Multi-agent DAG decomposition · AgentPool · SharedMemoryBus · SSE progress · SQLite task persistence |
| **v3.1** | ✅ Done | 6 new intent domains · multi-intent detection · subtype extraction · per-intent adaptive compute |
| **v3.2** | ✅ Done | 6 new domain experts · external prompt files · multi-expert routing · dynamic few-shot |
| **v3.3** | ✅ Done | Request-ID middleware · structured error responses · enhanced admin classify endpoint |
| **v3.4** | ✅ Done | `expert_used` propagation fix · agent enabled-flag check · real intent in agent responses |
| **v3.5** | ✅ Done | Two-tier semantic cache (BGE cosine) · expanded cacheable intents · inference optimization (threads/ctx/tokens) · Vulkan iGPU support |
| **v4.0** | Planned | Whisper ASR integration · voice-first agent mode |
| **v4.1** | Planned | Vision model support · multi-modal context fusion |
| **v5.0** | Planned | Federated multi-node inference · cross-device memory sync |

---

## Philosophy

> *Intelligence should be orchestrated, not monolithic.*

SynapseOS treats inference as a **distributed reasoning problem**:

- **Complexity-aware routing** — simple queries go direct; complex multi-step tasks are decomposed into specialized sub-agents
- **Domain specialization** — 15 intent classes, 11 experts, each tuned for its domain; the orchestrator composes them
- **Semantic memory** — identical and similar queries are cached via embedding similarity; the model is called only when genuinely needed
- **Compute-aware scheduling** — agent parallelism is capped to the llama server's actual slot count; no timeout flooding
- **Memory as infrastructure** — context is retrieved and ranked, not re-derived from scratch each turn
- **Fusion over selection** — all sources are scored and merged within the token budget, not picked arbitrarily
- **Resilience by default** — circuit breaker, 429/503 retry, cascade-failure propagation, async compression with backpressure
- **Privacy by default** — runs entirely on-device; no data is sent to any external service

---

## Contributing

1. Fork the repo and create a feature branch
2. Run the test suite: `pytest orchestrator/tests/ -v`
3. Open a pull request — all 700+ tests must pass

```bash
git clone https://github.com/ihtesham-jahangir/SynapseOS
cd SynapseOS
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pytest orchestrator/tests/ -v
```

---

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

**SynapseOS v3.5** · Built for developers who believe AI infrastructure should be open, modular, and private.

*Llama-3.2 · llama.cpp · FastAPI · FAISS · BGE · BM25 · Prometheus · Grafana · asyncio · Vulkan*

[GitHub](https://github.com/ihtesham-jahangir/SynapseOS) · [Issues](https://github.com/ihtesham-jahangir/SynapseOS/issues)

</div>
