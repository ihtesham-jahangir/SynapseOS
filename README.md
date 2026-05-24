<div align="center">

<img src="logo_main.png" alt="SynapseOS Logo" width="350" />

# SynapseOS

### AI Operating System — Private, On-Device, Orchestrated Intelligence

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776ab?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![SynapseOS](https://img.shields.io/badge/SynapseOS-v3.0-blueviolet?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij48cGF0aCBmaWxsPSJ3aGl0ZSIgZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6bTAgMThjLTQuNDEgMC04LTMuNTktOC04czMuNTktOCA4LTggOCAzLjU5IDggOC0zLjU5IDgtOCA4eiIvPjwvc3ZnPg==&logoColor=white)](https://github.com/ihtesham-jahangir/SynapseOS)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-b9279-ff6b35?logo=meta&logoColor=white)](https://github.com/ggerganov/llama.cpp)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-353%20passing-brightgreen)](orchestrator/tests/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ed?logo=docker&logoColor=white)](docker-compose.yml)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/ihtesham-jahangir/SynapseOS)

**Run a full AI inference stack — locally, privately, without any cloud API.**

[Quick Start](#quick-start) · [Architecture](#architecture) · [API Reference](#api-reference) · [Multi-Agent System](#multi-agent-system-v30) · [Configuration](#configuration) · [Monitoring](#monitoring) · [GitHub Repository](https://github.com/ihtesham-jahangir/SynapseOS)

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

SynapseOS is an **AI Operating System** — a multi-model orchestration runtime that sits above the model layer. It classifies intent, routes requests through single-agent or multi-agent execution paths, fuses knowledge from memory and documents, and returns a grounded, contextually-aware response.

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
   │   keyword + semantic  │      │   JSON DAG decomposition  │
   └───────────┬───────────┘      └──────────────┬───────────┘
               │                                 │
       ┌───────┴───────┐                ┌────────▼────────┐
       ▼               ▼                │   AgentPool      │
  Memory + RAG    Expert Router         │  topological DAG │
  L1·L2·L3·L4   code·math·etc          │  parallel waves  │
       │               │               └────────┬────────┘
       └───────┬───────┘               Research·Coder·Reasoner
               ▼                       Summarizer·General
      Adaptive Fusion Engine                     │
      score · dedup · rank                       ▼
               │                       Synthesized Answer
               ▼
         llama.cpp Runtime           SharedMemoryBus (pub/sub)
      Llama-3.2 · CPU/GPU            SSE progress streaming
               │
               ▼
   Streaming Response (SSE / REST / WebSocket)
```

---

## Key Features

### Core Engine (v1.0–v1.3)
- **Intent-aware routing** — 7 intent classes (coding, math, translation, summarization, reasoning, retrieval, conversation), each mapped to the optimal execution path
- **Four-tier memory** — L1 hot cache → L2 session summaries → L3 FAISS vector store → L4 persistent knowledge base
- **Adaptive context fusion** — scores, deduplicates, and ranks context from all active paths; fills the token budget with maximum information density
- **Streaming inference** — SSE token streaming and WebSocket with TTFT measurement
- **Response cache** — LRU + TTL cache (BLAKE2b keys) for deterministic intents including reasoning; skip logic for personal/contextual queries
- **API key authentication** — `X-API-Key` header; public paths always open
- **Prometheus + Grafana** — 14-panel auto-provisioned dashboard; 24+ metrics
- **Document deletion** — soft-delete + FAISS index rebuild; BM25 kept in sync
- **GPU auto-detection** — `nvidia-smi` detection; falls back gracefully to CPU

### Advanced RAG (v1.2)
- **Hybrid search** — BM25 keyword + FAISS semantic combined via Reciprocal Rank Fusion; catches exact matches that pure vector search misses
- **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` scores query-document pairs jointly
- **Semantic chunking** — splits at topic boundaries using embedding cosine similarity, not fixed character counts

### Performance (v2.0–v2.1)
- **Speculative decoding** — draft model pre-generates tokens, verifier accepts/rejects; 2–3× CPU speedup when a small draft model is available
- **SQLite WAL mode** — write-ahead logging on L2 and L4 caches for concurrent read safety
- **Async compression queue** — single-worker asyncio queue for L1→L2 summarization; per-session dedup, backpressure, ordered shutdown
- **Inference batcher** — semaphore gate capping concurrent llama.cpp requests to the server's `--parallel` slot count
- **Circuit breaker** — three-state (CLOSED → OPEN → HALF_OPEN) with configurable failure threshold and recovery timeout; prevents request pile-up when the backend is down
- **429/503 retry** — tenacity exponential back-off for transient server-busy responses

### Multi-Agent System (v3.0)
- **LLM task planner** — decomposes complex queries into a JSON dependency DAG; falls back to a single task on parse error
- **Topological DAG executor** — waves of parallel sub-tasks respecting dependency order; cascade-failure propagation for multi-hop chains
- **Five specialized agents** — Research, Coder, Reasoner, Summarizer, General; each with role-tuned system prompts and temperature
- **SharedMemoryBus** — asyncio pub/sub with topic history, blocking `wait_for()`, and subscriber queues for inter-agent coordination
- **SSE progress streaming** — `GET /v1/agents/tasks/{id}/stream` replays completed subtasks then subscribes for live events with 60-second keepalive
- **SQLite task persistence** — completed `TaskGraph` objects survive restarts; in-process LRU registry checked first (O(1)), SQLite as fallback
- **Automatic complexity routing** — `POST /v1/chat` auto-routes complex queries through multi-agent; simple queries bypass decomposition entirely
- **Agent-aware response cache** — multi-agent results stored in cache; identical repeated queries served in <10ms

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
│   │   │   └── logging_middleware.py
│   │   └── routes/
│   │       ├── chat.py             # POST /v1/chat, /v1/chat/stream, WS /v1/ws/{id}
│   │       ├── agents.py           # POST /v1/agents/tasks, GET /tasks/{id}, GET /tasks/{id}/stream
│   │       ├── documents.py        # POST/DELETE /v1/documents, /v1/knowledge
│   │       ├── admin.py            # classify, benchmark, cache, experts, stats
│   │       └── health.py           # /health (7 components), /health/live, /health/ready
│   ├── agents/                     # v3.0 multi-agent system
│   │   ├── task_types.py           # SubTask, TaskGraph, AgentRole, TaskStatus, BusMessage
│   │   ├── memory_bus.py           # SharedMemoryBus — asyncio pub/sub
│   │   ├── base_agent.py           # BaseAgent ABC — timing, error capture, bus publish
│   │   ├── specialized_agents.py   # ResearchAgent, CoderAgent, ReasonerAgent, SummarizerAgent, GeneralAgent
│   │   ├── task_planner.py         # TaskPlanner — LLM JSON DAG decomposition
│   │   ├── agent_pool.py           # AgentPool — topological DAG executor, cascade failure
│   │   ├── multi_agent_engine.py   # MultiAgentEngine — top-level routing + synthesis
│   │   └── task_store.py           # TaskStore — SQLite WAL persistence
│   ├── config/
│   │   ├── settings.py             # Pydantic-settings, all env vars, MultiAgentSettings
│   │   └── model_selector.py       # Priority-ranked .gguf auto-discovery
│   ├── core/
│   │   ├── types.py                # All domain types (Intent, FusedContext, GenerationPlan, ...)
│   │   ├── base.py                 # Abstract base classes
│   │   └── exceptions.py           # Typed exception hierarchy
│   ├── router/
│   │   ├── intent_classifier.py    # Keyword + semantic hybrid classifier, LRU cached
│   │   └── routing_engine.py       # Routing decisions, RAG/expert flags
│   ├── memory/
│   │   ├── memory_manager.py       # MemoryManager — orchestrates L1–L4 + compression queue
│   │   ├── l1_cache.py             # In-process session turn cache
│   │   ├── l2_cache.py             # SQLite session summary store (WAL)
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
│   │   ├── expert_manager.py       # ExpertManager — lazy-loaded plugin registry
│   │   └── implementations/        # Code, Math, Translation, Summarization, Reasoning
│   ├── runtime/
│   │   ├── inference_engine.py     # InferenceEngine + GenerationPlan + build_generation_plan()
│   │   ├── llama_client.py         # LlamaClient — circuit breaker, 429/503 retry, EOS stripping
│   │   ├── speculative.py          # SpeculativeDecoder — draft + verify + bonus token
│   │   ├── request_batcher.py      # InferenceBatcher — semaphore gate, Prometheus metrics
│   │   └── adaptive_compute.py     # AdaptiveComputeController — temperature clamping [0.05, 2.0]
│   ├── cache/
│   │   ├── response_cache.py       # ResponseCache — BLAKE2b keys, LRU + TTL
│   │   └── kv_manager.py           # KVCacheManager — token tracking
│   ├── streaming/
│   │   ├── token_streamer.py       # TokenStreamer — SSE events, TTFT, callbacks
│   │   └── websocket_manager.py    # WebSocketManager — session multiplexing
│   ├── utils/
│   │   ├── metrics.py              # 24+ Prometheus metrics incl. v3.0 agent metrics
│   │   ├── logging_utils.py        # structlog configuration
│   │   └── async_utils.py          # gather_with_fallback (logged), get_running_loop, AsyncLRUCache
│   └── tests/                      # 353 tests — unit + integration
│       ├── test_api.py
│       ├── test_rag_pipeline.py
│       ├── test_memory_manager.py
│       ├── test_tier2.py           # BM25, hybrid retriever, cross-encoder, semantic chunking
│       ├── test_tier3.py           # Auth, FAISS delete/rebuild, doc deletion endpoint
│       ├── test_v30.py             # 34 tests — multi-agent system end-to-end
│       └── test_improvements.py    # 24 tests — circuit breaker, BLAKE2b, cascade failure, etc.
├── monitoring/
│   ├── prometheus.yml
│   ├── prometheus-local.yml
│   └── grafana/
│       ├── provisioning/           # Auto-provisioned datasource + dashboard
│       └── dashboards/
│           └── synapseos.json      # 14-panel Grafana dashboard
├── models/                         # GGUF model files (gitignored)
├── bin/                            # llama.cpp binary (build b9279)
├── docker-compose.yml
├── Dockerfile
├── start_llama.sh                  # GPU-aware llama server launcher with model auto-discovery
├── start_draft.sh                  # Draft model launcher for speculative decoding
├── start_api.sh                    # SynapseOS API launcher
└── .env                            # All configuration
```

### Request Lifecycle

```
POST /v1/chat
  │
  ├─ RateLimitMiddleware          — token bucket per IP
  ├─ APIKeyMiddleware             — key validation (if enabled)
  ├─ RequestLoggingMiddleware     — structured request/response log
  │
  ▼
MultiAgentEngine.process_request()
  │
  ├─ ResponseCache.get()          — cache check before ANY work (math/coding/reasoning/…)
  │
  ├─ TaskPlanner.is_complex()     — keyword + length heuristic
  │
  ├─ [complex path] ──────────────────────────────────────────────────────┐
  │   TaskPlanner.plan()          — LLM → JSON DAG, cache_prompt=True     │
  │   AgentPool.execute()         — topological waves, per-task 180s timeout│
  │     ├─ ResearchAgent / CoderAgent / ReasonerAgent / …                  │
  │     ├─ Cascade-fail on dep failure (multi-hop propagation fixed)       │
  │     └─ SharedMemoryBus.publish("progress.{task_id}")                  │
  │   MultiAgentEngine._synthesize() — prefer SUMMARIZER, else concatenate│
  │   ResponseCache.set()         — store result for future cache hits     │
  │   TaskStore.save()            — async SQLite WAL persistence           │
  │   return ChatResponse(metadata={"multi_agent": True, "task_id": …})   │
  │                                                                        │
  └─ [simple path] ──────────────────────────────────────────────────────┐
      InferenceEngine.build_generation_plan()                             │
        ├─ IntentClassifier       — keyword + embedding hybrid            │
        ├─ RoutingEngine          — sets run_rag, expert_type flags       │
        ├─ [parallel fan-out]                                             │
        │    ├─ MemoryManager.retrieve_all()  — L1+L2+L3+L4              │
        │    ├─ RAGPipeline.retrieve()        — BM25+FAISS → cross-encoder│
        │    └─ ExpertManager.get_guidance()  — specialist prompt         │
        ├─ AdaptiveFusionEngine   — score + deduplicate + rank            │
        └─ AdaptiveComputeController — temperature clamp [0.05, 2.0]     │
      InferenceEngine.generate() / stream()                               │
        ├─ InferenceBatcher       — semaphore gate (n_parallel slots)     │
        ├─ SpeculativeDecoder     — draft+verify (if configured)          │
        └─ LlamaClient.chat()                                             │
              ├─ CircuitBreaker guard (CLOSED / OPEN / HALF_OPEN)        │
              ├─ 429/503 retry (tenacity, exponential back-off)           │
              └─ EOS token stripping (<|im_end|>, </s>, <|eot_id|>, …)   │
      ResponseCache.set()        — store if cacheable intent              │
      MemoryManager.record_turn() — write L1, async CompressionQueue→L2   │
      PrometheusMetrics          — latency histogram, TTFT, token count   │
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

# 3. Start llama.cpp server (GPU auto-detected if nvidia-smi present)
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
#   websocket_manager, compression_queue, agent_pool, circuit_breaker]}
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
  "metadata": {"multi_agent": false}
}
```

**Response (complex query — auto multi-agent):**
```json
{
  "session_id": "my-session",
  "content": "**Research (t1)**:\n...\n\n---\n\n**Coder (t2)**:\n...",
  "intent": "reasoning",
  "total_time_ms": 185000.0,
  "metadata": {"multi_agent": true, "task_id": "abc123", "sub_tasks": 3}
}
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

The multi-agent engine automatically activates for queries containing complexity signals (`first … then`, `compare`, `step by step`, `analyze`, or queries longer than 200 characters).

### Submit a Task

```bash
curl -X POST http://localhost:8000/v1/agents/tasks \
  -H "Content-Type: application/json" \
  -H "X-API-Key: changeme-in-production" \
  -d '{
    "session_id": "research-session",
    "query": "Compare REST and GraphQL APIs: define each, list pros/cons, and recommend which to use for a mobile app",
    "max_subtasks": 4
  }'
```

**Response:**
```json
{
  "task_id": "9e976ff0-...",
  "session_id": "research-session",
  "status": "done",
  "sub_tasks": [
    {"id": "t1", "role": "research",   "status": "done", "dependencies": [],     "elapsed_ms": 53278},
    {"id": "t2", "role": "reasoner",   "status": "done", "dependencies": ["t1"], "elapsed_ms": 42957},
    {"id": "t3", "role": "summarizer", "status": "done", "dependencies": ["t1", "t2"], "elapsed_ms": 38100}
  ],
  "final_answer": "...",
  "total_elapsed_ms": 134335
}
```

### Poll Task Status

```bash
curl http://localhost:8000/v1/agents/tasks/9e976ff0-... \
  -H "X-API-Key: changeme-in-production"
```

Tasks are persisted to SQLite — retrievable across server restarts.

### Stream Progress (SSE)

```bash
curl -N http://localhost:8000/v1/agents/tasks/9e976ff0-.../stream \
  -H "X-API-Key: changeme-in-production"
```

```
data: {"type": "subtask_done", "subtask_id": "t1", "role": "research", "elapsed_ms": 53278, "result_preview": "REST is..."}
data: {"type": "subtask_done", "subtask_id": "t2", "role": "reasoner", "elapsed_ms": 42957, "result_preview": "GraphQL offers..."}
data: {"type": "complete", "status": "done", "final_answer": "...", "total_elapsed_ms": 134335}
```

### How Multi-Agent Routing Works

```
Query: "First explain X, then compare with Y, and also analyze Z"
  │
  ├─ is_complex() → True  (keywords: "first", "then", "also", "compare")
  │
  ├─ TaskPlanner.plan() → JSON DAG:
  │     t1: research X       (deps: [])
  │     t2: research Y       (deps: [])
  │     t3: compare X vs Y   (deps: [t1, t2])
  │     t4: summarize         (deps: [t1, t2, t3])
  │
  ├─ AgentPool.execute():
  │     Wave 1: t1, t2 concurrently (no deps)
  │     Wave 2: t3 (t1+t2 done)
  │     Wave 3: t4 (t3 done)
  │
  ├─ _synthesize(): returns t4 (SUMMARIZER result)
  │
  └─ Cache the result → next identical query returns in <10ms
```

### Agent Roles

| Role | System prompt focus | Temperature | Max tokens |
|------|---------------------|-------------|------------|
| **Research** | Information gathering and synthesis | 0.3 | 300 |
| **Coder** | Clean, correct, well-commented code | 0.2 | 350 |
| **Reasoner** | Step-by-step logical analysis | 0.4 | 300 |
| **Summarizer** | Coherent synthesis of partial results | 0.5 | 350 |
| **General** | Fallback for unclassified sub-tasks | 0.7 | 300 |

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

# Index statistics
curl http://localhost:8000/v1/documents/stats

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
| `POST` | `/v1/chat` | Chat — auto-routes simple/complex queries |
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
| `POST` | `/v1/admin/classify` | Classify intent without inference |
| `POST` | `/v1/admin/benchmark` | Run latency benchmark |
| `GET`  | `/v1/admin/cache/stats` | Response cache statistics |
| `DELETE` | `/v1/admin/cache` | Flush response cache |
| `GET`  | `/v1/admin/experts` | List available experts |
| `DELETE` | `/v1/admin/sessions/{id}` | Clear session memory |
| `GET`  | `/health` | Full component health check |
| `GET`  | `/health/live` | Kubernetes liveness probe |
| `GET`  | `/health/ready` | Kubernetes readiness probe |
| `GET`  | `/metrics` | Prometheus metrics |
| `GET`  | `/docs` | OpenAPI interactive docs |

---

## Configuration

```env
# ── llama.cpp ────────────────────────────────────────────────────────────────
LLAMA_SERVER_URL=http://localhost:8080
LLAMA_CONTEXT_SIZE=4096
LLAMA_THREADS=8
LLAMA_GPU_LAYERS=0          # set to 99 when a GPU is available
LLAMA_TIMEOUT=120
LLAMA_N_PARALLEL=1          # match --parallel N in start_llama.sh

# ── Speculative decoding (optional — 2-3x speedup) ───────────────────────────
# LLAMA_DRAFT_MODEL_URL=http://localhost:8081
# LLAMA_DRAFT_K_TOKENS=5

# ── Embedding model ──────────────────────────────────────────────────────────
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DEVICE=cpu

# ── RAG ──────────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH=data/faiss_index
CHUNK_SIZE=512
CHUNK_OVERLAP=64
RAG_TOP_K=6
RAG_RERANK_TOP_N=3

# Hybrid search (BM25 + vector) — catches exact ID/name matches
RAG_USE_HYBRID_SEARCH=false
RAG_HYBRID_ALPHA=0.6        # 0.0 = pure BM25, 1.0 = pure semantic

# Cross-encoder reranking — more accurate, ~85MB extra model
RAG_USE_CROSS_ENCODER=false
RAG_CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Semantic chunking — splits at topic boundaries
RAG_USE_SEMANTIC_CHUNKING=false
RAG_SEMANTIC_CHUNK_THRESHOLD=0.75

# ── Memory tiers ─────────────────────────────────────────────────────────────
L1_MAX_TURNS=20
L1_MAX_TOKENS=2048
L2_MAX_TOKENS=4096
L3_SIMILARITY_THRESHOLD=0.72
L4_MAX_RESULTS=5
SQLITE_PATH=data/synapseos.db

# ── Context fusion ───────────────────────────────────────────────────────────
CONTEXT_TOKEN_BUDGET=3200
FUSION_SEMANTIC_WEIGHT=0.45
FUSION_RECENCY_WEIGHT=0.25
FUSION_PRIORITY_WEIGHT=0.20
FUSION_IMPORTANCE_WEIGHT=0.10

# ── Multi-agent system ───────────────────────────────────────────────────────
AGENT_MAX_SUBTASKS=5        # maximum sub-tasks per decomposition
AGENT_MAX_PARALLEL=4        # concurrent agents (capped to LLAMA_N_PARALLEL)
AGENT_TASK_TIMEOUT_S=180    # per-sub-task timeout in seconds

# ── API ──────────────────────────────────────────────────────────────────────
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=changeme-in-production
API_AUTH_ENABLED=false      # set to true to enforce API key
```

### Model Selection

`start_llama.sh` auto-discovers the best `.gguf` from `./models/` by priority:

```
1. Qwen2.5-7B / Qwen2.5-Coder-7B
2. Llama-3.1/3.2-8B
3. Mistral-7B / Mistral-Nemo
4. Gemma-2-9B / Gemma-2-2B
5. DeepSeek-R1-7B
6. Llama-3.2-3B (default bundled model)
7. Phi-3.5-mini / Phi-3-medium
8. Largest .gguf (fallback)
```

Override: `LLAMA_MODEL_PATH=/path/to/model.gguf ./start_llama.sh`

### GPU Acceleration

```bash
./start_llama.sh                    # auto-detect via nvidia-smi
LLAMA_GPU_LAYERS=32 ./start_llama.sh   # partial offload
LLAMA_GPU_LAYERS=99 ./start_llama.sh   # full offload
```

---

## Monitoring

SynapseOS exposes Prometheus metrics at `/metrics`. The Grafana dashboard provides 14 panels:

| Panel | What it shows |
|-------|---------------|
| Request Rate | Requests/s by intent |
| Error Rate | Errors/s |
| Active Sessions | Current open sessions |
| Cache Hit Rate | Response cache effectiveness |
| Latency p50/p95/p99 | End-to-end latency distribution |
| Time to First Token | TTFT p50/p95 |
| Token Throughput | Tokens/second |
| Memory Cache Hits | L1/L2/L3 hits by tier |
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

All results from a local CPU-only run (Llama-3.2-3B-Instruct-Q5_K_M, i5-8350U, 8 threads):

| Test | Result |
|------|--------|
| Simple query → InferenceEngine (not decomposed) | PASS |
| Complex query → multi-agent auto-routing | PASS |
| Task planner decomposition (2–4 subtasks) | PASS |
| Dependency chain (t2 runs after t1 done) | PASS |
| Cascade failure (A→B→C, A fails → B+C skipped) | PASS |
| Task poll `GET /v1/agents/tasks/{id}` | PASS |
| SSE replay for completed task | PASS |
| SQLite persistence survives restart | PASS |
| 404 for unknown task ID | PASS |
| Multi-agent result cached → 152,000ms → 0ms on repeat | PASS |
| Response cache hit (math) → 43,000ms → 9ms | PASS |
| SSE token streaming | PASS |
| WebSocket bidirectional | PASS |
| Multi-turn memory (name recalled across turns) | PASS |
| Circuit breaker state in `/health` | PASS |
| 7 health components all reported | PASS |
| Prometheus metrics (24+ counters/histograms) | PASS |

---

## Testing

```bash
source venv/bin/activate

# Full suite (353 tests)
pytest orchestrator/tests/ -v

# By area
pytest orchestrator/tests/test_v30.py -v             # multi-agent (34 tests)
pytest orchestrator/tests/test_improvements.py -v    # circuit breaker, BLAKE2b, cascade (24 tests)
pytest orchestrator/tests/test_tier2.py -v           # BM25, hybrid RAG, cross-encoder
pytest orchestrator/tests/test_tier3.py -v           # auth, FAISS deletion, doc delete

# With coverage
pytest orchestrator/tests/ --cov=orchestrator --cov-report=term-missing
```

| Test file | What it covers |
|-----------|----------------|
| `test_api.py` | API endpoints — chat, documents, admin, health |
| `test_router.py` | Intent classifier + routing engine |
| `test_memory_manager.py` | MemoryManager fan-out, overflow, queue |
| `test_rag_pipeline.py` | RAG ingest, retrieve, delete |
| `test_fusion.py` | Fusion engine, scorer, deduplicator |
| `test_inference_engine.py` | InferenceEngine, GenerationPlan, cache |
| `test_llama_client.py` | LlamaClient mock + HTTP (circuit breaker, retry) |
| `test_streaming.py` | TokenStreamer, SSE events |
| `test_experts.py` | Expert system plugin registry |
| `test_tier2.py` | BM25, hybrid retriever, cross-encoder, semantic chunking |
| `test_tier3.py` | Auth middleware, FAISS delete/rebuild, doc deletion |
| `test_v30.py` | SharedMemoryBus, TaskPlanner, AgentPool, MultiAgentEngine (34 tests) |
| `test_improvements.py` | Circuit breaker, BLAKE2b keys, temperature clamp, cascade failure (24 tests) |

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
| **v3.0** | ✅ Done | Multi-agent DAG decomposition · AgentPool · SharedMemoryBus · SSE progress · SQLite task persistence · auto-routing in `/v1/chat` |
| **v3.1** | Planned | Whisper ASR integration · voice-first agent mode |
| **v4.0** | Planned | Vision model support · multi-modal context fusion |

---

## Philosophy

> *Intelligence should be orchestrated, not monolithic.*

SynapseOS treats inference as a **distributed reasoning problem**:

- **Complexity-aware routing** — simple queries go direct; complex multi-step tasks are decomposed into specialized sub-agents
- **Compute-aware scheduling** — agent parallelism is capped to the llama server's actual slot count; no timeout flooding
- **Modular specialization** — each agent and expert does one thing well; the orchestrator composes them
- **Memory as infrastructure** — context is retrieved and ranked, not re-derived from scratch each turn
- **Fusion over selection** — all sources are scored and merged within the token budget, not picked arbitrarily
- **Resilience by default** — circuit breaker, 429/503 retry, cascade-failure propagation, async compression with backpressure
- **Privacy by default** — runs entirely on-device; no data is sent to any external service

---

## Contributing

1. Fork the repo and create a feature branch
2. Run the test suite: `pytest orchestrator/tests/ -v`
3. Open a pull request — all 353 tests must pass

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

**SynapseOS v3.0** · Built for developers who believe AI infrastructure should be open, modular, and private.

*Llama-3.2 · llama.cpp · FastAPI · FAISS · BGE · BM25 · Prometheus · Grafana · asyncio*

[GitHub](https://github.com/ihtesham-jahangir/SynapseOS) · [Issues](https://github.com/ihtesham-jahangir/SynapseOS/issues)

</div>
