"""
Adaptive Context Fusion Engine — SynapseOS core.

Combines outputs from all active knowledge paths into a single,
token-budget-optimal context for the LLM runtime.

Active paths:
  ┌─ L1 memory ──── recent conversation turns
  ├─ L4 knowledge ─ curated persistent facts
  ├─ L2 summaries ─ compressed session history
  ├─ L3 vectors ─── semantically retrieved past context
  └─ RAG chunks ─── retrieved document knowledge
         │
         ▼
  [Score each candidate]
  [Deduplicate near-duplicates]
  [Sort by composite relevance score]
  [Fill token budget greedily]
  [Inject expert / system guidance]
         │
         ▼
  FusedContext → LLM runtime
"""
from __future__ import annotations

import time
import uuid
from typing import List, Optional, Tuple

from orchestrator.core.base import BaseFusionEngine
from orchestrator.core.types import (
    ContextItem,
    DocumentChunk,
    ExpertGuidance,
    FusedContext,
    Intent,
    MemoryItem,
    MemoryLevel,
    MemoryResult,
    Message,
    RAGResult,
)
from orchestrator.core.exceptions import FusionError
from orchestrator.utils.logging_utils import get_logger
from orchestrator.utils.token_counter import count_tokens
from orchestrator.utils.metrics import FUSION_ITEMS

from .relevance_scorer import CompositeScorer
from .deduplicator import SemanticDeduplicator, ScoredItem
from .context_builder import AdaptiveContextBuilder

log = get_logger(__name__)


class AdaptiveFusionEngine(BaseFusionEngine):
    """
    Multi-path context fusion for SynapseOS.

    Fusion algorithm:
    1. Collect candidate items from memory tiers + RAG paths
    2. Score each against the query (semantic + recency + priority + importance)
    3. Deduplicate using semantic similarity threshold
    4. Sort by descending composite score
    5. Wrap in ContextItem with token counts
    6. Pass to AdaptiveContextBuilder for budget-aware assembly
    7. Return FusedContext

    Mandatory vs. optional items:
    - L1 conversation turns  → mandatory (always included)
    - L4 knowledge base items → highest-priority optional
    - L2/L3/RAG              → scored and compete for remaining budget
    """

    def __init__(self) -> None:
        self._scorer = CompositeScorer()
        self._dedup = SemanticDeduplicator()
        self._builder = AdaptiveContextBuilder()

    async def fuse(
        self,
        query: str,
        intent: Intent,
        memory_result: MemoryResult,
        rag_result: RAGResult,
        expert_guidance: Optional[ExpertGuidance],
        conversation: List[Message],
        system_prompt: str = "",
    ) -> FusedContext:
        t0 = time.perf_counter()

        try:
            # Step 1: Score all memory items
            scored_memory = self._score_memory_items(memory_result.items)

            # Step 2: Score all RAG chunks
            scored_rag = self._score_rag_items(rag_result.chunks, rag_result.scores)

            # Step 3: Combine into unified candidate pool
            all_candidates: List[ScoredItem] = scored_memory + scored_rag

            FUSION_ITEMS.observe(len(all_candidates))

            # Step 4: Deduplicate
            deduped = self._dedup.deduplicate(all_candidates)

            # Step 5: Convert to ContextItems (sorted by score desc)
            context_items = self._to_context_items(deduped)

            # Step 6: Build final context within token budget
            fused = self._builder.build(
                system_prompt=system_prompt,
                scored_context_items=[(item, item.composite_score) for item in context_items],
                conversation=conversation,
                expert_guidance=expert_guidance,
            )

            elapsed_ms = (time.perf_counter() - t0) * 1000
            log.info(
                "Context fusion complete",
                query_preview=query[:50],
                candidates_in=len(all_candidates),
                after_dedup=len(deduped),
                context_items_out=len(fused.context_items),
                token_budget_used=f"{fused.token_budget_used:.1%}",
                elapsed_ms=f"{elapsed_ms:.1f}",
            )

            fused.metadata.update(
                {
                    "fusion_elapsed_ms": elapsed_ms,
                    "candidates_pre_dedup": len(all_candidates),
                    "candidates_post_dedup": len(deduped),
                    "intent": intent.intent_type.value,
                    "expert": expert_guidance.expert_type.value if expert_guidance else None,
                }
            )

            return fused

        except Exception as exc:
            raise FusionError(f"Context fusion failed: {exc}") from exc

    def _score_memory_items(
        self,
        items: List[MemoryItem],
    ) -> List[ScoredItem]:
        """Score and wrap memory items for deduplication."""
        scored: List[ScoredItem] = []
        for item in items:
            score = self._scorer.score_memory_item(item)
            ctx_item = ContextItem(
                id=item.id,
                content=item.content,
                source=f"memory:{item.level.value}",
                composite_score=score,
                token_count=item.token_count or count_tokens(item.content),
                level=item.level,
                is_mandatory=(item.level == MemoryLevel.L1),
            )
            scored.append((ctx_item, score, None))
        return scored

    def _score_rag_items(
        self,
        chunks: List[DocumentChunk],
        scores: List[float],
    ) -> List[ScoredItem]:
        """Score and wrap RAG chunks for deduplication."""
        scored: List[ScoredItem] = []
        for chunk, retrieval_score in zip(chunks, scores):
            composite = self._scorer.score_rag_chunk(chunk, retrieval_score)
            ctx_item = ContextItem(
                id=chunk.id,
                content=chunk.content,
                source=f"rag:{chunk.source}",
                composite_score=composite,
                token_count=chunk.token_count,
                level=None,
                is_mandatory=False,
            )
            scored.append((ctx_item, composite, None))
        return scored

    def _to_context_items(self, deduped: List[ScoredItem]) -> List[ContextItem]:
        """Extract and sort ContextItems from deduped candidates."""
        items = [item for item, _, _ in deduped]
        # Mandatory items always first, then by score descending
        mandatory = [i for i in items if i.is_mandatory]
        optional = sorted(
            [i for i in items if not i.is_mandatory],
            key=lambda x: x.composite_score,
            reverse=True,
        )
        return mandatory + optional
