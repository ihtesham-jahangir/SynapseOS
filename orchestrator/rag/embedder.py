"""
BGE embedding engine (BAAI/bge-small-en-v1.5).
Lazy-loaded singleton to avoid GPU/CPU memory waste until first use.
Thread-safe batching with async executor wrapper.
"""
from __future__ import annotations

import asyncio
import hashlib
from typing import Dict, List, Optional

import numpy as np

from orchestrator.core.base import BaseEmbedder
from orchestrator.core.exceptions import EmbeddingError
from orchestrator.config.settings import get_settings
from orchestrator.utils.logging_utils import get_logger
from orchestrator.utils.async_utils import run_in_executor, AsyncLRUCache
from orchestrator.utils.circuit_breaker import CircuitBreaker

log = get_logger(__name__)


class BGEEmbedder(BaseEmbedder):
    """
    BAAI/bge-small-en-v1.5 embedding model.

    Produces 384-dimensional L2-normalized embeddings suitable for
    cosine similarity via inner product.
    Uses an async LRU cache for repeated queries.
    """

    def __init__(self) -> None:
        self._model = None
        self._model_lock = asyncio.Lock()
        self._cache: AsyncLRUCache = AsyncLRUCache(maxsize=1024)
        self._settings = get_settings().embedding
        self._circuit = CircuitBreaker(
            name="embedding", failure_threshold=3, recovery_timeout_s=60.0
        )

    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if self._circuit.is_open():
            raise EmbeddingError("Embedding model circuit breaker is open — load failed recently")
        async with self._model_lock:
            if self._model is not None:
                return
            log.info("Loading embedding model", model=self._settings.model)
            try:
                self._model = await run_in_executor(
                    self._load_model, self._settings.model, self._settings.device
                )
                self._circuit.record_success()
                log.info("Embedding model loaded", model=self._settings.model)
            except Exception as exc:
                self._circuit.record_failure()
                raise EmbeddingError(f"Failed to load embedding model: {exc}") from exc

    @staticmethod
    def _load_model(model_name: str, device: str):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name, device=device)

    def _encode_sync(self, texts: List[str]) -> np.ndarray:
        """Synchronous encode call – runs inside thread-pool executor."""
        vectors = self._model.encode(
            texts,
            batch_size=self._settings.batch_size,
            normalize_embeddings=self._settings.normalize,
            show_progress_bar=False,
        )
        return np.array(vectors, dtype=np.float32)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts. Caches individual results to avoid
        re-embedding identical strings (common with system prompts).
        """
        await self._ensure_loaded()

        # Separate cached vs. uncached
        cache_keys = [self._cache_key(t) for t in texts]
        results: Dict[int, List[float]] = {}
        uncached_idx: List[int] = []
        uncached_texts: List[str] = []

        for i, (text, key) in enumerate(zip(texts, cache_keys)):
            cached = await self._cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                uncached_idx.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            if self._circuit.is_open():
                raise EmbeddingError("Embedding circuit breaker is open — skipping inference")
            try:
                vectors = await run_in_executor(self._encode_sync, uncached_texts)
                self._circuit.record_success()
            except Exception as exc:
                self._circuit.record_failure()
                raise EmbeddingError(f"Embedding inference failed: {exc}") from exc

            for local_i, global_i in enumerate(uncached_idx):
                vec = vectors[local_i].tolist()
                results[global_i] = vec
                await self._cache.set(cache_keys[global_i], vec)

        return [results[i] for i in range(len(texts))]

    async def embed_query(self, query: str) -> List[float]:
        """
        Embed a retrieval query with BGE instruction prefix.
        BGE models perform better with the instruction prefix at query time.
        """
        prefixed = f"{self._settings.query_instruction}{query}"
        results = await self.embed([prefixed])
        return results[0]

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    @property
    def dimension(self) -> int:
        return 384  # bge-small-en-v1.5 output dimension
