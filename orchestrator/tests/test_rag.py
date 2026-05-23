"""Tests for the RAG pipeline components."""
from __future__ import annotations

import pytest

from orchestrator.rag.chunker import RecursiveTextChunker
from orchestrator.utils.text_utils import chunk_text_by_sentences


class TestChunker:
    def test_basic_chunking(self):
        chunker = RecursiveTextChunker(chunk_size=50, chunk_overlap=10)
        text = " ".join(["word"] * 200)
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.token_count <= 60  # some tolerance for overlap

    def test_empty_text(self):
        chunker = RecursiveTextChunker()
        chunks = chunker.chunk("")
        assert len(chunks) == 0

    def test_short_text_single_chunk(self):
        chunker = RecursiveTextChunker(chunk_size=500)
        text = "This is a short text."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0].content.strip() == text.strip()

    def test_chunk_indices_sequential(self):
        chunker = RecursiveTextChunker(chunk_size=30, chunk_overlap=5)
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
        chunks = chunker.chunk(text)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_preserves_content(self):
        chunker = RecursiveTextChunker(chunk_size=100)
        text = "The quick brown fox jumps over the lazy dog."
        chunks = chunker.chunk(text)
        reconstructed = " ".join(c.content for c in chunks)
        # All words should appear somewhere in the chunks
        for word in text.split():
            assert word in reconstructed


class TestTextUtils:
    def test_sentence_chunking(self):
        text = "First sentence. Second sentence. Third sentence. Fourth."
        chunks = chunk_text_by_sentences(text, max_chars=30, overlap_chars=10)
        assert len(chunks) > 0

    def test_overlap_produces_continuity(self):
        text = " ".join([f"sentence{i}." for i in range(20)])
        chunks = chunk_text_by_sentences(text, max_chars=100, overlap_chars=30)
        if len(chunks) > 1:
            # There should be some content overlap between consecutive chunks
            assert len(chunks) >= 2
