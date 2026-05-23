"""
Document management API – ingest and manage RAG knowledge base.

POST   /v1/documents          – ingest a document
GET    /v1/documents/stats    – index statistics
DELETE /v1/documents/index    – rebuild index (admin)
POST   /v1/knowledge          – add to L4 knowledge base
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel

from orchestrator.core.types import DocumentIngestionResponse, IngestRequest
from orchestrator.rag.pipeline import RAGPipeline
from orchestrator.memory.l4_cache import L4KnowledgeBase
from orchestrator.api.dependencies import get_rag_pipeline, get_l4
from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/v1/documents", tags=["documents"])


class KnowledgeAddRequest(BaseModel):
    content: str
    category: str = "general"
    keywords: str = ""
    priority: float = 0.8


class KnowledgeAddResponse(BaseModel):
    id: str
    category: str
    tokens: int


@router.post("", response_model=DocumentIngestionResponse)
async def ingest_document(
    request: IngestRequest,
    rag: RAGPipeline = Depends(get_rag_pipeline),
) -> DocumentIngestionResponse:
    """
    Ingest a text document into the RAG vector index.
    The document is chunked, embedded, and indexed automatically.
    """
    t0 = time.perf_counter()
    try:
        chunks = await rag.ingest(
            content=request.content,
            source=request.source,
            metadata=request.metadata,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return DocumentIngestionResponse(
        document_id=request.metadata.get("doc_id", "unknown"),
        chunks_created=chunks,
        tokens_indexed=sum(
            len(request.content.split()) // max(chunks, 1) for _ in range(chunks)
        ),  # approximate
        time_ms=elapsed_ms,
    )


@router.post("/upload")
async def ingest_file(
    file: UploadFile = File(...),
    rag: RAGPipeline = Depends(get_rag_pipeline),
) -> DocumentIngestionResponse:
    """Upload and ingest a text file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    t0 = time.perf_counter()
    content_bytes = await file.read()

    try:
        content = content_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode file: {exc}") from exc

    try:
        chunks = await rag.ingest(
            content=content,
            source=file.filename,
            metadata={"filename": file.filename},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return DocumentIngestionResponse(
        document_id=file.filename,
        chunks_created=chunks,
        tokens_indexed=len(content.split()),
        time_ms=elapsed_ms,
    )


@router.get("/stats")
async def get_index_stats(
    rag: RAGPipeline = Depends(get_rag_pipeline),
) -> Dict[str, Any]:
    """Return statistics about the RAG vector index."""
    return {
        "total_vectors": rag.index_size,
        "status": "ready",
    }


@router.post("/knowledge", response_model=KnowledgeAddResponse)
async def add_knowledge(
    request: KnowledgeAddRequest,
    l4: L4KnowledgeBase = Depends(get_l4),
) -> KnowledgeAddResponse:
    """
    Add a fact or piece of information to the L4 persistent knowledge base.
    L4 items have highest retrieval priority and don't decay over time.
    """
    doc_id = await l4.add_knowledge(
        content=request.content,
        category=request.category,
        keywords=request.keywords,
        priority=request.priority,
    )
    return KnowledgeAddResponse(
        id=doc_id,
        category=request.category,
        tokens=len(request.content.split()),
    )


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
    rag: RAGPipeline = Depends(get_rag_pipeline),
) -> Dict[str, Any]:
    """
    Remove a document and all its chunks from the vector index.

    FAISS does not support in-place deletion, so the index is rebuilt
    from surviving entries after the soft-delete. This may take a few
    seconds for large indexes.

    Returns the number of chunks removed and the count of remaining vectors.
    """
    try:
        result = await rag.delete_document(doc_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {exc}") from exc
    return result


@router.get("/knowledge/categories")
async def list_knowledge_categories(
    l4: L4KnowledgeBase = Depends(get_l4),
) -> List[str]:
    """List all categories in the L4 knowledge base."""
    return await l4.list_categories()
