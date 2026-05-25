"""
Document management API – ingest and manage RAG knowledge base.

POST   /v1/documents          – ingest a document
GET    /v1/documents/stats    – index statistics
DELETE /v1/documents/index    – rebuild index (admin)
POST   /v1/knowledge          – add to L4 knowledge base
"""
from __future__ import annotations

import asyncio
import time
import uuid
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


class BatchIngestRequest(BaseModel):
    documents: List[IngestRequest]


class BatchIngestResponse(BaseModel):
    results: List[DocumentIngestionResponse]
    total_chunks: int
    total_time_ms: float


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
    # Ensure every ingested document has a stable, returnable ID
    doc_id = request.metadata.get("doc_id") or str(uuid.uuid4())
    metadata = {**request.metadata, "doc_id": doc_id}

    t0 = time.perf_counter()
    try:
        chunks = await rag.ingest(
            content=request.content,
            source=request.source,
            metadata=metadata,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return DocumentIngestionResponse(
        document_id=doc_id,
        chunks_created=chunks,
        tokens_indexed=len(request.content.split()),
        time_ms=elapsed_ms,
    )


@router.post("/batch", response_model=BatchIngestResponse)
async def ingest_batch(
    request: BatchIngestRequest,
    rag: RAGPipeline = Depends(get_rag_pipeline),
) -> BatchIngestResponse:
    """
    Ingest multiple documents concurrently in a single request.
    Each document is chunked and indexed independently; partial failures
    are reported per-document without aborting the rest of the batch.
    """
    if not request.documents:
        raise HTTPException(status_code=400, detail="documents list must not be empty")

    t_batch = time.perf_counter()

    async def _ingest_one(req: IngestRequest) -> DocumentIngestionResponse:
        doc_id = req.metadata.get("doc_id") or str(uuid.uuid4())
        meta = {**req.metadata, "doc_id": doc_id}
        t0 = time.perf_counter()
        chunks = await rag.ingest(content=req.content, source=req.source, metadata=meta)
        return DocumentIngestionResponse(
            document_id=doc_id,
            chunks_created=chunks,
            tokens_indexed=len(req.content.split()),
            time_ms=(time.perf_counter() - t0) * 1000,
        )

    results = await asyncio.gather(
        *[_ingest_one(doc) for doc in request.documents],
        return_exceptions=True,
    )

    responses: List[DocumentIngestionResponse] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            responses.append(DocumentIngestionResponse(
                document_id="error",
                chunks_created=0,
                tokens_indexed=0,
                time_ms=0.0,
            ))
            log.error("Batch ingest item failed", index=i, error=str(result))
        else:
            responses.append(result)

    return BatchIngestResponse(
        results=responses,
        total_chunks=sum(r.chunks_created for r in responses),
        total_time_ms=(time.perf_counter() - t_batch) * 1000,
    )


@router.put("/{doc_id}", response_model=DocumentIngestionResponse)
async def update_document(
    doc_id: str,
    request: IngestRequest,
    rag: RAGPipeline = Depends(get_rag_pipeline),
) -> DocumentIngestionResponse:
    """
    Replace a document's content atomically: delete old chunks then re-ingest
    under the same doc_id.  If the document didn't previously exist, it is
    created (identical to POST with a pre-set doc_id).
    """
    try:
        await rag.delete_document(doc_id)
    except (ValueError, Exception):
        pass  # didn't exist — create fresh

    metadata = {**request.metadata, "doc_id": doc_id}
    t0 = time.perf_counter()
    try:
        chunks = await rag.ingest(content=request.content, source=request.source, metadata=metadata)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Update failed: {exc}") from exc

    return DocumentIngestionResponse(
        document_id=doc_id,
        chunks_created=chunks,
        tokens_indexed=len(request.content.split()),
        time_ms=(time.perf_counter() - t0) * 1000,
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
