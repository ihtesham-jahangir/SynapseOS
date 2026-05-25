"""
Chat API routes – the primary inference endpoints.

POST /v1/chat        – full response (non-streaming)
POST /v1/chat/stream – SSE token streaming
WS   /v1/ws/{session_id} – WebSocket bidirectional
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from orchestrator.core.types import (
    ChatRequest,
    ChatResponse,
    GenerationParams,
    MessageRole,
    Message,
    StreamEventType,
)
from orchestrator.core.exceptions import LlamaServerError, LlamaTimeoutError, OrchestratorError
from orchestrator.runtime.inference_engine import InferenceEngine
from orchestrator.agents.multi_agent_engine import MultiAgentEngine
from orchestrator.streaming.token_streamer import TokenStreamer
from orchestrator.streaming.websocket_manager import WebSocketManager
from orchestrator.utils.logging_utils import get_logger, bind_request_context

from orchestrator.api.dependencies import get_inference_engine, get_multi_agent_engine, get_ws_manager
from orchestrator.config.settings import get_settings

log = get_logger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    engine: MultiAgentEngine = Depends(get_multi_agent_engine),
) -> ChatResponse:
    """
    Full-response chat endpoint.
    Complex queries are automatically decomposed and executed via the multi-agent
    system; simple queries are forwarded directly to InferenceEngine.
    """
    try:
        return await engine.process_request(request)
    except LlamaTimeoutError as exc:
        raise HTTPException(status_code=504, detail="LLM inference timed out") from exc
    except LlamaServerError as exc:
        raise HTTPException(status_code=502, detail=f"LLM backend error: {exc.message}") from exc
    except OrchestratorError as exc:
        raise HTTPException(status_code=500, detail=exc.message) from exc


@router.post("/chat/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    inference: InferenceEngine = Depends(get_inference_engine),
    multi_agent: MultiAgentEngine = Depends(get_multi_agent_engine),
) -> StreamingResponse:
    """
    Server-Sent Events streaming endpoint.

    Simple queries stream token-by-token via InferenceEngine.
    Complex multi-step queries are decomposed by MultiAgentEngine; sub-task
    progress events are emitted, then the final answer is streamed.

    Event format:
        data: {"event_type": "token", "content": "Hello", "done": false}
        data: {"event_type": "subtask_done", "role": "...", "done": false}
        data: {"event_type": "done", "done": true, "metadata": {...}}
    """
    from orchestrator.api.dependencies import get_container
    session_id = request.session_id
    query = request.last_user_message

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            cfg = get_settings()
            is_complex = (
                cfg.agent.enabled
                and multi_agent._planner is not None
                and multi_agent._planner.is_complex(query)
            )

            if is_complex:
                # Multi-agent path: run decomposed plan, then stream the result word-by-word
                result = await multi_agent.process_request(request)
                final_text = result.content or ""
                for chunk in final_text.split():
                    event = {"event_type": "token", "content": chunk + " ", "done": False}
                    yield f"data: {json.dumps(event)}\n\n"
                done_event = {
                    "event_type": "done",
                    "done": True,
                    "metadata": {
                        "session_id": session_id,
                        "multi_agent": True,
                        **result.metadata,
                    },
                }
                yield f"data: {json.dumps(done_event)}\n\n"
            else:
                # Direct streaming path
                plan = await inference.build_generation_plan(request)
                streamer = TokenStreamer(session_id=session_id)
                full_content = []

                async for event in streamer.stream(inference.stream(plan.fused, plan.params)):
                    yield f"data: {json.dumps(event.model_dump())}\n\n"
                    if event.event_type == StreamEventType.TOKEN:
                        full_content.append(event.content)

                user_msg = Message(role=MessageRole.USER, content=query)
                asst_msg = Message(role=MessageRole.ASSISTANT, content="".join(full_content))
                asyncio.create_task(inference._memory.record_turn(session_id, user_msg, asst_msg))

        except Exception as exc:
            log.error("SSE stream error", error=str(exc))
            error_event = json.dumps({"event_type": "error", "content": str(exc), "done": True})
            yield f"data: {error_event}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/chat/history/{session_id}")
async def get_session_history(
    session_id: str,
    engine: InferenceEngine = Depends(get_inference_engine),
) -> dict:
    """
    Return the conversation history for a session.

    Includes L1 turns (recent in-memory) and L2 summaries (older compressed).
    Use this to restore a chat UI after page reload without resending messages.
    """
    turns = await engine._memory.get_conversation(session_id)
    l2_items = await engine._memory._l2.retrieve("", session_id, limit=20)
    return {
        "session_id": session_id,
        "turns": [
            {
                "role": m.role.value,
                "content": m.content,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None,
            }
            for m in turns
        ],
        "summaries": [
            {
                "id": s.id,
                "content": s.content,
                "importance": s.importance_score,
                "timestamp": s.timestamp.isoformat() if s.timestamp else None,
            }
            for s in l2_items
        ],
        "total_turns": len(turns),
    }


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    api_key: Optional[str] = Query(default=None, alias="api_key"),
    engine: InferenceEngine = Depends(get_inference_engine),
    ws_manager: WebSocketManager = Depends(get_ws_manager),
) -> None:
    """
    WebSocket endpoint for bidirectional streaming.

    Client sends:
        {"messages": [...], "stream": true}

    Server sends token-by-token:
        {"event_type": "token", "content": "Hello", "done": false}
        {"event_type": "done", "done": true, "metadata": {...}}
    """
    cfg = get_settings().api
    if cfg.auth_enabled and api_key != cfg.key:
        await websocket.close(code=4003, reason="Invalid or missing API key")
        return

    connected = await ws_manager.connect(websocket, session_id)
    if not connected:
        return
    bind_request_context(session_id)

    _WS_HEARTBEAT_INTERVAL = 30  # seconds between server→client pings

    async def _heartbeat() -> None:
        """Send a ping frame every 30 s to detect dead TCP connections early."""
        try:
            while True:
                await asyncio.sleep(_WS_HEARTBEAT_INTERVAL)
                await websocket.send_text(json.dumps({"event_type": "ping"}))
        except Exception:
            pass  # connection already dead; outer loop will handle cleanup

    heartbeat_task = asyncio.create_task(_heartbeat())
    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=300.0)
            except asyncio.TimeoutError:
                log.debug("WebSocket idle timeout — closing", session=session_id)
                break
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await ws_manager.send_error(session_id, "Invalid JSON")
                continue

            def _safe_role(raw: str) -> MessageRole:
                try:
                    return MessageRole(raw)
                except ValueError:
                    return MessageRole.USER

            request = ChatRequest(
                session_id=session_id,
                messages=[
                    Message(role=_safe_role(m.get("role", "user")), content=m.get("content", ""))
                    for m in data.get("messages", [])
                ],
                stream=True,
            )

            query = request.last_user_message
            if not query:
                continue

            plan = await engine.build_generation_plan(request)
            full_content: list = []
            streamer = TokenStreamer(session_id=session_id)

            async for event in streamer.stream(engine.stream(plan.fused, plan.params)):
                await ws_manager.send_event(session_id, event)
                if event.event_type == StreamEventType.TOKEN:
                    full_content.append(event.content)

            user_msg = Message(role=MessageRole.USER, content=query)
            asst_msg = Message(role=MessageRole.ASSISTANT, content="".join(full_content))
            asyncio.create_task(engine._memory.record_turn(session_id, user_msg, asst_msg))

    except WebSocketDisconnect:
        log.debug("WebSocket client disconnected", session=session_id)
    except Exception as exc:
        log.error("WebSocket error", session=session_id, error=str(exc))
        try:
            await ws_manager.send_error(session_id, str(exc))
        except Exception:
            pass
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        await ws_manager.disconnect(websocket, session_id)
