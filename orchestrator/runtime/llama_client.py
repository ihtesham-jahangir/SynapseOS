"""
llama.cpp HTTP server client.

Connects to a running llama.cpp server (llama-server or llama-cpp-python's
OpenAI-compatible endpoint) and exposes:
  - chat()         – full response
  - stream_chat()  – token-by-token generator
  - complete()     – raw /completion endpoint
  - tokenize()     – token count via server
  - health_check() – liveness probe

Retries with exponential back-off on transient errors.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from orchestrator.core.exceptions import LlamaServerError, LlamaTimeoutError
from orchestrator.config.settings import get_settings
from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)

_RETRY_NETWORK_EXCEPTIONS = (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadTimeout)
_RETRY_STATUS_CODES = frozenset({429, 503})  # rate-limited or server busy

# ChatML / model-specific end-of-turn tokens that leak into content when not
# listed in the server's stop list.  Strip them as a safety net.
_EOS_TOKENS: frozenset[str] = frozenset({
    "<|im_end|>",    # ChatML (Qwen, Mistral-ChatML, many others)
    "<|im_start|>",  # ChatML start (shouldn't appear in output but guard anyway)
    "<|eot_id|>",    # Llama-3 end-of-turn
    "<|end|>",       # Phi-3
    "</s>",          # SentencePiece EOS
})
_DEFAULT_STOP = list(_EOS_TOKENS)


def _is_retryable(exc: BaseException) -> bool:
    """Tenacity predicate: retry on transient network errors and server-busy responses."""
    if isinstance(exc, _RETRY_NETWORK_EXCEPTIONS):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRY_STATUS_CODES
    return False


from orchestrator.utils.circuit_breaker import CircuitBreaker  # shared implementation


class LlamaClient:
    """
    Async HTTP client for the llama.cpp server.

    Supports both the native /completion endpoint and the
    OpenAI-compatible /v1/chat/completions endpoint.
    Set MOCK_LLM=true in the environment to return stub responses without
    a running llama.cpp server (useful for development and testing).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        cfg = get_settings()
        llama_cfg = cfg.llama
        self._base_url = (base_url or llama_cfg.server_url).rstrip("/")
        self._timeout = timeout or llama_cfg.timeout
        self._mock = getattr(cfg, "mock_llm", False)
        self._client: Optional[httpx.AsyncClient] = None
        self._circuit = CircuitBreaker(name="llama_client")

    def _mock_response(self, messages: List[Dict[str, str]]) -> str:
        last_content = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        return (
            f"[MOCK] Received: '{last_content[:60]}'. "
            "Set MOCK_LLM=false and start llama.cpp to get real responses."
        )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout, connect=5.0),
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── OpenAI-compatible chat ────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        reraise=True,
    )
    async def _post_chat(self, client: httpx.AsyncClient, payload: Dict) -> str:
        """Inner call that lets tenacity see raw httpx exceptions for 429/503 retry."""
        resp = await client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        try:
            content: str = data["choices"][0]["message"]["content"]
            for tok in _EOS_TOKENS:
                content = content.replace(tok, "")
            return content.rstrip()
        except (KeyError, IndexError) as exc:
            raise LlamaServerError(f"Unexpected response format: {data}") from exc

    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        cache_prompt: bool = True,
    ) -> str:
        if self._mock:
            return self._mock_response(messages)

        if self._circuit.is_open():
            raise LlamaServerError("Circuit breaker OPEN — llama.cpp is unreachable")

        effective_stop = list(dict.fromkeys(_DEFAULT_STOP + (stop or [])))
        payload: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "stream": False,
            "cache_prompt": cache_prompt,
            "stop": effective_stop,
        }

        client = await self._get_client()
        try:
            result = await self._post_chat(client, payload)
            self._circuit.record_success()
            return result
        except httpx.TimeoutException as exc:
            self._circuit.record_failure()
            raise LlamaTimeoutError() from exc
        except httpx.HTTPStatusError as exc:
            self._circuit.record_failure()
            raise LlamaServerError(
                f"llama.cpp returned {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        cache_prompt: bool = True,
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens from /v1/chat/completions via Server-Sent Events.
        Yields individual token strings as they arrive.
        """
        if self._mock:
            mock_text = self._mock_response(messages)
            for word in mock_text.split():
                yield word + " "
                await asyncio.sleep(0.01)
            return
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "stream": True,
            "cache_prompt": cache_prompt,
            "stop": list(dict.fromkeys(_DEFAULT_STOP + (stop or []))),
        }

        client = await self._get_client()
        try:
            async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip() or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    try:
                        chunk = json.loads(line)
                        delta = chunk["choices"][0].get("delta", {})
                        token = delta.get("content", "")
                        if token and token not in _EOS_TOKENS:
                            yield token
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        except httpx.TimeoutException as exc:
            raise LlamaTimeoutError() from exc
        except httpx.HTTPStatusError as exc:
            raise LlamaServerError(
                f"Streaming error {exc.response.status_code}"
            ) from exc

    # ── Native /completion endpoint ───────────────────────────────────────────

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.3,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Call the native /completion endpoint (single turn, no chat template)."""
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        client = await self._get_client()
        try:
            resp = await client.post("/completion", json=payload)
            resp.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LlamaTimeoutError() from exc
        except httpx.HTTPStatusError as exc:
            raise LlamaServerError(f"Completion error: {exc.response.text[:200]}") from exc

        data = resp.json()
        return data.get("content", "")

    # ── Utility endpoints ─────────────────────────────────────────────────────

    async def tokenize(self, text: str) -> List[int]:
        """Return token IDs for text (uses server's actual tokenizer)."""
        client = await self._get_client()
        try:
            resp = await client.post("/tokenize", json={"content": text})
            resp.raise_for_status()
            return resp.json().get("tokens", [])
        except Exception:
            return []

    async def health_check(self) -> bool:
        client = await self._get_client()
        try:
            resp = await client.get("/health", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def get_model_info(self) -> Dict[str, Any]:
        client = await self._get_client()
        try:
            resp = await client.get("/props")
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return {}
