"""
Tests for LlamaClient – mocked at the httpx layer.

Verifies mock mode, timeout handling, error propagation,
streaming SSE parsing, and health check behaviour.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from orchestrator.core.exceptions import LlamaServerError, LlamaTimeoutError
from orchestrator.runtime.llama_client import LlamaClient


MESSAGES = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"},
]

CHAT_RESPONSE = {
    "choices": [{"message": {"role": "assistant", "content": "Hi there!"}}]
}


# ── Mock mode ─────────────────────────────────────────────────────────────────

class TestMockMode:
    @pytest.mark.asyncio
    async def test_mock_mode_returns_stub(self):
        with patch.dict("os.environ", {"MOCK_LLM": "true"}):
            from orchestrator.config.settings import get_settings
            # Force re-parse settings
            client = LlamaClient.__new__(LlamaClient)
            client._base_url = "http://localhost:8080"
            client._timeout = 120
            client._mock = True
            client._client = None

            result = await client.chat(MESSAGES)
            assert "[MOCK]" in result
            assert "Hello" in result

    @pytest.mark.asyncio
    async def test_mock_stream_yields_tokens(self):
        client = LlamaClient.__new__(LlamaClient)
        client._base_url = "http://localhost:8080"
        client._timeout = 120
        client._mock = True
        client._client = None

        tokens = [t async for t in client.stream_chat(MESSAGES)]
        assert len(tokens) > 0
        reconstructed = "".join(tokens)
        assert "Hello" in reconstructed or "MOCK" in reconstructed


# ── chat endpoint ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestChat:
    async def test_successful_response(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = CHAT_RESPONSE

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        mock_http.is_closed = False

        client = LlamaClient(base_url="http://localhost:8080")
        client._mock = False
        client._client = mock_http

        result = await client.chat(MESSAGES)
        assert result == "Hi there!"

    async def test_timeout_raises_llama_timeout(self):
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))
        mock_http.is_closed = False

        client = LlamaClient(base_url="http://localhost:8080")
        client._mock = False
        client._client = mock_http

        with pytest.raises(LlamaTimeoutError):
            await client.chat(MESSAGES)

    async def test_http_error_raises_llama_server_error(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.text = "Service Unavailable"
        http_error = httpx.HTTPStatusError(
            "503", request=MagicMock(), response=mock_resp
        )

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=http_error)
        mock_http.is_closed = False

        client = LlamaClient(base_url="http://localhost:8080")
        client._mock = False
        client._client = mock_http

        with pytest.raises(LlamaServerError):
            await client.chat(MESSAGES)

    async def test_unexpected_response_shape_raises(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"unexpected": "format"}

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        mock_http.is_closed = False

        client = LlamaClient(base_url="http://localhost:8080")
        client._mock = False
        client._client = mock_http

        with pytest.raises(LlamaServerError):
            await client.chat(MESSAGES)


# ── health_check ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestHealthCheck:
    async def test_returns_true_on_200(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.is_closed = False

        client = LlamaClient(base_url="http://localhost:8080")
        client._mock = False
        client._client = mock_http

        assert await client.health_check() is True

    async def test_returns_false_on_connection_error(self):
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_http.is_closed = False

        client = LlamaClient(base_url="http://localhost:8080")
        client._mock = False
        client._client = mock_http

        assert await client.health_check() is False

    async def test_returns_false_on_non_200(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 503

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.is_closed = False

        client = LlamaClient(base_url="http://localhost:8080")
        client._mock = False
        client._client = mock_http

        assert await client.health_check() is False


# ── close ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestClose:
    async def test_close_calls_aclose(self):
        mock_http = AsyncMock()
        mock_http.is_closed = False

        client = LlamaClient(base_url="http://localhost:8080")
        client._client = mock_http

        await client.close()
        mock_http.aclose.assert_awaited_once()

    async def test_close_when_no_client_is_safe(self):
        client = LlamaClient(base_url="http://localhost:8080")
        client._client = None
        await client.close()  # must not raise
