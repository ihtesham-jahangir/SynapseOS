"""
Tests for v2.0 Speculative Decoding.

Covers:
  - Token utilities (_split_tokens, _tokens_match, _build_context)
  - SpeculativeDecoder disabled path (pass-through to verifier)
  - SpeculativeDecoder enabled path (full concurrent loop)
  - Acceptance rule: partial match, full match + bonus token
  - Error fallback (draft/verifier failure)
  - EOS stop detection
  - Prometheus metrics incremented
  - stream_generate disabled and enabled paths
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.runtime.speculative import (
    SpeculativeDecoder,
    _split_tokens,
    _tokens_match,
    _build_context,
    _contains_eos,
    _strip_eos,
)


# ── Token utility tests ────────────────────────────────────────────────────────

class TestSplitTokens:
    def test_basic_sentence(self):
        tokens = _split_tokens("hello world")
        assert "".join(tokens) == "hello world"
        assert "hello" in tokens
        assert " " in tokens
        assert "world" in tokens

    def test_preserves_whitespace(self):
        text = "  spaces   everywhere  "
        assert "".join(_split_tokens(text)) == text

    def test_empty_string(self):
        assert _split_tokens("") == []

    def test_single_word(self):
        assert _split_tokens("word") == ["word"]

    def test_punctuation_preserved(self):
        tokens = _split_tokens("hello, world!")
        assert "".join(tokens) == "hello, world!"

    def test_newlines_preserved(self):
        text = "line1\nline2"
        assert "".join(_split_tokens(text)) == text


class TestTokensMatch:
    def test_exact_match(self):
        assert _tokens_match("hello", "hello") is True

    def test_case_insensitive(self):
        assert _tokens_match("Hello", "hello") is True
        assert _tokens_match("WORLD", "world") is True

    def test_whitespace_stripped(self):
        assert _tokens_match(" hello ", "hello") is True
        assert _tokens_match("hello", " hello ") is True

    def test_mismatch(self):
        assert _tokens_match("cat", "dog") is False

    def test_punctuation_mismatch(self):
        assert _tokens_match("hello.", "hello") is False


class TestBuildContext:
    def test_no_generated(self):
        messages = [{"role": "user", "content": "hi"}]
        result = _build_context(messages, [])
        assert result == messages

    def test_with_generated(self):
        messages = [{"role": "user", "content": "hi"}]
        result = _build_context(messages, ["hello", " ", "world"])
        assert len(result) == 2
        assert result[-1]["role"] == "assistant"
        assert result[-1]["content"] == "hello world"

    def test_original_messages_unchanged(self):
        messages = [{"role": "user", "content": "test"}]
        _build_context(messages, ["token"])
        assert len(messages) == 1  # original not mutated


class TestEosHandling:
    def test_contains_eos_positive(self):
        assert _contains_eos("some text </s> more") is True
        assert _contains_eos("[/INST] trailing") is True
        assert _contains_eos("<|im_end|>") is True

    def test_contains_eos_negative(self):
        assert _contains_eos("normal text here") is False

    def test_strip_eos(self):
        result = _strip_eos("hello world </s>")
        assert "</s>" not in result
        assert "hello world" in result

    def test_strip_eos_clean_text(self):
        assert _strip_eos("clean text") == "clean text"


# ── SpeculativeDecoder: disabled path ─────────────────────────────────────────

@pytest.mark.asyncio
class TestSpeculativeDecoderDisabled:
    def _make_verifier(self, response: str = "verifier response") -> MagicMock:
        verifier = MagicMock()
        verifier.chat = AsyncMock(return_value=response)
        return verifier

    async def test_enabled_false_when_no_draft(self):
        decoder = SpeculativeDecoder(verifier_client=self._make_verifier())
        assert decoder.enabled is False

    async def test_passes_through_to_verifier(self):
        verifier = self._make_verifier("direct answer")
        decoder = SpeculativeDecoder(verifier_client=verifier)
        messages = [{"role": "user", "content": "hello"}]
        result = await decoder.generate(messages, max_tokens=32)
        assert result == "direct answer"
        verifier.chat.assert_called_once()

    async def test_stream_passes_through_to_verifier(self):
        verifier = MagicMock()

        async def _stream(*args, **kwargs):
            for tok in ["hello", " ", "world"]:
                yield tok

        verifier.stream_chat = _stream
        decoder = SpeculativeDecoder(verifier_client=verifier)
        messages = [{"role": "user", "content": "hi"}]
        tokens = [t async for t in decoder.stream_generate(messages)]
        assert "".join(tokens) == "hello world"


# ── SpeculativeDecoder: enabled path ──────────────────────────────────────────

def _make_client(response: str) -> MagicMock:
    client = MagicMock()
    client.chat = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
class TestSpeculativeDecoderEnabled:
    async def test_enabled_true_when_draft_provided(self):
        decoder = SpeculativeDecoder(
            verifier_client=_make_client("v"),
            draft_client=_make_client("d"),
        )
        assert decoder.enabled is True

    async def test_all_tokens_accepted(self):
        """When draft and verifier agree, all K tokens are accepted."""
        # Both return same text → full acceptance
        draft = _make_client("the cat sat")
        verifier = _make_client("the cat sat on")  # K+1 tokens
        decoder = SpeculativeDecoder(verifier_client=verifier, draft_client=draft, k_tokens=3)
        messages = [{"role": "user", "content": "test"}]

        result = await decoder.generate(messages, max_tokens=10)
        # Should contain the accepted tokens
        assert "the" in result or "cat" in result or "sat" in result

    async def test_divergence_takes_verifier_correction(self):
        """At the first mismatch, verifier's token is used as the correction."""
        # Draft: "cat dog bird"  Verifier: "cat fox eagle"
        # Accept "cat", then diverge → take "fox"
        draft = _make_client("cat dog bird")
        verifier = _make_client("cat fox eagle bonus")
        decoder = SpeculativeDecoder(verifier_client=verifier, draft_client=draft, k_tokens=3)
        messages = [{"role": "user", "content": "animals"}]

        result = await decoder.generate(messages, max_tokens=5)
        assert "cat" in result
        assert "fox" in result
        # "dog" and "bird" should NOT appear (rejected)
        assert "dog" not in result

    async def test_bonus_token_on_full_acceptance(self):
        """When all K draft tokens match, the K+1 verifier token is appended.

        With k=2, _split_tokens("hello world") = ["hello", " ", "world"] and we
        take draft[:2] = ["hello", " "]. The verifier produces ["hello", " ", "world",
        " ", "extra"], so the bonus at index 2 is "world". Using max_tokens=3 forces
        the loop to exit after exactly one step with 3 tokens: "hello", " ", "world".
        """
        draft = _make_client("hello world")
        verifier = _make_client("hello world extra")
        decoder = SpeculativeDecoder(verifier_client=verifier, draft_client=draft, k_tokens=2)
        messages = [{"role": "user", "content": "greet"}]

        # max_tokens=3 → loop exits after 1 step (2 accepted + 1 bonus = 3 tokens)
        result = await decoder.generate(messages, max_tokens=3)
        assert "hello" in result
        assert "world" in result  # the bonus token at position k+1

    async def test_eos_stops_generation(self):
        """EOS token in the output terminates the loop early."""
        draft = _make_client("some text </s> more text")
        verifier = _make_client("some text </s> trailing")
        decoder = SpeculativeDecoder(verifier_client=verifier, draft_client=draft, k_tokens=5)
        messages = [{"role": "user", "content": "test"}]

        result = await decoder.generate(messages, max_tokens=100)
        assert "</s>" not in result  # stripped
        assert "trailing" not in result  # after EOS

    async def test_fallback_on_draft_error(self):
        """If draft call raises, fall back to verifier-only for remaining tokens."""
        draft = MagicMock()
        draft.chat = AsyncMock(side_effect=RuntimeError("draft server down"))
        verifier = _make_client("fallback response text")
        decoder = SpeculativeDecoder(verifier_client=verifier, draft_client=draft, k_tokens=3)
        messages = [{"role": "user", "content": "test"}]

        # Should not raise; should return verifier's fallback text
        result = await decoder.generate(messages, max_tokens=20)
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_empty_verifier_response_stops_loop(self):
        """Empty verifier response terminates the generation loop."""
        draft = _make_client("some tokens here")
        verifier = _make_client("")
        decoder = SpeculativeDecoder(verifier_client=verifier, draft_client=draft, k_tokens=3)
        messages = [{"role": "user", "content": "test"}]

        result = await decoder.generate(messages, max_tokens=50)
        assert isinstance(result, str)

    async def test_stream_generate_returns_all_tokens(self):
        """stream_generate pre-generates then yields token-by-token."""
        draft = _make_client("hello world")
        verifier = _make_client("hello world end")
        decoder = SpeculativeDecoder(verifier_client=verifier, draft_client=draft, k_tokens=2)
        messages = [{"role": "user", "content": "greet"}]

        tokens = [t async for t in decoder.stream_generate(messages, max_tokens=10)]
        assert len(tokens) > 0
        full_text = "".join(tokens)
        assert "hello" in full_text


# ── Metrics ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestSpeculativeMetrics:
    async def test_metrics_incremented_on_generation(self):
        draft = _make_client("one two three")
        verifier = _make_client("one two three bonus")
        decoder = SpeculativeDecoder(verifier_client=verifier, draft_client=draft, k_tokens=3)
        messages = [{"role": "user", "content": "count"}]

        with patch(
            "orchestrator.runtime.speculative.SPECULATIVE_TOKENS_DRAFTED"
        ) as mock_drafted, patch(
            "orchestrator.runtime.speculative.SPECULATIVE_TOKENS_ACCEPTED"
        ) as mock_accepted, patch(
            "orchestrator.runtime.speculative.SPECULATIVE_ACCEPTANCE_RATE"
        ) as mock_rate:
            mock_drafted.inc = MagicMock()
            mock_accepted.inc = MagicMock()
            mock_rate.observe = MagicMock()

            await decoder.generate(messages, max_tokens=20)

            mock_drafted.inc.assert_called_once()
            mock_accepted.inc.assert_called_once()
            mock_rate.observe.assert_called_once()

            # Acceptance rate must be in [0, 1] — bonus tokens are not counted as drafted
            rate_arg = mock_rate.observe.call_args[0][0]
            assert 0.0 <= rate_arg <= 1.0, f"acceptance_rate={rate_arg} out of range"
