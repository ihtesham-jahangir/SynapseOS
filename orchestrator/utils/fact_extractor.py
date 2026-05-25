"""
Lightweight fact extractor for L3 semantic memory auto-population.

Extracts personally meaningful facts from conversation turns using
regex heuristics — no extra LLM call needed.

Facts are categorised by type and assigned importance scores so the
fusion engine can weight them correctly later.

Extracted categories:
  • Identity    — name, age, gender
  • Location    — city, country, workplace
  • Profession  — job title, employer
  • Preference  — likes, dislikes, tools used
  • Goal        — what the user is trying to achieve
  • Technology  — programming languages, frameworks mentioned
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

# ── Pattern definitions ───────────────────────────────────────────────────────
# Each entry: (compiled_regex, fact_template, importance_score)
# fact_template uses {match} placeholder for the captured group.

@dataclass(frozen=True)
class FactPattern:
    regex: re.Pattern
    template: str        # human-readable sentence to store
    importance: float
    category: str


_RAW_PATTERNS: List[Tuple[str, str, float, str]] = [
    # ── Identity ──────────────────────────────────────────────────────────────
    (
        r"my name is ([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)?)",
        "User's name is {match}.",
        0.97, "identity",
    ),
    (
        r"(?:i am|i'm) ([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)?)",
        "User is {match}.",
        0.80, "identity",
    ),
    (
        r"(?:i am|i'm) (\d{1,2}) years? old",
        "User is {match} years old.",
        0.85, "identity",
    ),
    # ── Location ─────────────────────────────────────────────────────────────
    (
        r"(?:i live|i'm based|i am based) in ([A-Z][a-zA-Z ,]+?)(?:\.|,|$)",
        "User lives in {match}.",
        0.90, "location",
    ),
    (
        r"(?:i work|i'm working|i am working) in ([A-Z][a-zA-Z ,]+?)(?:\.|,|$)",
        "User works in {match}.",
        0.88, "location",
    ),
    (
        r"(?:i'm from|i am from) ([A-Z][a-zA-Z ,]+?)(?:\.|,|$)",
        "User is from {match}.",
        0.88, "location",
    ),
    # ── Profession ────────────────────────────────────────────────────────────
    (
        r"(?:i am|i'm) (?:a|an) ([a-zA-Z ]+?(?:engineer|developer|designer|scientist|manager|"
        r"analyst|researcher|architect|consultant|teacher|doctor|lawyer|writer|student))",
        "User is a {match}.",
        0.93, "profession",
    ),
    (
        r"i work (?:as|for) ([a-zA-Z ]+?)(?:\.|,|$)",
        "User works as/for {match}.",
        0.88, "profession",
    ),
    (
        r"i(?:'m| am) (?:employed|hired) (?:at|by) ([A-Z][a-zA-Z ]+?)(?:\.|,|$)",
        "User is employed at {match}.",
        0.90, "profession",
    ),
    # ── Preferences ───────────────────────────────────────────────────────────
    (
        r"i (?:prefer|like|love|enjoy|use|use mainly) ([a-zA-Z0-9\. ]+?)(?:\.|,| over | instead|$)",
        "User prefers/uses {match}.",
        0.75, "preference",
    ),
    (
        r"my (?:favorite|favourite|preferred|go-to) .{0,20}? is ([a-zA-Z0-9\. ]+?)(?:\.|,|$)",
        "User's favourite is {match}.",
        0.78, "preference",
    ),
    (
        r"i (?:don't like|hate|dislike|avoid) ([a-zA-Z0-9\. ]+?)(?:\.|,|$)",
        "User dislikes {match}.",
        0.72, "preference",
    ),
    # ── Technology / Tools ────────────────────────────────────────────────────
    (
        r"i (?:use|write|program in|code in|develop in|build with) "
        r"(Python|JavaScript|TypeScript|Rust|Go|Java|C\+\+|C#|Ruby|Swift|Kotlin|PHP|"
        r"React|Vue|Angular|Django|FastAPI|Node\.js|TensorFlow|PyTorch|Docker|Kubernetes)",
        "User works with {match}.",
        0.80, "technology",
    ),
    # ── Goals ────────────────────────────────────────────────────────────────
    (
        r"i(?:'m| am) (?:trying|working|learning) to ([a-zA-Z ]+?)(?:\.|,|$)",
        "User is trying to {match}.",
        0.70, "goal",
    ),
    (
        r"i want to ([a-zA-Z ]+?)(?:\.|,|$)",
        "User wants to {match}.",
        0.65, "goal",
    ),
]

_PATTERNS: List[FactPattern] = [
    FactPattern(
        regex=re.compile(raw, re.IGNORECASE),
        template=tmpl,
        importance=imp,
        category=cat,
    )
    for raw, tmpl, imp, cat in _RAW_PATTERNS
]

# Minimum word count for a captured match to be meaningful
_MIN_WORDS = 1
_MAX_WORDS = 8


def extract_facts(text: str) -> List[Tuple[str, float]]:
    """
    Extract (fact_sentence, importance) pairs from a user utterance.

    Designed to be fast (<1 ms) — no model calls.
    Returns an empty list when nothing meaningful is found.
    """
    facts: List[Tuple[str, float]] = []
    seen_content: set = set()

    for pattern in _PATTERNS:
        m = pattern.regex.search(text)
        if m is None:
            continue

        captured = m.group(1).strip().rstrip(".,;:")
        word_count = len(captured.split())
        if word_count < _MIN_WORDS or word_count > _MAX_WORDS:
            continue

        fact_text = pattern.template.format(match=captured)
        if fact_text in seen_content:
            continue

        seen_content.add(fact_text)
        facts.append((fact_text, pattern.importance))

    return facts


def should_extract(user_message: str) -> bool:
    """
    Quick gate: is this message worth running through the extractor?
    Avoids regex overhead on very short or pure-question messages.
    """
    text = user_message.strip()
    if len(text) < 10:
        return False
    # Must contain a first-person reference to be about the user
    return bool(re.search(r"\b(i|my|i'm|i am|i've|i work|i live)\b", text, re.IGNORECASE))
