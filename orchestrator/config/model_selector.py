"""
Model auto-discovery and selection.

Scans the models/ directory for .gguf files and picks the best one
based on a priority list.  Falls back gracefully so the system always
starts even if no preferred model is present.

Priority logic (first match wins):
  1. High-capability 7-9 B models  (Qwen2.5, Llama-3.1/3.2-8B, Mistral-7B …)
  2. Mid-tier 3-5 B models         (Llama-3.2-3B, Phi-3.5-mini …)
  3. Small fallback models         (Llama-3.2-1B, TinyLlama …)
  4. Largest .gguf found           (unknown model — use biggest file)

Usage:
    from orchestrator.config.model_selector import select_model

    path, reason = select_model()          # uses models/ in cwd
    path, reason = select_model("/abs/path/models")
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

from orchestrator.utils.logging_utils import get_logger

log = get_logger(__name__)


# ── Priority list  ────────────────────────────────────────────────────────────
# Each entry: (substring_to_match_in_filename_lowercase, human_label)
# Case-insensitive substring match against the .gguf filename.
# Put the best models first.

MODEL_PRIORITY: List[Tuple[str, str]] = [
    # ── Tier A: 7-9 B high-capability ────────────────────────────────────────
    ("qwen2.5-7b",          "Qwen2.5-7B — best coding + math"),
    ("qwen2.5-coder-7b",    "Qwen2.5-Coder-7B — elite coding"),
    ("llama-3.1-8b",        "Llama-3.1-8B — strong all-around"),
    ("llama-3.2-8b",        "Llama-3.2-8B — strong all-around"),
    ("llama-3-8b",          "Llama-3-8B — strong all-around"),
    ("mistral-7b",          "Mistral-7B — strong all-around"),
    ("mistral-nemo",        "Mistral-Nemo — strong all-around"),
    ("gemma-2-9b",          "Gemma-2-9B — strong reasoning"),
    ("gemma-2-2b",          "Gemma-2-2B — efficient"),
    ("deepseek-r1-7b",      "DeepSeek-R1-7B — strong reasoning"),
    ("deepseek-coder-6.7b", "DeepSeek-Coder-6.7B — coding"),

    # ── Tier B: 3-5 B mid-range ───────────────────────────────────────────────
    ("llama-3.2-3b",        "Llama-3.2-3B — current default"),
    ("llama-3.2-3",         "Llama-3.2-3B — current default"),
    ("phi-3.5-mini",        "Phi-3.5-mini — fast and capable"),
    ("phi-3-mini",          "Phi-3-mini — fast and capable"),
    ("phi-3-medium",        "Phi-3-medium"),
    ("qwen2.5-3b",          "Qwen2.5-3B"),
    ("stablelm-3b",         "StableLM-3B"),
    ("openhermes",          "OpenHermes — strong instruction following"),

    # ── Tier C: < 2 B fallback ────────────────────────────────────────────────
    ("llama-3.2-1b",        "Llama-3.2-1B — minimal, last resort"),
    ("tinyllama",           "TinyLlama-1.1B — last resort"),
    ("smollm",              "SmolLM — very small"),
]


def select_model(
    models_dir: Optional[str] = None,
) -> Tuple[Optional[Path], str]:
    """
    Discover and return the best available model.

    Args:
        models_dir: Directory to scan. Defaults to ``models/`` relative to
                    the project root (two levels up from this file).

    Returns:
        (path, reason) — path is None only when no .gguf exists at all.
    """
    if models_dir is None:
        # __file__ = orchestrator/config/model_selector.py
        # project root = two levels up
        project_root = Path(__file__).resolve().parent.parent.parent
        models_dir = project_root / "models"

    models_path = Path(models_dir)

    if not models_path.exists():
        return None, f"Models directory not found: {models_path}"

    gguf_files = sorted(models_path.glob("*.gguf"))
    if not gguf_files:
        return None, f"No .gguf files found in {models_path}"

    # Try priority list (case-insensitive substring match)
    for fragment, label in MODEL_PRIORITY:
        for gguf in gguf_files:
            if fragment.lower() in gguf.name.lower():
                reason = f"[model-selector] {label} → {gguf.name}"
                log.info("Model selected", model=gguf.name, tier=label)
                return gguf, reason

    # Fallback: largest file (most parameters = biggest file for same quant)
    largest = max(gguf_files, key=lambda f: f.stat().st_size)
    reason = (
        f"[model-selector] No priority match — falling back to largest file: "
        f"{largest.name} ({largest.stat().st_size // 1_048_576} MB)"
    )
    log.warning("Model fallback", model=largest.name)
    return largest, reason


def list_available_models(models_dir: Optional[str] = None) -> List[dict]:
    """Return metadata for every .gguf found, sorted by priority rank."""
    if models_dir is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        models_dir = project_root / "models"

    models_path = Path(models_dir)
    if not models_path.exists():
        return []

    results = []
    for gguf in sorted(models_path.glob("*.gguf")):
        # Find its priority rank
        rank = len(MODEL_PRIORITY)  # unranked = lowest priority
        label = "unknown"
        for i, (fragment, lbl) in enumerate(MODEL_PRIORITY):
            if fragment.lower() in gguf.name.lower():
                rank = i
                label = lbl
                break
        results.append({
            "name": gguf.name,
            "path": str(gguf),
            "size_mb": round(gguf.stat().st_size / 1_048_576, 1),
            "priority_rank": rank,
            "label": label,
        })

    results.sort(key=lambda x: x["priority_rank"])
    return results
