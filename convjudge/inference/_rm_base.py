"""Shared helpers for reward-model scoring pipelines.

Used by ``rm_classifier.py`` (classifier-head RMs like Skywork, ArmoRM) and
``rm_generative_scalar.py`` (generative RMs that emit a numeric score in
their text output, e.g. DeepSeek-GRM).
"""
from __future__ import annotations

from typing import Any

from convjudge.common.shared import normalize_category


def _get_guideline_text(
    guidelines: dict[str, Any], category: str, key: str, phase: int
) -> str | None:
    """Look up the oracle text for (category, key, phase)."""
    norm = normalize_category(category)
    for cat_name, cat_data in guidelines.items():
        if normalize_category(cat_name) != norm:
            continue
        if not isinstance(cat_data, dict) or key not in cat_data:
            continue
        val = cat_data[key]
        if isinstance(val, dict):
            text = val.get(f"Phase {phase}") or val.get(str(phase))
            return str(text) if text is not None else None
        return str(val)
    return None


def collect_labeled_turns(convo: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every labeled assistant turn, violation-overriding.

    Each entry has: ``turn_index, category, key, phase, guideline_text,
    is_violation_truth``. Violation turns (from ``mistakes``) override
    non-violation entries at the same ``turn_index``.
    """
    guidelines = convo.get("assistant_guidelines", {})

    violation_index: dict[int, dict[str, Any]] = {}
    for m in convo.get("mistakes", []):
        ti = int(m.get("turn_index", -1))
        cat = str(m.get("guidance category", ""))
        key = str(m.get("guidance key", ""))
        phase = int(m.get("guideline_phase", -1))
        violation_index[ti] = {
            "turn_index": ti,
            "category": cat,
            "key": key,
            "phase": phase,
            "guideline_text": _get_guideline_text(guidelines, cat, key, phase) or str(m.get("guideline", "")),
            "is_violation_truth": True,
        }

    result: dict[int, dict[str, Any]] = {}
    for msg in convo.get("message_list", []):
        if msg.get("role") != "assistant":
            continue
        cat = str(msg.get("category", "")).strip()
        key = str(msg.get("key", "")).strip()
        if not cat or not key:
            continue
        ti = int(msg.get("turn_index", -1))
        if ti in violation_index:
            result[ti] = violation_index[ti]
        else:
            phase = int(msg.get("phase", -1))
            result[ti] = {
                "turn_index": ti,
                "category": cat,
                "key": key,
                "phase": phase,
                "guideline_text": _get_guideline_text(guidelines, cat, key, phase) or str(msg.get("guideline_text", "")),
                "is_violation_truth": False,
            }

    for ti, vt in violation_index.items():
        result.setdefault(ti, vt)

    return sorted(result.values(), key=lambda x: x["turn_index"])


__all__ = ["collect_labeled_turns"]
