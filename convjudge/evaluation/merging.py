"""Cross-attempt error merging for LLM-judge evaluation.

When ``--n N`` is passed to ``convjudge.inference.llm_api`` / ``llm_vllm``,
each conversation has N ``_usage_attempt{k}.json`` outputs. This module
consolidates the per-attempt error lists into a single ``_usage_errors.json``
per conversation, recording which turns were wrong in how many attempts and
which distinct predictions the model made.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def add_merged(
    target: dict[int, dict[str, Any]],
    turn: dict[str, Any],
    attempt_id: str,
) -> None:
    ti = turn.get("turn_index")
    if not isinstance(ti, int):
        return
    rec = target.setdefault(ti, {
        "turn_index": ti,
        "truth": turn.get("truth"),
        "debug": turn.get("debug"),
        "_attempt_ids": set(),
        "_pred_variants": {},
    })
    rec["_attempt_ids"].add(attempt_id)
    pred = turn.get("pred") if isinstance(turn.get("pred"), dict) else {}
    pred_key = json.dumps(pred, sort_keys=True, ensure_ascii=False)
    pv = rec["_pred_variants"].setdefault(pred_key, {"pred": pred, "count": 0})
    pv["count"] += 1


def finalize_merged(
    merged: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int], int, int]:
    """Flatten a merged dict into (incorrect_turns, turn_counts, attempts_with_errors, total)."""
    incorrect_turns: list[dict[str, Any]] = []
    turn_error_attempt_counts: dict[str, int] = {}
    attempts_with_errors: set[str] = set()
    incorrect_turns_total = 0

    for ti in sorted(merged):
        rec = merged[ti]
        attempt_ids: set[str] = rec.get("_attempt_ids") or set()
        pred_variants = sorted(
            rec.get("_pred_variants", {}).values(),
            key=lambda x: -int(x.get("count") or 0),
        )
        attempts_with_errors.update(attempt_ids)
        turn_error_attempt_counts[str(ti)] = len(attempt_ids)
        incorrect_turns_total += sum(int(v.get("count") or 0) for v in pred_variants)
        incorrect_turns.append({
            "turn_index": ti,
            "attempt_count": len(attempt_ids),
            "pred_variants": pred_variants,
            "truth": rec.get("truth"),
            "debug": rec.get("debug"),
        })

    return incorrect_turns, turn_error_attempt_counts, len(attempts_with_errors), incorrect_turns_total


def write_or_delete(
    dir_path: Path,
    merged: dict[int, dict[str, Any]],
    *,
    file_name: str,
    conv_id: str,
    convo_file: str | None,
    convo: dict[str, Any] | None,
    num_paths: int,
    error_group: str,
) -> None:
    """Write an error JSON, or delete an existing one when no errors remain."""
    path = dir_path / file_name
    incorrect_turns, turn_counts, attempts_with_errors, incorrect_total = finalize_merged(merged)
    if incorrect_turns:
        dir_path.mkdir(parents=True, exist_ok=True)
        doc = {
            "error_group": error_group,
            "conversation_id": conv_id,
            "conversation_file": convo_file,
            "attempts_total": num_paths,
            "attempts_with_errors": attempts_with_errors,
            "incorrect_turns_unique": len(incorrect_turns),
            "incorrect_turns_total": incorrect_total,
            "turn_error_attempt_counts": turn_counts,
            "incorrect_turns": incorrect_turns,
            "conversation": convo,
        }
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


__all__ = ["add_merged", "finalize_merged", "write_or_delete"]
