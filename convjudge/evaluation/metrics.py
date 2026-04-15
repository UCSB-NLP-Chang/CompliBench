"""Per-turn evaluation metrics for guideline usage accuracy."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from convjudge.common.shared import normalize_category


def _get_oracle_guideline_text(
    convo: dict[str, Any], category: str, key: str, phase: int
) -> str | None:
    for cat_name, cat_data in convo.get("assistant_guidelines", {}).items():
        if normalize_category(cat_name) != category:
            continue
        if not isinstance(cat_data, dict) or key not in cat_data:
            continue
        val = cat_data[key]
        return val.get(f"Phase {phase}") if isinstance(val, dict) else str(val)
    return None


def _get_violation_directive(
    convo: dict[str, Any], category: str, key: str, phase: int
) -> dict[str, Any] | None:
    """Look up the violation directive (original vs modified) for a given turn."""
    for ov in convo.get("cat2_overrides", []):
        if (
            normalize_category(str(ov.get("category", ""))) == category
            and ov.get("intent") == key
            and ov.get("phase") == f"Phase {phase}"
        ):
            return {
                "category": ov["category"],
                "key": key,
                "phase": phase,
                "original": ov.get("original"),
                "modified": ov.get("modified"),
            }
    for ov in convo.get("cat3_overrides", []):
        if (
            normalize_category(str(ov.get("category", ""))) == category
            and ov.get("key") == key
        ):
            return {
                "category": ov["category"],
                "key": key,
                "phase": phase,
                "original": ov.get("original"),
                "modified": ov.get("modified"),
            }
    return None


def find_convo_path(usage_path: Path, data_root: Path) -> Path:
    """Resolve the conversation JSON path from a usage output path."""
    stem = usage_path.stem
    if stem.endswith("_usage"):
        convo_name = stem[:-6] + ".json"
    elif "_usage_attempt" in stem:
        convo_name = stem.split("_usage_attempt", 1)[0] + ".json"
    else:
        convo_name = stem + ".json"
    for candidate in (
        data_root / usage_path.parent.name / convo_name,
        data_root / usage_path.parent.parent.name / convo_name,
        data_root / convo_name,
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Conversation JSON not found for {usage_path}")


def build_truth(convo: dict[str, Any]) -> dict[int, tuple[str, str, int, bool]]:
    """Extract ground truth turn annotations from a conversation."""
    truth: dict[int, tuple[str, str, int, bool]] = {}
    for msg in convo.get("message_list", []):
        if msg.get("role") != "assistant" or not msg.get("category") or not msg.get("key"):
            continue
        ti = int(msg.get("turn_index", -1))
        truth[ti] = (
            normalize_category(str(msg["category"])),
            str(msg["key"]),
            int(msg.get("phase", -1)),
            False,
        )
    for item in convo.get("mistakes", []):
        ti = int(item.get("turn_index", -1))
        truth[ti] = (
            normalize_category(str(item.get("guidance category", ""))),
            str(item.get("guidance key", "")),
            int(item.get("guideline_phase", -1)),
            True,
        )
    return truth


def build_pred(usage: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Extract turn-level predictions from a usage output."""
    pred: dict[int, dict[str, Any]] = {}
    for item in usage.get("turn_guidelines", []):
        ti = int(item.get("turn_index", -1))
        guid = item.get("guideline_used") or {}
        pred[ti] = {
            "category": normalize_category(str(guid.get("guidance_category", ""))),
            "key": str(guid.get("guidance_key", "")),
            "phase": int(guid.get("guideline_phase", -1)),
            "is_violation": bool(guid.get("is_violation")),
            "judge_reason": str(guid.get("judge_reason", "")),
        }
    return pred


def evaluate_file(usage_path: Path, data_root: Path) -> dict[str, Any]:
    """Evaluate a single usage file against its conversation ground truth."""
    usage = json.loads(usage_path.read_text(encoding="utf-8"))
    convo_path = find_convo_path(usage_path, data_root)
    convo = json.loads(convo_path.read_text(encoding="utf-8"))

    truth_by_turn = build_truth(convo)
    pred_by_turn = build_pred(usage)
    msg_by_turn: dict[int, dict[str, Any]] = {
        int(m.get("turn_index", -1)): m
        for m in convo.get("message_list", [])
        if m.get("role") == "assistant"
    }

    non_violation_total = non_violation_correct = 0
    m1_mismatch_count = 0  # any wrong guideline match (incl. phase/key/category/off-by-phase/missing)
    m1_false_violation_count = 0  # guideline matches but incorrectly flagged as violation
    violation_total = violation_detect_correct = violation_strict_correct = 0
    incorrect_turns_metric1: list[dict[str, Any]] = []
    incorrect_turns_metric2: list[dict[str, Any]] = []
    incorrect_turns_metric3: list[dict[str, Any]] = []

    def _build_error_turn(
        *,
        ti: int,
        truth_cat: str,
        truth_key: str,
        truth_phase: int,
        truth_violation: bool,
        pred: dict[str, Any] | None,
        pred_cat: str,
        pred_key: str,
        pred_phase: int,
        pred_violation: bool,
    ) -> dict[str, Any]:
        judge_reason = pred.get("judge_reason", "") if pred else ""
        msg = msg_by_turn.get(ti, {})
        vdir = _get_violation_directive(convo, truth_cat, truth_key, truth_phase)
        oracle_text = _get_oracle_guideline_text(convo, truth_cat, truth_key, truth_phase)
        pred_guideline_text = None
        if pred_cat and pred_key:
            pred_guideline_text = _get_oracle_guideline_text(convo, pred_cat, pred_key, pred_phase)
        return {
            "turn_index": ti,
            "pred": {
                "category": pred_cat,
                "key": pred_key,
                "phase": pred_phase,
                "is_violation": pred_violation,
                "judge_reason": judge_reason,
                "guideline_text": pred_guideline_text,
            },
            "truth": {
                "category": truth_cat,
                "key": truth_key,
                "phase": truth_phase,
                "is_violation": truth_violation,
                "judge_reason": judge_reason,
            },
            "debug": {
                "assistant_content": msg.get("content"),
                "message_guideline_text": msg.get("guideline_text"),
                "oracle_guideline_text": oracle_text,
                "violation_directive": vdir,
                "modified_guideline_text": vdir["modified"] if vdir else None,
            },
        }

    err_kwargs = dict(
        truth_cat="", truth_key="", truth_phase=-1, truth_violation=False,
        pred=None, pred_cat="", pred_key="", pred_phase=-1, pred_violation=False,
    )

    for ti, (truth_cat, truth_key, truth_phase, truth_violation) in truth_by_turn.items():
        pred = pred_by_turn.get(ti)
        pred_violation = bool(pred.get("is_violation")) if pred else False
        pred_cat = pred["category"] if pred else ""
        pred_key = pred["key"] if pred else ""
        pred_phase = pred["phase"] if pred else -1
        err_turn_kwargs = dict(
            ti=ti,
            truth_cat=truth_cat, truth_key=truth_key, truth_phase=truth_phase, truth_violation=truth_violation,
            pred=pred, pred_cat=pred_cat, pred_key=pred_key, pred_phase=pred_phase, pred_violation=pred_violation,
        )

        if truth_violation:
            violation_total += 1
            if pred_violation:
                violation_detect_correct += 1
            else:
                incorrect_turns_metric2.append(_build_error_turn(**err_turn_kwargs))
            violation_strict_ok = (
                pred and pred_cat == truth_cat and pred_key == truth_key
                and pred_phase == truth_phase and pred_violation is True
            )
            if violation_strict_ok:
                violation_strict_correct += 1
            else:
                incorrect_turns_metric3.append(_build_error_turn(**err_turn_kwargs))
        else:
            non_violation_total += 1
            non_violation_ok = (
                pred and pred_cat == truth_cat and pred_key == truth_key
                and pred_phase == truth_phase and pred_violation is False
            )
            if non_violation_ok:
                non_violation_correct += 1
            else:
                if (
                    pred
                    and pred_cat == truth_cat
                    and pred_key == truth_key
                    and pred_phase == truth_phase
                    and pred_violation
                ):
                    m1_false_violation_count += 1
                else:
                    m1_mismatch_count += 1
                incorrect_turns_metric1.append(_build_error_turn(**err_turn_kwargs))

    for lst in (incorrect_turns_metric1, incorrect_turns_metric2, incorrect_turns_metric3):
        lst.sort(key=lambda x: x["turn_index"])

    def _safe_rate(num: int, denom: int) -> float | None:
        return round(num / denom, 4) if denom else None

    non_viol_ok = non_violation_total == 0 or non_violation_correct == non_violation_total
    viol_ok = violation_total == 0 or violation_detect_correct == violation_total
    viol_strict_ok = violation_total == 0 or violation_strict_correct == violation_total

    return {
        "conversation_file": str(convo_path),
        "usage_file": str(usage_path),
        "macro_correct_accuracy_strict": _safe_rate(non_violation_correct, non_violation_total),
        "macro_accuracy_violation_detect": _safe_rate(violation_detect_correct, violation_total),
        "macro_correct_accuracy_violation_strict": _safe_rate(violation_strict_correct, violation_total),
        "file_correct_accuracy_strict": non_viol_ok and viol_ok,
        "file_correct_accuracy_strict_m1_m3": non_viol_ok and viol_strict_ok,
        "non_violation_total": non_violation_total,
        "non_violation_correct": non_violation_correct,
        "m1_mismatch_count": m1_mismatch_count,
        "m1_false_violation_count": m1_false_violation_count,
        "violation_total": violation_total,
        "violation_detect_correct": violation_detect_correct,
        "violation_strict_correct": violation_strict_correct,
        "incorrect_turns_metric1": incorrect_turns_metric1,
        "incorrect_turns_metric2": incorrect_turns_metric2,
        "incorrect_turns_metric3": incorrect_turns_metric3,
        "conversation": convo,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-file evaluation results into micro and macro summary metrics."""
    if not results:
        return {
            "micro_correct_accuracy_strict": 0.0,
            "micro_accuracy_violation_detect": 0.0,
            "micro_correct_accuracy_violation_strict": 0.0,
            "macro_correct_accuracy_strict": 0.0,
            "macro_accuracy_violation_detect": 0.0,
            "macro_correct_accuracy_violation_strict": 0.0,
            "file_correct_accuracy_strict": 0.0,
            "file_correct_accuracy_strict_m1_m3": 0.0,
        }

    def _micro(num_key: str, denom_key: str) -> float | None:
        num = sum(int(r.get(num_key) or 0) for r in results)
        denom = sum(int(r.get(denom_key) or 0) for r in results)
        return round(num / denom, 4) if denom else None

    def _macro(key: str) -> float | None:
        vals = [r.get(key) for r in results if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    file_correct_count = sum(1 for r in results if r["file_correct_accuracy_strict"])
    file_correct_m1_m3_count = sum(1 for r in results if r.get("file_correct_accuracy_strict_m1_m3"))
    file_correct = round(file_correct_count / len(results), 4)
    file_correct_m1_m3 = round(file_correct_m1_m3_count / len(results), 4)

    return {
        "micro_correct_accuracy_strict": _micro("non_violation_correct", "non_violation_total"),
        "micro_accuracy_violation_detect": _micro("violation_detect_correct", "violation_total"),
        "micro_correct_accuracy_violation_strict": _micro("violation_strict_correct", "violation_total"),
        "macro_correct_accuracy_strict": _macro("macro_correct_accuracy_strict"),
        "macro_accuracy_violation_detect": _macro("macro_accuracy_violation_detect"),
        "macro_correct_accuracy_violation_strict": _macro("macro_correct_accuracy_violation_strict"),
        "file_correct_accuracy_strict": file_correct,
        "file_correct_accuracy_strict_m1_m3": file_correct_m1_m3,
    }
