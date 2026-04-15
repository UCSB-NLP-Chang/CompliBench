"""Formatting helpers for evaluation prompts and parsing."""
from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence


def infer_category_titles(oracle: Mapping[str, Any]) -> dict[str, str]:
    """Infer the exact category titles used in an oracle guideline mapping."""
    titles = {
        "cat1": "Category 1: Universal Compliance",
        "cat2": "Category 2: Intent Triggered Guidelines",
        "cat3": "Category 3: Condition Triggered Guidelines",
    }
    for key in oracle.keys():
        low = key.lower()
        if low.startswith("category 1"):
            titles["cat1"] = key
        elif low.startswith("category 2"):
            titles["cat2"] = key
        elif low.startswith("category 3"):
            titles["cat3"] = key
    return titles


def format_guidelines(oracle: Mapping[str, Any], titles: Mapping[str, str]) -> str:
    """Render the oracle guideline mapping into a judge-friendly text block."""
    cat1 = oracle.get(titles["cat1"], {}) or {}
    cat2 = oracle.get(titles["cat2"], {}) or {}
    cat3 = oracle.get(titles["cat3"], {}) or {}
    lines: list[str] = []
    lines.append(f"{titles['cat1'].upper()} (Keys must match exactly)")
    for key, value in cat1.items():
        lines.append(f"- Key: {key}\n  Text: {value}")
    lines.append("")
    lines.append(f"{titles['cat2'].upper()} (Keys are intents; include Phase number)")
    if isinstance(cat2, dict):
        for intent, phases in cat2.items():
            lines.append(f"- Intent Key: {intent}")
            if isinstance(phases, dict):
                for idx, (phase_key, step_val) in enumerate(phases.items(), 1):
                    match = re.search(r"(\d+)", str(phase_key))
                    phase_num = int(match.group(1)) if match else idx
                    text = (
                        "; ".join(str(x) for x in step_val)
                        if isinstance(step_val, list)
                        else str(step_val)
                    )
                    lines.append(f"  Phase {phase_num} — {text}")
            elif isinstance(phases, list):
                for idx, step_val in enumerate(phases, 1):
                    text = (
                        "; ".join(str(x) for x in step_val)
                        if isinstance(step_val, list)
                        else str(step_val)
                    )
                    lines.append(f"  Phase {idx} — {text}")
    lines.append("")
    lines.append(f"{titles['cat3'].upper()} (Keys must match exactly)")
    for key, value in cat3.items():
        lines.append(f"- Key: {key}\n  Text: {value}")
    return "\n".join(lines)


def format_conversation(message_list: Sequence[Mapping[str, Any]]) -> str:
    """Render a conversation message list into a compact transcript string."""
    out: list[str] = []
    for msg in message_list:
        idx = msg.get("turn_index")
        role = msg.get("role", "")
        content = msg.get("content", "")
        out.append(f"{idx} | {role.upper()}: {content}")
    return "\n".join(out)


def extract_first_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a free-form model response."""
    t = text.strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 2:
            code = parts[1]
            if code.startswith("json"):
                code = code[len("json"):]
            t = code.strip()
    if not t.startswith("{"):
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            t = t[start:end + 1]
    return json.loads(t)


__all__ = [
    "extract_first_json",
    "format_conversation",
    "format_guidelines",
    "infer_category_titles",
]
