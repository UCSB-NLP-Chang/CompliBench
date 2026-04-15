"""Prompt formatting for reward-model scoring."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def format_full_conversation_for_rm(
    message_list: Sequence[Mapping[str, Any]],
    guidelines_text: str,
    domain: str = "",
) -> list[dict[str, str]]:
    """Build the full multi-turn dialogue for single-pass reward-model scoring.

    All oracle guidelines go into a system message. The real user/assistant
    turns follow in order. The reward model is then scored at each assistant
    turn boundary.
    """
    domain_context = f" for {domain}" if domain else ""
    system_content = (
        f"You are a professional customer service agent{domain_context}. "
        f"Follow these guidelines strictly.\n\n"
        f"GUIDELINES:\n{guidelines_text}"
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
    for msg in message_list:
        role = str(msg.get("role", ""))
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": str(msg.get("content", ""))})
    return messages


__all__ = ["format_full_conversation_for_rm"]
