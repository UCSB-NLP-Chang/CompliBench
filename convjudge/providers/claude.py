"""Anthropic Claude API caller."""
from __future__ import annotations

import os
from typing import Any, Iterable, Mapping, Optional, Type

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_MAX_TOKENS = 4096

_client = None


def _get_client():
    global _client
    if _client is None:
        try:
            from anthropic import Anthropic  # type: ignore
        except Exception as exc:
            raise ImportError(
                "anthropic package is required for Claude API calls. "
                "Install with: pip install anthropic"
            ) from exc
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required.")
        _client = Anthropic(api_key=api_key)
    return _client


def _split_system_and_messages(
    messages: Iterable[Mapping[str, Any]], fallback_system: str
) -> tuple[str, list[dict[str, str]]]:
    system_text = fallback_system
    chat: list[dict[str, str]] = []
    for m in messages:
        role = str(m.get("role", "")).strip().lower()
        content = m.get("content", "")
        text = content if isinstance(content, str) else str(content)
        if role == "system":
            if text.strip():
                system_text = text
            continue
        if role in {"user", "assistant"}:
            chat.append({"role": role, "content": text})
    if not chat:
        chat = [{"role": "user", "content": ""}]
    return system_text, chat


def _extract_text(response: Any) -> str:
    blocks = getattr(response, "content", None)
    if isinstance(blocks, list):
        parts = [getattr(b, "text", "") for b in blocks if isinstance(getattr(b, "text", None), str)]
        if parts:
            return "\n".join(parts).strip()
    return str(response).strip()


def call_chat_completion(
    model_type: str,
    messages: Iterable[Mapping[str, Any]],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    reasoning_effort: Optional[str] = None,
    response_model: Optional[Type[Any]] = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> str:
    del reasoning_effort, seed
    if response_model is not None:
        raise NotImplementedError("Claude caller does not support structured parsing.")

    system_text, chat = _split_system_and_messages(messages, system_prompt)
    max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", DEFAULT_MAX_TOKENS))
    payload: dict[str, Any] = {
        "model": model_type,
        "system": system_text,
        "max_tokens": max_tokens,
        "messages": chat,
    }
    payload.update(kwargs)
    return _extract_text(_get_client().messages.create(**payload))


__all__ = ["call_chat_completion"]
