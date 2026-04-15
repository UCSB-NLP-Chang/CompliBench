"""Moonshot Kimi API caller (OpenAI-compatible endpoint)."""
from __future__ import annotations

import os
from typing import Any, Iterable, Mapping, Optional, Type

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
_DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY", "")
        if not api_key:
            raise ValueError("KIMI_API_KEY (or MOONSHOT_API_KEY) environment variable is required.")
        base_url = os.getenv("KIMI_API_BASE_URL", _DEFAULT_BASE_URL)
        _client = OpenAI(base_url=base_url, api_key=api_key)
    return _client


@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(6))
def _chat_create_with_retry(**kwargs):
    return _get_client().chat.completions.create(**kwargs)


def _with_system_prompt(messages: Iterable[Mapping[str, Any]], system_prompt: str):
    msgs = [dict(m) for m in messages]
    if not msgs or msgs[0].get("role") != "system":
        msgs.insert(0, {"role": "system", "content": system_prompt})
    return msgs


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
    del reasoning_effort
    if response_model is not None:
        raise NotImplementedError("Kimi caller does not support structured parsing.")

    payload = {
        "model": model_type,
        "messages": _with_system_prompt(messages, system_prompt),
    }
    if seed is not None:
        payload["seed"] = int(seed)
    payload.update(kwargs)

    result = _chat_create_with_retry(**payload)
    content = result.choices[0].message.content
    return (content or "").strip() if isinstance(content, str) else str(content).strip()


__all__ = ["call_chat_completion"]
