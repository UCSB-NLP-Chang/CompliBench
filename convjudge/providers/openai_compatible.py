"""Generic OpenAI-compatible chat-completion caller.

Works with any server that speaks the OpenAI ``/v1/chat/completions`` schema,
including:

  * self-hosted vLLM OpenAI server
  * sglang / LM Studio / llama.cpp server
  * hosted endpoints that expose the OpenAI schema (DashScope, etc.)

Configured via environment variables:

  OPENAI_COMPAT_BASE_URL   (default: http://127.0.0.1:30001/v1)
  OPENAI_COMPAT_API_KEY    (default: "EMPTY"; some servers ignore this)
  OPENAI_COMPAT_MODEL      (default: "default"; fallback when caller passes
                            a placeholder like "openai_compatible")
  OPENAI_COMPAT_TEMPERATURE, OPENAI_COMPAT_MAX_TOKENS (both optional)

Any explicit ``model`` argument passed to ``call_chat_completion`` wins over
the env var, unless it is one of the aliases {"openai_compatible", "local"}.
"""
from __future__ import annotations

import os
from typing import Any, Iterable, Mapping, Optional, Type

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
_ALIASES = {"openai_compatible", "openai-compatible", "local"}

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        base_url = os.getenv("OPENAI_COMPAT_BASE_URL", "http://127.0.0.1:30001/v1")
        api_key = os.getenv("OPENAI_COMPAT_API_KEY", "EMPTY")
        _client = OpenAI(base_url=base_url, api_key=api_key)
    return _client


@retry(wait=wait_random_exponential(min=2, max=60), stop=stop_after_attempt(10))
def _chat_create_with_retry(**kwargs):
    return _get_client().chat.completions.create(**kwargs)


def _with_system_prompt(messages: Iterable[Mapping[str, Any]], system_prompt: str) -> list[dict[str, Any]]:
    msgs = [dict(m) for m in messages]
    if not msgs or msgs[0].get("role") != "system":
        msgs.insert(0, {"role": "system", "content": system_prompt})
    return msgs


def _strip_think_block(text: str) -> str:
    """Remove anything before and including ``</think>`` so reasoning models
    return only the final answer."""
    if not isinstance(text, str) or "</think>" not in text:
        return text or ""
    return text.split("</think>", 1)[1].lstrip()


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
        raise NotImplementedError("openai_compatible caller does not support structured parsing.")

    target_model = (
        os.getenv("OPENAI_COMPAT_MODEL", "default")
        if (model_type or "").lower() in _ALIASES
        else model_type
    )

    payload: dict[str, Any] = {
        "model": target_model,
        "messages": _with_system_prompt(messages, system_prompt),
    }
    if (t := os.getenv("OPENAI_COMPAT_TEMPERATURE")):
        payload["temperature"] = float(t)
    if (mt := os.getenv("OPENAI_COMPAT_MAX_TOKENS")):
        payload["max_tokens"] = int(mt)
    if seed is not None:
        payload["seed"] = int(seed)
    payload.update(kwargs)

    result = _chat_create_with_retry(**payload)
    content = result.choices[0].message.content or ""
    return _strip_think_block(content).strip()


__all__ = ["call_chat_completion"]
