"""Google Gemini caller (google.genai SDK)."""
from __future__ import annotations

import os
from typing import Any, Iterable, Mapping, Optional, Type

from tenacity import retry, stop_after_attempt, wait_random_exponential

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

_client = None  # lazy
_types = None


def _get_client():
    global _client, _types
    if _client is None:
        try:
            from google import genai  # type: ignore
            from google.genai import types as _t  # type: ignore
        except Exception as exc:
            raise ImportError(
                "google-generativeai package is required for Gemini calls. "
                "Install with: pip install google-generativeai"
            ) from exc
        if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable is required.")
        _client = genai.Client()
        _types = _t
    return _client, _types


def _build_contents(messages: Iterable[Mapping[str, Any]], system_prompt: str):
    _, types = _get_client()
    msgs = [dict(m) for m in messages]
    sys_instr = system_prompt
    contents: list = []
    for m in msgs:
        role = m.get("role", "user")
        text = m.get("content", "") or ""
        if role == "system":
            sys_instr = text or sys_instr
            continue
        gemini_role = "model" if role == "assistant" else "user"
        contents.append(types.ContentDict(role=gemini_role, parts=[types.PartDict(text=text)]))
    return contents, sys_instr


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def _generate_with_retry(model_name: str, contents, system_instruction: Optional[str]) -> str:
    client, types = _get_client()
    config = types.GenerateContentConfig(system_instruction=system_instruction) if system_instruction else None
    resp = client.models.generate_content(model=model_name, contents=contents, config=config)
    return getattr(resp, "text", "") or ""


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
    del reasoning_effort, seed, kwargs, response_model
    contents, sys_instr = _build_contents(messages, system_prompt)
    return _generate_with_retry(model_type, contents, sys_instr)


__all__ = ["call_chat_completion"]
