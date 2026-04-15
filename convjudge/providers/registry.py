"""Provider registry — maps ``--provider`` names to callable backends."""
from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, Dict


def _optional(module_path: str) -> Callable[..., Any] | None:
    try:
        module = import_module(module_path)
        fn = getattr(module, "call_chat_completion", None)
        return fn if callable(fn) else None
    except Exception:
        return None


CALL_PROVIDERS: Dict[str, Callable[..., Any]] = {
    name: fn
    for name, fn in {
        "claude":             _optional("convjudge.providers.claude"),
        "deepseek":           _optional("convjudge.providers.deepseek"),
        "gemini":             _optional("convjudge.providers.gemini"),
        "kimi":               _optional("convjudge.providers.kimi"),
        "qwen":               _optional("convjudge.providers.qwen"),
        "openai_compatible":  _optional("convjudge.providers.openai_compatible"),
    }.items()
    if fn is not None
}

# Normalized name aliases.
_ALIASES = {
    "qwen_api": "qwen",
    "qwen-api": "qwen",
    "openai-compatible": "openai_compatible",
    "local": "openai_compatible",
    "vllm_server": "openai_compatible",
    "sglang": "openai_compatible",
}


def resolve_provider(name: str | None, model: str | None = None) -> Callable[..., Any]:
    """Return the provider callable registered under ``name``.

    Raises ``ValueError`` if the name is not supported.
    """
    del model  # kept for backwards compatibility
    raw = (name or "deepseek").strip().lower()
    key = _ALIASES.get(raw, raw)
    if key not in CALL_PROVIDERS:
        raise ValueError(
            f"Unknown provider '{name}'. Supported: {sorted(CALL_PROVIDERS)}"
        )
    return CALL_PROVIDERS[key]


__all__ = ["CALL_PROVIDERS", "resolve_provider"]
