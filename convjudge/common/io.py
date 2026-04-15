"""Filesystem / YAML / path utilities shared by every entry script."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional
    yaml = None  # type: ignore


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML file as a mapping. Raises if pyyaml is missing or the
    top level isn't an object."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to load YAML configs; `pip install pyyaml`.")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping/object: {path}")
    return data


def resolve_path(raw: str | Path | None, *, root: Path) -> Path | None:
    """Turn a string/Path into an absolute Path, resolving relatives against ``root``.
    Returns None for empty/None input."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    p = Path(s)
    return p if p.is_absolute() else (root / p)


def as_optional_str(cfg: dict[str, Any], key: str) -> str | None:
    raw = cfg.get(key)
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError(f"Config field '{key}' must be a string or null.")
    s = raw.strip()
    return s or None


def as_optional_int(cfg: dict[str, Any], *keys: str) -> int | None:
    """Return the first set int-castable value among ``keys``, else None."""
    for key in keys:
        raw = cfg.get(key)
        if raw is None:
            continue
        try:
            return int(raw)
        except Exception as exc:
            raise ValueError(f"Config field '{key}' must be an int or null.") from exc
    return None


def parse_domains(raw: Any) -> list[str] | None:
    """Accept a comma string, a list, or None; return a normalized list (or None)."""
    if raw is None:
        return None
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return parts or None
    if isinstance(raw, list):
        out = [str(item or "").strip() for item in raw if str(item or "").strip()]
        return out or None
    raise ValueError("'domains' must be a comma-separated string, a list of strings, or null.")


def discover_conversation_files(root: Path) -> list[tuple[str, Path]]:
    """Return ``[(domain, path)]`` for every ``conversation_*.json`` under root.

    Two layouts are supported:
      - flat:   ``root/conversation_*.json``         → domain = ""
      - pooled: ``root/<domain>/conversation_*.json`` → domain = <domain>
    """
    direct = sorted(root.glob("conversation_*.json"))
    if direct:
        return [("", p) for p in direct]
    files: list[tuple[str, Path]] = []
    for domain_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(domain_dir.glob("conversation_*.json")):
            files.append((domain_dir.name, path))
    return files


__all__ = [
    "read_json",
    "load_yaml_mapping",
    "resolve_path",
    "as_optional_str",
    "as_optional_int",
    "parse_domains",
    "discover_conversation_files",
]
