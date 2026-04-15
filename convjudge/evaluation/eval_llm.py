#!/usr/bin/env python3
"""Aggregate per-turn LLM-judge accuracy into a single ``summary.json``.

Consumes ``*_usage*.json`` files produced by ``convjudge.inference.llm_api`` /
``llm_vllm`` and compares them against the ground truth in the original
conversation JSONs. Emits three summary metrics (see
:data:`convjudge.evaluation.SUMMARY_KEYS`) plus per-model and (when
applicable) per-model-per-domain breakdowns.

When N attempts exist per conversation, each attempt is counted independently
in the aggregate rates, but per-turn error files are merged across attempts to
record how many attempts made each kind of mistake.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from convjudge.common.io import load_yaml_mapping, read_json, resolve_path
from convjudge.evaluation import SUMMARY_KEYS, filter_summary
from convjudge.evaluation.merging import add_merged, write_or_delete
from convjudge.evaluation.metrics import evaluate_file, summarize


def _infer_model_name(usage_path: Path, usage_root: Path, usage: dict[str, Any]) -> str:
    model = str(usage.get("model") or "").strip()
    if model:
        return model
    try:
        rel = usage_path.relative_to(usage_root)
        if len(rel.parts) >= 2:
            return rel.parts[0]
    except Exception:
        pass
    return usage_path.parent.name


def _infer_domain_name(usage_path: Path, usage_root: Path, data_root: Path, model: str) -> str:
    try:
        parts = usage_path.relative_to(usage_root).parts
    except Exception:
        parts = ()
    if len(parts) >= 3:
        return parts[1]
    if len(parts) == 2:
        return data_root.name if parts[0] == model else parts[0]
    return data_root.name


def _usage_conversation_id(usage_path: Path) -> str:
    stem = usage_path.stem
    if stem.endswith("_usage"):
        return stem[:-6]
    if "_usage_attempt" in stem:
        return stem.split("_usage_attempt", 1)[0]
    return stem


def _usage_attempt_id(usage_path: Path) -> str | None:
    stem = usage_path.stem
    if "_usage_attempt" not in stem:
        return None
    return stem.split("_usage_attempt", 1)[1].strip() or None


def _sanitize(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value or "").strip("._")
    return cleaned or "unknown"


def _error_file_name(conv_id: str, domain: str, *, include_domain: bool) -> str:
    if include_domain:
        return f"{conv_id}__{_sanitize(domain)}_usage_errors.json"
    return f"{conv_id}_usage_errors.json"


def _cleanup(usage_accuracy_root: Path) -> None:
    if not usage_accuracy_root.exists():
        return
    for model_errors_dir in usage_accuracy_root.rglob("model_errors"):
        if not model_errors_dir.is_dir():
            continue
        for error_file in model_errors_dir.rglob("*_usage_errors.json"):
            try:
                error_file.unlink()
            except FileNotFoundError:
                pass


def _prune_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for dir_path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):
        try:
            dir_path.rmdir()
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--usage-root", required=False)
    parser.add_argument("--data-root", required=False)
    parser.add_argument("--output", required=False, help="Write summary JSON to this exact path.")
    parser.add_argument("--output-dir", required=False, help="Write summary JSON to <dir>/summary.json.")
    args = parser.parse_args()

    cfg: dict[str, Any] = {}
    cfg_path = Path(args.config) if args.config else (ROOT / "configs" / "default.yaml")
    if cfg_path.exists():
        try:
            cfg = read_json(cfg_path)
        except Exception:
            cfg = load_yaml_mapping(cfg_path)

    usage_root = resolve_path(args.usage_root, root=ROOT) or resolve_path(cfg.get("output_dir"), root=ROOT)
    data_root = resolve_path(args.data_root, root=ROOT) or resolve_path(cfg.get("data_dir"), root=ROOT)
    if usage_root is None or not usage_root.exists():
        raise SystemExit(f"--usage-root is required and must exist: {usage_root}")
    if data_root is None or not data_root.exists():
        raise SystemExit(f"--data-root is required and must exist: {data_root}")

    files = [
        p for p in usage_root.rglob("*_usage*.json")
        if "usage_accuracy" not in p.parts
        and "model_errors" not in p.parts
        and not p.name.endswith("_usage_errors.json")
        and (p.name.endswith("_usage.json") or "_usage_attempt" in p.name)
    ]
    if not files:
        raise SystemExit(f"No valid usage files found under {usage_root}")

    # Group by (model, domain, conv_id); prefer attempt files when present.
    grouped: dict[tuple[str, str, str], list[Path]] = {}
    attempt_seen: set[tuple[str, str, str]] = set()
    for p in sorted(files):
        try:
            usage = read_json(p)
        except Exception:
            usage = {}
        model = _infer_model_name(p, usage_root, usage if isinstance(usage, dict) else {})
        domain = _infer_domain_name(p, usage_root, data_root, model)
        conv_id = _usage_conversation_id(p)
        key = (model, domain, conv_id)
        grouped.setdefault(key, []).append(p)
        if "_usage_attempt" in p.stem:
            attempt_seen.add(key)

    selected: dict[tuple[str, str, str], list[Path]] = {
        key: [p for p in paths if "_usage_attempt" in p.stem] if key in attempt_seen else list(paths)
        for key, paths in grouped.items()
    }
    usage_accuracy_root = usage_root / "usage_accuracy"
    include_domain_in_error_name = len({domain for _, domain, _ in selected}) > 1

    _cleanup(usage_accuracy_root)
    _prune_empty_dirs(usage_accuracy_root)

    overall: list[dict[str, Any]] = []
    by_model: dict[str, list[dict[str, Any]]] = {}
    by_md: dict[str, dict[str, list[dict[str, Any]]]] = {}
    skipped = 0

    for (model, domain, conv_id), paths in sorted(selected.items()):
        paths = sorted(paths)
        err_base = usage_accuracy_root / model / "model_errors"
        err_file = _error_file_name(conv_id, domain, include_domain=include_domain_in_error_name)

        convo: dict[str, Any] | None = None
        convo_file: str | None = None
        m1: dict[int, dict[str, Any]] = {}
        m2: dict[int, dict[str, Any]] = {}
        m3: dict[int, dict[str, Any]] = {}
        m_all: dict[int, dict[str, Any]] = {}

        for p in paths:
            try:
                res = evaluate_file(p, data_root)
            except FileNotFoundError as exc:
                skipped += 1
                print(f"Skipping usage file with missing conversation JSON: {p} ({exc})", file=sys.stderr)
                continue

            overall.append(res)
            by_model.setdefault(model, []).append(res)
            by_md.setdefault(model, {}).setdefault(domain, []).append(res)

            if convo is None:
                convo = res.get("conversation") or {}
                convo_file = str(res.get("conversation_file") or "")

            attempt_id = _usage_attempt_id(p) or "usage"
            for t in res.get("incorrect_turns_metric1") or []:
                add_merged(m1, t, attempt_id)
                add_merged(m_all, t, attempt_id)
            for t in res.get("incorrect_turns_metric2") or []:
                add_merged(m2, t, attempt_id)
            for t in res.get("incorrect_turns_metric3") or []:
                add_merged(m3, t, attempt_id)
                add_merged(m_all, t, attempt_id)

        common = dict(
            file_name=err_file, conv_id=conv_id, convo_file=convo_file,
            convo=convo, num_paths=len(paths),
        )
        write_or_delete(err_base / "metric1", m1, error_group="metric1", **common)
        write_or_delete(err_base / "metric2", m2, error_group="metric2", **common)
        write_or_delete(err_base / "metric3", m3, error_group="metric3", **common)
        write_or_delete(err_base / "combined", m_all, error_group="combined", **common)

    summary: dict[str, Any] = {
        **filter_summary(summarize(overall)),
        "by_model": {
            m: filter_summary(summarize(results))
            for m, results in sorted(by_model.items())
        },
        "skipped_missing_conversation_count": skipped,
    }
    if include_domain_in_error_name:
        summary["by_model_domain"] = {
            m: {d: filter_summary(summarize(rs)) for d, rs in sorted(dm.items())}
            for m, dm in sorted(by_md.items())
        }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    out_path = resolve_path(args.output, root=ROOT) if args.output else None
    if out_path is None and args.output_dir:
        od = resolve_path(args.output_dir, root=ROOT)
        if od is None:
            raise SystemExit("--output-dir must be a non-empty path.")
        od.mkdir(parents=True, exist_ok=True)
        out_path = od / "summary.json"
    if out_path is None:
        out_path = usage_root / "summary.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _prune_empty_dirs(usage_accuracy_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
