#!/usr/bin/env python3
"""Compute violation-detection accuracy from reward-model scores.

Reads ``*_rm_scores.json`` / ``*_grm_scores.json`` produced by
``convjudge.inference.rm_classifier`` / ``rm_generative_scalar`` and emits
a ``summary.json`` parallel to ``convjudge.evaluation.eval_llm``:

  micro_correct_accuracy_strict    : non-violation turns with score >= T
                                     / total non-violation turns
  micro_accuracy_violation_detect  : violation turns with score < T
                                     / total violation turns
  file_correct_accuracy_strict     : fraction of conversations where EVERY
                                     turn is correctly classified
                                     (all violations detected AND no non-
                                     violation turn flagged)

Threshold selection:
  - ``--threshold <float>`` uses that value as-is.
  - Otherwise the script auto-searches per (model, domain), maximising
    ``file_correct_accuracy_strict`` via ``rm_threshold_search``.
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

from convjudge.common.io import resolve_path
from convjudge.evaluation import SUMMARY_KEYS, filter_summary
from convjudge.evaluation.rm_threshold_search import (
    compute_metrics_at_threshold,
    search_threshold,
)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _score_files(root: Path) -> list[Path]:
    patterns = ("*_rm_scores.json", "*_grm_scores.json")
    return sorted({p for pat in patterns for p in root.rglob(pat)})


def _infer_model_and_domain(path: Path, root: Path, data: dict[str, Any]) -> tuple[str, str]:
    """Derive (model, domain) for a scores file. Metadata wins; path layout is fallback."""
    model = str(data.get("rm_model") or "").strip()
    domain = str(data.get("domain") or "").strip()
    try:
        rel_parts = path.relative_to(root).parts[:-1]
    except Exception:
        rel_parts = ()
    if not model:
        model = rel_parts[0] if len(rel_parts) >= 1 else "(unknown_model)"
    if not domain:
        if len(rel_parts) >= 2:
            domain = rel_parts[1]
        elif len(rel_parts) == 1:
            domain = rel_parts[0] if rel_parts[0] != model else ""
    return model or "(unknown_model)", domain or ""


def _load_group(paths: list[Path]) -> list[dict[str, Any]]:
    convs: list[dict[str, Any]] = []
    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[warn] cannot read {p}: {exc}", file=sys.stderr)
            continue
        turns = [
            {"score": float(t["score"]), "is_violation_truth": bool(t.get("is_violation_truth", False))}
            for t in data.get("turn_scores", [])
            if "score" in t and t["score"] is not None
        ]
        if not turns:
            continue
        convs.append({
            "conversation_id": data.get("conversation_id", p.stem),
            "turns": turns,
        })
    return convs


def _metrics_block(
    convs: list[dict[str, Any]],
    *,
    threshold: float | None,
    n_steps: int,
) -> dict[str, Any]:
    if not convs:
        return {k: None for k in SUMMARY_KEYS} | {"threshold": None, "n_conversations": 0}

    chosen = float(threshold) if threshold is not None else search_threshold(convs, n_steps=n_steps)[0]
    m = compute_metrics_at_threshold(convs, chosen)
    return {
        "threshold": round(chosen, 4),
        "n_conversations": len(convs),
        "micro_correct_accuracy_strict": m["oracle_turn_acc"],
        "micro_accuracy_violation_detect": m["violation_turn_acc"],
        "file_correct_accuracy_strict": m["file_acc"],
        "counts": {
            "non_violation_correct": m["oracle_correct"],
            "non_violation_total": m["oracle_total"],
            "violation_correct": m["violation_correct"],
            "violation_total": m["violation_total"],
            "file_correct": m["file_correct"],
            "file_total": m["file_total"],
        },
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(
    *,
    scores_dir: Path,
    output: Path | None,
    output_dir: Path | None,
    threshold: float | None,
    n_steps: int,
    domains: list[str] | None,
) -> int:
    if not scores_dir.exists():
        raise SystemExit(f"Scores directory not found: {scores_dir}")
    files = _score_files(scores_dir)
    if not files:
        raise SystemExit(f"No *_rm_scores.json / *_grm_scores.json under {scores_dir}")

    grouped: dict[tuple[str, str], list[Path]] = {}
    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        model, domain = _infer_model_and_domain(p, scores_dir, data if isinstance(data, dict) else {})
        if domains and domain not in domains:
            continue
        grouped.setdefault((model, domain), []).append(p)

    if not grouped:
        raise SystemExit("No scores files matched the given filters.")

    by_model_domain: dict[str, dict[str, dict[str, Any]]] = {}
    convs_by_model: dict[str, list[dict[str, Any]]] = {}
    all_convs: list[dict[str, Any]] = []

    for (model, domain), paths in sorted(grouped.items()):
        convs = _load_group(paths)
        block = _metrics_block(convs, threshold=threshold, n_steps=n_steps)
        by_model_domain.setdefault(model, {})[domain or "(root)"] = block
        convs_by_model.setdefault(model, []).extend(convs)
        all_convs.extend(convs)

    by_model = {m: _metrics_block(cs, threshold=threshold, n_steps=n_steps) for m, cs in sorted(convs_by_model.items())}
    overall = _metrics_block(all_convs, threshold=threshold, n_steps=n_steps)

    multi_domain = any(len(d) > 1 for d in by_model_domain.values())
    summary: dict[str, Any] = {
        **filter_summary(overall),
        "threshold": overall["threshold"],
        "n_conversations": overall["n_conversations"],
        "by_model": {
            m: {**filter_summary(b), "threshold": b["threshold"], "n_conversations": b["n_conversations"]}
            for m, b in by_model.items()
        },
    }
    if multi_domain or len(by_model) > 1:
        summary["by_model_domain"] = {
            m: {
                d: {**filter_summary(b), "threshold": b["threshold"], "n_conversations": b["n_conversations"]}
                for d, b in sorted(dm.items())
            }
            for m, dm in sorted(by_model_domain.items())
        }
    summary["detail"] = {
        "overall": overall,
        "by_model": by_model,
        "by_model_domain": by_model_domain,
    }

    print(json.dumps({k: summary.get(k) for k in (*SUMMARY_KEYS, "threshold", "n_conversations")},
                     indent=2, ensure_ascii=False))

    out_path: Path | None = resolve_path(output, root=ROOT) if output else None
    if out_path is None and output_dir:
        od = resolve_path(output_dir, root=ROOT)
        if od is None:
            raise SystemExit("--output-dir must be a non-empty path.")
        od.mkdir(parents=True, exist_ok=True)
        out_path = od / "summary.json"
    if out_path is None:
        out_path = scores_dir / "summary.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[wrote] {out_path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--scores-dir", required=True)
    p.add_argument("--output", default=None, help="Write summary JSON to this exact path.")
    p.add_argument("--output-dir", default=None, help="Write summary JSON to <dir>/summary.json.")
    p.add_argument("--threshold", type=float, default=None, help="Fixed threshold; if omitted, auto-search per (model, domain).")
    p.add_argument("--n-steps", type=int, default=1000, help="Grid steps when auto-searching.")
    p.add_argument("--domains", default=None)
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    scores_dir = resolve_path(args.scores_dir, root=ROOT)
    if scores_dir is None:
        raise SystemExit("--scores-dir is required.")
    domains = None
    if args.domains:
        domains = [d.strip() for d in args.domains.split(",") if d.strip()] or None
    return run(
        scores_dir=scores_dir,
        output=args.output,
        output_dir=args.output_dir,
        threshold=args.threshold,
        n_steps=int(args.n_steps),
        domains=domains,
    )


if __name__ == "__main__":
    raise SystemExit(main())
