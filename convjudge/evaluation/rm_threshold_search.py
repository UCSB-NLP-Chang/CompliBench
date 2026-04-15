#!/usr/bin/env python3
"""Search for the best per-domain threshold for reward-model-based violation detection.

Reads ``*_rm_scores.json`` / ``*_grm_scores.json`` files and searches for a
threshold T per domain. At threshold T, a turn is predicted as violation when
its score < T.

  oracle_turn_acc    : non-violation turns with score >= T / total non-violation
  violation_turn_acc : violation turns with score < T / total violation
  file_acc           : conversations where ALL turns are correctly classified

Usage example:
    python -m convjudge.evaluation.rm_threshold_search \\
        --scores-dir results/rm_classifier/airlines \\
        --output results/rm_classifier/airlines/threshold.json \\
        --n-steps 1000
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_score_files(scores_dir: Path) -> list[tuple[str, Path]]:
    """Return [(domain, path)] for all *_rm_scores.json / *_grm_scores.json files."""
    patterns = ("*_rm_scores.json", "*_grm_scores.json")
    direct = sorted(p for pat in patterns for p in scores_dir.glob(pat))
    if direct:
        return [("", p) for p in direct]
    files: list[tuple[str, Path]] = []
    for domain_dir in sorted(p for p in scores_dir.iterdir() if p.is_dir()):
        for pat in patterns:
            for path in sorted(domain_dir.glob(pat)):
                files.append((domain_dir.name, path))
    return files


# ---------------------------------------------------------------------------
# Metric computation at a single threshold
# ---------------------------------------------------------------------------

def compute_metrics_at_threshold(
    conv_with_turns: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    """Compute metrics at a given threshold.

    Args:
        conv_with_turns: list of dicts, each with:
            - "turns": list of {"score": float, "is_violation_truth": bool}
            - "conversation_id": str
        threshold: score < threshold → predict violation.

    Returns dict with:
        oracle_turn_acc    : fraction of non-violation turns correctly kept above threshold
        violation_turn_acc : fraction of violation turns correctly caught below threshold
        file_acc           : fraction of conversations where ALL turns are correctly classified
    """
    oracle_total = oracle_correct = 0
    violation_total = violation_correct = 0
    file_total = file_correct = 0

    for c in conv_with_turns:
        turns = c.get("turns", [])
        if not turns:
            continue
        file_total += 1
        file_all_correct = True

        for t in turns:
            score = t["score"]
            is_violation = t["is_violation_truth"]
            turn_correct = (is_violation and score < threshold) or (not is_violation and score >= threshold)

            if is_violation:
                violation_total += 1
                if score < threshold:
                    violation_correct += 1
            else:
                oracle_total += 1
                if score >= threshold:
                    oracle_correct += 1

            if not turn_correct:
                file_all_correct = False

        if file_all_correct:
            file_correct += 1

    return {
        "threshold": threshold,
        "oracle_turn_acc": round(oracle_correct / oracle_total, 4) if oracle_total else None,
        "violation_turn_acc": round(violation_correct / violation_total, 4) if violation_total else None,
        "file_acc": round(file_correct / file_total, 4) if file_total else None,
        "oracle_correct": oracle_correct,
        "oracle_total": oracle_total,
        "violation_correct": violation_correct,
        "violation_total": violation_total,
        "file_correct": file_correct,
        "file_total": file_total,
    }


# ---------------------------------------------------------------------------
# Threshold search
# ---------------------------------------------------------------------------

def search_threshold(
    conv_with_turns: list[dict[str, Any]],
    n_steps: int = 1000,
) -> tuple[float, list[dict[str, Any]]]:
    """Grid-search the best threshold. Returns (best_threshold, full_sweep_table).

    Best threshold = argmax file_acc, tiebroken by oracle_turn_acc + violation_turn_acc.
    Search range covers all per-turn scores in the dataset.
    """
    all_scores = [t["score"] for c in conv_with_turns for t in c.get("turns", [])]
    if not all_scores:
        return 0.0, []

    lo = min(all_scores)
    hi = max(all_scores)
    step = (hi - lo) / max(n_steps - 1, 1)

    best_threshold = lo
    best_file_acc = -1.0
    best_tiebreak = -1.0
    sweep: list[dict[str, Any]] = []

    t = lo
    for _ in range(n_steps):
        m = compute_metrics_at_threshold(conv_with_turns, t)
        sweep.append({
            "threshold": round(t, 4),
            "oracle_turn_acc": m["oracle_turn_acc"],
            "violation_turn_acc": m["violation_turn_acc"],
            "file_acc": m["file_acc"],
        })

        file_acc = m["file_acc"] or 0.0
        tiebreak = (m["oracle_turn_acc"] or 0.0) + (m["violation_turn_acc"] or 0.0)
        if file_acc > best_file_acc or (file_acc == best_file_acc and tiebreak > best_tiebreak):
            best_file_acc = file_acc
            best_tiebreak = tiebreak
            best_threshold = t

        t += step

    return best_threshold, sweep


# ---------------------------------------------------------------------------
# Per-domain evaluation
# ---------------------------------------------------------------------------

def evaluate_domain(
    score_files: list[tuple[str, Path]],
    domain: str,
    n_steps: int,
) -> dict[str, Any]:
    """Load score files for one domain and run threshold search."""
    conv_with_turns: list[dict[str, Any]] = []
    rm_model = ""

    for _, path in score_files:
        data = _read_json(path)
        rm_model = data.get("rm_model", rm_model)
        turns = [
            {"score": t["score"], "is_violation_truth": bool(t.get("is_violation_truth", False))}
            for t in data.get("turn_scores", [])
        ]
        if not turns:
            continue
        conv_with_turns.append({
            "conversation_id": data.get("conversation_id", path.stem),
            "turns": turns,
            "has_violation_truth": bool(data.get("has_violation_truth", False)),
        })

    best_threshold, sweep = search_threshold(conv_with_turns, n_steps=n_steps)
    best_metrics = compute_metrics_at_threshold(conv_with_turns, best_threshold)

    n_violation_convos = sum(1 for c in conv_with_turns if c["has_violation_truth"])
    n_oracle_turns = sum(1 for c in conv_with_turns for t in c["turns"] if not t["is_violation_truth"])
    n_violation_turns = sum(1 for c in conv_with_turns for t in c["turns"] if t["is_violation_truth"])

    return {
        "domain": domain or "(root)",
        "rm_model": rm_model,
        "n_conversations": len(conv_with_turns),
        "n_violation_convos": n_violation_convos,
        "n_oracle_turns": n_oracle_turns,
        "n_violation_turns": n_violation_turns,
        "best_threshold": round(best_threshold, 4),
        "at_best_threshold": {
            "oracle_turn_acc": best_metrics["oracle_turn_acc"],
            "violation_turn_acc": best_metrics["violation_turn_acc"],
            "file_acc": best_metrics["file_acc"],
        },
        "threshold_sweep": sweep,
    }


def _print_domain_summary(result: dict[str, Any]) -> None:
    domain = result["domain"]
    n = result["n_conversations"]
    n_viol = result["n_violation_convos"]
    thresh = result["best_threshold"]
    m = result["at_best_threshold"]
    print(
        f"[{domain}] n={n} (violation={n_viol}) | threshold={thresh:.4f} | "
        f"oracle_turn_acc={m['oracle_turn_acc']} | violation_turn_acc={m['violation_turn_acc']} | "
        f"file_acc={m['file_acc']}"
    )


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(
    *,
    scores_dir: Path,
    output: Path | None,
    n_steps: int,
    domains: list[str] | None,
) -> int:
    if not scores_dir.exists():
        raise FileNotFoundError(f"Scores directory not found: {scores_dir}")

    all_files = _discover_score_files(scores_dir)
    if not all_files:
        print(f"No *_rm_scores.json files found in {scores_dir}")
        return 0

    grouped: dict[str, list[tuple[str, Path]]] = {}
    for domain, path in all_files:
        if domains and domain not in domains:
            continue
        grouped.setdefault(domain, []).append((domain, path))

    if not grouped:
        print("No matching domain files found.")
        return 0

    all_results: list[dict[str, Any]] = []
    for domain, files in sorted(grouped.items()):
        print(f"\nProcessing domain '{domain or '(root)'}' ({len(files)} files)...")
        result = evaluate_domain(files, domain, n_steps=n_steps)
        all_results.append(result)
        _print_domain_summary(result)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = all_results if len(all_results) > 1 else all_results[0]
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nResults written to {output}")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search best threshold per domain for reward-model violation detection."
    )
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--domains", default=None)
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    scores_dir = Path(args.scores_dir)
    if not scores_dir.is_absolute():
        scores_dir = ROOT / scores_dir

    output: Path | None = None
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = ROOT / output

    domains: list[str] | None = None
    if args.domains:
        domains = [d.strip() for d in args.domains.split(",") if d.strip()] or None

    return run(scores_dir=scores_dir, output=output, n_steps=args.n_steps, domains=domains)


if __name__ == "__main__":
    raise SystemExit(main())
