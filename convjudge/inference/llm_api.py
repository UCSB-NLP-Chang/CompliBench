#!/usr/bin/env python3
"""Run the LLM-as-judge on conversations via a remote API provider.

For each conversation the script loads oracle guidelines, builds a single
judge prompt, and asks the provider which guideline was applied (and whether
violated) for each assistant turn. Outputs ``<conv>_usage[_attemptN].json``
per conversation under ``<output_dir>/<sanitized_model>/[<domain>/]``.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from convjudge.common.formatting import (
    extract_first_json,
    format_conversation,
    format_guidelines,
    infer_category_titles,
)
from convjudge.common.io import (
    as_optional_int,
    as_optional_str,
    discover_conversation_files,
    load_yaml_mapping,
    parse_domains,
    read_json,
    resolve_path,
)
from convjudge.common.prompts import SYSTEM_PROMPT, build_user_prompt
from convjudge.providers.registry import CALL_PROVIDERS, resolve_provider


def _evaluate_one(
    model: str,
    oracle: dict[str, Any],
    convo_path: Path,
    out_dir: Path,
    call_fn,
    *,
    attempt: int | None,
    seed: int | None,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    convo = read_json(convo_path)
    conv_id = convo_path.stem
    cat_titles = infer_category_titles(oracle)
    user_prompt = build_user_prompt(
        format_guidelines(oracle, cat_titles),
        format_conversation(convo.get("message_list", [])),
        conv_id,
    )
    response_text = call_fn(
        model,
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
        system_prompt=SYSTEM_PROMPT,
        reasoning_effort=reasoning_effort,
        seed=seed,
    )

    try:
        resp = extract_first_json(response_text)
    except Exception:
        resp = {"conversation_id": conv_id, "guidelines_used": []}

    used_raw = resp.get("turn_guidelines") or resp.get("guidelines_used") or []
    if not isinstance(used_raw, list):
        used_raw = []

    normalized: list[dict[str, Any]] = []
    for item in used_raw:
        if not isinstance(item, dict):
            continue
        try:
            turn_idx = int(item.get("turn_index", -1))
        except Exception:
            continue
        guid = item.get("guideline_used") or item.get("guidelines_used")
        if not isinstance(guid, dict):
            continue
        cat = str(guid.get("guidance_category", "")).strip()
        key = str(guid.get("guidance_key", "")).strip()
        try:
            phase = int(guid.get("guideline_phase", -1))
        except Exception:
            phase = -1
        if cat.lower().startswith("category 1") or cat.lower().startswith("category 3"):
            phase = -1
        is_violation = bool(guid.get("is_violation")) if isinstance(guid.get("is_violation"), bool) else False
        vr = guid.get("judge_reason", guid.get("violation_reason"))
        judge_reason = vr.strip() if isinstance(vr, str) and vr.strip() else "judgment rationale required"
        normalized.append({
            "turn_index": turn_idx,
            "guideline_used": {
                "guidance_category": cat,
                "guidance_key": key,
                "guideline_phase": phase,
                "is_violation": is_violation,
                "judge_reason": judge_reason,
            },
        })

    out: dict[str, Any] = {
        "conversation_id": conv_id,
        "model": model,
        "turn_guidelines": normalized,
        "raw_response": response_text,
    }
    if attempt:
        out["attempt"] = attempt

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_usage_attempt{attempt}" if attempt else "_usage"
    (out_dir / f"{convo_path.stem}{suffix}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


def run(
    *,
    model: str,
    provider: str,
    reasoning_effort: str | None,
    data_dir: Path,
    output_dir: Path,
    guidelines_path: Path | None,
    limit: int | None,
    num_workers: int,
    n: int,
    domains: list[str] | None,
    overwrite: bool,
) -> int:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    fallback_oracle: dict[str, Any] | None = None
    if guidelines_path is not None:
        if not guidelines_path.exists():
            raise FileNotFoundError(f"Guidelines not found: {guidelines_path}")
        obj = read_json(guidelines_path)
        if not isinstance(obj, dict):
            raise ValueError(f"Guidelines must be a JSON object: {guidelines_path}")
        fallback_oracle = obj

    call_fn = resolve_provider(provider, model)
    sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(model or "model"))
    output_dir = output_dir / sanitized

    domain_files = discover_conversation_files(data_dir)
    if domains:
        allowed = set(domains)
        domain_files = [(d, p) for d, p in domain_files if d in allowed]
    if not domain_files:
        print(f"No simulated conversations in {data_dir}")
        return 0

    max_attempt = max(1, int(n))
    tasks: list[tuple[str, Path, int | None]] = []
    for domain, path in domain_files:
        out_subdir = output_dir / domain if domain else output_dir
        for attempt_idx in range(1, max_attempt + 1):
            attempt_val: int | None = attempt_idx if max_attempt > 1 else None
            suffix = f"_usage_attempt{attempt_val}" if attempt_val else "_usage"
            if (out_subdir / f"{path.stem}{suffix}.json").exists() and not overwrite:
                continue
            tasks.append((domain, path, attempt_val))

    if not tasks:
        print("All conversations already have usage outputs.")
        return 0
    if limit and limit > 0:
        tasks = tasks[:limit]

    print(f"Evaluating guideline usage for {len(tasks)} item(s) with {max(1, num_workers)} workers.")
    progress = tqdm(total=len(tasks), desc="GuidelineUsage", dynamic_ncols=True) if tqdm else None

    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
        future_map: dict = {}
        for domain, path, attempt_idx in tasks:
            oracle: Any = None
            try:
                oracle = read_json(path).get("assistant_guidelines")
            except Exception:
                pass
            if not isinstance(oracle, dict) or not oracle:
                oracle = fallback_oracle
            if oracle is None:
                raise ValueError(f"Missing assistant_guidelines and no fallback for {path}")
            out_subdir = (output_dir / domain) if domain else output_dir
            fut = executor.submit(
                _evaluate_one, model, oracle, path, out_subdir, call_fn,
                attempt=attempt_idx, seed=attempt_idx, reasoning_effort=reasoning_effort,
            )
            future_map[fut] = (domain, path, attempt_idx)

        for fut in as_completed(future_map):
            domain, path, attempt_idx = future_map[fut]
            prefix = f"{domain}/" if domain else ""
            label = f"{prefix}{path.name} (attempt {attempt_idx})" if attempt_idx else f"{prefix}{path.name}"
            try:
                res = fut.result()
                print(f"[{label}] wrote ({len(res.get('turn_guidelines', []))} turns)")
            except Exception as exc:
                import traceback
                print(f"Failed on {label}: {exc}\n{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}")
            finally:
                if progress:
                    progress.update(1)

    if progress:
        progress.close()
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", default=None, choices=sorted(list(CALL_PROVIDERS.keys()) + ["qwen_api", "local"]))
    parser.add_argument("--guidelines", default=None, help="Fallback oracle JSON if a conversation is missing assistant_guidelines.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--n", type=int, default=None, help="Number of independent attempts per conversation.")
    parser.add_argument("--domains", type=str, default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _load_cfg(args: argparse.Namespace) -> dict[str, Any]:
    cfg_path: Path | None = None
    if args.config:
        cfg_path = resolve_path(args.config, root=ROOT) or Path(str(args.config))
    else:
        default = ROOT / "configs" / "default.yaml"
        if default.exists():
            cfg_path = default
    return load_yaml_mapping(cfg_path) if cfg_path else {}


def _resolve_params(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    model = str(args.model or as_optional_str(cfg, "model") or "deepseek-chat")
    provider = str(args.provider or as_optional_str(cfg, "provider") or "deepseek")
    reasoning_effort = (
        str(args.reasoning_effort).strip()
        if args.reasoning_effort is not None
        else as_optional_str(cfg, "reasoning_effort") or as_optional_str(cfg, "reasoning-effort")
    )
    data_dir = resolve_path(args.data_dir, root=ROOT) or resolve_path(as_optional_str(cfg, "data_dir"), root=ROOT)
    if data_dir is None:
        raise SystemExit("--data-dir is required (or set data_dir in the config).")
    output_dir = (
        resolve_path(args.output_dir, root=ROOT)
        or resolve_path(as_optional_str(cfg, "output_dir"), root=ROOT)
        or (ROOT / "results" / "llm_api")
    )
    guidelines_path = resolve_path(args.guidelines, root=ROOT) or resolve_path(as_optional_str(cfg, "guidelines"), root=ROOT)
    limit = args.limit if args.limit is not None else as_optional_int(cfg, "limit")
    num_workers = int(
        args.num_workers if args.num_workers is not None
        else (as_optional_int(cfg, "num_workers", "num-workers") or 32)
    )
    if num_workers < 1:
        raise SystemExit("num_workers must be >= 1")
    n = int(args.n if args.n is not None else (as_optional_int(cfg, "n") or 1))
    if n < 1:
        raise SystemExit("n must be >= 1")
    domains = parse_domains(args.domains) or parse_domains(cfg.get("domains"))
    return dict(
        model=model, provider=provider, reasoning_effort=reasoning_effort,
        data_dir=data_dir, output_dir=output_dir, guidelines_path=guidelines_path,
        limit=limit, num_workers=num_workers, n=n, domains=domains,
        overwrite=bool(args.overwrite),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    cfg = _load_cfg(args)
    return run(**_resolve_params(args, cfg))


if __name__ == "__main__":
    raise SystemExit(main())
