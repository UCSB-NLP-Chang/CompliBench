#!/usr/bin/env python3
"""Score conversations with a generative reward model that emits a numeric score.

Unlike the classifier-head path, this calls a causal LM per labeled turn,
prompts it to rate guideline adherence on 1-10, then extracts the score
from ``\\boxed{N}`` or ``[N]`` in the generated text.

Output: ``*_grm_scores.json`` (same schema as classifier path, so
``convjudge.evaluation.rm_threshold_search`` / ``eval_rm`` can consume either).

Tested with: BBQGOD/DeepSeek-GRM-16B.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any

try:
    import torch
    from transformers import AutoTokenizer, pipeline
except ImportError as exc:
    raise SystemExit(f"transformers and torch are required: {exc}") from exc

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from convjudge.common.io import (
    discover_conversation_files,
    load_yaml_mapping,
    read_json,
    resolve_path,
)
from convjudge.inference._rm_base import collect_labeled_turns


_JUDGE_TEMPLATE = (
    "You are a skilled expert at evaluating assistant responses for guideline compliance.\n"
    "Given the conversation context and a single response from the assistant, score how well "
    "the response follows the provided guideline.\n\n"
    "#### Evaluation Criteria ####\n"
    "1. Guideline Adherence:\n"
    "   - Fully Adhered (9-10): The response fully follows the guideline's required steps, "
    "constraints, and routing behaviour.\n"
    "   - Partially Adhered (6-8): The response mostly follows the guideline but has minor "
    "omissions or deviations.\n"
    "   - Barely Adhered (3-5): The response partially satisfies the guideline but misses "
    "key requirements.\n"
    "   - Not Adhered (1-2): The response clearly violates or ignores the guideline.\n\n"
    "#### Oracle Guideline for This Turn ####\n"
    "{guideline_text}\n\n"
    "#### Conversation Context ####\n"
    "{conversation_context}\n\n"
    "#### Response to Score ####\n"
    "{assistant_response}\n\n"
    "#### Output Format ####\n"
    "Analysis: <brief analysis of how well the response follows the guideline>.\n"
    r"Score: \boxed{<integer 1-10>}"
)

_SCORE_RE = re.compile(r'(?:\\{1,2}boxed\{|\[)\s*([^\]\}]+?)\s*(?:\}|\])')


def _extract_score(text: str) -> float | None:
    matches = list(_SCORE_RE.finditer(text))
    if not matches:
        return None
    last = matches[-1].group(1).strip()
    for part in re.split(r"\s*,\s*", last):
        try:
            return float(part)
        except ValueError:
            continue
    return None


def _build_context(message_list: list[dict[str, Any]], up_to: int) -> str:
    lines: list[str] = []
    for msg in message_list:
        ti = int(msg.get("turn_index", -1))
        role = str(msg.get("role", ""))
        if ti >= up_to or role not in ("user", "assistant"):
            continue
        lines.append(f"{role.capitalize()}: {msg.get('content', '')}")
    return "\n\n".join(lines) if lines else "(start of conversation)"


def _build_judge_messages(guideline_text: str, context: str, response: str) -> list[dict[str, str]]:
    return [{
        "role": "user",
        "content": _JUDGE_TEMPLATE.format(
            guideline_text=guideline_text,
            conversation_context=context,
            assistant_response=response,
        ),
    }]


def load_pipeline(model_name: str, device: str):
    print(f"Loading GRM pipeline: {model_name}  device={device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return pipeline(
        "text-generation",
        tokenizer=tokenizer,
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device,
        trust_remote_code=True,
    )


def score_conversation(convo: dict[str, Any], pipe, model_name: str, max_new_tokens: int) -> dict[str, Any]:
    message_list = convo.get("message_list", [])
    domain = str(convo.get("domain", ""))
    labeled_turns = collect_labeled_turns(convo)

    content_by_ti = {
        int(m.get("turn_index", -1)): str(m.get("content", ""))
        for m in message_list if m.get("role") == "assistant"
    }

    turn_scores: list[dict[str, Any]] = []
    for lt in labeled_turns:
        ti = lt["turn_index"]
        guideline_text = lt.get("guideline_text", "")
        if not guideline_text:
            continue
        response = content_by_ti.get(ti, "")
        if not response:
            continue

        messages = _build_judge_messages(
            guideline_text, _build_context(message_list, up_to=ti), response,
        )
        try:
            outputs = pipe(messages, max_new_tokens=max_new_tokens, temperature=1.0, do_sample=True)
            judgement = outputs[0]["generated_text"][-1]["content"].strip()
            score = _extract_score(judgement)
        except Exception as exc:
            print(f"  [warn] turn {ti}: generation failed: {exc}")
            judgement, score = "", None

        turn_scores.append({
            "turn_index": ti,
            "score": score,
            "is_violation_truth": lt["is_violation_truth"],
            "category": lt["category"],
            "key": lt["key"],
            "phase": lt["phase"],
            "guideline_text": guideline_text,
            "judgement": judgement,
        })

    valid = [t["score"] for t in turn_scores if t["score"] is not None]
    return {
        "conversation_id": convo.get("conversation_id", ""),
        "rm_model": model_name,
        "domain": domain,
        "has_violation_truth": bool(convo.get("mistakes")),
        "score_min": min(valid) if valid else None,
        "score_mean": sum(valid) / len(valid) if valid else None,
        "turn_scores": turn_scores,
    }


def run(
    *,
    model_name: str,
    data_dir: Path,
    output_dir: Path,
    device: str,
    max_new_tokens: int,
    limit: int | None,
    domains: list[str] | None,
    overwrite: bool,
) -> int:
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")
    pipe = load_pipeline(model_name, device)

    domain_files = discover_conversation_files(data_dir)
    if domains:
        domain_files = [(d, p) for d, p in domain_files if d in set(domains)]
    if not domain_files:
        print(f"No conversation files found in {data_dir}")
        return 0

    def _out_path(domain: str, path: Path) -> Path:
        d = output_dir / domain if domain else output_dir
        return d / f"{path.stem}_grm_scores.json"

    tasks = [(d, p) for d, p in domain_files if overwrite or not _out_path(d, p).exists()]
    if not tasks:
        print("All conversations already scored.")
        return 0
    if limit:
        tasks = tasks[:limit]

    print(f"Scoring {len(tasks)} conversation(s) — 1 generation per labeled turn.")
    bar = tqdm(total=len(tasks), desc="GRMScore", dynamic_ncols=True) if tqdm else None
    for domain, path in tasks:
        out_file = _out_path(domain, path)
        try:
            convo = read_json(path)
            convo.setdefault("conversation_id", path.stem)
            result = score_conversation(convo, pipe, model_name, max_new_tokens)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            label = f"{domain}/{path.name}" if domain else path.name
            print(f"[{label}] {len(result['turn_scores'])} turns | min={result['score_min']} mean={result['score_mean']}")
        except Exception as exc:
            print(f"Failed on {path}: {exc}\n{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}")
        finally:
            if bar:
                bar.update(1)

    if bar:
        bar.close()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--config", default=None)
    p.add_argument("--model-name", default=None, help="HuggingFace model name or local path")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--domains", default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    cfg: dict[str, Any] = {}
    if args.config:
        cfg = load_yaml_mapping(resolve_path(args.config, root=ROOT) or Path(str(args.config)))
    model_name = args.model_name or cfg.get("model_name")
    if not model_name:
        raise SystemExit("--model-name is required.")
    data_dir = resolve_path(args.data_dir, root=ROOT) or resolve_path(str(cfg.get("data_dir", "")), root=ROOT)
    if data_dir is None:
        raise SystemExit("--data-dir is required.")
    output_dir = (
        resolve_path(args.output_dir, root=ROOT)
        or resolve_path(str(cfg.get("output_dir", "")), root=ROOT)
        or (ROOT / "results" / "rm_generative_scalar")
    )
    domains_raw = args.domains or cfg.get("domains")
    domains = None
    if isinstance(domains_raw, str):
        domains = [d.strip() for d in domains_raw.split(",") if d.strip()] or None
    elif isinstance(domains_raw, list):
        domains = [str(d).strip() for d in domains_raw if str(d).strip()] or None

    return run(
        model_name=model_name, data_dir=data_dir, output_dir=output_dir,
        device=args.device, max_new_tokens=args.max_new_tokens,
        limit=args.limit, domains=domains, overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    raise SystemExit(main())
