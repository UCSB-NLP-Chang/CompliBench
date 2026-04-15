#!/usr/bin/env python3
"""Score conversations with a classifier-head reward model.

One GPU forward pass per conversation: the full multi-turn dialogue is
tokenized once, hidden states are read at the last token of each labeled
assistant turn, and the score head's scalar is saved per turn.

Compatible with Skywork-Reward, ArmoRM, etc. Output: ``*_rm_scores.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as exc:
    raise SystemExit(f"transformers and torch are required: {exc}") from exc

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from convjudge.common.formatting import format_guidelines, infer_category_titles
from convjudge.common.io import (
    discover_conversation_files,
    load_yaml_mapping,
    read_json,
    resolve_path,
)
from convjudge.common.rm_prompts import format_full_conversation_for_rm
from convjudge.inference._rm_base import collect_labeled_turns


def _score_head(rm):
    for attr in ("score", "classifier"):
        h = getattr(rm, attr, None)
        if h is not None:
            return h
    raise AttributeError("Cannot find score head on reward model (tried 'score', 'classifier').")


def _base_model(rm):
    for attr in ("model", "transformer"):
        b = getattr(rm, attr, None)
        if b is not None:
            return b
    raise AttributeError("Cannot find base model (tried 'model', 'transformer').")


def load_reward_model(model_name: str, device: str):
    print(f"Loading reward model: {model_name}  device={device}")
    kwargs = dict(torch_dtype=torch.bfloat16, device_map=device, num_labels=1)
    try:
        rm = AutoModelForSequenceClassification.from_pretrained(
            model_name, attn_implementation="flash_attention_2", **kwargs
        )
    except Exception:
        rm = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    rm.eval()
    return rm, tokenizer


def score_conversation(convo: dict[str, Any], rm, tokenizer, device: str, model_name: str) -> dict[str, Any]:
    message_list = convo.get("message_list", [])
    domain = str(convo.get("domain", ""))
    guidelines = convo.get("assistant_guidelines", {})
    labeled_turns = collect_labeled_turns(convo)
    labeled_index = {lt["turn_index"]: lt for lt in labeled_turns}

    guidelines_text = format_guidelines(guidelines, infer_category_titles(guidelines))
    full_messages = format_full_conversation_for_rm(message_list, guidelines_text, domain)
    bos = tokenizer.bos_token

    full_fmt = tokenizer.apply_chat_template(full_messages, tokenize=False)
    if bos and full_fmt.startswith(bos):
        full_fmt = full_fmt[len(bos):]
    full_inputs = tokenizer(full_fmt, return_tensors="pt").to(device)
    full_seq_len = full_inputs["input_ids"].shape[1]

    prefix_messages: list[dict[str, str]] = [full_messages[0]]
    turn_positions: list[tuple[int, int]] = []
    for msg in message_list:
        role = str(msg.get("role", ""))
        ti = int(msg.get("turn_index", -1))
        if role not in ("user", "assistant"):
            continue
        prefix_messages.append({"role": role, "content": str(msg.get("content", ""))})
        if role == "assistant" and ti in labeled_index:
            prefix_fmt = tokenizer.apply_chat_template(prefix_messages, tokenize=False)
            if bos and prefix_fmt.startswith(bos):
                prefix_fmt = prefix_fmt[len(bos):]
            n_tokens = len(tokenizer(prefix_fmt, return_tensors="pt")["input_ids"][0])
            turn_positions.append((ti, min(n_tokens - 1, full_seq_len - 1)))

    with torch.no_grad():
        hidden = _base_model(rm)(
            input_ids=full_inputs["input_ids"],
            attention_mask=full_inputs["attention_mask"],
        ).last_hidden_state
        all_scores = _score_head(rm)(hidden).squeeze(-1).squeeze(0)

    turn_scores: list[dict[str, Any]] = []
    for ti, pos in turn_positions:
        lt = labeled_index[ti]
        if not lt.get("guideline_text"):
            continue
        turn_scores.append({
            "turn_index": ti,
            "score": all_scores[pos].item(),
            "is_violation_truth": lt["is_violation_truth"],
            "category": lt["category"],
            "key": lt["key"],
            "phase": lt["phase"],
            "guideline_text": lt["guideline_text"],
        })

    raw = [t["score"] for t in turn_scores]
    return {
        "conversation_id": convo.get("conversation_id", ""),
        "rm_model": model_name,
        "domain": domain,
        "has_violation_truth": bool(convo.get("mistakes")),
        "score_min": min(raw) if raw else None,
        "score_mean": sum(raw) / len(raw) if raw else None,
        "turn_scores": turn_scores,
    }


def run(
    *,
    model_name: str,
    data_dir: Path,
    output_dir: Path,
    device: str,
    limit: int | None,
    domains: list[str] | None,
    overwrite: bool,
) -> int:
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    rm, tokenizer = load_reward_model(model_name, device)

    domain_files = discover_conversation_files(data_dir)
    if domains:
        domain_files = [(d, p) for d, p in domain_files if d in set(domains)]
    if not domain_files:
        print(f"No conversation files found in {data_dir}")
        return 0

    def _out_path(domain: str, path: Path) -> Path:
        d = output_dir / domain if domain else output_dir
        return d / f"{path.stem}_rm_scores.json"

    tasks = [(d, p) for d, p in domain_files if overwrite or not _out_path(d, p).exists()]
    if not tasks:
        print("All conversations already scored.")
        return 0
    if limit:
        tasks = tasks[:limit]

    print(f"Scoring {len(tasks)} conversation(s) — 1 forward pass each.")
    bar = tqdm(total=len(tasks), desc="RMScore", dynamic_ncols=True) if tqdm else None
    for domain, path in tasks:
        out_file = _out_path(domain, path)
        try:
            convo = read_json(path)
            convo.setdefault("conversation_id", path.stem)
            result = score_conversation(convo, rm, tokenizer, device, model_name)
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
        or (ROOT / "results" / "rm_classifier")
    )
    domains_raw = args.domains or cfg.get("domains")
    domains = None
    if isinstance(domains_raw, str):
        domains = [d.strip() for d in domains_raw.split(",") if d.strip()] or None
    elif isinstance(domains_raw, list):
        domains = [str(d).strip() for d in domains_raw if str(d).strip()] or None

    return run(
        model_name=model_name, data_dir=data_dir, output_dir=output_dir,
        device=args.device, limit=args.limit, domains=domains, overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    raise SystemExit(main())
