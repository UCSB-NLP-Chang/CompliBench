#!/usr/bin/env python3
"""Run the LLM-as-judge locally via vLLM (in-process batched inference).

Input/output schema matches ``convjudge.inference.llm_api`` so the downstream
evaluation script can consume either. Requires ``pip install vllm torch``.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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


def _as_optional_float(cfg: dict[str, Any], key: str) -> float | None:
    raw = cfg.get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except Exception as exc:
        raise ValueError(f"Config field '{key}' must be a float or null.") from exc


@dataclass
class EvalTask:
    domain: str
    path: Path
    attempt: int | None
    out_path: Path
    conversation_id: str
    messages: list[dict[str, str]]


def _sanitize_model_name(model: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in model)


def _build_messages(conversation: dict[str, Any], oracle: dict[str, Any], conv_id: str) -> list[dict[str, str]]:
    cat_titles = infer_category_titles(oracle)
    user_prompt = build_user_prompt(
        format_guidelines(oracle, cat_titles),
        format_conversation(conversation.get("message_list", [])),
        conv_id,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _extract_post_think(text: str) -> str:
    if not text:
        return ""
    marker = "</think>"
    idx = text.rfind(marker)
    return text[idx + len(marker):].strip() if idx >= 0 else text.strip()


def _normalize_turn_guidelines(obj: dict[str, Any]) -> list[dict[str, Any]]:
    used = obj.get("turn_guidelines") or obj.get("guidelines_used") or []
    if not isinstance(used, list):
        return []
    out: list[dict[str, Any]] = []
    for item in used:
        if not isinstance(item, dict):
            continue
        try:
            ti = int(item.get("turn_index", -1))
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
        is_v = bool(guid.get("is_violation")) if isinstance(guid.get("is_violation"), bool) else False
        reason = guid.get("judge_reason", guid.get("violation_reason"))
        reason_str = reason.strip() if isinstance(reason, str) and reason.strip() else "judgment rationale required"
        out.append({
            "turn_index": ti,
            "guideline_used": {
                "guidance_category": cat,
                "guidance_key": key,
                "guideline_phase": phase,
                "is_violation": is_v,
                "judge_reason": reason_str,
            },
        })
    return out


def _build_tasks(
    *,
    model: str,
    data_dir: Path,
    output_dir: Path,
    fallback_oracle: dict[str, Any] | None,
    domains: list[str] | None,
    n: int,
    overwrite: bool,
) -> list[EvalTask]:
    domain_files = discover_conversation_files(data_dir)
    if domains:
        allowed = set(domains)
        domain_files = [(d, p) for d, p in domain_files if d in allowed]
    if not domain_files:
        return []

    model_subdir = output_dir / _sanitize_model_name(model)
    tasks: list[EvalTask] = []
    max_attempt = max(1, int(n))
    for domain, path in domain_files:
        conversation = read_json(path)
        oracle = conversation.get("assistant_guidelines")
        if not isinstance(oracle, dict) or not oracle:
            oracle = fallback_oracle
        if oracle is None:
            raise ValueError(f"Missing assistant_guidelines and no fallback for {path}")
        conv_id = path.stem
        messages = _build_messages(conversation, oracle, conv_id)
        out_subdir = model_subdir / domain if domain else model_subdir

        for attempt_idx in range(1, max_attempt + 1):
            attempt_val = attempt_idx if max_attempt > 1 else None
            suffix = f"_usage_attempt{attempt_val}" if attempt_val else "_usage"
            out_path = out_subdir / f"{path.stem}{suffix}.json"
            if out_path.exists() and not overwrite:
                continue
            tasks.append(EvalTask(
                domain=domain, path=path, attempt=attempt_val,
                out_path=out_path, conversation_id=conv_id, messages=messages,
            ))
    return tasks


def run(
    *,
    model: str,
    data_dir: Path,
    output_dir: Path,
    guidelines_path: Path | None,
    limit: int | None,
    domains: list[str] | None,
    n: int,
    overwrite: bool,
    batch_size: int,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    min_p: float | None,
    max_tokens: int | None,
    tensor_parallel_size: int,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    trust_remote_code: bool,
    enforce_eager: bool,
    seed: int,
) -> int:
    from vllm import LLM, SamplingParams  # imported here so CLI help works without vllm

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    fallback_oracle: dict[str, Any] | None = None
    if guidelines_path is not None:
        obj = read_json(guidelines_path)
        if not isinstance(obj, dict):
            raise ValueError(f"Guidelines must be a JSON object: {guidelines_path}")
        fallback_oracle = obj

    tasks = _build_tasks(
        model=model, data_dir=data_dir, output_dir=output_dir,
        fallback_oracle=fallback_oracle, domains=domains, n=n, overwrite=overwrite,
    )
    if not tasks:
        print("All conversations already have usage outputs.")
        return 0
    if limit and limit > 0:
        tasks = tasks[:limit]

    print(f"Running vLLM inference for {len(tasks)} item(s), batch_size={batch_size}, tp={tensor_parallel_size}.")

    llm_kwargs: dict[str, Any] = dict(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=trust_remote_code,
        enforce_eager=enforce_eager,
        seed=seed,
    )
    if max_model_len is not None and max_model_len > 0:
        llm_kwargs["max_model_len"] = max_model_len
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    sampling_kwargs: dict[str, Any] = {}
    if max_tokens is not None:
        sampling_kwargs["max_tokens"] = int(max_tokens)
    if temperature is not None:
        sampling_kwargs["temperature"] = float(temperature)
    if top_p is not None:
        sampling_kwargs["top_p"] = float(top_p)
    if top_k is not None:
        sampling_kwargs["top_k"] = int(top_k)
    if min_p is not None:
        sampling_kwargs["min_p"] = float(min_p)
    base_params = SamplingParams(**sampling_kwargs)

    def _sp_for(task: EvalTask) -> SamplingParams:
        if sampling_kwargs.get("temperature", 0) == 0:
            return base_params
        attempt_seed = hash((seed, task.attempt or 0)) % (2**31)
        return SamplingParams(**{**sampling_kwargs, "seed": attempt_seed})

    progress = tqdm(total=len(tasks), desc="vLLMUsage", dynamic_ncols=True) if tqdm else None

    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(t.messages, tokenize=False, add_generation_prompt=True)
            for t in batch
        ]
        outputs = llm.generate(prompts, sampling_params=[_sp_for(t) for t in batch], use_tqdm=False)
        if len(outputs) != len(batch):
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(batch)} prompts.")

        for task, output in zip(batch, outputs):
            response_text = str(output.outputs[0].text or "") if getattr(output, "outputs", None) else ""
            parsed = _extract_post_think(response_text)
            try:
                parsed_obj = extract_first_json(parsed)
            except Exception:
                parsed_obj = {"conversation_id": task.conversation_id, "guidelines_used": []}
            payload: dict[str, Any] = {
                "conversation_id": task.conversation_id,
                "model": model,
                "turn_guidelines": _normalize_turn_guidelines(parsed_obj),
                "raw_response": response_text,
                "parsed_response": parsed,
            }
            if task.attempt:
                payload["attempt"] = task.attempt
            task.out_path.parent.mkdir(parents=True, exist_ok=True)
            task.out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            if progress:
                progress.update(1)

    if progress:
        progress.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--guidelines", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--domains", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg: dict[str, Any] = {}
    cfg_path = None
    if args.config:
        cfg_path = resolve_path(args.config, root=ROOT) or Path(str(args.config))
    else:
        default = ROOT / "configs" / "default.yaml"
        if default.exists():
            cfg_path = default
    if cfg_path:
        cfg = load_yaml_mapping(cfg_path)

    model = str(args.model or as_optional_str(cfg, "model") or "Qwen/Qwen3-8B")
    data_dir = resolve_path(args.data_dir, root=ROOT) or resolve_path(as_optional_str(cfg, "data_dir"), root=ROOT)
    if data_dir is None:
        raise SystemExit("--data-dir is required.")
    output_dir = (
        resolve_path(args.output_dir, root=ROOT)
        or resolve_path(as_optional_str(cfg, "output_dir"), root=ROOT)
        or (ROOT / "results" / "llm_vllm")
    )
    guidelines_path = resolve_path(args.guidelines, root=ROOT) or resolve_path(as_optional_str(cfg, "guidelines"), root=ROOT)
    limit = args.limit if args.limit is not None else as_optional_int(cfg, "limit")
    n = int(args.n if args.n is not None else (as_optional_int(cfg, "n") or 1))
    domains = parse_domains(args.domains) or parse_domains(cfg.get("domains"))
    batch_size = int(args.batch_size if args.batch_size is not None else (as_optional_int(cfg, "batch_size") or 8))
    temperature = args.temperature if args.temperature is not None else _as_optional_float(cfg, "temperature")
    top_p = args.top_p if args.top_p is not None else _as_optional_float(cfg, "top_p")
    top_k = args.top_k if args.top_k is not None else as_optional_int(cfg, "top_k")
    min_p = args.min_p if args.min_p is not None else _as_optional_float(cfg, "min_p")
    max_tokens = int(args.max_tokens if args.max_tokens is not None else (as_optional_int(cfg, "max_tokens") or 38912))
    tensor_parallel_size = int(args.tensor_parallel_size if args.tensor_parallel_size is not None else (as_optional_int(cfg, "tensor_parallel_size") or 1))
    dtype = str(args.dtype or as_optional_str(cfg, "dtype") or "bfloat16")
    gpu_memory_utilization = float(
        args.gpu_memory_utilization if args.gpu_memory_utilization is not None
        else (_as_optional_float(cfg, "gpu_memory_utilization") or 0.9)
    )
    max_model_len = args.max_model_len if args.max_model_len is not None else as_optional_int(cfg, "max_model_len")
    seed = int(args.seed if args.seed is not None else (as_optional_int(cfg, "seed") or 0))

    return run(
        model=model, data_dir=data_dir, output_dir=output_dir, guidelines_path=guidelines_path,
        limit=limit, domains=domains, n=n, overwrite=bool(args.overwrite),
        batch_size=batch_size, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
        max_tokens=max_tokens, tensor_parallel_size=tensor_parallel_size, dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
        trust_remote_code=bool(args.trust_remote_code), enforce_eager=bool(args.enforce_eager),
        seed=seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
