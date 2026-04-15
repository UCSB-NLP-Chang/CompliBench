# CompliBench: Benchmarking LLM Judges for Compliance Violation Detection in Dialogue Systems

This is the official implementation of the paper *CompliBench: Benchmarking LLM Judges for Compliance Violation Detection in Dialogue Systems*.

## Abstract

Dialogue systems in regulated domains (airlines, healthcare, insurance) must follow domain-specific compliance guidelines at every assistant turn. CompliBench benchmarks how well modern LLM judges and reward models detect compliance violations in multi-turn conversations against an oracle of intent-triggered and condition-triggered guidelines. We release (1) three labeled dialogue datasets spanning airlines, healthcare, and insurance; (2) a unified harness that runs the same benchmark across remote-API LLM judges, locally served LLM judges, classifier reward models, and generative reward models; and (3) a metric suite that treats non-violation identification, violation detection, and conversation-level correctness as three directly comparable axes.

## Requirements

### Environment

```
python -m pip install -r requirements.txt
```

Extra packages are needed for some pipelines — uncomment the relevant block in `requirements.txt`:

* `anthropic` / `google-generativeai` for the Claude / Gemini APIs
* `torch vllm transformers` for the local-vLLM LLM judge
* `torch transformers accelerate` for the classifier / generative reward models

### API credentials

```
cp .env.example .env
```

Fill in only the providers you plan to use. Supported providers and their env vars:

| Provider              | Env vars                                                                       |
| --------------------- | ------------------------------------------------------------------------------ |
| `claude`            | `ANTHROPIC_API_KEY`                                                          |
| `deepseek`          | `DEEPSEEK_API_KEY`                                                           |
| `gemini`            | `GEMINI_API_KEY`                                                             |
| `kimi`              | `KIMI_API_KEY`                                                               |
| `qwen`              | `DASHSCOPE_API_KEY`                                                          |
| `openai_compatible` | `OPENAI_COMPAT_BASE_URL`, `OPENAI_COMPAT_API_KEY`, `OPENAI_COMPAT_MODEL` |

### Data

The three evaluation datasets ship in `data/{airlines,healthcare,insurance}/` (83 / 109 / 117 conversations). No download step required. Each `data/<domain>/conversation_*.json`:

```jsonc
{
  "domain": "airlines",
  "assistant_guidelines": {
    "Category 1: Universal Compliance":            { "<key>": "<text>" },
    "Category 2: Intent Triggered Guidelines":     { "<intent>": { "Phase 1": "...", "Phase 2": "..." } },
    "Category 3: Condition Triggered Guidelines":  { "<key>": "<text>" }
  },
  "message_list": [
    { "turn_index": 1, "role": "assistant", "content": "...",
      "category": "Category 2: ...", "key": "<intent>", "phase": 1 }
  ],
  "mistakes":       [ /* ground-truth violations; override message_list at same turn_index */ ],
  "cat2_overrides": [ /* original vs. modified guideline text for each injected violation */ ],
  "cat3_overrides": [ ... ]
}
```

Evaluation uses Category 2 (intent-triggered with ordered phases) and Category 3 (condition-triggered) guidelines only; Category 1 (universal compliance) is excluded by design.

## Usage

### Run a judge

All pipelines go through `python -m convjudge.<module>`. The main arguments:

```text
--provider      Provider name (API pipelines only)
--model         Model identifier
--data-dir      Dataset directory (e.g. data/airlines)
--output-dir    Where to write per-conversation outputs
--n             Number of attempts per conversation
--num-workers   Concurrent requests (API pipelines only)
```

Example — run DeepSeek-R1 on the airlines dataset:

```shell
python -m convjudge.inference.llm_api \
    --provider deepseek --model deepseek-reasoner \
    --data-dir data/airlines --output-dir results/llm_api/airlines \
    --n 1 --num-workers 32
```

Example — run Qwen3-8B locally via vLLM:

```shell
python -m convjudge.inference.llm_vllm \
    --model Qwen/Qwen3-8B \
    --data-dir data/airlines --output-dir results/llm_vllm/airlines \
    --tensor-parallel-size 1 --batch-size 8 \
    --temperature 0.6 --top-p 0.95 --top-k 20 --n 4 \
    --trust-remote-code
```

Example — score with a classifier reward model:

```shell
python -m convjudge.inference.rm_classifier \
    --model-name Skywork/Skywork-Reward-V2-Llama-3.1-8B \
    --data-dir data/airlines \
    --output-dir results/rm_classifier/airlines/Skywork_Skywork-Reward-V2-Llama-3.1-8B \
    --device cuda:0
```

### Evaluate

```shell
# LLM judges
python -m convjudge.evaluation.eval_llm \
    --usage-root results/llm_api/airlines --data-root data/airlines \
    --output-dir results/llm_api/airlines/usage_accuracy

# Reward models
python -m convjudge.evaluation.eval_rm \
    --scores-dir results/rm_classifier/airlines/<model_slug> \
    --output-dir results/rm_classifier/airlines/<model_slug>
```

Each run writes a `summary.json` with three headline fields, shared across LLM and reward-model pipelines:

* `micro_correct_accuracy_strict` — fraction of non-violation assistant turns answered correctly. For LLM judges: exact `(category, key, phase)` match with `is_violation=false`. For reward models at threshold `T`: `score ≥ T`.
* `micro_accuracy_violation_detect` — fraction of violation turns flagged as such. LLM: `is_violation=true` (guideline identity not required). RM: `score < T`.
* `file_correct_accuracy_strict` — fraction of conversations where every labeled turn is classified correctly (both axes above at 100%).

For reward-model runs the threshold `T` is auto-searched per `(model, domain)` to maximise `file_correct_accuracy_strict`; pass `--threshold <float>` to `eval_rm` to fix it.

Example `summary.json`:

```json
{
  "micro_correct_accuracy_strict": 0.83,
  "micro_accuracy_violation_detect": 0.76,
  "file_correct_accuracy_strict": 0.41,
  "by_model": {
    "deepseek-reasoner": {
      "micro_correct_accuracy_strict": 0.83,
      "micro_accuracy_violation_detect": 0.76,
      "file_correct_accuracy_strict": 0.41
    }
  }
}
```

`by_model_domain` breakdowns are nested in the same JSON when multiple domains are evaluated together. Per-turn error files for qualitative analysis live under `usage_accuracy/<model>/model_errors/`.

## Reproduce results in our paper

Shell wrappers in `scripts/` loop over all three datasets and chain inference + evaluation:

```shell
PROVIDER=deepseek MODEL=deepseek-reasoner         bash scripts/run_llm_api.sh
MODEL=Qwen/Qwen3-8B TP=1 TEMPERATURE=0.6 N=4      bash scripts/run_llm_vllm.sh
MODEL=Skywork/Skywork-Reward-V2-Llama-3.1-8B      bash scripts/run_rm_classifier.sh
MODEL=BBQGOD/DeepSeek-GRM-16B                     bash scripts/run_rm_generative.sh
```

Outputs land in `results/<pipeline>/<dataset>/usage_accuracy/summary.json`.

Per-pipeline runnable env vars: `DATASETS` (subset), `N`, `NUM_WORKERS`, `OUT_ROOT`, and for vLLM / RM pipelines `CUDA_VISIBLE_DEVICES`, `TP`, `BATCH_SIZE`, `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`.

## Repository layout

```
CompliBench/
├── data/{airlines,healthcare,insurance}/   # 83 / 109 / 117 conversations
├── configs/default.yaml                    # defaults (override via CLI flags)
├── convjudge/
│   ├── common/                             # prompts, formatting, IO helpers
│   ├── providers/                          # 6 API callers + registry
│   ├── inference/                          # llm_api · llm_vllm · rm_classifier · rm_generative_scalar
│   └── evaluation/                         # eval_llm · eval_rm · metrics · rm_threshold_search
└── scripts/run_{llm_api,llm_vllm,rm_classifier,rm_generative}.sh
```

## License

Apache 2.0. See [LICENSE](LICENSE).
