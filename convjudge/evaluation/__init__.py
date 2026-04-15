"""Evaluation metric computation for LLM-judge and RM outputs."""

#: The three summary fields surfaced in ``summary.json`` for both pipelines.
#: Semantics differ (LLM checks cat/key/phase strict match; RM checks
#: score-vs-threshold), but the keys match so results can be compared
#: side-by-side.
SUMMARY_KEYS: tuple[str, ...] = (
    "micro_correct_accuracy_strict",
    "micro_accuracy_violation_detect",
    "file_correct_accuracy_strict",
)


def filter_summary(d: dict) -> dict:
    """Return a copy of ``d`` restricted to :data:`SUMMARY_KEYS`."""
    return {k: d.get(k) for k in SUMMARY_KEYS if k in d}


__all__ = ["SUMMARY_KEYS", "filter_summary"]
