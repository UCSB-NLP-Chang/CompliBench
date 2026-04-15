"""Domain constants and category-name normalization."""
from __future__ import annotations


def normalize_category(cat: str) -> str:
    """Map free-form category strings to the canonical Category-N titles."""
    c = (cat or "").strip().lower()
    if not c:
        return ""
    if c.startswith(("category 1", "cat 1", "cat1")):
        return "Category 1: Universal Compliance"
    if c.startswith(("category 2", "cat 2", "cat2")):
        return "Category 2: Intent Triggered Guidelines"
    if c.startswith(("category 3", "cat 3", "cat3")):
        return "Category 3: Condition Triggered Guidelines"
    if "universal" in c or "compliance" in c:
        return "Category 1: Universal Compliance"
    if "intent" in c or "triggered" in c:
        return "Category 2: Intent Triggered Guidelines"
    if "condition" in c or "conditional" in c:
        return "Category 3: Condition Triggered Guidelines"
    return cat


__all__ = ["normalize_category"]
