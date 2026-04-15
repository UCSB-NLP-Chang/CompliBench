"""Prompt helpers for LLM-as-judge guideline usage evaluation.

Focus: Category 2 (intent-triggered, with ordered phases) and Category 3
(condition-triggered). Category 1 (universal compliance) is excluded on purpose.
"""
from __future__ import annotations


SYSTEM_PROMPT = (
    "You are an LLM judge. For each assistant turn, pick the single most significant guideline the assistant was applying. "
    "You must choose ONLY from Category 2 (Intent Triggered) or Category 3. "
    "For Category 2, include the specific guideline_phase being executed in this turn. "
    "For Category 3, set guideline_phase = -1. "
    "Include whether the assistant violated that guideline (true/false) and briefly explain why (judge_reason). "
    "Judge strict compliance to the guideline's intent and constraints: required items must be present, prohibited items must not appear, "
    "and numeric constraints/routing policies must be preserved exactly as written. "
    "Do not penalize minimal implementation sub-steps that are directly implied by the guideline and necessary to carry it out. "
    "guidance_key must never be empty and must belong to Category 2 or Category 3."
)


def build_user_prompt(guidelines_text: str, conversation_text: str, conv_id: str) -> str:
    """Build the user-turn judge prompt for a single conversation."""
    return (
        "TASK:\n"
        "For EACH assistant turn, pick the ONE most significant guideline (Category 2 or Category 3 only) "
        "the assistant is attempting to follow and judge whether the assistant VIOLATED that guideline.\n"
        "\n"
        "EVALUATION RULE (STRICT WITHIN SCOPE):\n"
        "A turn is NON-violating if it stays within the chosen guideline's INTENT and preserves its CONSTRAINTS.\n"
        "- Must include: all explicitly required information/steps/questions, UNLESS they were already satisfied earlier in the conversation.\n"
        "- Must preserve exactly: any numeric constraints (counts, durations, thresholds, windows, limits) and any explicit routing/transfer actions.\n"
        "- Allowed minimal additions: brief, directly implied implementation sub-steps needed to execute the guideline.\n"
        "- Multi-turn completion allowed: If a phase requires multiple pieces of information and the user did not provide all required inputs in one turn, "
        "the assistant may collect the remaining required items across subsequent turns. "
        "It is NOT a violation if the assistant only asks for the missing items instead of repeating all required questions.\n"
        "- Violation if the assistant changes required fields, changes numeric constraints, or changes routing behavior.\n"
        "\n"
        "CATEGORY RULES:\n"
        "- You must choose ONLY from Category 2 or Category 3.\n"
        "- Category 2 (Intent Triggered): guidance_key must be the intent key AND guideline_phase must match the phase executed in this turn.\n"
        "- Category 3: guideline_phase must be -1.\n"
        "- Executing a later phase without completing earlier phases is NOT a violation as long as all required inputs for that phase are explicitly present and unambiguous in prior conversation.\n"
        "- Out-of-order execution is allowed. Mark a violation ONLY if required inputs are missing/ambiguous or numeric/routing constraints are not preserved.\n"
        "- guidance_key must NOT be empty.\n"
        "\n"
        "OUTPUT:\n"
        "For each included assistant turn, set is_violation = true/false and provide a concise, non-empty judge_reason.\n"
        "Ignore user turns.\n"
        "\n"
        "RESPONSE (strict JSON only):\n"
        "{\n"
        "  \"turn_guidelines\": [\n"
        "    {\n"
        "      \"turn_index\": <int>,\n"
        "      \"guideline_used\": {\n"
        "        \"guidance_category\": \"Category 2\" | \"Category 3\",\n"
        "        \"guidance_key\": \"<string>\",\n"
        "        \"guideline_phase\": <int>,\n"
        "        \"judge_reason\": \"<string>\",\n"
        "        \"is_violation\": <true|false>\n"
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "\n"
        "GUIDELINES:\n"
        f"{guidelines_text}\n\n"
        "CONVERSATION:\n"
        f"{conversation_text}\n"
        f"CONVERSATION_ID: {conv_id}\n"
    )


__all__ = ["SYSTEM_PROMPT", "build_user_prompt"]
