# airlines_grouped_violation_03-fixed-v2

**95 conversations** in the `airlines_short_grouped` domain (Celestar Air). This is the **eval split** of the violation dataset.

Each file is a synthetic airline customer service conversation where some of the assistant's guidelines were intentionally replaced with bad/harmful instructions. The assistant follows these injected bad guidelines, and the resulting violations are annotated. Each conversation contains at least one violation.

## Quick Reading Guide

1. **`violation_directives`** — what bad instructions were injected
2. **`mistakes`** — which turns executed them (with evidence)
3. **`message_list[turn_index].content`** — the assistant's actual output at each violated turn

---

## Field Reference

Each `conversation_XXXXX.json` contains the fields below.

### Identity & Context

| Field | Description |
|-------|-------------|
| `domain` | Always `airlines_short_grouped` in this dataset |
| `intent_name` | User intent for this conversation (e.g. `loyalty_account_assistance`) |
| `workflow_index` | Variant index for the same intent workflow (0-based) |
| `pool_dir` | Path to the source guideline pool |

### Guidelines

**Three guideline categories:**
- **Cat1 (Universal Compliance)** — general tone rules, apply to all conversations
- **Cat2 (Intent Triggered)** — step-by-step workflow (Phase 1–N), triggered by the user's intent
- **Cat3 (Condition Triggered)** — rules for special situations (e.g. medical emergencies, payment disputes)

| Field | Description |
|-------|-------------|
| `assistant_guidelines` | The **original, correct** guidelines (all three categories) |
| `assistant_guidelines_shuffled` | The **modified version** the assistant actually follows. Cat2 phases are shuffled and some replaced with bad instructions. Only contains Cat2. |
| `cat2_modified_ratio` | Fraction of Cat2 phases modified (0.3 in this dataset) |
| `cat3_modified_ratio` | Fraction of Cat3 conditions modified (0.5 in this dataset) |
| `cat2_overrides` | Cat2 rules that were replaced. Each entry: `phase`, `original`, `modified` |
| `cat3_overrides` | Cat3 rules that were replaced. Each entry: `key`, `original`, `modified`. May be `[]` |
| `violation_directives` | Union of `cat2_overrides` + `cat3_overrides`. Quickest way to see all injected bad instructions |

### Persona

The `persona` field contains a synthetic user identity for realistic conversation content:

```
name, gender, phone, email, address (street/city/state/zip),
record_locator, pnr, ticket_number, flight_number,
origin, destination, travel_date, return_date,
seat_number, seat_preference, cabin_class, fare_type,
payment_method, card_last4
```

### `message_list`

The conversation, one entry per turn:

| Field | Present on | Description |
|-------|------------|-------------|
| `turn_index` | all | Turn index (0-based) |
| `role` | all | `"assistant"` or `"user"` |
| `content` | all | Message text |
| `category` | assistant | Guideline category (`Category 2` or `Category 3`) |
| `key` | assistant | Intent name (Cat2) or condition name (Cat3) |
| `phase` | assistant | Phase number for Cat2; `-1` for Cat3 |
| `guideline_text` | assistant | The guideline this turn follows (from `assistant_guidelines_shuffled`) — may be an injected bad instruction |
| `is_violation` | some assistant | `true` when flagged as a violation |
| `content_judge` | some assistant | Compliance judgment; `final_is_compliant: false` = assistant did not strictly follow `guideline_text` |

### `mistakes`

Ground-truth list of all violated turns:

| Field | Description |
|-------|-------------|
| `turn_index` | Turn where the violation occurred |
| `guidance category` | `Category 2` or `Category 3` |
| `guidance key` | Intent or condition name |
| `guideline_phase` | Phase number (Cat2) or `-1` (Cat3) |
| `guideline` | The bad guideline that was followed |
| `evidence` | The assistant's actual output (violation evidence) |
