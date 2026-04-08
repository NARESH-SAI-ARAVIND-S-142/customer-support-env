"""
customer_support_env/graders.py

Deterministic graders for all three task levels.

Scoring rubric
--------------
Task 1 (Easy)  – classification only
  classification_score : 1.0  (binary)
  response_score       : 0.0  (not evaluated)
  resolution_score     : 0.0  (not evaluated)
  total                : classification_score * 1.0

Task 2 (Medium) – classification + response
  classification_score : 0.40
  response_score       : 0.60
  resolution_score     : 0.00
  total                : weighted sum

Task 3 (Hard) – all three
  classification_score : 0.30
  response_score       : 0.30
  resolution_score     : 0.40
  total                : weighted sum − penalty
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from env import Action, Reward
    from tasks import Task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_CATEGORIES = {
    "delivery", "billing", "damaged_product", "account_access"
}

VALID_RESOLUTIONS = {
    "refund", "resend", "escalate", "reset_password", "account_unlock",
    "discount_applied", "schedule_pickup", "investigation_opened",
}

# Minimum response length to be considered non-trivial
MIN_RESPONSE_LENGTH = 30


def _normalise(text: str) -> str:
    """Lowercase and strip punctuation for fuzzy matching."""
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


def _keyword_hit_rate(response: str, keywords: List[str]) -> float:
    """
    Fraction of expected keywords found in the response.
    Matching is case-insensitive, partial-word aware.
    """
    if not keywords:
        return 1.0
    normalised = _normalise(response)
    hits = sum(1 for kw in keywords if _normalise(kw) in normalised)
    return hits / len(keywords)


def _grade_classification(action_category: str, expected_category: str) -> float:
    """Binary: 1.0 if exact match, 0.0 otherwise."""
    return 1.0 if _normalise(action_category) == _normalise(expected_category) else 0.0


def _grade_response(response: str, expected_keywords: List[str]) -> float:
    """
    Score a customer-facing response on two sub-criteria:

    1. Length adequacy  (40 % of response score)
       - 0 pts if < MIN_RESPONSE_LENGTH chars
       - scales up to 1.0 proportionally until 100 chars, then caps

    2. Keyword coverage (60 % of response score)
       - Fraction of expected_keywords found in the response
    """
    if not response or len(response.strip()) < MIN_RESPONSE_LENGTH:
        return 0.0

    length_score = min(len(response.strip()) / 150.0, 1.0)
    keyword_score = _keyword_hit_rate(response, expected_keywords)

    return round(0.40 * length_score + 0.60 * keyword_score, 4)


def _grade_resolution(
    resolution_action: str,
    expected_resolution: str,
    task_id: int,
) -> float:
    """
    Binary grade for resolution action.
    Only evaluated in Task 3; returns 0 for Tasks 1 and 2.
    """
    if task_id != 3:
        return 0.0
    norm = _normalise(resolution_action.strip())
    # Accept exact match or the word appearing inside a longer phrase
    return 1.0 if _normalise(expected_resolution) in norm else 0.0


def _compute_repeat_penalty(
    action_dict: Dict[str, Any],
    previous_actions: List[Dict[str, Any]],
) -> float:
    """
    Penalise exact duplicate actions: -0.1 per repeat, max -0.3.
    Duplicates are defined as identical (classification, resolution_action) pairs.
    """
    penalty = 0.0
    current_sig = (
        action_dict.get("classification", "").lower(),
        action_dict.get("resolution_action", "").lower(),
    )
    count = sum(
        1 for prev in previous_actions
        if (
            prev.get("classification", "").lower(),
            prev.get("resolution_action", "").lower(),
        ) == current_sig
    )
    penalty = min(count * 0.10, 0.30)
    return round(penalty, 4)


# ---------------------------------------------------------------------------
# Main grader entry point
# ---------------------------------------------------------------------------

def grade_action(
    task: "Task",
    action: "Action",
    previous_actions: List[Dict[str, Any]],
) -> "Reward":
    """
    Grade an agent action against the task ground truth.

    Returns a Reward model with per-component scores and the total.
    """
    from env import Reward  # local import to avoid circular at module load

    task_id = task.task_id

    # --- Component scores --------------------------------------------------
    classification_score = _grade_classification(
        action.classification, task.expected_category
    )
    response_score = _grade_response(action.response, task.expected_keywords)
    resolution_score = _grade_resolution(
        action.resolution_action, task.expected_resolution, task_id
    )

    # --- Weights per task ---------------------------------------------------
    weights = {
        1: (1.00, 0.00, 0.00),   # (classification, response, resolution)
        2: (0.40, 0.60, 0.00),
        3: (0.30, 0.30, 0.40),
    }
    w_cls, w_resp, w_res = weights[task_id]

    weighted = (
        w_cls * classification_score
        + w_resp * response_score
        + w_res * resolution_score
    )

    # --- Repeat-action penalty (only after first step) ----------------------
    penalty = _compute_repeat_penalty(action.model_dump(), previous_actions)

    # --- Penalties for empty / invalid fields ------------------------------
    if action.classification and action.classification not in VALID_CATEGORIES:
        # Agent used an unknown category — light penalty
        penalty += 0.05

    if task_id == 3 and action.resolution_action not in VALID_RESOLUTIONS:
        # Invalid resolution action in hard task
        penalty += 0.05

    penalty = round(min(penalty, 0.40), 4)   # cap total penalty at 0.40
    total = round(max(0.0, min(1.0, weighted - penalty)), 4)

    breakdown = {
        "task_id": task_id,
        "expected_category": task.expected_category,
        "agent_category": action.classification,
        "classification_correct": bool(classification_score),
        "keyword_hit_rate": _keyword_hit_rate(action.response, task.expected_keywords),
        "response_length": len(action.response),
        "expected_resolution": task.expected_resolution if task_id == 3 else "N/A",
        "agent_resolution": action.resolution_action if task_id == 3 else "N/A",
        "resolution_correct": bool(resolution_score),
        "weights": {"classification": w_cls, "response": w_resp, "resolution": w_res},
        "penalty": penalty,
    }

    return Reward(
        total=total,
        classification_score=round(classification_score, 4),
        response_score=round(response_score, 4),
        resolution_score=round(resolution_score, 4),
        penalty=penalty,
        breakdown=breakdown,
    )
