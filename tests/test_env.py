"""
tests/test_env.py

Unit tests for the Customer Support Ticket Resolver environment.
No API key required — all tests use the deterministic local grader.

Run:
  python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
import os
import pytest

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env import Action, CustomerSupportEnv, Observation, Reward, make_env
from graders import grade_action, _grade_classification, _grade_response, _grade_resolution
from tasks import TASKS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def perfect_action_for_task(task_id: int) -> Action:
    """Return a hand-crafted perfect action for the first task of a given level."""
    tasks = [t for t in TASKS if t.task_id == task_id]
    task = tasks[0]
    return Action(
        classification=task.expected_category,
        response=(
            "Dear customer, we sincerely apologise for the inconvenience you have "
            "experienced. We have thoroughly reviewed your ticket and we understand "
            "how frustrating this situation must be. Please rest assured that we "
            "are taking immediate action to resolve the issue. We will keep you "
            "updated at every step until full resolution. Thank you for your patience."
        ),
        resolution_action=task.expected_resolution,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task catalogue tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTaskCatalogue:
    def test_at_least_5_tasks_per_level(self):
        for task_id in (1, 2, 3):
            tasks = [t for t in TASKS if t.task_id == task_id]
            assert len(tasks) >= 5, f"Task {task_id} has fewer than 5 entries"

    def test_all_categories_covered(self):
        categories = {t.expected_category for t in TASKS}
        assert "delivery" in categories
        assert "billing" in categories
        assert "damaged_product" in categories
        assert "account_access" in categories

    def test_unique_ticket_ids(self):
        ids = [t.ticket_id for t in TASKS if t.ticket_id]
        assert len(ids) == len(set(ids)), "Duplicate ticket IDs found"

    def test_ticket_text_non_empty(self):
        for task in TASKS:
            assert len(task.ticket_text.strip()) > 20, f"Ticket {task.ticket_id} is too short"


# ─────────────────────────────────────────────────────────────────────────────
# Grader unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGraders:
    def test_classification_correct(self):
        assert _grade_classification("delivery", "delivery") == 1.0

    def test_classification_wrong(self):
        assert _grade_classification("billing", "delivery") == 0.0

    def test_classification_case_insensitive(self):
        assert _grade_classification("BILLING", "billing") == 1.0

    def test_response_empty_gives_zero(self):
        assert _grade_response("", ["apologize"]) == 0.0

    def test_response_too_short_gives_zero(self):
        assert _grade_response("Hi", ["apologize"]) == 0.0

    def test_response_long_with_keywords_high_score(self):
        text = (
            "We sincerely apologize for the issue with your order. We have investigated "
            "and will take immediate action to resolve this for you quickly and professionally."
        )
        score = _grade_response(text, ["apologize", "order", "investigate", "resolve"])
        assert score >= 0.70

    def test_resolution_wrong_task_always_zero(self):
        assert _grade_resolution("refund", "refund", task_id=1) == 0.0
        assert _grade_resolution("refund", "refund", task_id=2) == 0.0

    def test_resolution_correct_task3(self):
        assert _grade_resolution("refund", "refund", task_id=3) == 1.0

    def test_resolution_wrong_task3(self):
        assert _grade_resolution("resend", "refund", task_id=3) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Environment interface tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvironmentInterface:
    def test_reset_returns_observation(self):
        env = make_env(task_id=1, seed=42)
        obs = env.reset()
        assert isinstance(obs, Observation)
        assert len(obs.ticket_text) > 10
        assert obs.task_id == 1
        assert obs.step == 0

    def test_step_returns_tuple(self):
        env = make_env(task_id=1, seed=42)
        env.reset()
        action = Action(classification="delivery")
        result = env.step(action)
        assert len(result) == 4
        obs, reward, done, info = result
        assert isinstance(obs, Observation)
        assert isinstance(reward, Reward)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_before_reset_raises(self):
        env = CustomerSupportEnv(task_id=1, seed=42)
        with pytest.raises(RuntimeError):
            env.step(Action(classification="delivery"))

    def test_step_after_done_raises(self):
        env = make_env(task_id=1, seed=42)
        env.reset()
        action = perfect_action_for_task(1)
        _, _, done, _ = env.step(action)
        assert done  # perfect action ends episode immediately
        with pytest.raises(RuntimeError):
            env.step(action)

    def test_state_returns_dict(self):
        env = make_env(task_id=2, seed=42)
        env.reset()
        state = env.state()
        assert isinstance(state, dict)
        assert "task_id" in state
        assert "ticket" in state
        assert "step" in state

    def test_determinism_same_seed(self):
        env_a = make_env(task_id=1, seed=99)
        env_b = make_env(task_id=1, seed=99)
        obs_a = env_a.reset()
        obs_b = env_b.reset()
        assert obs_a.ticket_text == obs_b.ticket_text

    def test_different_seed_may_differ(self):
        env_a = make_env(task_id=1, seed=1)
        env_b = make_env(task_id=1, seed=2)
        obs_a = env_a.reset()
        obs_b = env_b.reset()
        # With only 5 tickets per task the chance of collision is 1/5 but seeds
        # 1 and 2 deterministically pick different tickets.
        # We just verify the environment doesn't crash.
        assert isinstance(obs_a.ticket_text, str)
        assert isinstance(obs_b.ticket_text, str)

    def test_max_steps_terminates_episode(self):
        env = make_env(task_id=2, seed=42)
        env.reset()
        # Submit wrong action MAX_STEPS times — should terminate by step limit
        done = False
        steps = 0
        while not done:
            _, _, done, info = env.step(Action(classification="billing"))
            steps = info["step"]
        assert steps <= env.MAX_STEPS


# ─────────────────────────────────────────────────────────────────────────────
# Scoring sanity tests
# ─────────────────────────────────────────────────────────────────────────────

class TestScoringLogic:
    def test_task1_perfect_action_score_1(self):
        env = make_env(task_id=1, seed=42)
        env.reset()
        action = perfect_action_for_task(1)
        _, reward, done, _ = env.step(action)
        assert reward.total == 1.0
        assert done is True

    def test_task1_wrong_classification_score_0(self):
        env = make_env(task_id=1, seed=42)
        env.reset()
        action = Action(classification="billing")  # wrong for a delivery ticket
        _, reward, _, _ = env.step(action)
        assert reward.total == 0.0

    def test_task2_partial_credit_no_response(self):
        env = make_env(task_id=2, seed=42)
        obs = env.reset()
        # Find the expected category for this ticket
        from tasks import TASKS as TASK_LIST
        tasks = [t for t in TASK_LIST if t.task_id == 2]
        import random
        rng = random.Random(42)
        chosen = rng.choice(tasks)
        action = Action(classification=chosen.expected_category, response="")
        _, reward, _, _ = env.step(action)
        # classification is correct (0.40 weight) but response is empty (0.60 weight)
        assert 0.30 <= reward.total <= 0.45

    def test_task3_all_correct_high_score(self):
        env = make_env(task_id=3, seed=42)
        env.reset()
        # Derive ground-truth fields from the seeded task so the test stays deterministic
        internal_task = env._task
        assert internal_task is not None
        # Build response that includes ALL expected keywords for maximum response_score
        keyword_sentence = " ".join(internal_task.expected_keywords)
        response = (
            f"Dear customer, we sincerely apologise for this issue regarding your {keyword_sentence}. "
            f"We have carefully reviewed your request and can confirm that we will {keyword_sentence} "
            f"immediately. We understand how important this is and will take every step to ensure a "
            f"satisfactory resolution. Please allow 3-5 business days and we will keep you informed."
        )
        action = Action(
            classification=internal_task.expected_category,
            response=response,
            resolution_action=internal_task.expected_resolution,
        )
        _, reward, done, _ = env.step(action)
        assert reward.total >= 0.80, f"Expected ≥0.80 but got {reward.total} — breakdown: {reward.breakdown}"
        # done fires at reward >= 0.95 OR at max_steps; high reward is the key assertion


    def test_reward_bounds(self):
        """Reward must always stay within [0, 1]."""
        for task_id in (1, 2, 3):
            env = make_env(task_id=task_id, seed=42)
            env.reset()
            for _ in range(env.MAX_STEPS):
                if env._done:
                    break
                action = Action(
                    classification="billing",
                    response="This is a repeated filler response.",
                    resolution_action="refund",
                )
                _, reward, _, _ = env.step(action)
                assert 0.0 <= reward.total <= 1.0

    def test_repeat_penalty_applied(self):
        env = make_env(task_id=3, seed=42)
        env.reset()
        # Force 2 identical actions in Task 3 to trigger repeat penalty
        # Use wrong classification so episode doesn't end on high score
        action = Action(
            classification="delivery",
            response="We apologize for the inconvenience and will resolve this promptly.",
            resolution_action="resend",
        )
        _, reward1, done1, _ = env.step(action)
        if not done1:
            _, reward2, _, _ = env.step(action)
            # Second identical action should have a penalty
            assert reward2.penalty >= 0.10


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic model tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPydanticModels:
    def test_action_defaults(self):
        a = Action()
        assert a.classification == ""
        assert a.response == ""
        assert a.resolution_action == ""

    def test_action_model_dump(self):
        a = Action(classification="billing", response="Hello", resolution_action="refund")
        d = a.model_dump()
        assert d["classification"] == "billing"
        assert d["resolution_action"] == "refund"

    def test_reward_bounds_enforced(self):
        with pytest.raises(Exception):
            Reward(
                total=1.5,  # > 1.0  — pydantic should reject
                classification_score=1.0,
                response_score=1.0,
                resolution_score=1.0,
                penalty=0.0,
                breakdown={},
            )
