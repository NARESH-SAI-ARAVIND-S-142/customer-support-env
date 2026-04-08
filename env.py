"""
customer_support_env/env.py

OpenEnv-compliant Customer Support Ticket Resolver Environment.

This module defines the main environment class that simulates a human
customer support agent resolving incoming support tickets through
classification, response generation, and resolution planning.
"""

from __future__ import annotations

import copy
import json
import random
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from tasks import TASKS, Task
from graders import grade_action


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """Agent action submitted each step.

    Fields mirror the three-part structure expected by the graders:
      - classification: e.g. "billing", "delivery", "account_access", "damaged_product"
      - response: the human-readable reply to the customer
      - resolution_action: concrete next step (e.g. "refund", "resend", "escalate")
    """
    classification: str = Field(
        default="",
        description="Ticket category assigned by the agent"
    )
    response: str = Field(
        default="",
        description="Written response to the customer"
    )
    resolution_action: str = Field(
        default="",
        description="Concrete resolution step to take"
    )


class Observation(BaseModel):
    """What the agent perceives at each step."""
    ticket_text: str = Field(description="The full customer support ticket body")
    priority: str = Field(
        default="medium",
        description="Ticket priority: low | medium | high | critical"
    )
    customer_type: str = Field(
        default="standard",
        description="Customer tier: standard | premium | enterprise"
    )
    task_id: int = Field(description="Active task identifier (1, 2, or 3)")
    task_description: str = Field(description="Human-readable task instructions")
    step: int = Field(default=0, description="Current step index within the episode")
    max_steps: int = Field(default=5, description="Maximum allowed steps per episode")
    previous_actions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of previously submitted actions this episode"
    )


class Reward(BaseModel):
    """Structured reward breakdown after grading an action."""
    total: float = Field(ge=0.0, le=1.0, description="Aggregate reward [0, 1]")
    classification_score: float = Field(ge=0.0, le=1.0)
    response_score: float = Field(ge=0.0, le=1.0)
    resolution_score: float = Field(ge=0.0, le=1.0)
    penalty: float = Field(ge=0.0, description="Deduction for bad behaviour")
    breakdown: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed grader feedback"
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CustomerSupportEnv:
    """
    OpenEnv-compliant environment that simulates customer support ticket resolution.

    Lifecycle
    ---------
    env = CustomerSupportEnv(task_id=1)
    obs  = env.reset()
    obs, reward, done, info = env.step(action)

    task_id  1  (Easy)   – classify only
    task_id  2  (Medium) – classify + respond
    task_id  3  (Hard)   – classify + respond + resolve
    """

    MAX_STEPS: int = 5

    def __init__(self, task_id: int = 1, seed: Optional[int] = 42):
        assert task_id in (1, 2, 3), "task_id must be 1, 2, or 3"
        self.task_id = task_id
        self.seed = seed
        self._rng = random.Random(seed)

        # Episode state
        self._task: Optional[Task] = None
        self._step: int = 0
        self._done: bool = False
        self._previous_actions: List[Dict[str, Any]] = []
        self._cumulative_reward: float = 0.0
        self._episode_rewards: List[Reward] = []

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset environment, select a ticket, and return initial observation."""
        self._rng = random.Random(self.seed)          # deterministic replay
        task_pool = [t for t in TASKS if t.task_id == self.task_id]
        self._task = self._rng.choice(task_pool)

        self._step = 0
        self._done = False
        self._previous_actions = []
        self._cumulative_reward = 0.0
        self._episode_rewards = []

        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Process one agent action.

        Returns
        -------
        (observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() before stepping.")
        if self._task is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # Grade the action
        reward = grade_action(
            task=self._task,
            action=action,
            previous_actions=self._previous_actions,
        )

        # Record action for repeat-penalty logic
        self._previous_actions.append(action.model_dump())
        self._episode_rewards.append(reward)
        self._cumulative_reward += reward.total

        self._step += 1
        self._done = (self._step >= self.MAX_STEPS) or (reward.total >= 0.95)

        obs = self._build_observation()

        info: Dict[str, Any] = {
            "step": self._step,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "episode_done": self._done,
            "reward_breakdown": reward.model_dump(),
        }

        return obs, reward, self._done, info

    def state(self) -> Dict[str, Any]:
        """Return full internal state snapshot (useful for debugging / logging)."""
        return {
            "task_id": self.task_id,
            "step": self._step,
            "done": self._done,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "ticket": self._task.ticket_text if self._task else None,
            "expected_category": self._task.expected_category if self._task else None,
            "previous_actions": copy.deepcopy(self._previous_actions),
            "episode_rewards": [r.model_dump() for r in self._episode_rewards],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        assert self._task is not None
        task_descriptions = {
            1: (
                "Task 1 (Easy): Classify the support ticket into one of the following "
                "categories: delivery, billing, damaged_product, account_access. "
                "Set only the 'classification' field in your action. "
                "Leave 'response' and 'resolution_action' empty."
            ),
            2: (
                "Task 2 (Medium): Classify the ticket AND write a professional "
                "customer-facing response. Set 'classification' and 'response'. "
                "Leave 'resolution_action' empty."
            ),
            3: (
                "Task 3 (Hard): Classify the ticket, write a professional response, "
                "AND specify the resolution action. Valid resolution actions: "
                "refund, resend, escalate, reset_password, account_unlock, "
                "discount_applied, schedule_pickup, investigation_opened. "
                "Fill in all three fields."
            ),
        }
        return Observation(
            ticket_text=self._task.ticket_text,
            priority=self._task.priority,
            customer_type=self._task.customer_type,
            task_id=self.task_id,
            task_description=task_descriptions[self.task_id],
            step=self._step,
            max_steps=self.MAX_STEPS,
            previous_actions=copy.deepcopy(self._previous_actions),
        )


# ---------------------------------------------------------------------------
# Convenience factory (matches OpenEnv loader convention)
# ---------------------------------------------------------------------------

def make_env(task_id: int = 1, seed: int = 42) -> CustomerSupportEnv:
    """Factory used by the OpenEnv runner and inference script."""
    return CustomerSupportEnv(task_id=task_id, seed=seed)
