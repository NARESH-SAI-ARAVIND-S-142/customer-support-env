#!/usr/bin/env python3
"""
inference.py  –  Baseline inference script for the Customer Support Ticket Resolver.

Usage
-----
  export API_BASE_URL=https://api.openai.com/v1
  export MODEL_NAME=gpt-4o-mini
  export HF_TOKEN=hf_...                   # optional, for gated models
  python inference.py

Environment variables
---------------------
  API_BASE_URL  : OpenAI-compatible base URL (default: https://api.openai.com/v1)
  MODEL_NAME    : Model identifier           (default: gpt-4o-mini)
  HF_TOKEN      : HuggingFace token for proxy auth (optional)

Logging format
--------------
  [START]
  [STEP]  task=X step=Y  ...
  [END]
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict

# Add project root to path so env/tasks/graders import correctly
sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI

from env import Action, CustomerSupportEnv, make_env

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str | None = os.environ.get("HF_TOKEN")

SEED: int = 42
MAX_STEPS_PER_EPISODE: int = 5

# ---------------------------------------------------------------------------
# OpenAI client factory
# ---------------------------------------------------------------------------

def _build_client() -> OpenAI:
    api_key = HF_TOKEN or os.environ.get("OPENAI_API_KEY", "sk-placeholder")
    return OpenAI(base_url=API_BASE_URL, api_key=api_key)


# ---------------------------------------------------------------------------
# System prompts per task
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: Dict[int, str] = {
    1: """You are a customer support ticket classifier.

Given a support ticket, you must output ONLY a JSON object with this exact schema:
{
  "classification": "<category>",
  "response": "",
  "resolution_action": ""
}

Valid categories: delivery | billing | damaged_product | account_access

Rules:
- Output ONLY the JSON. No explanation, no markdown.
- Pick exactly ONE category that best describes the ticket.
""",

    2: """You are an expert customer support agent.

Given a support ticket, you must output ONLY a JSON object with this exact schema:
{
  "classification": "<category>",
  "response": "<your professional response to the customer>",
  "resolution_action": ""
}

Valid categories: delivery | billing | damaged_product | account_access

Rules:
- Output ONLY the JSON. No explanation, no markdown.
- The response must be professional, empathetic, and at least 50 words.
- Include an apology, acknowledgement of the issue, and next steps.
""",

    3: """You are a senior customer support agent with resolution authority.

Given a support ticket, you must output ONLY a JSON object with this exact schema:
{
  "classification": "<category>",
  "response": "<your professional response to the customer>",
  "resolution_action": "<action>"
}

Valid categories: delivery | billing | damaged_product | account_access

Valid resolution actions:
  refund | resend | escalate | reset_password | account_unlock |
  discount_applied | schedule_pickup | investigation_opened

Rules:
- Output ONLY the JSON. No explanation, no markdown.
- The response must be professional, empathetic, and at least 60 words.
- Include an apology, acknowledgement, next steps, and the resolution you've initiated.
- Choose the SINGLE most appropriate resolution action.
""",
}


# ---------------------------------------------------------------------------
# Single-step agent call
# ---------------------------------------------------------------------------

def call_agent(client: OpenAI, task_id: int, ticket_text: str, step: int) -> Action:
    """Call the LLM and parse the JSON action."""
    user_message = f"Support Ticket:\n\n{ticket_text}"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[task_id]},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        raw = response.choices[0].message.content or ""
    except Exception as exc:
        print(f"  [WARN] API error on step {step}: {exc}")
        return Action()

    # Strip potential markdown fences
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if len(lines) > 2 else ""

    try:
        data: Dict[str, Any] = json.loads(raw)
        return Action(
            classification=str(data.get("classification", "")).strip(),
            response=str(data.get("response", "")).strip(),
            resolution_action=str(data.get("resolution_action", "")).strip(),
        )
    except json.JSONDecodeError:
        print(f"  [WARN] Could not parse JSON from model output: {raw[:120]!r}")
        return Action()


# ---------------------------------------------------------------------------
# Run one full task episode
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_id: int) -> Dict[str, Any]:
    """
    Run one episode for the given task_id.

    Returns a summary dict with scores and per-step info.
    """
    env: CustomerSupportEnv = make_env(task_id=task_id, seed=SEED)
    obs = env.reset()

    print(f"\n[START] task={task_id}  ticket_preview={obs.ticket_text[:60]!r}...")
    print(f"        priority={obs.priority}  customer_type={obs.customer_type}")

    steps_log = []
    cumulative_reward = 0.0
    done = False
    step = 0

    while not done and step < MAX_STEPS_PER_EPISODE:
        step += 1
        action = call_agent(client, task_id, obs.ticket_text, step)

        obs, reward, done, info = env.step(action)
        cumulative_reward = info["cumulative_reward"]

        print(
            f"[STEP]  task={task_id} step={step}  "
            f"cls={action.classification!r:20s}  "
            f"total_reward={reward.total:.3f}  "
            f"cls_score={reward.classification_score:.2f}  "
            f"resp_score={reward.response_score:.2f}  "
            f"res_score={reward.resolution_score:.2f}  "
            f"penalty={reward.penalty:.2f}  "
            f"done={done}"
        )

        steps_log.append({
            "step": step,
            "classification": action.classification,
            "resolution_action": action.resolution_action,
            "reward": reward.total,
            "breakdown": reward.breakdown,
        })

        if done:
            break

        # Brief pause to avoid rate-limit bursts
        time.sleep(0.5)

    final_state = env.state()
    print(
        f"[END]   task={task_id}  steps={step}  "
        f"cumulative_reward={cumulative_reward:.4f}  "
        f"expected_category={final_state['expected_category']}"
    )

    return {
        "task_id": task_id,
        "steps": step,
        "cumulative_reward": cumulative_reward,
        "final_reward": steps_log[-1]["reward"] if steps_log else 0.0,
        "expected_category": final_state["expected_category"],
        "step_log": steps_log,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("  Customer Support Ticket Resolver – Baseline Inference")
    print(f"  API_BASE_URL : {API_BASE_URL}")
    print(f"  MODEL_NAME   : {MODEL_NAME}")
    print(f"  SEED         : {SEED}")
    print("=" * 70)

    client = _build_client()
    results = []

    for task_id in (1, 2, 3):
        result = run_task(client, task_id)
        results.append(result)

    # Summary table
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"  {'Task':<8} {'Steps':<8} {'Final Reward':<16} {'Cumulative':<14}")
    print("  " + "-" * 50)
    for r in results:
        print(
            f"  Task {r['task_id']:<4} "
            f"{r['steps']:<8} "
            f"{r['final_reward']:<16.4f} "
            f"{r['cumulative_reward']:.4f}"
        )

    avg_final = sum(r["final_reward"] for r in results) / len(results)
    avg_cumulative = sum(r["cumulative_reward"] for r in results) / len(results)
    print("  " + "-" * 50)
    print(f"  {'Average':<8} {'':8} {avg_final:<16.4f} {avg_cumulative:.4f}")
    print("=" * 70)

    # Persist results as JSON
    output_path = os.path.join(os.path.dirname(__file__), "inference_results.json")
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "model": MODEL_NAME,
                "api_base": API_BASE_URL,
                "seed": SEED,
                "results": results,
                "avg_final_reward": round(avg_final, 4),
                "avg_cumulative_reward": round(avg_cumulative, 4),
            },
            fh,
            indent=2,
        )
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
