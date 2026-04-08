"""
customer_support_env/tasks.py

Defines the Task dataclass and the full deterministic task catalogue.

Each task bundles a realistic customer support ticket with:
  - expected_category    – ground-truth classification label
  - expected_keywords    – words a good response MUST include
  - expected_resolution  – correct resolution action for Task 3
  - task_id              – 1 (easy) | 2 (medium) | 3 (hard)
  - priority / customer_type – metadata passed to the agent via Observation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

@dataclass
class Task:
    task_id: int
    ticket_text: str
    expected_category: str
    expected_keywords: List[str]    # required words in a quality response
    expected_resolution: str        # expected resolution_action (Task 3)
    priority: str = "medium"
    customer_type: str = "standard"
    ticket_id: str = ""             # human-readable ID for logging


# ---------------------------------------------------------------------------
# Task catalogue
# 15 tickets across 3 task levels (5 per level) covering 4 categories
# ---------------------------------------------------------------------------

TASKS: List[Task] = [

    # -----------------------------------------------------------------------
    # TASK 1 – EASY: Classification only
    # -----------------------------------------------------------------------

    Task(
        task_id=1,
        ticket_id="T1-001",
        priority="medium",
        customer_type="standard",
        expected_category="delivery",
        expected_keywords=["deliver", "order", "ship"],
        expected_resolution="resend",
        ticket_text=(
            "Hi, I placed an order 10 days ago (order #ORD-29834) and it still hasn't "
            "arrived. The tracking page just says 'In Transit' and hasn't updated in "
            "5 days. I need my package urgently. Please help!"
        ),
    ),

    Task(
        task_id=1,
        ticket_id="T1-002",
        priority="high",
        customer_type="premium",
        expected_category="billing",
        expected_keywords=["charge", "invoice", "payment"],
        expected_resolution="refund",
        ticket_text=(
            "I was charged twice for my subscription this month. My bank statement "
            "shows two separate charges of $49.99 on the 1st and 3rd. I only have "
            "one account. Please refund the duplicate charge immediately."
        ),
    ),

    Task(
        task_id=1,
        ticket_id="T1-003",
        priority="low",
        customer_type="standard",
        expected_category="damaged_product",
        expected_keywords=["damaged", "broken", "replacement"],
        expected_resolution="schedule_pickup",
        ticket_text=(
            "The laptop stand I ordered arrived completely shattered. The box looked "
            "fine on the outside but inside everything was broken into pieces. "
            "I have photos. What do I do?"
        ),
    ),

    Task(
        task_id=1,
        ticket_id="T1-004",
        priority="critical",
        customer_type="enterprise",
        expected_category="account_access",
        expected_keywords=["password", "login", "access"],
        expected_resolution="reset_password",
        ticket_text=(
            "We cannot log in to our enterprise dashboard. Our entire team has been "
            "locked out since this morning. We have a live product demo in two hours. "
            "This is critical — please fix ASAP!"
        ),
    ),

    Task(
        task_id=1,
        ticket_id="T1-005",
        priority="medium",
        customer_type="standard",
        expected_category="billing",
        expected_keywords=["charge", "refund", "cancel"],
        expected_resolution="refund",
        ticket_text=(
            "I cancelled my subscription last week but I was still charged for next "
            "month. The cancellation confirmation email says effective immediately but "
            "you still took the money. I want a full refund."
        ),
    ),

    # -----------------------------------------------------------------------
    # TASK 2 – MEDIUM: Classification + Response
    # -----------------------------------------------------------------------

    Task(
        task_id=2,
        ticket_id="T2-001",
        priority="high",
        customer_type="premium",
        expected_category="delivery",
        expected_keywords=["apologize", "order", "investigation", "update"],
        expected_resolution="investigation_opened",
        ticket_text=(
            "My order #ORD-55621 was marked as delivered yesterday but I never received "
            "it. I was home all day. The courier says it was left at the door but "
            "there's nothing here. My neighbor didn't receive it either."
        ),
    ),

    Task(
        task_id=2,
        ticket_id="T2-002",
        priority="medium",
        customer_type="standard",
        expected_category="billing",
        expected_keywords=["apologize", "refund", "charge", "review"],
        expected_resolution="refund",
        ticket_text=(
            "I see an unrecognized charge of $19.99 labelled 'PREMIUM UPGRADE' on my "
            "statement. I never agreed to any upgrade. I am on the free plan. "
            "Please explain and reverse this charge."
        ),
    ),

    Task(
        task_id=2,
        ticket_id="T2-003",
        priority="high",
        customer_type="standard",
        expected_category="damaged_product",
        expected_keywords=["apologize", "damaged", "replacement", "photo"],
        expected_resolution="schedule_pickup",
        ticket_text=(
            "The coffee machine I ordered for my office arrived with a cracked water "
            "tank. It's completely unusable. This was an expensive purchase and I'm "
            "really disappointed. What is the fastest way to get a replacement?"
        ),
    ),

    Task(
        task_id=2,
        ticket_id="T2-004",
        priority="critical",
        customer_type="enterprise",
        expected_category="account_access",
        expected_keywords=["apologize", "access", "reset", "team", "urgency"],
        expected_resolution="account_unlock",
        ticket_text=(
            "Our entire company account has been suspended without warning. We have 50 "
            "employees who cannot work. We have not violated any terms of service. "
            "This is causing massive revenue loss. Escalate immediately."
        ),
    ),

    Task(
        task_id=2,
        ticket_id="T2-005",
        priority="low",
        customer_type="standard",
        expected_category="delivery",
        expected_keywords=["apologize", "delay", "ship", "track"],
        expected_resolution="resend",
        ticket_text=(
            "My order was supposed to arrive within 3 business days. It's now been "
            "9 days and I still haven't received a tracking number. Can you please "
            "tell me where my order is?"
        ),
    ),

    # -----------------------------------------------------------------------
    # TASK 3 – HARD: Classification + Response + Resolution Action
    # -----------------------------------------------------------------------

    Task(
        task_id=3,
        ticket_id="T3-001",
        priority="high",
        customer_type="premium",
        expected_category="billing",
        expected_keywords=["apologize", "refund", "charge", "confirm", "investigate"],
        expected_resolution="refund",
        ticket_text=(
            "Hello, I have been billed $299 for an annual plan I downgraded from "
            "two months ago. I have the downgrade confirmation email dated Feb 15. "
            "I am currently on the Basic monthly plan at $9.99. This looks like "
            "a billing system error. I want the $299 refunded to my card."
        ),
    ),

    Task(
        task_id=3,
        ticket_id="T3-002",
        priority="critical",
        customer_type="enterprise",
        expected_category="account_access",
        expected_keywords=["apologize", "unlock", "security", "team", "restore"],
        expected_resolution="account_unlock",
        ticket_text=(
            "Our admin account appears to have been compromised. We noticed "
            "unauthorized logins from an unknown IP at 3 AM. We've already changed "
            "the password but we suspect there are still active sessions. "
            "We need all sessions terminated and our account secured now."
        ),
    ),

    Task(
        task_id=3,
        ticket_id="T3-003",
        priority="medium",
        customer_type="standard",
        expected_category="damaged_product",
        expected_keywords=["apologize", "damaged", "replacement", "pickup", "sorry"],
        expected_resolution="schedule_pickup",
        ticket_text=(
            "The bluetooth headphones I ordered arrived with one ear cup completely "
            "detached from the headband. The packaging was also torn open — it "
            "seems like someone returned them and they were resold to me. I don't "
            "want a repair, I want a brand new sealed unit."
        ),
    ),

    Task(
        task_id=3,
        ticket_id="T3-004",
        priority="high",
        customer_type="premium",
        expected_category="delivery",
        expected_keywords=["apologize", "resend", "order", "ship", "replacement"],
        expected_resolution="resend",
        ticket_text=(
            "My package shows delivered to a completely different address than mine. "
            "The tracking shows 'Delivered to neighbor' but my neighbors say they "
            "never received anything. This is my third order this year with a "
            "delivery problem. I need the items reshipped immediately."
        ),
    ),

    Task(
        task_id=3,
        ticket_id="T3-005",
        priority="low",
        customer_type="standard",
        expected_category="account_access",
        expected_keywords=["apologize", "password", "reset", "secure", "help"],
        expected_resolution="reset_password",
        ticket_text=(
            "I have forgotten my password and the reset email is not arriving. "
            "I've checked my spam folder. I tried three times in the last hour. "
            "I also tried the alternate email option but that address no longer "
            "exists. Please help me regain access to my account."
        ),
    ),
]
