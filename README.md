---
title: Customer Support Ticket Resolver
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# Customer Support Ticket Resolver 🎫


[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-1.0-blue)](https://openenv.ai)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-green)](https://python.org)
[![HuggingFace Spaces](https://img.shields.io/badge/HF-Spaces-yellow)](https://huggingface.co/spaces)

---

## 🌍 Environment Description

The **Customer Support Ticket Resolver** is an OpenEnv-compliant reinforcement-learning environment that simulates a human customer-support agent handling incoming support tickets.

At each episode the agent receives a realistic customer ticket and must:

1. **Classify** it into the correct category
2. **Respond** professionally and empathetically
3. **Select** the correct back-office resolution action

This environment is designed to train, evaluate, and benchmark language models on real-world customer-support workflows.

---

## 🎯 Real-World Motivation

Customer support is one of the highest-volume enterprise AI applications. A model that can:

* Accurately classify customer intent
* Generate policy-compliant responses
* Initiate the correct resolution (refund, resend, escalate, …)

can dramatically reduce resolution times, improve CSAT scores, and free human agents for complex escalations.

This environment provides a **controlled, reproducible benchmark** for that capability.

---

## 📋 Ticket Categories

| Category | Description |
|---|---|
| `delivery` | Lost, delayed, or misdelivered packages |
| `billing` | Duplicate charges, unexpected fees, refund requests |
| `damaged_product` | Items arrived broken, defective, or wrong |
| `account_access` | Locked accounts, forgotten passwords, compromised credentials |

---

## 🏆 Tasks

### Task 1 – Easy: Ticket Classification

| Property | Value |
|---|---|
| Input | Ticket text |
| Output | `classification` only |
| Scoring | 1.0 for correct category, 0.0 otherwise |
| Max steps | 5 |

The agent only needs to pick the right category label. No response writing required.

---

### Task 2 – Medium: Classify + Respond

| Property | Value |
|---|---|
| Input | Ticket text |
| Output | `classification` + `response` |
| Scoring | classification ×0.40, response quality ×0.60 |
| Max steps | 5 |

The agent must classify correctly AND write a professional customer-facing reply that:
- Acknowledges the issue empathetically
- Contains relevant keywords (apology, issue reference, next steps)
- Is at least 50 words

---

### Task 3 – Hard: Full Resolution

| Property | Value |
|---|---|
| Input | Ticket text |
|Output | `classification` + `response` + `resolution_action` |
| Scoring | classification ×0.30, response ×0.30, resolution ×0.40 |
| Max steps | 5 |

The agent must do everything in Task 2 AND choose the correct resolution action from:

```
refund | resend | escalate | reset_password | account_unlock |
discount_applied | schedule_pickup | investigation_opened
```

---

## 📐 Action Schema

```json
{
  "classification":    "billing",
  "response":          "Dear customer, we sincerely apologise for the duplicate charge...",
  "resolution_action": "refund"
}
```

| Field | Type | Required |
|---|---|---|
| `classification` | enum string | Always |
| `response` | string (≥50 words) | Tasks 2 & 3 |
| `resolution_action` | enum string | Task 3 only |

---

## 👁 Observation Schema

```json
{
  "ticket_text":       "I was charged twice for my subscription…",
  "priority":          "high",
  "customer_type":     "premium",
  "task_id":           3,
  "task_description":  "Task 3 (Hard): ...",
  "step":              0,
  "max_steps":         5,
  "previous_actions":  []
}
```

---

## 🎁 Reward Design

### Weights per task

| Component | Task 1 | Task 2 | Task 3 |
|---|--------|--------|--------|
| Classification | 1.00 | 0.40 | 0.30 |
| Response | 0.00 | 0.60 | 0.30 |
| Resolution | 0.00 | 0.00 | 0.40 |

### Response scoring

Response quality is evaluated as:
```
response_score = 0.40 × length_score + 0.60 × keyword_hit_rate
```

Where `keyword_hit_rate` = fraction of expected keywords found in the response.

### Penalties

| Trigger | Deduction |
|---|---|
| Exact duplicate action | −0.10 (max −0.30) |
| Invalid category name | −0.05 |
| Invalid resolution action (Task 3) | −0.05 |

### Expected Baseline Scores

Scores when running a capable LLM (e.g. `gpt-4o-mini`) with `temperature=0`:

| Task | Expected Final Reward |
|---|---|
| Task 1 | 0.90 – 1.00 |
| Task 2 | 0.70 – 0.90 |
| Task 3 | 0.60 – 0.85 |

---

## 🗂 Project Structure

```
customer_support_env/
├── env.py            # Core OpenEnv environment class + Pydantic models
├── tasks.py          # 15-ticket deterministic task catalogue
├── graders.py        # Deterministic graders for all 3 tasks
├── server.py         # FastAPI REST server (/reset /step /state)
├── inference.py      # Baseline inference script (OpenAI client)
├── openenv.yaml      # OpenEnv manifest (passes openenv validate)
├── requirements.txt  # Python dependencies
├── Dockerfile        # Production-ready container
├── tests/
│   └── test_env.py   # Unit tests (no API key required)
└── README.md
```

---

## ⚙ Setup

### Prerequisites

- Python 3.11+
- An OpenAI-compatible API key

### Install

```bash
git clone <your-repo>
cd customer_support_env
pip install -r requirements.txt
```

---

## 🚀 How to Run Locally

### 1. Run baseline inference

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export OPENAI_API_KEY=sk-...

python inference.py
```

Expected output:

```
======================================================================
  Customer Support Ticket Resolver – Baseline Inference
  API_BASE_URL : https://api.openai.com/v1
  MODEL_NAME   : gpt-4o-mini
  SEED         : 42
======================================================================

[START] task=1  ticket_preview='Hi, I placed an order 10 days ago ...'
        priority=medium  customer_type=standard
[STEP]  task=1 step=1  cls='delivery'   total_reward=1.000  ...  done=True
[END]   task=1  steps=1  cumulative_reward=1.0000  expected_category=delivery
...
```

### 2. Start the REST API server

```bash
python server.py
# or via uvicorn:
uvicorn server:app --host 0.0.0.0 --port 7860 --reload
```

Then use the API:

```bash
# Reset for Task 3
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 3, "seed": 42}'

# Submit an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "classification": "billing",
    "response": "We sincerely apologise for the erroneous charge...",
    "resolution_action": "refund"
  }'

# View state
curl http://localhost:7860/state
```

### 3. Run unit tests

```bash
python -m pytest tests/ -v
```

---

## 🐳 Docker

### Build

```bash
docker build -t cs-resolver .
```

### Run the REST API

```bash
docker run -p 7860:7860 \
  -e MODEL_NAME=gpt-4o-mini \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e OPENAI_API_KEY=sk-... \
  cs-resolver
```

### Run inference directly

```bash
docker run --rm \
  -e MODEL_NAME=gpt-4o-mini \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e OPENAI_API_KEY=sk-... \
  cs-resolver python inference.py
```

---

## ☁ Hugging Face Spaces Deployment

1. Create a new HF Space (Docker SDK)
2. Push the entire `customer_support_env/` directory as the Space root
3. Add Secrets:
   - `OPENAI_API_KEY` (or `HF_TOKEN` for proxy endpoints)
   - `API_BASE_URL`
   - `MODEL_NAME`
4. The Space will start the FastAPI server on port 7860 automatically

---

## 🔍 OpenEnv Validation

```bash
openenv validate openenv.yaml
```

The manifest declares all required fields: environment name, tasks, action space, observation space, reward structure, and runtime constraints.

---

## 📊 Resource Constraints

| Resource | Limit |
|---|---|
| vCPU | 2 |
| RAM | 8 GB |
| Runtime | < 20 minutes |
| External deps | None (no heavy ML) |

---

## 📝 License

MIT License. See LICENSE for details.
