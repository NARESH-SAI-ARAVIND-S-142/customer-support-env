"""
customer_support_env/server.py

FastAPI server exposing the OpenEnv-compatible REST API.

Endpoints
---------
  POST /reset         – reset environment, returns Observation
  POST /step          – take one action, returns (obs, reward, done, info)
  GET  /state         – return internal state snapshot
  GET  /health        – liveness probe
  GET  /              – root info page

Hugging Face Spaces entry point:
  uvicorn server:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env import Action, CustomerSupportEnv, Observation, Reward

# ---------------------------------------------------------------------------
# App singleton
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Customer Support Ticket Resolver",
    description=(
        "OpenEnv-compliant REST API that simulates a customer support agent "
        "resolving incoming tickets across three difficulty levels."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One environment instance per server process
_env: Optional[CustomerSupportEnv] = None


def _get_env() -> CustomerSupportEnv:
    global _env
    if _env is None:
        # Auto-initialise with defaults if not yet reset
        _env = CustomerSupportEnv(task_id=1, seed=42)
        _env.reset()
    return _env


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: int = 1
    seed: int = 42


class StepRequest(BaseModel):
    classification: str = ""
    response: str = ""
    resolution_action: str = ""


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> Dict[str, Any]:
    """Root endpoint — basic info."""
    return {
        "name": "Customer Support Ticket Resolver",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/health", "/docs"],
        "status": "running",
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "customer-support-env"}


@app.post("/reset")
async def reset(request: Request) -> Dict[str, Any]:
    """
    Initialise (or re-initialise) the environment and return the first observation.
    Accepts optional JSON body: {"task_id": 1, "seed": 42}
    Works with empty body too.
    """
    global _env

    # Safely parse body — works with empty body, missing fields, or full body
    try:
        body = await request.json()
    except Exception:
        body = {}

    task_id = int(body.get("task_id", 1)) if isinstance(body, dict) else 1
    seed = int(body.get("seed", 42)) if isinstance(body, dict) else 42

    # Clamp task_id to valid range
    if task_id not in (1, 2, 3):
        task_id = 1

    _env = CustomerSupportEnv(task_id=task_id, seed=seed)
    obs: Observation = _env.reset()
    return obs.model_dump()


@app.post("/step")
async def step(request: Request) -> Dict[str, Any]:
    """Submit one action and receive the next observation, reward, done flag and info."""
    env = _get_env()

    try:
        body = await request.json()
    except Exception:
        body = {}

    if not isinstance(body, dict):
        body = {}

    action = Action(
        classification=str(body.get("classification", "")),
        response=str(body.get("response", "")),
        resolution_action=str(body.get("resolution_action", "")),
    )

    obs, reward, done, info = env.step(action)

    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> Dict[str, Any]:
    """Return a full internal state snapshot for debugging."""
    return _get_env().state()


# ---------------------------------------------------------------------------
# Dev-server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
