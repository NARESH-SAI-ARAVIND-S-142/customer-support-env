"""
customer_support_env/server.py

FastAPI server exposing the OpenEnv-compatible REST API.

Endpoints
---------
  POST /reset         – reset environment, returns Observation
  POST /step          – take one action, returns (obs, reward, done, info)
  GET  /state         – return internal state snapshot
  GET  /health        – liveness probe

Hugging Face Spaces entry point:
  uvicorn server:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# One environment instance per server process (suitable for single-user Spaces)
_env: Optional[CustomerSupportEnv] = None


def _get_env() -> CustomerSupportEnv:
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
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

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "customer-support-env"}


@app.post("/reset", response_model=Dict[str, Any])
def reset(req: ResetRequest) -> Dict[str, Any]:
    """Initialise (or re-initialise) the environment and return the first observation."""
    global _env
    _env = CustomerSupportEnv(task_id=req.task_id, seed=req.seed)
    obs: Observation = _env.reset()
    return obs.model_dump()


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    """Submit one action and receive the next observation, reward, done flag and info."""
    env = _get_env()
    action = Action(
        classification=req.classification,
        response=req.response,
        resolution_action=req.resolution_action,
    )
    obs, reward, done, info = env.step(action)
    return StepResponse(
        observation=obs.model_dump(),
        reward=reward.model_dump(),
        done=done,
        info=info,
    )


@app.get("/state", response_model=Dict[str, Any])
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
