"""
server/app.py

FastAPI application for the Customer Support Ticket Resolver.
This is the OpenEnv-required server entry point.

Endpoints:
  GET  /          – root info
  GET  /health    – liveness probe
  POST /reset     – reset environment, returns Observation
  POST /step      – take one action, returns (obs, reward, done, info)
  GET  /state     – return internal state snapshot

Entry point (pyproject.toml):
  serve = "server.app:main"

Uvicorn:
  uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

# Ensure the repo root is on the path so env/tasks/graders are importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from env import Action, CustomerSupportEnv, Observation, Reward

# ---------------------------------------------------------------------------
# App
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

_env: Optional[CustomerSupportEnv] = None


def _get_env() -> CustomerSupportEnv:
    global _env
    if _env is None:
        _env = CustomerSupportEnv(task_id=1, seed=42)
        _env.reset()
    return _env


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> Dict[str, Any]:
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
    Reset the environment. Accepts optional JSON body:
      {"task_id": 1, "seed": 42}
    Works with no body too.
    """
    global _env

    try:
        body = await request.json()
    except Exception:
        body = {}

    if not isinstance(body, dict):
        body = {}

    task_id = int(body.get("task_id", 1))
    seed = int(body.get("seed", 42))
    if task_id not in (1, 2, 3):
        task_id = 1

    _env = CustomerSupportEnv(task_id=task_id, seed=seed)
    obs: Observation = _env.reset()
    return obs.model_dump()


@app.post("/step")
async def step(request: Request) -> Dict[str, Any]:
    """Submit one action and receive next observation, reward, done and info."""
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
    """Return full internal state for debugging."""
    return _get_env().state()


# ---------------------------------------------------------------------------
# Entry point for `serve` script (pyproject.toml [project.scripts])
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
