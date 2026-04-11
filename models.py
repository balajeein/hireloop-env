"""
HireLoop Models
===============
Domain models + openenv-core compliant Action/Observation types.

HireLoopAction and HireLoopObservation are pure Pydantic BaseModel classes.
HireLoopState is used internally by the environment orchestrator.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# ── OpenEnv-core base class shim ──────────────────────────────────
# When openenv-core is available (Python 3.10+), we note it.
# But our Action/Observation are pure Pydantic — no dataclass inheritance.
try:
    from openenv_core.env_server.types import State as BaseState
    _OPENENV_AVAILABLE = True
except (ImportError, TypeError):
    _OPENENV_AVAILABLE = False

    class BaseState:
        """Shim for openenv_core.env_server.types.State"""
        def __init__(self, episode_id=None, step_count=0):
            self.episode_id = episode_id
            self.step_count = step_count


# ── Domain models (preserved exactly) ─────────────────────────────

class Candidate(BaseModel):
    id: str
    name: str
    skills: List[str]
    years_experience: int
    expected_salary: int  # Salary in USD (e.g., 70000 = $70,000/year)
    gender: Optional[str] = None
    nationality: Optional[str] = None


class JobDescription(BaseModel):
    role: str
    required_skills: List[str]
    max_salary: int  # Maximum salary in USD
    seniority: str  # "junior" / "mid" / "senior"


# ── OpenEnv-compliant Action ──────────────────────────────────────

class HireLoopAction(BaseModel):
    type: str = ""
    candidate_id: str = ""
    content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ── OpenEnv-compliant Observation ─────────────────────────────────

class HireLoopObservation(BaseModel):
    done: bool = False
    reward: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    job_description: Optional[Dict] = None
    candidates: Optional[List[Dict]] = None
    shortlisted: Optional[List[str]] = None
    rejected: Optional[List[str]] = None
    step_count: int = 0
    task_type: str = "resume"
    budget: Optional[int] = None
    offers_made: Optional[List[Dict]] = None
    emails_sent: Optional[List[Dict]] = None
    counterfactual: Optional[Dict] = None
    negotiation_hints: Optional[Dict] = None


# ── Internal state model (used inside hireloop/env.py) ─────────────
# This is a Pydantic model for internal env logic — NOT returned to API

class HireLoopState(BaseModel):
    job_description: JobDescription
    candidates: List[Candidate]
    shortlisted: List[str]
    rejected: List[str]
    step_count: int
    task_type: str = "resume"
    budget: Optional[int] = None
    offers_made: Optional[List[Dict]] = None
    emails_sent: Optional[List[Dict]] = None
    counterfactual: Optional[Dict] = None


# ── Session request model (kept for api.py backward compat) ───────

class StepRequest(BaseModel):
    """Request body for POST /step with session isolation."""
    session_id: str
    action: Dict[str, Any]