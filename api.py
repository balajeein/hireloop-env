from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from fastapi import FastAPI, Query
from typing import Dict, Optional

from hireloop.env import HireLoopEnv
from hireloop.session import (
    create_session, get_session, delete_session, get_or_create_legacy_session
)
from hireloop.tasks.offer import check_negotiation_eligibility
from models import Action, StepRequest

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "HireLoop Environment API is running"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "env": "hireloop",
        "version": "1.0.0",
        "tasks": ["resume", "offer", "communication"]
    }

@app.get("/ui", response_class=HTMLResponse)
def web_interface():
    html_path = Path(__file__).parent / "web_interface.html"
    if html_path.exists():
        return html_path.read_text()
    return "<h1>web_interface.html not found</h1>"


# -----------------------------------------------------------------------
# RESET — POST creates a new session; GET is legacy backward compat
# POST /reset          → new session, random task
# POST /reset?task=offer → new session, specific task
# GET  /reset          → legacy mode (default session slot)
# -----------------------------------------------------------------------
@app.post("/reset")
def reset_post(task: Optional[str] = Query(None, description="Task type: resume, offer, or communication")):
    session_id, env = create_session()
    if task and task in ("resume", "offer", "communication"):
        state = env.reset_with_task(task)
    else:
        state = env.reset()
    return {"session_id": session_id, "state": state}


@app.get("/reset")
def reset_get(task: Optional[str] = Query(None, description="Task type: resume, offer, or communication")):
    """Legacy backward-compatible reset — uses a default session slot."""
    _, env = get_or_create_legacy_session()
    if task and task in ("resume", "offer", "communication"):
        state = env.reset_with_task(task)
    else:
        state = env.reset()
    return {"state": state}


# -----------------------------------------------------------------------
# STEP — accepts action + session_id, routes based on current task_type
# Supports both session-based and legacy (no session_id) requests
# -----------------------------------------------------------------------
@app.post("/step")
def step(body: dict):
    # Support both StepRequest (with session_id) and plain Action
    session_id = body.get("session_id")

    if session_id:
        env = get_session(session_id)
        if env is None:
            return {"error": f"Session {session_id} not found. Call POST /reset first."}
        action_data = body.get("action", body)
    else:
        # Legacy mode: no session_id, treat entire body as action
        _, env = get_or_create_legacy_session()
        action_data = body

    obs, reward, done, info = env.step(action_data)

    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info,
    }


# -----------------------------------------------------------------------
# STATE — supports session_id query param
# -----------------------------------------------------------------------
@app.get("/state")
def state(session_id: Optional[str] = Query(None)):
    if session_id:
        env = get_session(session_id)
        if env is None:
            return {"error": f"Session {session_id} not found."}
    else:
        _, env = get_or_create_legacy_session()
    return {"state": env.state_view()}


# -----------------------------------------------------------------------
# GRADER — supports session_id query param
# -----------------------------------------------------------------------
@app.get("/grader")
def grader(session_id: Optional[str] = Query(None)):
    if session_id:
        env = get_session(session_id)
        if env is None:
            return {"error": f"Session {session_id} not found."}
    else:
        _, env = get_or_create_legacy_session()
    score = env.compute_final_score()
    task_type = env.state.task_type if env.state else "unknown"
    return {
        "task_type": task_type,
        "score": score,
    }


# -----------------------------------------------------------------------
# Shared heuristic logic — used by /baseline and /eval to avoid duplication
# -----------------------------------------------------------------------
def _run_heuristic_task(env: HireLoopEnv, task_type: str) -> dict:
    """
    Run a heuristic agent on one task and return results.
    Used by both /baseline and /eval endpoints.
    """
    state = env.reset_with_task(task_type)
    total_reward = 0.0
    steps_taken = 0

    # ------------------ RESUME ------------------
    if task_type == "resume":
        job_skills = set(state.job_description.required_skills)
        for candidate in state.candidates:
            candidate_skills = set(candidate.skills)
            if len(candidate_skills & job_skills) >= 1:
                action = {"type": "accept", "candidate_id": candidate.id}
            else:
                action = {"type": "reject", "candidate_id": candidate.id}
            _, reward, done, _ = env.step(action)
            total_reward += reward
            steps_taken += 1
            if done:
                break

    # ------------------ OFFER ------------------
    elif task_type == "offer":
        sorted_candidates = sorted(
            state.candidates,
            key=lambda c: c.expected_salary
        )
        for candidate in sorted_candidates:
            eligibility = check_negotiation_eligibility(
                candidate.skills,
                list(state.job_description.required_skills)
            )
            if eligibility["negotiable"]:
                action = {"type": "negotiate", "candidate_id": candidate.id}
            elif eligibility["eligible"]:
                action = {"type": "offer", "candidate_id": candidate.id}
            else:
                continue
            _, reward, done, _ = env.step(action)
            total_reward += reward
            steps_taken += 1
            if done:
                break

    # ------------------ COMMUNICATION ------------------
    elif task_type == "communication":
        for candidate in state.candidates:
            action = {
                "type": "write_email",
                "candidate_id": candidate.id,
                "content": (
                    f"Dear {candidate.name}, "
                    "Thank you for applying to our role. "
                    "Unfortunately, we have decided not to move forward "
                    "with your application at this time. "
                    "We appreciate your interest and wish you the best "
                    "in your job search. "
                    "Sincerely, The Hiring Team"
                ),
            }
            _, reward, done, _ = env.step(action)
            total_reward += reward
            steps_taken += 1
            if done:
                break

    final_score = env.compute_final_score()
    quality = env._get_decision_quality(final_score)
    bias_report = getattr(env, "_last_bias_explanation", "n/a")

    return {
        "task": task_type,
        "role": env.state.job_description.role,
        "scenario_id": getattr(env, "current_scenario_id", "unknown"),
        "final_score": final_score,
        "decision_quality": quality,
        "total_reward": round(total_reward, 4),
        "steps_taken": steps_taken,
        "bias_report": bias_report if task_type == "resume" else "n/a",
    }


# -----------------------------------------------------------------------
# BASELINE — runs a simple heuristic per task (uses internal session)
# -----------------------------------------------------------------------
@app.get("/baseline")
def baseline():
    env = HireLoopEnv()
    tasks = ["resume", "offer", "communication"]
    scores = []
    results = []

    for task in tasks:
        result = _run_heuristic_task(env, task)
        scores.append(result["final_score"])
        results.append({
            "task": result["task"],
            "score": result["final_score"],
            "total_reward": result["total_reward"],
        })

    return {
        "baseline_score": round(sum(scores) / len(scores), 4),
        "task_breakdown": results
    }


# -----------------------------------------------------------------------
# TASKS — all 3 tasks with action schemas
# -----------------------------------------------------------------------
@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "name": "Resume Screening",
                "task_type": "resume",
                "difficulty": "easy",
                "expected_score": "~0.75",
                "objective": "Select top 3 candidates based on skill match and experience",
                "max_steps": 15,
                "num_candidates": 10,
                "reward_signals": [
                    "+skill_match",
                    "+experience_fit",
                    "-obvious_rejects_kept",
                    "-diversity_penalty",
                ],
                "action_schema": {
                    "type": "accept | reject",
                    "candidate_id": "string",
                },
            },
            {
                "name": "Offer Decision",
                "task_type": "offer",
                "difficulty": "medium",
                "expected_score": "~0.55",
                "objective": "Make offers to shortlisted candidates within budget constraints",
                "max_steps": 10,
                "num_candidates": 5,
                "budget": "dynamic (~2.2x median candidate salary in USD)",
                "reward_signals": [
                    "+role_fit_score",
                    "+budget_efficiency",
                    "-over_budget_penalty",
                    "+request_details_bonus",
                ],
                "action_schema": {
                    "type": "offer | negotiate",
                    "candidate_id": "string",
                    "note": "Use 'negotiate' when candidate has partial skill match to get 10% salary discount"
                },
            },
            {
                "name": "Communication Drafting",
                "task_type": "communication",
                "difficulty": "hard",
                "expected_score": "~0.30",
                "objective": "Write professional and safe rejection emails. Includes 1 adversarial candidate.",
                "max_steps": 10,
                "num_candidates": 8,
                "reward_signals": [
                    "+polite_tone",
                    "+clear_rejection",
                    "+structured_response",
                    "+personalization",
                    "+job_role_context",
                    "+missing_skill_reference",
                    "+existing_skill_acknowledgment",
                    "+encouragement",
                    "-unsafe_discriminatory_words",
                    "-prompt_injection_caught",
                    "+counterfactual_audit",
                ],
                "action_schema": {
                    "type": "write_email",
                    "candidate_id": "string",
                    "content": "string (the email body)",
                },
            },
        ],
    }


# -----------------------------------------------------------------------
# EVAL — runs all 3 tasks with baseline heuristic, returns full report
# -----------------------------------------------------------------------
@app.get("/eval")
def eval_all():
    env = HireLoopEnv()
    tasks = ["resume", "offer", "communication"]
    results = []

    for task_type in tasks:
        result = _run_heuristic_task(env, task_type)
        results.append(result)

    average_score = round(
        sum(r["final_score"] for r in results) / len(results), 4
    )
    overall_quality = env._get_decision_quality(average_score)

    return {
        "env": "hireloop",
        "version": "1.0.0",
        "average_score": average_score,
        "overall_quality": overall_quality,
        "tasks": results,
    }