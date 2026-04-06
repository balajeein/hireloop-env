"""
HireLoop Inference Script
=========================
Runs a baseline LLM agent against all 3 tasks and reports scores.

Required environment variables:
    API_BASE_URL   The API endpoint for the LLM
    MODEL_NAME     The model identifier
    HF_TOKEN       Your Hugging Face / API key
"""

import os
import json
import requests
from typing import List, Optional
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are an expert hiring manager agent operating in a recruitment environment.

You will receive the current state of a hiring pipeline and must respond with exactly one action as valid JSON.

For resume screening tasks, respond with:
{"type": "accept", "candidate_id": "X"} or {"type": "reject", "candidate_id": "X"}

For offer decision tasks, respond with:
{"type": "offer", "candidate_id": "X"} if the candidate has full skill match
{"type": "negotiate", "candidate_id": "X"} if the candidate has partial skill match (saves 10% salary, use this to stay within budget). Check the negotiation_hints field in state to identify which candidates are negotiable before deciding.

For communication tasks, respond with:
{"type": "write_email", "candidate_id": "X", "content": "Dear [Name], ..."}

Rules:
- Only use candidate_ids that exist in the candidates list
- For emails: always be polite, professional, and never mention protected characteristics
- For offers: stay within the budget constraint
- Respond with raw JSON only, no explanation
"""


# ==============================================================================
# REQUIRED STDOUT LOGGING FUNCTIONS
# ==============================================================================

def log_start(task: str, env: str, model: str) -> None:
    """Emit START line to stdout."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Emit STEP line to stdout."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit END line to stdout."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = str(success).lower()
    print(
        f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ==============================================================================
# AGENT LOGIC
# ==============================================================================

def get_llm_action(state: dict, task_type: str) -> dict:
    """Ask the LLM what action to take given current state."""
    state_for_llm = {k: v for k, v in state.items() if k != "negotiation_hints"}
    state_str = json.dumps(state_for_llm, indent=2, default=str)

    already_shortlisted = state.get("shortlisted", [])
    already_rejected = state.get("rejected", [])
    all_ids = [c["id"] for c in state.get("candidates", [])]
    remaining = [cid for cid in all_ids if cid not in already_shortlisted and cid not in already_rejected]

    prompt = f"""Current environment state:
    {state_str}

    Task type: {task_type}
    Already shortlisted: {already_shortlisted}
    Already rejected: {already_rejected}
    Remaining candidates to process: {remaining}

    Pick from remaining candidates only. Do not repeat actions on already processed candidates.
    What is your next action? Respond with a single JSON object only."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.1,
        )
        content = response.choices[0].message.content.strip()

        # Clean up response if wrapped in markdown
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        action = json.loads(content)
        return action

    except Exception as e:
        return None


def run_task(task_type: str) -> dict:
    """Run one full episode for a given task type."""
    
    # Reset environment
    resp = requests.get(f"{ENV_BASE_URL}/reset", params={"task": task_type})
    state = resp.json()["state"]

    total_reward = 0.0
    step = 0
    done = False
    max_steps = 20  # safety cap
    rewards: List[float] = []
    success = False

    # Emit START
    log_start(task=task_type, env="hireloop", model=MODEL_NAME)

    while not done and step < max_steps:
        step += 1

        # Get action from LLM
        action = get_llm_action(state, task_type)

        if action is None:
            # Log failed step
            log_step(step=step, action="invalid", reward=0.0, done=False, error="LLM returned invalid action")
            continue

        # Format action string for logging
        action_type = action.get('type', 'unknown')
        candidate_id = action.get('candidate_id', 'none')
        action_str = f"{action_type}('{candidate_id}')"

        # Execute action
        try:
            resp = requests.post(
                f"{ENV_BASE_URL}/step",
                json=action,
                headers={"Content-Type": "application/json"}
            )
            result = resp.json()
            state = result["observation"]
            reward = result["reward"]
            done = result["done"]
            
            rewards.append(reward)
            total_reward += reward

            # Log this step
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

        except Exception as e:
            error_msg = str(e)
            log_step(step=step, action=action_str, reward=0.0, done=False, error=error_msg)
            break

    # Get final score
    try:
        resp = requests.get(f"{ENV_BASE_URL}/grader")
        final_score = resp.json()["score"]
    except Exception:
        final_score = 0.0
    
    # Determine success (threshold: 0.5)
    success = final_score >= 0.5

    # Emit END
    log_end(success=success, steps=step, score=final_score, rewards=rewards)

    return {
        "task": task_type,
        "final_score": final_score,
        "total_reward": round(total_reward, 4),
        "steps_taken": step,
        "success": success
    }


def main():
    # Check environment is alive
    try:
        resp = requests.get(f"{ENV_BASE_URL}/health")
        assert resp.json()["status"] == "ok"
    except Exception:
        print("[ERROR] Environment not reachable. Start with: uvicorn api:app --port 7860", flush=True)
        return

    tasks = ["resume", "offer", "communication"]
    results = []

    for task in tasks:
        result = run_task(task)
        results.append(result)

    # Print summary (optional, not part of required format but helpful for humans)
    print("\n" + "="*50, flush=True)
    print("SUMMARY", flush=True)
    print("="*50, flush=True)
    for r in results:
        print(f"  {r['task']:<15} score={r['final_score']:.3f}   success={r['success']}", flush=True)
    
    avg_score = sum(r["final_score"] for r in results) / len(results)
    print(f"\n  Average score: {avg_score:.3f}", flush=True)
    print("="*50, flush=True)


if __name__ == "__main__":
    main()