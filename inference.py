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


def get_llm_action(state: dict, task_type: str) -> dict:
    """Ask the LLM what action to take given current state."""
    state_str = json.dumps(state, indent=2, default=str)

    prompt = f"""Current environment state:
{state_str}

Task type: {task_type}

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
        print(f"  LLM error: {e}")
        return None


def run_task(task_type: str) -> dict:
    """Run one full episode for a given task type."""
    print(f"\n{'='*50}")
    print(f"Running task: {task_type.upper()}")
    print(f"{'='*50}")

    # Reset environment
    resp = requests.get(f"{ENV_BASE_URL}/reset", params={"task": task_type})
    state = resp.json()["state"]
    print(f"Candidates: {len(state['candidates'])}")

    total_reward = 0.0
    step = 0
    done = False
    max_steps = 20  # safety cap

    while not done and step < max_steps:
        step += 1

        # Get action from LLM
        action = get_llm_action(state, task_type)

        if action is None:
            print(f"  Step {step}: LLM returned invalid action, skipping")
            continue

        print(f"  Step {step}: {action.get('type')} → candidate {action.get('candidate_id')}")

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
            total_reward += reward
            print(f"           reward={reward:.4f}, done={done}")

        except Exception as e:
            print(f"  Step error: {e}")
            break

    # Get final score
    resp = requests.get(f"{ENV_BASE_URL}/grader")
    final_score = resp.json()["score"]
    print(f"\nFinal score: {final_score:.4f}")
    print(f"Total reward accumulated: {total_reward:.4f}")

    return {
        "task": task_type,
        "final_score": final_score,
        "total_reward": round(total_reward, 4),
        "steps_taken": step
    }


def main():
    print("HireLoop — LLM Agent Baseline")
    print(f"Model: {MODEL_NAME}")
    print(f"Environment: {ENV_BASE_URL}")

    # Check environment is alive
    try:
        resp = requests.get(f"{ENV_BASE_URL}/health")
        assert resp.json()["status"] == "ok"
        print("Environment: online\n")
    except Exception:
        print("ERROR: Environment not reachable. Start with: uvicorn api:app --port 7860")
        return

    tasks = ["resume", "offer", "communication"]
    results = []

    for task in tasks:
        result = run_task(task)
        results.append(result)

    # Summary
    print(f"\n{'='*50}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*50}")
    for r in results:
        print(f"  {r['task']:<20} score={r['final_score']:.4f}   steps={r['steps_taken']}")

    avg_score = sum(r["final_score"] for r in results) / len(results)
    print(f"\n  Average score: {avg_score:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()