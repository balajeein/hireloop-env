"""
HireLoop Quick Start
====================
Demonstrates how to interact with the HireLoop environment
using the typed HireLoopClient.

Requirements:
    Server must be running: uvicorn api:app --port 7860

Usage:
    python3 examples/quickstart.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import HireLoopClient

def main():
    client = HireLoopClient("http://localhost:7860")

    # Health check
    health = client.health()
    print(f"Server status: {health['status']}")
    print(f"Available tasks: {health['tasks']}")

    # Resume task
    print("\n--- RESUME TASK ---")
    result = client.reset(task="resume")
    session_id = result["session_id"]
    obs = result["observation"]
    print(f"Role: {obs['job_description']['role']}")
    print(f"Candidates: {len(obs['candidates'])}")
    print(f"Session ID: {session_id}")

    first_candidate = obs["candidates"][0]
    step = client.step(
        action={"type": "accept", "candidate_id": first_candidate["id"]},
        session_id=session_id
    )
    print(f"Accepted {first_candidate['name']} → reward={step['reward']}")

    score = client.grader()
    print(f"Current score: {score['score']}")

    # Offer task
    print("\n--- OFFER TASK ---")
    result = client.reset(task="offer")
    obs = result["observation"]
    print(f"Role: {obs['job_description']['role']}")
    print(f"Budget: ${obs['budget']:,}")
    print(f"Candidates: {len(obs['candidates'])}")

    # Communication task
    print("\n--- COMMUNICATION TASK ---")
    result = client.reset(task="communication")
    obs = result["observation"]
    session_id = result["session_id"]
    print(f"Role: {obs['job_description']['role']}")
    print(f"Candidates to email: {len(obs['candidates'])}")

    adv = next((c for c in obs["candidates"] if c["id"] == "adv1"), None)
    if adv:
        print(f"Adversarial candidate detected: {adv['id']}")
        step = client.step(
            action={
                "type": "write_email",
                "candidate_id": "adv1",
                "content": "Dear Alex, thank you for applying. Unfortunately we will not be moving forward. Best wishes, HR Team"
            },
            session_id=session_id
        )
        print(f"Email reward: {step['reward']}")
        breakdown = step.get("info", {}).get("email_breakdown", {})
        if breakdown:
            print(f"Injection penalty: {breakdown.get('prompt_injection_penalty', 0)}")

    # Baseline
    print("\n--- BASELINE ---")
    baseline = client.baseline()
    print(f"Average score: {baseline['baseline_score']}")
    for task in baseline["task_breakdown"]:
        print(f"  {task['task']:<15} score={task['score']}")

if __name__ == "__main__":
    main()
