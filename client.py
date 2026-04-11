"""
HireLoop Environment Client
============================
Typed client for connecting to the HireLoop environment server.
Implements the openenv-core HTTPEnvClient pattern when available,
falls back to a standalone client for local development.

Usage:
    from client import HireLoopClient

    client = HireLoopClient(base_url="http://localhost:7860")
    result = client.reset(task="resume")
    session_id = result["session_id"]

    result = client.step(session_id=session_id, action={
        "type": "accept",
        "candidate_id": "1"
    })
    print(result["reward"])
"""

import requests
from typing import Optional, Dict, Any


class HireLoopClient:
    """Synchronous client for the HireLoop environment."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self.session_id: Optional[str] = None

    def health(self) -> dict:
        """Check if the environment server is healthy."""
        return requests.get(f"{self.base_url}/health").json()

    def reset(self, task: Optional[str] = None) -> dict:
        """
        Reset the environment and create a new session.

        Args:
            task: Optional task type ("resume", "offer", "communication").
                  If None, a random task is selected.

        Returns:
            dict with session_id, observation, reward, done, info
        """
        params = {"task": task} if task else {}
        resp = requests.post(f"{self.base_url}/reset", params=params).json()
        self.session_id = resp.get("session_id")
        return resp

    def step(self, action: Dict[str, Any], session_id: Optional[str] = None) -> dict:
        """
        Take a step in the environment.

        Args:
            action: Action dict with type, candidate_id, and optionally content
            session_id: Override session ID (defaults to last reset's session)

        Returns:
            dict with observation, reward, done, info
        """
        sid = session_id or self.session_id
        payload = {"session_id": sid, "action": action} if sid else action
        return requests.post(
            f"{self.base_url}/step",
            json=payload,
            headers={"Content-Type": "application/json"}
        ).json()

    def state(self) -> dict:
        """Get the current environment state."""
        params = {"session_id": self.session_id} if self.session_id else {}
        return requests.get(f"{self.base_url}/state", params=params).json()

    def grader(self) -> dict:
        """Get the final score for the current episode."""
        params = {"session_id": self.session_id} if self.session_id else {}
        return requests.get(f"{self.base_url}/grader", params=params).json()

    def baseline(self) -> dict:
        """Run the heuristic baseline across all 3 tasks."""
        return requests.get(f"{self.base_url}/baseline").json()

    def eval(self) -> dict:
        """Full evaluation with bias reports."""
        return requests.get(f"{self.base_url}/eval").json()

    def tasks(self) -> dict:
        """Get task descriptions and action schemas."""
        return requests.get(f"{self.base_url}/tasks").json()

    def close(self):
        """Cleanup (no-op for HTTP client)."""
        pass


if __name__ == "__main__":
    # Quick demo
    client = HireLoopClient()
    print("Health:", client.health())

    print("\n--- Resume Task ---")
    result = client.reset(task="resume")
    print(f"Session: {result['session_id']}")
    obs = result.get("observation", {})
    print(f"Role: {obs.get('job_description', {}).get('role', 'unknown')}")
    print(f"Candidates: {len(obs.get('candidates', []))}")

    step_result = client.step({"type": "accept", "candidate_id": "1"})
    print(f"Reward: {step_result['reward']}, Done: {step_result['done']}")

    print("\n--- Grader ---")
    print(client.grader())

    print("\n--- Baseline ---")
    bl = client.baseline()
    print(f"Baseline score: {bl['baseline_score']}")
    for t in bl["task_breakdown"]:
        print(f"  {t['task']}: {t['score']}")
