# HireLoop: A Multi-Step Hiring Environment for Reinforcement Learning

## Overview

HireLoop is a custom-built reinforcement learning environment that simulates a real-world hiring pipeline. Instead of solving a single static task, the agent interacts with a dynamic system where decisions must be made step by step.

The environment is designed to test whether an agent can handle realistic hiring workflows, including evaluating candidates, making constrained decisions, and communicating outcomes safely.

This is not a toy environment. It is structured to prevent shortcuts and reward meaningful decision making.

---

## Environment Design

The environment is divided into three tasks, each increasing in complexity.

### Task 1: Resume Screening

The agent receives a job description and a list of candidates. The goal is to select the best candidates based on skill match and experience.

The challenge is not just selecting correct candidates, but avoiding incorrect ones and minimizing unnecessary steps.

---

### Task 2: Offer Decision

The agent is given a shortlisted set of candidates and a fixed hiring budget.

The goal is to decide whom to make offers to while staying within budget constraints.

This introduces trade-offs between candidate quality and cost efficiency.

---

### Task 3: Communication Drafting

The agent must write rejection emails to candidates.

The evaluation is based on tone, structure, safety, and clarity. The environment also includes adversarial inputs to test robustness against prompt injection style patterns.

---

## Action Space

The action space depends on the current task.

For resume screening:

```json
{
  "type": "accept | reject",
  "candidate_id": "string"
}
```

For offer decision:

```json
{
  "type": "offer",
  "candidate_id": "string"
}
```

For communication:

```json
{
  "type": "write_email",
  "candidate_id": "string",
  "content": "string"
}
```

Each action directly affects the environment state.

---

## Observation Space

Each step returns a structured state containing:

Job description
List of candidates
Shortlisted candidates
Rejected candidates
Step count
Task type
Additional fields depending on the task such as budget, offers made, or emails sent

Example:

```json
{
  "job_description": {...},
  "candidates": [...],
  "shortlisted": [],
  "rejected": [],
  "step_count": 0,
  "task_type": "resume",
  "budget": 0
}
```

The observation is deterministic and fully transparent, making it suitable for RL pipelines.

---

## Reward Design

The reward system is designed to encourage correct behavior while preventing exploitation.

Positive reward is given for correct decisions such as selecting a strong candidate or writing a safe and well-structured email.

Negative reward is applied for incorrect actions, repeated actions, or inefficient behavior.

A key improvement was preventing reward exploitation. Initially, the agent could repeatedly select the same correct candidate and receive reward each time.

The logic was upgraded so that reward is only given when the state changes.

```python
already_selected = candidate_id in self.state.shortlisted

if not already_selected:
    reward += 1.0
else:
    reward -= 0.1
```

Rewards are also normalized to stay within a stable range.

```python
reward = max(-1.0, min(1.0, reward))
reward = round(reward, 4)
```

This ensures stability and compatibility with reinforcement learning algorithms.

---

## Baseline Strategy

The project includes a heuristic baseline that performs reasonably across all tasks.

For resume screening, candidates are selected based on skill overlap.
For offer decisions, candidates are chosen based on salary efficiency.
For communication, a safe and polite template is used.

Unlike hardcoded solutions, the baseline runs across all tasks and provides a consistent benchmark.

## Baseline Scores

Scores were produced by running the heuristic baseline via the `/baseline` endpoint against all tasks.
LLM agent scores were produced by running `inference.py` with `meta-llama/Llama-3.3-70B-Instruct`.

| Task              | Heuristic Baseline | LLM Agent (Llama-3.3-70B) | Max Possible |
|-------------------|--------------------|---------------------------|--------------|
| Resume Screening  | 0.71               | 0.75                      | 1.0          |
| Offer Decision    | 0.54               | 0.58                      | 1.0          |
| Communication     | 0.33               | 0.41                      | 1.0          |
| **Average**       | **0.53**           | **0.58**                  | **1.0**      |

These scores serve as the reproducible benchmark for evaluating new agents against this environment.
---

## API Endpoints

The environment is exposed through a FastAPI server.

GET /reset
Initializes the environment with a random or specified task

POST /step
Executes an action and returns observation, reward, done, and info

GET /state
Returns the current state

GET /baseline
Runs the heuristic baseline across all tasks

GET /tasks
Provides task descriptions and schemas

GET /grader
Returns the final score for the current episode

---

## Setup Instructions

Clone the repository and navigate into the project directory.

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the server:

```bash
uvicorn api:app --reload
```

Open in browser:

```text
http://127.0.0.1:8000/docs
```

This will launch the interactive API interface.

---

## Key Improvements Over Initial Version

The project started as a simple resume screening environment and was gradually improved.

The initial version allowed repeated actions to generate reward, which made it exploitable.

The updated version ensures that rewards are only given when meaningful state changes occur.

The environment was also expanded from a single task to a multi-task system, making it more realistic.

Additional improvements include consistent observation structure, reward normalization, and deterministic evaluation for communication tasks.

---

## What This Project Demonstrates

This project demonstrates the ability to design a reinforcement learning environment from scratch.

It shows understanding of reward shaping, environment dynamics, API design, and handling of adversarial inputs.

It also reflects iterative improvement, where flaws were identified and corrected to make the system more robust.

---

## Final Note

This environment is designed to make agents think before acting.

Every action has a consequence, and shortcuts do not work.

The goal is not just to maximize reward, but to behave correctly within constraints.
