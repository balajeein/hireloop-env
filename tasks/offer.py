"""
Offer Decision Task Module
===========================
Handles reset, step, and scoring logic for the offer decision task.
Includes skill category matching and negotiation eligibility logic.
"""

from typing import List, Dict, Optional
from models import Candidate, JobDescription, HireLoopState
from utils.skills import (
    SKILL_CATEGORIES, get_skill_category,
    are_skills_similar, check_negotiation_eligibility
)

MIN_STRICT_SCORE = 0.01
MAX_STRICT_SCORE = 0.99


def reset(scenario: dict, rng) -> tuple:
    """
    Initialize an offer decision episode from a scenario.

    Returns:
        (state, correct_shortlist, max_steps, current_scenario_id, negotiation_hints)
    """
    job = JobDescription(
        role=scenario["job"]["role"],
        required_skills=scenario["job"]["required_skills"],
        max_salary=scenario["job"]["max_salary"],
        seniority=scenario["job"]["seniority"],
    )

    # For offer task, only take candidates who match at least 1 required skill
    # These simulate pre-shortlisted candidates from resume round
    required_skills = set(scenario["job"]["required_skills"])
    qualified = [
        c for c in scenario["candidates"]
        if set(c["skills"]) & required_skills
    ]

    # If less than 2 qualified, fall back to all candidates
    if len(qualified) < 2:
        qualified = scenario["candidates"]

    candidates = [
        Candidate(
            id=c["id"],
            name=c["name"],
            skills=c["skills"],
            years_experience=c["years_experience"],
            expected_salary=c["expected_salary"],
            gender=c.get("gender"),
            nationality=c.get("nationality"),
        )
        for c in qualified
    ]

    # Use correct_shortlist directly — these are already the best candidates
    correct_candidates = [c for c in candidates if c.id in scenario["correct_shortlist"]]
    correct_candidates.sort(key=lambda c: c.expected_salary)

    # Pick top 3 cheapest from correct shortlist
    optimal_set = correct_candidates[:3]

    # Budget = sum of their salaries with negotiation applied
    budget = 0
    for c in optimal_set:
        eligibility = check_negotiation_eligibility(c.skills, list(job.required_skills))
        if eligibility["negotiable"]:
            budget += round(c.expected_salary * 0.9)
        else:
            budget += c.expected_salary

    correct_shortlist = scenario["correct_shortlist"]
    max_steps = 10

    # Build negotiation hints so agent can reason about each candidate
    negotiation_hints = {}
    for c in candidates:
        eligibility = check_negotiation_eligibility(
            c.skills,
            list(job.required_skills)
        )
        negotiation_hints[c.id] = {
            "eligible": eligibility["eligible"],
            "negotiable": eligibility["negotiable"],
            "reason": eligibility["reason"],
        }

    state = HireLoopState(
        job_description=job,
        candidates=candidates,
        shortlisted=[],
        rejected=[],
        step_count=0,
        task_type="offer",
        budget=budget,
        offers_made=[],
        emails_sent=[],
    )

    return state, correct_shortlist, max_steps, scenario["id"], negotiation_hints



def step(state: HireLoopState, action: Dict, correct_shortlist: List[str],
         last_action, max_steps: int, negotiation_hints: dict) -> tuple:
    """
    Execute one step of the offer decision task.

    Returns:
        (state_dict, reward, done, info)
    """
    reward = 0.0
    done = False
    info = {}
    info["explanation"] = ""

    action_type = action.get("type")
    candidate_id = action.get("candidate_id")

    all_ids = [c.id for c in state.candidates]

    if action_type not in ("offer", "negotiate"):
        reward -= 0.5
        info["error"] = "Invalid action type. Use 'offer' or 'negotiate'."
        info["task_type"] = state.task_type
        state.step_count += 1
        info["step_count"] = state.step_count
        info["budget"] = state.budget
        info["offers_count"] = len(state.offers_made)
        reward = max(-1.0, min(1.0, reward))
        reward = round(reward, 4)
        info["explanation"] = f"Invalid action type '{action_type}'. Use 'offer' for full offers or 'negotiate' for partial skill matches."
        return state, reward, done, info

    if candidate_id not in all_ids:
        reward -= 0.5
        info["error"] = f"Candidate {candidate_id} not found."
        info["task_type"] = state.task_type
        state.step_count += 1
        info["step_count"] = state.step_count
        info["budget"] = state.budget
        info["offers_count"] = len(state.offers_made)
        info["explanation"] = "Invalid candidate ID provided."
        reward = max(-1.0, min(1.0, reward))
        reward = round(reward, 4)
        return state, reward, done, info

    # Prevent duplicate offers
    already_offered = [o["candidate_id"] for o in (state.offers_made or [])]
    if candidate_id in already_offered:
        reward -= 0.2
        info["error"] = "Duplicate offer."
        info["explanation"] = f"Duplicate offer for candidate {candidate_id}. Already offered."
        info["task_type"] = state.task_type
        state.step_count += 1
        info["step_count"] = state.step_count
        info["budget"] = state.budget
        info["offers_count"] = len(state.offers_made)
        reward = max(-1.0, min(1.0, reward))
        reward = round(reward, 4)
        return state, reward, done, info

    # Find candidate
    candidate = next(c for c in state.candidates if c.id == candidate_id)

    # Check negotiation eligibility
    job_skills = list(state.job_description.required_skills)
    eligibility = check_negotiation_eligibility(candidate.skills, job_skills)




    if action_type == "offer":
        # Standard full offer — only valid for full skill match
        if not eligibility["eligible"]:
            # Zero matches — direct reject, penalize agent
            reward -= 0.5
            info["explanation"] = (
                f"Candidate {candidate_id} rejected. "
                f"{eligibility['reason']}"
            )
            info["negotiation"] = eligibility
            state.step_count += 1
            state.rejected.append(candidate_id)

            reward = max(-1.0, min(1.0, reward))
            reward = round(reward, 4)
            info["task_type"] = state.task_type
            info["step_count"] = state.step_count
            info["budget"] = state.budget
            info["offers_count"] = len(state.offers_made)
            return state, reward, done, info

        if eligibility["negotiable"]:
            # Agent used full offer on a negotiable candidate — small penalty
            # They should have used negotiate action instead
            reward -= 0.15
            info["explanation"] = (
                f"Used full offer on negotiable candidate {candidate_id}. "
                f"Consider using 'negotiate' action to save budget."
            )
        else:
            info["explanation"] = f"Full offer to candidate {candidate_id}. Perfect skill match."

        # Calculate spend with full salary
        actual_salary = candidate.expected_salary

    elif action_type == "negotiate":
        # Negotiation action — only valid when candidate is negotiable
        if not eligibility["eligible"]:
            reward -= 0.5
            info["explanation"] = (
                f"Cannot negotiate with candidate {candidate_id}. "
                f"{eligibility['reason']}"
            )
            info["negotiation"] = eligibility
            state.step_count += 1

            reward = max(-1.0, min(1.0, reward))
            reward = round(reward, 4)
            info["task_type"] = state.task_type
            info["step_count"] = state.step_count
            info["budget"] = state.budget
            info["offers_count"] = len(state.offers_made)
            return state, reward, done, info

        if not eligibility["negotiable"]:
            # Agent tried to negotiate with a perfect match candidate
            # That's unnecessary — small penalty
            reward -= 0.1
            info["explanation"] = (
                f"Unnecessary negotiation with perfect match candidate {candidate_id}. "
                f"Use standard 'offer' action."
            )
            actual_salary = candidate.expected_salary
        else:
            # Perfect use of negotiate action
            discount = eligibility["discount"]
            actual_salary = round(candidate.expected_salary * (1 - discount))
            reward += 0.2  # bonus for smart negotiation
            info["explanation"] = (
                f"Smart negotiation with candidate {candidate_id}. "
                f"Salary reduced from {candidate.expected_salary} "
                f"to {actual_salary} (10% discount). "
                f"Similar skills: {eligibility['similar_matches']}"
            )
            info["negotiation"] = eligibility
            info["negotiated_salary"] = actual_salary
    else:
        reward -= 0.5
        info["explanation"] = f"Invalid action type '{action_type}'. Use 'offer' or 'negotiate'."
        reward = max(-1.0, min(1.0, reward))
        reward = round(reward, 4)
        info["task_type"] = state.task_type
        info["step_count"] = state.step_count
        info["budget"] = state.budget
        info["offers_count"] = len(state.offers_made)
        return state, reward, done, info

    # Calculate current total spend
    current_spend = sum(
        o.get("actual_salary", next(
            c for c in state.candidates if c.id == o["candidate_id"]
        ).expected_salary)
        for o in (state.offers_made or [])
    )

    new_spend = current_spend + actual_salary




    # Role fit score
    job_skills_set = set(state.job_description.required_skills)
    candidate_skills_set = set(candidate.skills)
    overlap = len(job_skills_set & candidate_skills_set) / len(job_skills_set) if job_skills_set else 0
    reward += overlap * 0.5

    # Budget efficiency
    if new_spend <= state.budget:
        efficiency = 1.0 - (new_spend / state.budget)
        reward += efficiency * 0.3
    else:
        overage = (new_spend - state.budget) / state.budget
        reward -= overage * 2.0

    # Experience bonus
    if candidate.years_experience >= 3:
        reward += 0.1

    # Record offer with actual salary
    state.offers_made.append({
        "candidate_id": candidate_id,
        "actual_salary": actual_salary,
        "negotiated": action_type == "negotiate" and eligibility["negotiable"],
    })
    state.shortlisted.append(candidate_id)

    # Step penalty
    reward -= 0.01

    # Loop penalty
    if last_action == action:
        reward -= 0.2

    state.step_count += 1

    # Count how many candidates are actually eligible (full or negotiable)
    eligible_count = sum(
        1 for c in state.candidates
        if check_negotiation_eligibility(c.skills, list(state.job_description.required_skills))["eligible"]
    )
    max_offers = min(3, eligible_count)

    if state.step_count >= max_steps or len(state.offers_made) >= max_offers:
        done = True

    reward = max(-1.0, min(1.0, reward))
    reward = round(reward, 4)

    # Append budget context to existing explanation (don't overwrite negotiation details)
    if new_spend <= state.budget:
        budget_note = f" Budget remaining: {state.budget - new_spend}."
    else:
        budget_note = f" WARNING: Exceeded budget by {new_spend - state.budget}."
    info["explanation"] = info.get("explanation", "") + budget_note

    info["task_type"] = state.task_type
    info["step_count"] = state.step_count
    info["budget"] = state.budget
    info["offers_count"] = len(state.offers_made)
    state_dict = state.model_dump()
    state_dict["negotiation_hints"] = negotiation_hints
    return state_dict, reward, done, info



def score(state: HireLoopState, correct_shortlist: List[str], max_steps: int) -> float:
    """Compute final episode score for offer decision."""
    if not state.offers_made:
        return MIN_STRICT_SCORE

    job_skills = set(state.job_description.required_skills)

    # Role fit: average skill overlap across offers
    fit_scores = []
    total_salary = 0
    for offer in state.offers_made:
        cand = next((c for c in state.candidates if c.id == offer["candidate_id"]), None)
        if cand:
            overlap = len(set(cand.skills) & job_skills) / len(job_skills) if job_skills else 0
            fit_scores.append(overlap)
            actual_salary = offer.get("actual_salary", cand.expected_salary)
            total_salary += actual_salary

    avg_fit = min(MAX_STRICT_SCORE, sum(fit_scores) / len(fit_scores) if fit_scores else MIN_STRICT_SCORE)

    # Budget efficiency
    budget = state.budget or 1
    if total_salary <= budget:
        budget_score = min(MAX_STRICT_SCORE, 1.0 - (total_salary / budget) * 0.5)
    else:
        # Over-budget penalty
        overage_ratio = (total_salary - budget) / budget
        budget_score = max(0.0, 0.5 - overage_ratio)

    # Speed bonus
    steps_used = state.step_count
    speed_bonus = max(0, (max_steps - steps_used) / max_steps) * 0.1

    final = (avg_fit * 0.5) + (budget_score * 0.4) + speed_bonus
    return round(min(MAX_STRICT_SCORE, max(MIN_STRICT_SCORE, final)), 4)
