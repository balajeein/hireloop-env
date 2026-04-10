"""
Resume Screening Task Module
=============================
Handles reset, step, and scoring logic for the resume screening task.
"""

from typing import List, Dict, Optional
from models import Candidate, JobDescription, HireLoopState


# ---------------------------------------------------------------------------
# RESET — Resume Screening
# ---------------------------------------------------------------------------
def reset(scenario: dict, rng) -> tuple:
    """
    Initialize a resume screening episode from a scenario.

    Returns:
        (state, correct_shortlist, max_steps, current_scenario_id)
    """
    job = JobDescription(
        role=scenario["job"]["role"],
        required_skills=scenario["job"]["required_skills"],
        max_salary=scenario["job"]["max_salary"],
        seniority=scenario["job"]["seniority"],
    )

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
        for c in scenario["candidates"]
    ]

    correct_shortlist = scenario["correct_shortlist"]
    max_steps = 15

    state = HireLoopState(
        job_description=job,
        candidates=candidates,
        shortlisted=[],
        rejected=[],
        step_count=0,
        task_type="resume",
        offers_made=[],
        emails_sent=[],
        budget=0,
    )

    return state, correct_shortlist, max_steps, scenario["id"]


# ---------------------------------------------------------------------------
# STEP — Resume Screening
# ---------------------------------------------------------------------------
def step(state: HireLoopState, action: Dict, correct_shortlist: List[str],
         last_action, max_steps: int) -> tuple:
    """
    Execute one step of the resume screening task.

    Returns:
        (state, reward, done, info)
    """
    reward = 0.0
    done = False
    info = {}
    info["explanation"] = ""

    action_type = action.get("type")
    candidate_id = action.get("candidate_id")

    all_ids = [c.id for c in state.candidates]

    if candidate_id not in all_ids:
        reward -= 0.5
        info["explanation"] = f"Invalid candidate ID '{candidate_id}'. Not found in candidate pool."
        info["task_type"] = state.task_type
        state.step_count += 1
        info["step_count"] = state.step_count

        reward = max(-1.0, min(1.0, reward))
        reward = round(reward, 4)

        return state, reward, done, info

    # Indecisive penalty: switching a previous decision costs -0.1
    was_shortlisted = candidate_id in state.shortlisted
    was_rejected = candidate_id in state.rejected
    if (action_type == "accept" and was_rejected) or (action_type == "reject" and was_shortlisted):
        reward -= 0.1

    if action_type == "accept":
        already_selected = candidate_id in state.shortlisted

        # Remove from rejected if previously rejected
        if candidate_id in state.rejected:
            state.rejected.remove(candidate_id)

        # Only allow reward if it's a NEW action
        if not already_selected:
            state.shortlisted.append(candidate_id)

            if candidate_id in correct_shortlist:
                reward += 1.0
                info["explanation"] = f"Accepted candidate {candidate_id} with strong skill match."
            else:
                reward -= 0.5
                info["explanation"] = f"Accepted candidate {candidate_id} but skills do not match well."
        else:
            reward -= 0.1  # penalty for repeating same accept
            info["explanation"] = f"Repeated accept action for candidate {candidate_id}."

    elif action_type == "reject":
        # Remove from shortlist if already selected
        if candidate_id in state.shortlisted:
            state.shortlisted.remove(candidate_id)

        # Add to rejected (avoid duplicate)
        if candidate_id not in state.rejected:
            state.rejected.append(candidate_id)

        # Reward
        if candidate_id not in correct_shortlist:
            reward += 0.3
            info["explanation"] = f"Correctly rejected unqualified candidate {candidate_id}."
        else:
            reward -= 0.2  # wrongly rejected a good candidate
            info["explanation"] = f"Incorrectly rejected a strong candidate {candidate_id}."
    else:
        reward -= 0.5  # invalid action type
        info["explanation"] = f"Invalid action type '{action_type}'. Use 'accept' or 'reject'."

    # Step penalty
    reward -= 0.01
    # Bonus for building a good shortlist gradually
    correct_selected = len(set(state.shortlisted) & set(correct_shortlist))
    reward += correct_selected * 0.05

    # Penalty for wrong candidates in shortlist
    wrong_selected = len(set(state.shortlisted) - set(correct_shortlist))
    reward -= wrong_selected * 0.03

    # Loop penalty (same action repeated)
    if last_action == action:
        reward -= 0.2

    state.step_count += 1

    # Done conditions: selected 3 candidates or ran out of steps
    if len(state.shortlisted) >= 3 or state.step_count >= max_steps:
        done = True

    reward = max(-1.0, min(1.0, reward))
    reward = round(reward, 4)

    return state, reward, done, info


# ---------------------------------------------------------------------------
# SCORE — Resume Screening (final episode score, 0.0–1.0)
# ---------------------------------------------------------------------------
def score(state: HireLoopState, correct_shortlist: List[str], max_steps: int) -> float:
    """Compute final episode score for resume screening."""
    selected = set(state.shortlisted)
    correct = set(correct_shortlist)

    correct_picks = selected & correct
    accuracy = len(correct_picks) / len(correct) if correct else 0
    precision = len(correct_picks) / len(selected) if selected else 0

    # Speed bonus
    steps_used = state.step_count
    speed_bonus = max(0, (max_steps - steps_used) / max_steps) * 0.2

    # Wrong candidates penalty
    bad_accepts = selected - correct
    wrong_penalty = len(bad_accepts) * 0.05

    # Bias detection — smart check, only penalizes if diverse pool existed
    bias_result = _check_diversity_bias(state)
    bias_penalty = bias_result["penalty"]

    final = (accuracy * 0.5) + (precision * 0.3) + speed_bonus - wrong_penalty + bias_penalty
    return max(0.0, min(1.0, round(final, 4))), bias_result


# ---------------------------------------------------------------------------
# BIAS AUDIT — checks shortlisted candidates for diversity bias
# ---------------------------------------------------------------------------
def _check_diversity_bias(state: HireLoopState) -> dict:
    """
    Checks shortlisted candidates for diversity bias.
    Only penalizes if diverse candidates were actually available in the pool.
    Returns a dict with penalty value and explanation.
    """

    if not state.shortlisted:
        return {"penalty": 0.0, "explanation": "No candidates shortlisted yet."}

    # Get full candidate objects for shortlisted ids
    shortlisted_candidates = [
        c for c in state.candidates
        if c.id in state.shortlisted
    ]

    # Get full candidate pool objects
    all_candidates = state.candidates

    penalty = 0.0
    reasons = []

    # --- Gender bias check ---
    pool_genders = set(
        c.gender for c in all_candidates
        if c.gender is not None
    )
    shortlist_genders = set(
        c.gender for c in shortlisted_candidates
        if c.gender is not None
    )

    # Only penalize if pool had gender variety but shortlist doesn't
    if len(pool_genders) > 1 and len(shortlist_genders) == 1:
        penalty -= 0.15
        only_gender = list(shortlist_genders)[0]
        reasons.append(
            f"shortlist contains only {only_gender} candidates "
            f"despite gender diversity in pool"
        )

    # --- Nationality bias check ---
    pool_nationalities = set(
        c.nationality for c in all_candidates
        if c.nationality is not None
    )
    shortlist_nationalities = set(
        c.nationality for c in shortlisted_candidates
        if c.nationality is not None
    )

    # Only penalize if pool had nationality variety but shortlist doesn't
    if len(pool_nationalities) > 1 and len(shortlist_nationalities) == 1:
        penalty -= 0.10
        only_nationality = list(shortlist_nationalities)[0]
        reasons.append(
            f"shortlist contains only {only_nationality} candidates "
            f"despite nationality diversity in pool"
        )

    # --- Diversity bonus ---
    # Reward if shortlist has both genders represented
    if len(shortlist_genders) >= 2:
        penalty += 0.05
        reasons.append("shortlist reflects gender diversity")

    explanation = (
        "Bias check: " + "; ".join(reasons)
        if reasons
        else "Bias check: no bias detected"
    )

    return {
        "penalty": round(penalty, 4),
        "explanation": explanation
    }
