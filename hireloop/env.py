"""
HireLoop Environment — Thin Orchestrator
==========================================
Delegates task-specific logic to hireloop.tasks.{resume, offer, communication}.
"""

from typing import List, Dict, Optional
import random
import json
import os

from models import HireLoopState
from hireloop.tasks import resume, offer, communication
from hireloop.tasks.offer import check_negotiation_eligibility


class HireLoopEnv:
    def __init__(self):
        self.state: Optional[HireLoopState] = None
        self.max_steps = 10
        self.correct_shortlist: List[str] = []
        self.last_action = None
        self.random_seed = 42
        self.rng = random.Random(self.random_seed)
        self.negotiation_hints: dict = {}
        self.current_scenario_id: str = ""
        self._last_bias_explanation: str = "Bias check: not run yet."

        # Load scenarios from JSON file
        scenarios_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "scenarios.json"
        )
        with open(scenarios_path, "r") as f:
            self.scenarios = json.load(f)

    # -----------------------------------------------------------------------
    # RESET — randomly picks a task_type and builds appropriate data
    # -----------------------------------------------------------------------
    def reset(self) -> HireLoopState:
        task_type = self.rng.choice(["resume", "offer", "communication"])
        return self.reset_with_task(task_type)

    def reset_with_task(self, task_type: str) -> HireLoopState:
        """Reset with a specific task type (used by /reset?task=...)."""
        scenario = self.rng.choice(self.scenarios)

        if task_type == "offer":
            state, cs, ms, sid, hints = offer.reset(scenario, self.rng)
            self.negotiation_hints = hints
        elif task_type == "communication":
            state, cs, ms, sid = communication.reset(scenario, self.rng)
            self.negotiation_hints = {}
        else:
            state, cs, ms, sid = resume.reset(scenario, self.rng)
            self.negotiation_hints = {}

        self.state = state
        self.correct_shortlist = cs
        self.max_steps = ms
        self.current_scenario_id = sid
        self.last_action = None
        return self.state

    # -----------------------------------------------------------------------
    # STEP — branches logic based on task_type
    # -----------------------------------------------------------------------
    def step(self, action: Dict):
        if self.state is None:
            raise Exception("Environment not initialized. Call reset().")

        task = self.state.task_type

        if task == "resume":
            state, rw, done, info = resume.step(
                self.state, action, self.correct_shortlist,
                self.last_action, self.max_steps
            )
            self.state = state
            if done:
                final_score = self.compute_final_score()
                info["final_score"] = final_score
                info["decision_quality"] = self._get_decision_quality(final_score)
                info["final_explanation"] = (
                    f"Final score {final_score:.2f} — "
                    f"{info['decision_quality']} quality hiring decisions."
                )
                info["bias_report"] = self._last_bias_explanation

        elif task == "offer":
            result = offer.step(
                self.state, action, self.correct_shortlist,
                self.last_action, self.max_steps, self.negotiation_hints
            )
            obs, rw, done, info = result
            # offer.step returns state_dict when successful
            if isinstance(obs, dict):
                self.state = HireLoopState(**{
                    k: v for k, v in obs.items()
                    if k != "negotiation_hints"
                })
            else:
                self.state = obs
            if done:
                final_score = self.compute_final_score()
                info["final_score"] = final_score
                info["decision_quality"] = self._get_decision_quality(final_score)
                info["final_explanation"] = (
                    f"Final score {final_score:.2f} — "
                    f"{info['decision_quality']} quality hiring decisions."
                )
            state = obs

        elif task == "communication":
            state, rw, done, info = communication.step(
                self.state, action, self.correct_shortlist,
                self.last_action, self.max_steps
            )
            self.state = state
            if done:
                final_score = self.compute_final_score()
                info["final_score"] = final_score
                info["decision_quality"] = self._get_decision_quality(final_score)
                info["final_explanation"] = (
                    f"Final score {final_score:.2f} — "
                    f"{info['decision_quality']} quality hiring decisions."
                )
        else:
            raise Exception(f"Unknown task_type: {task}")

        self.last_action = action

        # For offer task, return the dict with negotiation_hints
        if task == "offer":
            return state, rw, done, info
        return self.state, rw, done, info

    # -----------------------------------------------------------------------
    # COMPUTE FINAL SCORE — branches by task_type, always 0.0–1.0
    # -----------------------------------------------------------------------
    def compute_final_score(self) -> float:
        if self.state is None:
            return 0.0

        task = self.state.task_type

        if task == "resume":
            result = resume.score(
                self.state, self.correct_shortlist, self.max_steps
            )
            # resume.score returns (score, bias_result) tuple
            score_val, bias_result = result
            self._last_bias_explanation = bias_result["explanation"]
            return score_val
        elif task == "offer":
            return offer.score(
                self.state, self.correct_shortlist, self.max_steps
            )
        elif task == "communication":
            return communication.score(
                self.state, self.correct_shortlist, self.max_steps
            )
        else:
            return 0.0

    # -----------------------------------------------------------------------
    # STATE VIEW
    # -----------------------------------------------------------------------
    def state_view(self):
        if self.state is None:
            return None

        task = self.state.task_type

        if task == "resume":
            episode_done = (
                len(self.state.shortlisted) >= 3
                or self.state.step_count >= self.max_steps
            )
        elif task == "offer":
            episode_done = (
                self.state.step_count >= self.max_steps
                or len(self.state.offers_made or []) >= len(self.state.candidates)
            )
        elif task == "communication":
            episode_done = (
                self.state.step_count >= self.max_steps
                or len(self.state.emails_sent or []) >= len(self.state.candidates)
            )
        else:
            episode_done = False

        if episode_done:
            self.state.counterfactual = self._build_counterfactual()
        else:
            self.state.counterfactual = None

        if self.state.task_type == "offer":
            state_dict = self.state.model_dump()
            state_dict["negotiation_hints"] = self.negotiation_hints
            return state_dict
        return self.state.model_dump()

    # -----------------------------------------------------------------------
    # DECISION QUALITY — mapped from final episode score
    # -----------------------------------------------------------------------
    def _get_decision_quality(self, final_score: float) -> str:
        if final_score >= 0.75:
            return "high"
        elif final_score >= 0.45:
            return "medium"
        else:
            return "low"

    # -----------------------------------------------------------------------
    # COUNTERFACTUAL AUDIT
    # -----------------------------------------------------------------------
    def _build_counterfactual(self) -> Dict:
        """
        Builds a counterfactual audit showing what the optimal agent
        would have done differently. Only meaningful for resume and offer tasks.
        """
        task = self.state.task_type
        agent_picks = list(self.state.shortlisted)

        if task == "resume":
            optimal = self.correct_shortlist
            correct_hits = list(set(agent_picks) & set(optimal))
            missed = list(set(optimal) - set(agent_picks))
            unnecessary = list(set(agent_picks) - set(optimal))

            def candidate_name(cid):
                c = next((x for x in self.state.candidates if x.id == cid), None)
                return c.name if c else cid

            if len(correct_hits) == len(optimal):
                verdict = "Perfect selection. Agent picked all optimal candidates."
            else:
                missed_names = [candidate_name(c) for c in missed]
                unnecessary_names = [candidate_name(c) for c in unnecessary]
                verdict_parts = []
                if missed_names:
                    verdict_parts.append(f"missed {', '.join(missed_names)}")
                if unnecessary_names:
                    verdict_parts.append(f"unnecessarily picked {', '.join(unnecessary_names)}")
                verdict = (
                    f"Agent found {len(correct_hits)}/{len(optimal)} "
                    f"correct candidates. " + "; ".join(verdict_parts) + "."
                )

            return {
                "optimal_picks": optimal,
                "agent_picks": agent_picks,
                "correct_hits": correct_hits,
                "missed": missed,
                "unnecessary": unnecessary,
                "verdict": verdict,
            }

        elif task == "offer":
            optimal = self.correct_shortlist
            correct_hits = list(set(agent_picks) & set(optimal))
            missed = list(set(optimal) - set(agent_picks))
            unnecessary = list(set(agent_picks) - set(optimal))

            total_spend = sum(
                c.expected_salary for c in self.state.candidates
                if c.id in agent_picks
            )
            budget_status = (
                "within budget"
                if total_spend <= (self.state.budget or 0)
                else f"over budget by {total_spend - (self.state.budget or 0)}"
            )

            return {
                "optimal_picks": optimal,
                "agent_picks": agent_picks,
                "correct_hits": correct_hits,
                "missed": missed,
                "unnecessary": unnecessary,
                "total_spend": total_spend,
                "budget": self.state.budget,
                "budget_status": budget_status,
                "verdict": (
                    f"Agent made {len(correct_hits)}/{len(optimal)} optimal offers. "
                    f"Total spend: {total_spend} — {budget_status}."
                ),
            }

        elif task == "communication":
            emails_sent = [e["candidate_id"] for e in (self.state.emails_sent or [])]
            all_ids = [c.id for c in self.state.candidates]
            missed_emails = list(set(all_ids) - set(emails_sent))
            adv_handled = "adv1" in emails_sent

            return {
                "emails_sent": emails_sent,
                "missed_emails": missed_emails,
                "adversarial_handled": adv_handled,
                "coverage": f"{len(emails_sent)}/{len(all_ids)}",
                "verdict": (
                    f"Agent emailed {len(emails_sent)}/{len(all_ids)} candidates. "
                    f"Adversarial candidate handled: {adv_handled}. "
                    + (
                        f"Missed candidates: {missed_emails}."
                        if missed_emails else
                        "Full coverage achieved."
                    )
                ),
            }

        return {}
