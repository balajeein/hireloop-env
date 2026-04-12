"""
HireLoop Environment — OpenEnv-Core Compliant Orchestrator
============================================================
Inherits from openenv_core Environment base class.
Delegates task-specific logic to hireloop.tasks.{resume, offer, communication}.
"""

from typing import List, Dict, Optional
import random
import json
import os
from models import (
    HireLoopAction, HireLoopObservation, HireLoopState,
    BaseState,
)

# Import Environment base class — real or shim
try:
    from openenv_core.env_server.interfaces import Environment
except (ImportError, TypeError):
    from abc import ABC, abstractmethod

    class Environment(ABC):
        """Shim for openenv_core.env_server.interfaces.Environment"""
        def __init__(self, transform=None):
            self.transform = transform

        @abstractmethod
        def reset(self) -> object:
            pass

        @abstractmethod
        def step(self, action) -> object:
            pass

        @property
        @abstractmethod
        def state(self) -> object:
            pass

        def _apply_transform(self, observation):
            if self.transform is not None:
                return self.transform(observation)
            return observation


from tasks import resume, offer, communication
from utils.skills import check_negotiation_eligibility

MIN_STRICT_SCORE = 0.0001
MAX_STRICT_SCORE = 0.9999


class HireLoopEnv(Environment):
    """
    HireLoop: Multi-step hiring pipeline RL environment.
    Tests agents on fairness, budget reasoning, safety, and adversarial robustness.

    Compliant with openenv-core Environment interface:
    - reset() -> HireLoopObservation
    - step(action) -> HireLoopObservation
    - state -> BaseState
    """

    # Tells openenv-core that this env supports concurrent sessions
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._state: Optional[HireLoopState] = None
        self.max_steps = 10
        self.correct_shortlist: List[str] = []
        self.last_action = None
        self.random_seed = 42
        self.rng = random.Random(self.random_seed)
        self.negotiation_hints: dict = {}
        self.current_scenario_id: str = ""
        self._last_bias_explanation: str = "Bias check: not run yet."
        self._episode_id: Optional[str] = None

        # Load scenarios from JSON file
        scenarios_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "scenarios.json"
        )
        with open(scenarios_path, "r") as f:
            self.scenarios = json.load(f)



    @property
    def state(self) -> BaseState:
        """Required abstract property from Environment base class."""
        return BaseState(
            episode_id=self._episode_id,
            step_count=self._state.step_count if self._state else 0,
        )

    def reset(self, **kwargs) -> HireLoopObservation:
        """
        Reset the environment (openenv-compliant).

        Kwargs:
            task: str — "resume", "offer", or "communication"
            seed: int — optional RNG seed
            episode_id: str — optional episode identifier
        """
        # Handle openenv kwargs
        seed = kwargs.get("seed")
        if seed is not None:
            self.rng = random.Random(seed)

        self._episode_id = kwargs.get("episode_id")

        task_type = kwargs.get("task")
        if task_type and task_type in ("resume", "offer", "communication"):
            return self.reset_with_task(task_type)
        task_type = self.rng.choice(["resume", "offer", "communication"])
        return self.reset_with_task(task_type)

    def step(self, action, **kwargs) -> HireLoopObservation:
        """
        Take a step in the environment (openenv-compliant).

        Accepts either a HireLoopAction dataclass or a plain dict.
        Returns a HireLoopObservation with done, reward, metadata fields.
        """
        if self._state is None:
            raise Exception("Environment not initialized. Call reset() first.")

        # Support both HireLoopAction dataclass and raw dict
        if isinstance(action, HireLoopAction):
            action_dict = {
                "type": action.type,
                "candidate_id": action.candidate_id,
            }
            if action.content is not None:
                action_dict["content"] = action.content
        elif isinstance(action, dict):
            action_dict = action
        else:
            # Fallback: try to convert to dict
            if hasattr(action, 'model_dump'):
                action_dict = action.model_dump()
            else:
                action_dict = vars(action) if hasattr(action, '__dict__') else {}

        task = self._state.task_type
        rw = 0.0
        done = False
        info = {}

        if task == "resume":
            new_state, rw, done, info = resume.step(
                self._state, action_dict, self.correct_shortlist,
                self.last_action, self.max_steps
            )
            self._state = new_state
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
                self._state, action_dict, self.correct_shortlist,
                self.last_action, self.max_steps, self.negotiation_hints
            )
            obs, rw, done, info = result
            # offer.step returns state_dict when successful
            if isinstance(obs, dict):
                self._state = HireLoopState(**{
                    k: v for k, v in obs.items()
                    if k != "negotiation_hints"
                })
            else:
                self._state = obs
            if done:
                final_score = self.compute_final_score()
                info["final_score"] = final_score
                info["decision_quality"] = self._get_decision_quality(final_score)
                info["final_explanation"] = (
                    f"Final score {final_score:.2f} — "
                    f"{info['decision_quality']} quality hiring decisions."
                )

        elif task == "communication":
            new_state, rw, done, info = communication.step(
                self._state, action_dict, self.correct_shortlist,
                self.last_action, self.max_steps
            )
            self._state = new_state
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

        self.last_action = action_dict

        # Return openenv-compliant Observation
        obs = self._build_observation(reward=rw, done=done, info=info)
        return self._apply_transform(obs)



    def reset_with_task(self, task_type: str) -> HireLoopObservation:
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

        self._state = state
        self.correct_shortlist = cs
        self.max_steps = ms
        self.current_scenario_id = sid
        self.last_action = None

        return self._build_observation(reward=None, done=False, info={})

    def compute_final_score(self) -> float:
        """Compute the final episode score (0.0–1.0)."""
        if self._state is None:
            return MIN_STRICT_SCORE

        task = self._state.task_type

        if task == "resume":
            result = resume.score(
                self._state, self.correct_shortlist, self.max_steps
            )
            # resume.score returns (score, bias_result) tuple
            score_val, bias_result = result
            self._last_bias_explanation = bias_result["explanation"]
            return round(min(MAX_STRICT_SCORE, max(MIN_STRICT_SCORE, score_val)), 4)
        elif task == "offer":
            score = offer.score(
                self._state, self.correct_shortlist, self.max_steps
            )
            return round(min(MAX_STRICT_SCORE, max(MIN_STRICT_SCORE, score)), 4)
        elif task == "communication":
            score = communication.score(
                self._state, self.correct_shortlist, self.max_steps
            )
            return round(min(MAX_STRICT_SCORE, max(MIN_STRICT_SCORE, score)), 4)
        else:
            return MIN_STRICT_SCORE


    def state_view(self):
        """Legacy method for /state endpoint. Returns raw dict."""
        if self._state is None:
            return None

        task = self._state.task_type

        if task == "resume":
            episode_done = (
                len(self._state.shortlisted) >= 3
                or self._state.step_count >= self.max_steps
            )
        elif task == "offer":
            episode_done = (
                self._state.step_count >= self.max_steps
                or len(self._state.offers_made or []) >= len(self._state.candidates)
            )
        elif task == "communication":
            episode_done = (
                self._state.step_count >= self.max_steps
                or len(self._state.emails_sent or []) >= len(self._state.candidates)
            )
        else:
            episode_done = False

        if episode_done:
            self._state.counterfactual = self._build_counterfactual()
        else:
            self._state.counterfactual = None

        if self._state.task_type == "offer":
            state_dict = self._state.model_dump()
            state_dict["negotiation_hints"] = self.negotiation_hints
            return state_dict
        return self._state.model_dump()


    def _get_decision_quality(self, final_score: float) -> str:
        if final_score >= 0.75:
            return "high"
        elif final_score >= 0.45:
            return "medium"
        else:
            return "low"


    def _build_observation(self, reward, done, info) -> HireLoopObservation:
        """Convert internal HireLoopState → HireLoopObservation (openenv format)."""
        if self._state is None:
            return HireLoopObservation(done=done, reward=reward, metadata=info)

        state_dict = self._state.model_dump()

        # Candidates and job_description: convert Pydantic to plain dicts
        candidates = [c.model_dump() for c in self._state.candidates]
        job_desc = self._state.job_description.model_dump()

        return HireLoopObservation(
            done=done,
            reward=reward,
            metadata=info,
            job_description=job_desc,
            candidates=candidates,
            shortlisted=list(self._state.shortlisted),
            rejected=list(self._state.rejected),
            step_count=self._state.step_count,
            task_type=self._state.task_type,
            budget=self._state.budget,
            offers_made=self._state.offers_made,
            emails_sent=self._state.emails_sent,
            counterfactual=self._state.counterfactual,
            negotiation_hints=self.negotiation_hints if self._state.task_type == "offer" else None,
        )


    def _build_counterfactual(self) -> Dict:
        """
        Builds a counterfactual audit showing what the optimal agent
        would have done differently. Only meaningful for resume and offer tasks.
        """
        task = self._state.task_type
        agent_picks = list(self._state.shortlisted)

        if task == "resume":
            optimal = self.correct_shortlist
            correct_hits = list(set(agent_picks) & set(optimal))
            missed = list(set(optimal) - set(agent_picks))
            unnecessary = list(set(agent_picks) - set(optimal))

            def candidate_name(cid):
                c = next((x for x in self._state.candidates if x.id == cid), None)
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
                c.expected_salary for c in self._state.candidates
                if c.id in agent_picks
            )
            budget_status = (
                "within budget"
                if total_spend <= (self._state.budget or 0)
                else f"over budget by {total_spend - (self._state.budget or 0)}"
            )

            return {
                "optimal_picks": optimal,
                "agent_picks": agent_picks,
                "correct_hits": correct_hits,
                "missed": missed,
                "unnecessary": unnecessary,
                "total_spend": total_spend,
                "budget": self._state.budget,
                "budget_status": budget_status,
                "verdict": (
                    f"Agent made {len(correct_hits)}/{len(optimal)} optimal offers. "
                    f"Total spend: {total_spend} — {budget_status}."
                ),
            }

        elif task == "communication":
            emails_sent = [e["candidate_id"] for e in (self._state.emails_sent or [])]
            all_ids = [c.id for c in self._state.candidates]
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
