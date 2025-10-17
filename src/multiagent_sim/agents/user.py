"""User agent logic for community-level simulation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional
import random

from ..community.persona import BeliefProfile, Persona
from ..community.experience import Experience, ExperienceRepository


@dataclass(slots=True)
class UserAgentState:
    persona: Persona
    belief: BeliefProfile
    inbox: List[str] = field(default_factory=list)


class UserAgent:
    """Captures how a community member consumes and propagates news."""

    def __init__(self, persona: Persona, belief: Optional[BeliefProfile] = None, *, experience_repo: Optional[ExperienceRepository] = None) -> None:
        self.state = UserAgentState(persona=persona, belief=belief or BeliefProfile())
        self._experience_repo = experience_repo or ExperienceRepository()
        self._rng = random.Random(hash(persona.user_id) & 0xFFFFFFFF)

    def receive_news(self, news_id: str, content: str, belief_delta: float) -> None:
        persona = self.state.persona
        if news_id not in self.state.inbox:
            self.state.inbox.append(news_id)
        openness = getattr(persona, "openness", 0.5)
        conscientiousness = getattr(persona, "conscientiousness", 0.5)
        agreeableness = getattr(persona, "agreeableness", 0.5)
        neuroticism = getattr(persona, "neuroticism", 0.5)
        skepticism = self.state.belief.skepticism
        modifier = 0.6 + 0.4 * openness - 0.3 * conscientiousness + 0.25 * neuroticism + 0.15 * (agreeableness - 0.5)
        modifier *= 1.0 - 0.4 * skepticism
        modifier = max(0.15, min(1.75, modifier))
        adjusted_delta = belief_delta * modifier
        self.state.belief.increase_belief(adjusted_delta)

    def seed_news(self, news_id: str, content: str) -> None:
        if news_id not in self.state.inbox:
            self.state.inbox.append(news_id)
        baseline = 0.8 + getattr(self.state.persona, "extraversion", 0.5) * 0.2
        self.state.belief.set_belief(min(1.0, max(0.95, baseline)))

    def ready_to_propagate(self) -> bool:
        return self.state.belief.ready_to_propagate()

    def propagate_targets(self, neighbours: Iterable[str]) -> List[str]:
        if not self.ready_to_propagate():
            return []
        neighbour_list = list(neighbours)
        if not neighbour_list:
            return []
        persona = self.state.persona
        extraversion = getattr(persona, "extraversion", 0.5)
        conscientiousness = getattr(persona, "conscientiousness", 0.5)
        agreeableness = getattr(persona, "agreeableness", 0.5)
        share_ratio = 0.25 + extraversion * 0.6 - conscientiousness * 0.2 + (agreeableness - 0.5) * 0.1
        share_ratio = max(0.1, min(1.0, share_ratio))
        share_count = max(1, int(round(len(neighbour_list) * share_ratio)))
        self._rng.shuffle(neighbour_list)
        selected = neighbour_list[:share_count]
        if agreeableness <= 0.3 and len(selected) > 1:
            selected = selected[: max(1, int(len(selected) * 0.7))]
        return selected

    def apply_detection_feedback(self, news_id: str, verdict: int) -> None:
        persona = self.state.persona
        self.state.belief.integrate_detection_feedback(verdict)
        adjust = 0.08 + getattr(persona, "conscientiousness", 0.5) * 0.18
        if verdict == 1:
            self.state.belief.belief_strength = max(0.0, self.state.belief.belief_strength - adjust)
            threshold_boost = adjust * (0.5 + getattr(persona, "neuroticism", 0.5) * 0.5)
            self.state.belief.propagation_threshold = min(1.3, self.state.belief.propagation_threshold + threshold_boost)
            if news_id in self.state.inbox:
                self.state.inbox.remove(news_id)
        else:
            reward = adjust * (0.4 + getattr(persona, "agreeableness", 0.5) * 0.3)
            self.state.belief.belief_strength = min(1.0, self.state.belief.belief_strength + reward)
            self.state.belief.propagation_threshold = max(0.3, self.state.belief.propagation_threshold - reward * (0.5 + getattr(persona, "extraversion", 0.5) * 0.4))

    def reinforce(self, experience: Experience) -> None:
        self._experience_repo.add(self.state.persona.user_id, experience)
        self.state.belief.record_experience(experience.rationale)

    def recall_experience(self, limit: int = 3) -> List[Experience]:
        return self._experience_repo.latest(self.state.persona.user_id, limit=limit)
