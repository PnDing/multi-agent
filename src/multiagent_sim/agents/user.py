"""User agent logic for community-level simulation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

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

    def receive_news(self, news_id: str, content: str, belief_delta: float) -> None:
        self.state.inbox.append(news_id)
        self.state.belief.belief_strength = max(0.0, min(1.0, self.state.belief.belief_strength + belief_delta))

    def ready_to_propagate(self) -> bool:
        return self.state.belief.ready_to_propagate()

    def propagate_targets(self, neighbours: Iterable[str]) -> List[str]:
        if not self.ready_to_propagate():
            return []
        return list(neighbours)

    def reinforce(self, experience: Experience) -> None:
        self._experience_repo.add(self.state.persona.user_id, experience)
        self.state.belief.record_experience(experience.rationale)

    def recall_experience(self, limit: int = 3) -> List[Experience]:
        return self._experience_repo.latest(self.state.persona.user_id, limit=limit)
