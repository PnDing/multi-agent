"""User persona definitions for the community simulation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence


@dataclass(slots=True)
class Persona:
    """Static attributes describing a user agent."""

    user_id: str
    name: str
    gender: str
    age: int
    location: str
    occupation: str
    education: str
    interests: Sequence[str] = field(default_factory=tuple)
    media_preferences: Sequence[str] = field(default_factory=tuple)
    political_tendency: str = "neutral"


@dataclass(slots=True)
class BeliefProfile:
    """Dynamic state that evolves with the simulation."""

    belief_strength: float = 0.0
    skepticism: float = 0.5
    propagation_threshold: float = 1.0
    experience_memory: List[str] = field(default_factory=list)
    trust_scores: Dict[str, float] = field(default_factory=dict)

    def update_trust(self, source: str, delta: float) -> None:
        self.trust_scores[source] = max(0.0, min(1.0, self.trust_scores.get(source, 0.5) + delta))

    def record_experience(self, summary: str) -> None:
        self.experience_memory.append(summary)

    def increase_belief(self, delta: float) -> None:
        self.belief_strength = max(0.0, min(1.0, self.belief_strength + delta))

    def set_belief(self, value: float) -> None:
        self.belief_strength = max(0.0, min(1.0, value))

    def ready_to_propagate(self) -> bool:
        return self.belief_strength >= self.propagation_threshold

    def integrate_detection_feedback(self, verdict: int) -> None:
        if verdict == 1:
            self.belief_strength = max(0.0, self.belief_strength - 0.5)
            self.propagation_threshold = min(1.0, self.propagation_threshold + 0.1)
        else:
            self.belief_strength = min(1.0, self.belief_strength + 0.2)
            self.propagation_threshold = max(0.3, self.propagation_threshold - 0.1)
