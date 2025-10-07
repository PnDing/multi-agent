"""Simulation state container."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from ..community.graph import CommunityGraph
from ..community.persona import Persona


@dataclass(slots=True)
class SimulationState:
    news_registry: Dict[str, dict] = field(default_factory=dict)
    personas: Dict[str, Persona] = field(default_factory=dict)
    graph: CommunityGraph = field(default_factory=CommunityGraph)
    round_index: int = 0

    def register_news(self, news_id: str, payload: dict) -> None:
        self.news_registry[news_id] = payload

    def next_round(self) -> int:
        self.round_index += 1
        return self.round_index
