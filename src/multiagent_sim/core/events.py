"""Event definitions driving the simulation workflow."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(slots=True)
class GenerationEvent:
    news_id: str
    generator_id: str
    strategy_hint: str | None


@dataclass(slots=True)
class PropagationEvent:
    news_id: str
    sender_id: str
    receivers: Sequence[str]


@dataclass(slots=True)
class DetectionEvent:
    news_id: str
    detector_id: str
    opinions: List[str]
