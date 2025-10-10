"""Adaptive strategy optimizer for the detector agent."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import inf
from typing import Dict, List, Optional, Sequence


DEFAULT_DETECTOR_STRATEGIES: Sequence[str] = (
    "multi-modal_cross_check",
    "authority_verification",
    "source_consensus",
    "temporal_consistency",
)


@dataclass(slots=True)
class DetectorStrategyStats:
    attempts: int = 0
    cumulative_score: float = 0.0

    def register(self, score: float) -> None:
        self.attempts += 1
        self.cumulative_score += score

    @property
    def average_score(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.cumulative_score / self.attempts


class DetectorStrategyOptimizer:
    """Tracks strategy performance and suggests the most promising option."""

    def __init__(
        self,
        *,
        strategies: Optional[Sequence[str]] = None,
        exploration_weight: float = 50.0,
    ) -> None:
        self._strategies = list(strategies or DEFAULT_DETECTOR_STRATEGIES)
        self._stats: Dict[str, DetectorStrategyStats] = defaultdict(DetectorStrategyStats)
        self._exploration_weight = max(0.0, exploration_weight)
        self.history: List[Dict[str, float]] = []

    def update(self, strategy: str, accuracy: float) -> None:
        if strategy not in self._strategies:
            self._strategies.append(strategy)
        stats = self._stats[strategy]
        stats.register(accuracy)
        self.history.append({strategy: accuracy})

    def suggest(self) -> str:
        if not self._strategies:
            return "multi-modal_cross_check"
        untried = [s for s in self._strategies if self._stats[s].attempts == 0]
        if untried:
            return untried[0]
        return max(self._strategies, key=self._score)

    def _score(self, strategy: str) -> float:
        stats = self._stats[strategy]
        if stats.attempts == 0:
            return inf
        exploration_bonus = self._exploration_weight / stats.attempts
        return stats.average_score + exploration_bonus
