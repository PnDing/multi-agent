"""Adaptive strategy optimizer for the generator agent."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import inf
from typing import Dict, List, Optional, Sequence


DEFAULT_GENERATOR_STRATEGIES: Sequence[str] = (
    "temporal_transplant",
    "truth_mixing",
    "authority_hijacking",
    "title_anchoring",
    "hotspot_mirroring",
    "evidence_pollution",
    "semantic_poisoning",
    "preset_traps",
    "double_standard_narrative",
)


@dataclass(slots=True)
class GeneratorStrategyStats:
    attempts: int = 0
    successes: int = 0

    def register(self, success: bool) -> None:
        self.attempts += 1
        if success:
            self.successes += 1

    @property
    def success_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts


class GeneratorStrategyOptimizer:
    """Simple explore-exploit selector for generator strategies."""

    def __init__(
        self,
        *,
        strategies: Optional[Sequence[str]] = None,
        exploration_weight: float = 0.5,
    ) -> None:
        self._strategies = list(strategies or DEFAULT_GENERATOR_STRATEGIES)
        self._stats: Dict[str, GeneratorStrategyStats] = defaultdict(GeneratorStrategyStats)
        self._exploration_weight = max(0.0, exploration_weight)
        self.history: List[Dict[str, float]] = []

    def update(self, strategy: str, success: bool) -> None:
        if strategy not in self._strategies:
            self._strategies.append(strategy)
        stats = self._stats[strategy]
        stats.register(success)
        self.history.append({strategy: 1.0 if success else 0.0})

    def suggest(self) -> str:
        if not self._strategies:
            return "temporal_transplant"
        untried = [s for s in self._strategies if self._stats[s].attempts == 0]
        if untried:
            return untried[0]
        return max(self._strategies, key=self._score)

    def _score(self, strategy: str) -> float:
        stats = self._stats[strategy]
        if stats.attempts == 0:
            return inf
        exploration_bonus = self._exploration_weight / stats.attempts
        return stats.success_rate + exploration_bonus
