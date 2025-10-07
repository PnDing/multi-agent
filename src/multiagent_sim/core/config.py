"""Global configuration holder for the simulation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SimulationConfig:
    max_rounds: int = 1
    seed_user_count: int = 1
    enable_tooling: bool = True
