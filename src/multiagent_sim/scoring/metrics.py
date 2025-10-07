"""Scoring utilities for generator and detector."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class GeneratorScore:
    total: float
    detection_evasion: float
    spread: float


@dataclass(slots=True)
class DetectorScore:
    total: float
    accuracy: float
    spread_incentive: float


@dataclass(slots=True)
class RoundScore:
    generator: GeneratorScore
    detector: DetectorScore
    notes: Optional[str] = None


class ScoreCalculator:
    """Implements the scoring rules from score_method.md."""

    def score(
        self,
        *,
        detector_label: int,
        ground_truth: int,
        infected_users: int,
        total_users: int,
        detector_confidence: float = 0.0,
    ) -> RoundScore:
        if total_users <= 0:
            raise ValueError("total_users must be positive")

        spread_ratio = infected_users / total_users
        spread_score = spread_ratio * 100.0

        detection_evasion = 100.0 if detector_label != ground_truth else 0.0
        generator_total = (detection_evasion + spread_score) / 2.0

        accuracy_score = 100.0 if detector_label == ground_truth else 0.0
        spread_incentive = (1 - 2 * ground_truth) * spread_score
        detector_total = (accuracy_score + spread_incentive) / 2.0

        notes = None
        if detector_confidence:
            notes = f"confidence={detector_confidence:.2f}" if detector_confidence else None

        generator = GeneratorScore(
            total=generator_total,
            detection_evasion=detection_evasion,
            spread=spread_score,
        )
        detector = DetectorScore(
            total=detector_total,
            accuracy=accuracy_score,
            spread_incentive=spread_incentive,
        )
        return RoundScore(generator=generator, detector=detector, notes=notes)
