"""Domain objects representing news items."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(slots=True)
class NewsItem:
    news_id: str
    original_text: str
    generated_text: Optional[str] = None
    operation_log: Optional[str] = None
    factuality_label: Optional[int] = None
    original_truth_label: Optional[int] = None
    rewrite_applied: Optional[bool] = None
    opinions: Sequence[str] | None = None
    generator_strategy: Optional[str] = None
    detector_strategy: Optional[str] = None
    detector_operation_log: Optional[str] = None
    generator_sources: Optional[List[str]] = None
    propagation_history: Optional[List[List[str]]] = None
    infected_users: Optional[List[str]] = None
    raw_generation: Optional[str] = None
    raw_detection: Optional[str] = None


