"""Reusable persona attribute pools and personality presets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple


CANDIDATE_NAMES = (
    "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy",
    "Mallory", "Niaj", "Olivia", "Peggy", "Rupert", "Sybil", "Trent", "Victor", "Walter", "Yolanda",
)

GENDER_OPTIONS = ("female", "male")

AGE_BUCKETS = (list(range(22, 31)), list(range(31, 41)), list(range(41, 51)))

CITY_OPTIONS = (
    "Beijing", "Shanghai", "Shenzhen", "Guangzhou", "Chengdu", "Hangzhou", "Wuhan", "Nanjing", "Xi'an", "Chongqing",
)

OCCUPATION_OPTIONS = (
    "Journalist", "Engineer", "Teacher", "Data Analyst", "Content Creator", "Healthcare Worker",
    "Researcher", "Marketing Manager", "Product Manager", "Financial Analyst",
)

EDUCATION_OPTIONS = ("Bachelor", "Master", "PhD")

INTEREST_POOL = (
    "technology", "finance", "sports", "healthcare", "travel", "photography", "cooking", "education", "politics", "science",
    "music", "film", "gaming", "environment", "startups", "literature", "art", "investment", "history", "parenting",
)

MEDIA_PREFERENCES_POOL = (
    "newspaper", "television", "podcast", "social_media", "short_video", "forum", "radio", "newsletter",
)


@dataclass(frozen=True)
class PersonalityRange:
    low: float
    high: float

    def sample(self, *, rng) -> float:
        return rng.uniform(self.low, self.high)


PERSONALITY_ARCHETYPES: Dict[str, Dict[str, PersonalityRange]] = {
    "skeptic": {
        "openness": PersonalityRange(0.2, 0.5),
        "conscientiousness": PersonalityRange(0.6, 0.9),
        "extraversion": PersonalityRange(0.2, 0.5),
        "agreeableness": PersonalityRange(0.3, 0.6),
        "neuroticism": PersonalityRange(0.4, 0.7),
    },
    "advocate": {
        "openness": PersonalityRange(0.6, 0.9),
        "conscientiousness": PersonalityRange(0.4, 0.7),
        "extraversion": PersonalityRange(0.6, 0.9),
        "agreeableness": PersonalityRange(0.6, 0.9),
        "neuroticism": PersonalityRange(0.2, 0.5),
    },
    "broadcaster": {
        "openness": PersonalityRange(0.5, 0.8),
        "conscientiousness": PersonalityRange(0.3, 0.6),
        "extraversion": PersonalityRange(0.7, 1.0),
        "agreeableness": PersonalityRange(0.4, 0.7),
        "neuroticism": PersonalityRange(0.3, 0.6),
    },
}

ARCHETYPE_DISTRIBUTION: Tuple[str, ...] = (
    "skeptic", "advocate", "broadcaster", "advocate", "skeptic", "broadcaster", "advocate", "skeptic",
)


def sample_age(*, rng) -> int:
    bucket = rng.choice(AGE_BUCKETS)
    return rng.choice(bucket)


def sample_interests(*, rng, k: int = 3) -> Sequence[str]:
    return tuple(rng.sample(INTEREST_POOL, k=k))


def sample_media_preferences(*, rng, k: int = 2) -> Sequence[str]:
    return tuple(rng.sample(MEDIA_PREFERENCES_POOL, k=k))


def sample_personality(archetype: str, *, rng) -> Dict[str, float]:
    ranges = PERSONALITY_ARCHETYPES[archetype]
    return {trait: rng.uniform(r.low, r.high) for trait, r in ranges.items()}
