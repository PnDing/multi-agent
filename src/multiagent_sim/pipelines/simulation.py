"""High-level builder utilities for running the simulation."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

from ..agents.detector import DetectorAgent, DetectorAgentConfig
from ..agents.factory import AgentFactory
from ..agents.generator import GeneratorAgent, GeneratorAgentConfig
from ..agents.user import UserAgent
from ..community.graph import CommunityGraph
from ..community.persona import BeliefProfile, Persona
from ..core.news import NewsItem
from ..core.orchestrator import OpinionCallback, SimulationOrchestrator
from ..optimization.detector_strategy_agent import DetectorStrategyAgent
from ..optimization.generator_optimizer import GeneratorStrategyOptimizer
from ..scoring.metrics import ScoreCalculator




def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _initial_belief_from_persona(persona: Persona) -> BeliefProfile:
    base_belief = _clamp(0.1 + persona.openness * 0.3 + persona.extraversion * 0.2 - persona.conscientiousness * 0.2, 0.0, 0.7)
    skepticism = _clamp(0.45 + (0.5 - persona.openness) * 0.35 + (0.5 - persona.agreeableness) * 0.25 + persona.neuroticism * 0.25, 0.05, 0.95)
    threshold = _clamp(0.85 + (0.5 - persona.extraversion) * 0.5 + persona.conscientiousness * 0.3 - persona.agreeableness * 0.1, 0.4, 1.2)
    return BeliefProfile(
        belief_strength=base_belief,
        skepticism=skepticism,
        propagation_threshold=threshold,
    )


def build_default_orchestrator(
    personas: Sequence[Persona],
    edges: Iterable[tuple[str, str]],
    *,
    generator_config: Optional[GeneratorAgentConfig] = None,
    detector_config: Optional[DetectorAgentConfig] = None,
    opinion_callback: Optional[OpinionCallback] = None,
    score_calculator: Optional[ScoreCalculator] = None,
    propagation_rounds: int = 2,
    generator_optimizer: Optional[GeneratorStrategyOptimizer] = None,
    detector_strategy_agent: Optional[DetectorStrategyAgent] = None,
) -> SimulationOrchestrator:
    generator = AgentFactory.build_generator(generator_config)
    detector = AgentFactory.build_detector(detector_config)
    community = {
        persona.user_id: UserAgent(persona, belief=_initial_belief_from_persona(persona))
        for persona in personas
    }
    graph = CommunityGraph()
    graph.ingest_edges(edges)
    return SimulationOrchestrator(
        generator=generator,
        detector=detector,
        community=community,
        graph=graph,
        score_calculator=score_calculator,
        opinion_callback=opinion_callback,
        propagation_rounds=propagation_rounds,
        generator_optimizer=generator_optimizer or GeneratorStrategyOptimizer(),
        detector_strategy_agent=detector_strategy_agent or DetectorStrategyAgent(),
    )


def default_opinion_callback(user: UserAgent, news: NewsItem) -> str:
    experiences = user.recall_experience(limit=1)
    experience_hint = experiences[0].rationale if experiences else "No prior experience"
    return f"{user.state.persona.name} believes {news.news_id} due to {experience_hint}"

# --- Added by patch: belief-based, reasoned opinion callbacks ---
import re
from typing import Tuple


def _current_reasons_for_user(user: UserAgent, news: NewsItem) -> Tuple[list[str], list[str]]:
    """Generate current-round supporting/opposing reasons based on news text, persona, and propagation depth.
    Does not read historical memory to keep opinions timely and grounded in the current round.
    """
    text = (news.generated_text or news.original_text or "") or ""
    low = text.lower()
    persona = user.state.persona

    pos: list[str] = []
    neg: list[str] = []

    # Relevance: location / occupation / interests
    if getattr(persona, "location", None) and persona.location.lower() in low:
        pos.append(f"it references my city ({persona.location})")
    else:
        neg.append("no direct relevance to my location")

    if getattr(persona, "occupation", None) and persona.occupation.lower() in low:
        pos.append(f"it concerns my profession ({persona.occupation})")
    else:
        neg.append("not related to my professional domain")

    interests = getattr(persona, "interests", ()) or ()
    hits = [kw for kw in interests if isinstance(kw, str) and kw and kw.lower() in low]
    if hits:
        pos.append(f"it matches my interests ({', '.join(hits[:3])})")
    else:
        neg.append("does not match my personal interests")

    # Authority / specificity heuristics
    authority_tokens = ("according to", "report", "study", "expert", "research", "who", "cdc")
    has_authority = any(tok in low for tok in authority_tokens) or ('"' in text)
    if has_authority:
        pos.append("it cites authorities or includes quotes")
    else:
        neg.append("no verifiable source or quote is cited")

    has_numbers = bool(re.search(r"\d{2,}", text))
    if has_numbers:
        pos.append("it provides concrete figures/dates")
    else:
        neg.append("lacks concrete numbers or dates")

    # Propagation depth: early exposure feels closer to source; many hops may distort
    uid = persona.user_id
    depth = None
    if news.propagation_history:
        for i, layer in enumerate(news.propagation_history):
            if uid in layer:
                depth = i
                break
    if depth is not None:
        if depth <= 1:
            pos.append("I saw it early in the cascade (direct/nearby source)")
        else:
            neg.append("I only saw it after multiple hops (possible distortion)")
    else:
        neg.append("unclear how the news reached me")

    # Personality-driven heuristics
    if getattr(persona, "openness", 0.5) >= 0.7:
        pos.append("I enjoy exploring novel ideas")
    elif getattr(persona, "openness", 0.5) <= 0.3:
        neg.append("too unconventional for my taste")

    if getattr(persona, "conscientiousness", 0.5) >= 0.7:
        neg.append("I prefer to verify facts before accepting them")
    elif getattr(persona, "conscientiousness", 0.5) <= 0.3:
        pos.append("I trust my intuition even without full evidence")

    if getattr(persona, "extraversion", 0.5) >= 0.7:
        pos.append("people around me are excited about it")
    elif getattr(persona, "extraversion", 0.5) <= 0.3:
        neg.append("I tend to observe quietly before engaging")

    if getattr(persona, "agreeableness", 0.5) >= 0.7:
        pos.append("I want to align with others to maintain harmony")
    elif getattr(persona, "agreeableness", 0.5) <= 0.3:
        neg.append("I naturally question others' intentions")

    if getattr(persona, "neuroticism", 0.5) >= 0.7:
        pos.append("it triggers serious concerns I cannot ignore")
    elif getattr(persona, "neuroticism", 0.5) <= 0.3:
        neg.append("it does not worry me enough to act")

    role = getattr(persona, "role", "bystander")
    if role == "broadcaster":
        pos.append("amplifying the story keeps everyone alert")
    elif role == "commentator":
        pos.append("I want to share my take on this news")
    elif role == "verifier":
        neg.append("I prefer to verify before endorsing it")
    elif role == "bystander":
        neg.append("I mostly observe unless it proves critical")

    return pos, neg


def belief_reason_opinion_callback(user: UserAgent, news: NewsItem) -> str:
    """Output stance (believes/skeptical) plus current-round reasons derived from content/persona/graph.
    We use belief vs threshold only to decide the stance, not as the textual reason.
    """
    belief = user.state.belief.belief_strength
    threshold = user.state.belief.propagation_threshold
    pos, neg = _current_reasons_for_user(user, news)

    believes = belief >= threshold  # used to choose stance only
    persona = user.state.persona
    role = getattr(persona, "role", "bystander")
    name = persona.name
    stance = "believes" if believes else "is skeptical of"

    reasons = (pos if believes else neg) or (neg if believes else pos) or ["insufficient evidence"]
    reason_text = "; ".join(reasons[:2])
    role_tag = {
        "broadcaster": "speaking as a broadcaster",
        "commentator": "weighing in as a commentator",
        "verifier": "reviewing it as a verifier",
        "bystander": "sharing cautiously as a bystander",
    }.get(role, "sharing my perspective")
    return f"{name} {stance} {news.news_id} because {reason_text} ({role_tag})"
