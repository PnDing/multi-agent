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
from ..optimization.detector_optimizer import DetectorStrategyOptimizer
from ..optimization.generator_optimizer import GeneratorStrategyOptimizer
from ..scoring.metrics import ScoreCalculator


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
    detector_optimizer: Optional[DetectorStrategyOptimizer] = None,
) -> SimulationOrchestrator:
    generator = AgentFactory.build_generator(generator_config)
    detector = AgentFactory.build_detector(detector_config)
    community = {
        persona.user_id: UserAgent(persona, belief=BeliefProfile()) for persona in personas
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
        detector_optimizer=detector_optimizer or DetectorStrategyOptimizer(),
    )


def default_opinion_callback(user: UserAgent, news: NewsItem) -> str:
    experiences = user.recall_experience(limit=1)
    experience_hint = experiences[0].rationale if experiences else "No prior experience"
    return f"{user.state.persona.name} believes {news.news_id} due to {experience_hint}"
