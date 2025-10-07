"""Minimal entrypoint demonstrating the simulation framework."""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from multiagent_sim.agents.generator import GeneratorAgentConfig, GENERATOR_PROMPT
from multiagent_sim.agents.detector import DetectorAgentConfig, DETECTOR_PROMPT
from multiagent_sim.community.persona import Persona
from multiagent_sim.pipelines.simulation import build_default_orchestrator, default_opinion_callback


def main() -> None:
    personas = [
        Persona(user_id="u0", name="Alice", gender="female", age=28, location="Beijing", occupation="Journalist", education="Bachelor"),
        Persona(user_id="u1", name="Bob", gender="male", age=35, location="Shanghai", occupation="Engineer", education="Master"),
        Persona(user_id="u2", name="Carol", gender="female", age=42, location="Shenzhen", occupation="Teacher", education="Bachelor"),
        Persona(user_id="u3", name="David", gender="male", age=31, location="Guangzhou", occupation="Data Analyst", education="Master"),
        Persona(user_id="u4", name="Eve", gender="female", age=29, location="Chengdu", occupation="Content Creator", education="Bachelor"),
        Persona(user_id="u5", name="Frank", gender="male", age=37, location="Wuhan", occupation="Healthcare Worker", education="Bachelor"),
    ]
    edges = [("u0", "u1"), ("u1", "u2"), ("u2", "u3"), ("u3", "u4"), ("u4", "u5"), ("u1", "u3"), ("u2", "u4"), ("u0", "u2")]

    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = "gpt-3.5-turbo-0125"

    generator_cfg = GeneratorAgentConfig(
        name="generator",
        system_prompt=GENERATOR_PROMPT,
        model=model_name,
        base_url=base_url,
    )
    detector_cfg = DetectorAgentConfig(
        name="detector",
        system_prompt=DETECTOR_PROMPT,
        model=model_name,
        base_url=base_url,
    )

    orchestrator = build_default_orchestrator(
        personas,
        edges,
        generator_config=generator_cfg,
        detector_config=detector_cfg,
        opinion_callback=default_opinion_callback,
        propagation_rounds=5,
    )
    result = orchestrator.simulate_round(
        "Scientists discover water on Mars again amid growing debates.",
        ground_truth=0,
        seed_users=["u0"],
    )
    print("Generated news:", result.news_item.generated_text)
    print("Propagation history:", result.propagation_history)
    print("Infected users:", result.infected_users)
    print("Opinions:", result.opinions)
    print("Detector payload:", result.detector_payload)
    print("Score - Generator:", result.score.generator)
    print("Score - Detector:", result.score.detector)


if __name__ == "__main__":
    main()
