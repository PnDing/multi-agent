"""Minimal entrypoint demonstrating the simulation framework."""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from multiagent_sim.agents.generator import GeneratorAgentConfig, GENERATOR_PROMPT
from multiagent_sim.agents.detector import DetectorAgentConfig, DETECTOR_PROMPT
from multiagent_sim.community.persona import Persona
from multiagent_sim.data.persona_catalog import (
    ARCHETYPE_DISTRIBUTION,
    CANDIDATE_NAMES,
    CITY_OPTIONS,
    EDUCATION_OPTIONS,
    GENDER_OPTIONS,
    OCCUPATION_OPTIONS,
    sample_age,
    sample_interests,
    sample_media_preferences,
    sample_personality,
    sample_role,
)
from multiagent_sim.pipelines.simulation import build_default_orchestrator, belief_reason_opinion_callback


def generate_personas(count: int, *, seed: int = 42) -> List[Persona]:
    rng = random.Random(seed)
    available_names = list(CANDIDATE_NAMES)
    rng.shuffle(available_names)
    if count > len(available_names):
        raise ValueError(f"Requested {count} personas but only {len(available_names)} names available")
    personas: List[Persona] = []
    archetypes = list(ARCHETYPE_DISTRIBUTION)
    for idx in range(count):
        archetype = archetypes[idx % len(archetypes)]
        personality = sample_personality(archetype, rng=rng)
        persona = Persona(
            user_id=f"u{idx}",
            name=available_names[idx],
            gender=rng.choice(GENDER_OPTIONS),
            age=sample_age(rng=rng),
            location=rng.choice(CITY_OPTIONS),
            occupation=rng.choice(OCCUPATION_OPTIONS),
            education=rng.choice(EDUCATION_OPTIONS),
            interests=sample_interests(rng=rng),
            media_preferences=sample_media_preferences(rng=rng),
            role=sample_role(archetype, rng=rng),
            openness=personality["openness"],
            conscientiousness=personality["conscientiousness"],
            extraversion=personality["extraversion"],
            agreeableness=personality["agreeableness"],
            neuroticism=personality["neuroticism"],
        )
        personas.append(persona)
    return personas


def main() -> None:
    personas = generate_personas(6, seed=42)
    print("Persona roster:")
    for persona in personas:
        print(f"  {persona.user_id} [{persona.role}] - age {persona.age}, city {persona.location}, occupation {persona.occupation}; O={persona.openness:.2f}, C={persona.conscientiousness:.2f}, E={persona.extraversion:.2f}, A={persona.agreeableness:.2f}, N={persona.neuroticism:.2f}")
    edges = [("u0", "u1"), ("u1", "u2"), ("u2", "u3"), ("u3", "u4"), ("u4", "u5"), ("u1", "u3"), ("u2", "u4"), ("u0", "u2")]

    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = "gpt-4o-2024-05-13"

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
        opinion_callback=belief_reason_opinion_callback,
        propagation_rounds=5,
    )

    simulation_rounds = [
        {"news": "Scientists discover water on Mars again amid growing debates."},
        {"news": "WHO confirms a global eradication of polio after coordinated vaccination efforts."},
        {"news": "Leaked documents reveal plans to move the Great Wall to boost tourism."},
        {"news": "Shanghai launches a new high-speed rail link reducing travel time to Beijing to 2 hours."},
        {"news": "A startup claims cold fusion generators will power every home by next year."},
    ]

    rewrite_probability = 0.5
    random.seed(42)

    for idx, config in enumerate(simulation_rounds, start=1):
        apply_rewrite = random.random() < rewrite_probability
        ground_truth_raw = 0
        ground_truth_final = 1 if apply_rewrite else 0

        result = orchestrator.simulate_round(
            config["news"],
            ground_truth_raw=ground_truth_raw,
            ground_truth_final=ground_truth_final,
            seed_users=["u0"],
            rewrite=apply_rewrite,
        )
        print(f"=== Round {idx} ===")
        print("News:", config["news"])
        print("Rewrite applied:", apply_rewrite)
        print("Ground truth (raw):", ground_truth_raw)
        print("Ground truth (final):", ground_truth_final)
        print("Generated news:", result.news_item.generated_text)
        print("Strategy used:", result.news_item.generator_strategy)
        print("Generator operation log:", result.news_item.operation_log)
        print("Generator evidence:", getattr(result.news_item, "generator_sources", None))
        print("Propagation history:", result.propagation_history)
        print("Infected users:", result.infected_users)
        print("Opinions:", result.opinions)
        print("Detector payload:", result.detector_payload)
        print("Score - Generator:", result.score.generator)
        print("Score - Detector:", result.score.detector)
        print()


if __name__ == "__main__":
    main()




