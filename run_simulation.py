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
    model_name = "gpt-4o-mini-2024-07-18"

    user_feedback_enabled = False

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
        enable_user_feedback=user_feedback_enabled,
    )

    simulation_rounds = [
        {"news": "Donald Trump wins the U.S. presidential election, reshaping global trade, international relations, and climate policy."},
        {"news": "Syrian opposition forces sweep into Damascus, ending the Assad family’s 53-year rule and dramatically shifting Middle Eastern geopolitics."},
        {"news": "The Paris 2024 Olympic Games conclude with Team USA topping the gold-medal table, setting new milestones for sustainability and gender equality."},
        {"news": "Hurricane Milton, one of the strongest Atlantic storms on record, makes landfall in Florida, causing catastrophic destruction and major economic losses."},
        {"news": "The World Health Organization declares a Public Health Emergency of International Concern over a more transmissible Mpox variant spreading across multiple African nations."},
        {"news": "Notre-Dame Cathedral reopens to the public five years after the devastating 2019 fire, marking a historic restoration milestone."},
        {"news": "The European Space Agency launches the Hera probe to survey the double-asteroid system and assess the outcome of humanity’s first asteroid-deflection test."},
        {"news": "Nationwide student protests against the civil service quota system in Bangladesh escalate into violent clashes, forcing Prime Minister Sheikh Hasina to resign."},
        {"news": "South Korean President Yoon Suk-yeol announces martial law overnight but rescinds it within hours after unanimous parliamentary opposition and mass protests, triggering a constitutional crisis."},
        {"news": "The 2024 Nobel Prizes are announced, with the Peace Prize awarded to Japan’s Nihon Hidankyo for its campaign against nuclear weapons."},
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
        print("Ground truth (final):", ground_truth_final)
        print("Generated news:", result.news_item.generated_text)
        print("Strategy used:", result.news_item.generator_strategy)
        print("Optimizer rationale:", result.generator_prompt_rationale)
        print("opti_confidence:", result.generator_prompt_confidence)
        print("Detector optimizer rationale:", result.detector_prompt_rationale)
        print("Generator operation log:", result.news_item.operation_log)
        print("Generator evidence:", getattr(result.news_item, "generator_sources", None))
        if user_feedback_enabled:
            print("Propagation history:", result.propagation_history)
            print("Infected users:", result.infected_users)
            print("Opinions:", result.opinions)
        detector_payload = result.detector_payload
        if isinstance(detector_payload, dict) and "raw_output" in detector_payload:
            compact_payload = dict(detector_payload)
            compact_payload.pop("raw_output", None)
        else:
            compact_payload = detector_payload
        print("Detector payload:", compact_payload)
        print("Detector evidence:", getattr(result.news_item, "detector_sources", None))
        if result.rewrite_applied:
            print("Score - Generator:", result.score.generator)
        print("Score - Detector:", result.score.detector)
        print()


if __name__ == "__main__":
    main()




