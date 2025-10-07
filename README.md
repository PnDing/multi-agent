# Multi-Agent Fake News Simulation (First Draw)

This repository hosts the first-draw implementation of the adversarial multi-agent framework described in the higher-level design document (`../½á¹¹¿ò¼Ü.md`).

## Layout

- `src/multiagent_sim/`
  - `agents/`: Agent abstractions and concrete implementations (generator, detector, user, evaluators).
  - `community/`: User persona models, social graph, and experience memory helpers.
  - `core/`: Simulation loop, event bus, configuration, and orchestrator.
  - `optimization/`: Strategy optimizers for generator and detector agents.
  - `pipelines/`: High-level workflows for data ingestion, simulation, and evaluation.
  - `scoring/`: Metrics and feedback calculation logic.
  - `tools/`: External tool adapters (e.g., Wikipedia search).
  - `utils/`: Shared helpers (logging, identifier generation, persistence adapters).
  - `data/`: Optional storage helpers and dataset interfaces.
- `tests/`: Future unit and integration tests (currently empty).

## Setup

1. Create a Python 3.10+ virtual environment.
2. Install the dependencies: `pip install -r requirements.txt`.
3. Provide credentials via environment variables before running the simulation:
   - `OPENAI_API_KEY`

These keys are not bundled in the repository; set them manually when you are ready to execute the workflow.
Note: Wikipedia search endpoints do not require an API key, but outbound network access must be enabled.

## Usage

The `src/multiagent_sim` package exposes a configurable `SimulationOrchestrator` that wires all agents together. A minimal usage example is provided in `run_simulation.py`.

```
python run_simulation.py
```

## Status

This is a scaffolding project: most modules contain extensible class skeletons and TODO markers awaiting domain-specific logic, strategy optimizers, and production-grade prompts.






