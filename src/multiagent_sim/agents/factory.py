"""Agent factory utilities to mirror the original report.py style."""
from __future__ import annotations

from typing import Optional

from .detector import DetectorAgent, DetectorAgentConfig
from .generator import GeneratorAgent, GeneratorAgentConfig


class AgentFactory:
    """Centralised builder for core agents."""

    @staticmethod
    def build_generator(config: Optional[GeneratorAgentConfig] = None) -> GeneratorAgent:
        return GeneratorAgent(config)

    @staticmethod
    def build_detector(config: Optional[DetectorAgentConfig] = None) -> DetectorAgent:
        return DetectorAgent(config)
