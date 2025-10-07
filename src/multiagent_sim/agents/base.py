"""Minimal LLM agent abstraction."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

try:
    from langchain_openai import ChatOpenAI  # type: ignore
except ImportError:  # pragma: no cover
    ChatOpenAI = None  # type: ignore

try:
    from ..config.api_keys import OPENAI_API_KEY as HARDCODED_OPENAI_KEY
    from ..config.api_keys import OPENAI_BASE_URL as HARDCODED_OPENAI_BASE_URL  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    HARDCODED_OPENAI_KEY = None
    HARDCODED_OPENAI_BASE_URL = None

PLACEHOLDER_KEY_TOKENS = ("your-openai-key", "sk-xxxxx")
DEFAULT_MODEL = "gpt-3.5-turbo-0125"


def _sanitize(candidate: Optional[str]) -> Optional[str]:
    if not candidate:
        return None
    lowered = candidate.lower()
    if any(token in lowered for token in PLACEHOLDER_KEY_TOKENS):
        return None
    return candidate


@dataclass(slots=True)
class AgentConfig:
    name: str
    system_prompt: str
    temperature: float = 0.0
    model: str = DEFAULT_MODEL
    tools: Sequence[Any] = field(default_factory=tuple)
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class BaseAgent(ABC):
    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    @abstractmethod
    def act(self, prompt: str) -> str:
        pass


class LLMAgent(BaseAgent):
    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)
        if ChatOpenAI is None:
            raise ImportError("langchain-openai is required for LLMAgent")
        api_key = _sanitize(config.api_key) or _sanitize(HARDCODED_OPENAI_KEY)
        base_url = config.base_url or HARDCODED_OPENAI_BASE_URL
        kwargs: dict[str, Any] = {
            "model": config.model or DEFAULT_MODEL,
            "temperature": config.temperature,
        }
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._llm = ChatOpenAI(**kwargs)

    def act(self, prompt: str) -> str:
        messages = [
            ("system", self.config.system_prompt),
            ("user", prompt),
        ]
        response = self._llm.invoke(messages)
        return getattr(response, "content", str(response))


def build_agent(config: AgentConfig) -> LLMAgent:
    return LLMAgent(config)
