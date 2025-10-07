"""Detector agent responsible for fact checking and verdict output."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from .base import AgentConfig, build_agent
from ..tools.search import wikipedia_search
from ..utils.parsing import coerce_int, ensure_dict_output

DETECTOR_PROMPT = (
    "You are a misinformation detector. Given a piece of news, optional community"
    " opinions, and optional external evidence, combine internal reasoning to determine"
    " authenticity. Output JSON with fields `authenticity` (1 for fake, 0 for real)"
    " and `operation_log` summarising the supporting evidence."
)


def _format_evidence(news: str) -> str:
    try:
        results = wikipedia_search(news)
    except Exception:
        return "Wikipedia search unavailable."
    snippets = []
    for item in results.get("results", [])[:3]:
        snippet = item.get("snippet", "").replace("\u201c", '"').replace("\u201d", '"')
        snippets.append(f"- {item.get('title')}: {snippet}")
    if not snippets:
        return "No relevant Wikipedia entries found."
    return "\n".join(snippets)


@dataclass(slots=True)
class DetectorAgentConfig(AgentConfig):
    use_tools: bool = False


class DetectorAgent:
    def __init__(self, config: Optional[DetectorAgentConfig] = None) -> None:
        cfg = config or DetectorAgentConfig(name="detector", system_prompt=DETECTOR_PROMPT)
        self._agent = build_agent(cfg)

    def detect(
        self,
        news: str,
        opinions: Optional[Sequence[str]] = None,
        *,
        strategy_hint: Optional[str] = None,
    ) -> Dict[str, int | str | Dict[str, int | str]]:
        sections = []
        if strategy_hint:
            sections.append(f"Detection strategy hint: {strategy_hint}")
        sections.append(f"News:\n{news}")
        if opinions:
            joined = "\n- ".join(opinions)
            sections.append(f"Community opinions:\n- {joined}")
        evidence = _format_evidence(news)
        sections.append(f"Wikipedia evidence:\n{evidence}")
        prompt = "\n\n".join(sections)
        raw_result = self._agent.act(prompt)
        parsed = ensure_dict_output(
            raw_result,
            expected_keys=("authenticity", "operation_log"),
            defaults={"authenticity": 0, "operation_log": ""},
        )
        parsed["authenticity"] = coerce_int(parsed.get("authenticity", 0), fallback=0)
        if strategy_hint:
            parsed.setdefault("strategy_hint", strategy_hint)
        return parsed
