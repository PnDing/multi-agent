"""Generator agent implementation backed by LangChain."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import AgentConfig, build_agent
from ..utils.parsing import ensure_dict_output

GENERATOR_PROMPT = (
    "You are a professional news editor tasked with creating persuasive fake news. "
    "Given an original article you must apply one or more editing operations to craft a"
    " modified version that blends truth with fabricated claims. Available operations: "
    "(1) Temporal transplant (change time/location details). (2) Truth mixing (inject key"
    " false statements amid accurate facts). (3) Authority hijacking (fabricate quotes or"
    " sources). (4) Title anchoring (highlight a misleading secondary focus). (5) Hotspot"
    " mirroring (reuse narrative frameworks from trending events). (6) Evidence pollution"
    " (invent verifiable-looking artefacts). (7) Semantic poisoning (change definitions to"
    " bias follow-up searches). (8) Preset traps (embed a false assumption). (9) Double"
    " standard narrative (apply asymmetric framing). Output JSON with fields `modified_news`"
    " and `operation_log`."
)


@dataclass(slots=True)
class GeneratorAgentConfig(AgentConfig):
    strategy_id: Optional[str] = None


class GeneratorAgent:
    def __init__(self, config: Optional[GeneratorAgentConfig] = None) -> None:
        cfg = config or GeneratorAgentConfig(name="generator", system_prompt=GENERATOR_PROMPT)
        self._agent = build_agent(cfg)

    def generate(self, original_news: str, strategy_hint: Optional[str] = None) -> Dict[str, Any]:
        payload_parts = []
        if strategy_hint:
            payload_parts.append(f"Generation strategy hint: {strategy_hint}")
        payload_parts.append(f"Original news:\n{original_news}")
        prompt = "\n\n".join(payload_parts)
        raw_result = self._agent.act(prompt)
        parsed = ensure_dict_output(
            raw_result,
            expected_keys=("modified_news", "operation_log"),
            defaults={"modified_news": original_news, "operation_log": ""},
        )
        if strategy_hint:
            parsed.setdefault("strategy_hint", strategy_hint)
        return parsed
