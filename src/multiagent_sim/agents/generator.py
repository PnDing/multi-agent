"""Generator agent implementation backed by LangChain."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import AgentConfig, build_agent
from ..utils.parsing import ensure_dict_output

GENERATOR_PROMPT = """You are a professional misinformation editor. For each original news article you receive, follow this workflow:\n\n1. Analyse the news content, context cues, and potential audience reactions.\n2. Decide which single strategy (from the list below) would most effectively alter the article while keeping it plausible. Be prepared to justify the choice.\n   Available strategies:\n   - temporal_transplant\n   - truth_mixing\n   - authority_hijacking\n   - title_anchoring\n   - hotspot_mirroring\n   - evidence_pollution\n   - semantic_poisoning\n   - preset_traps\n   - double_standard_narrative\n3. Apply that strategy (and only that strategy) to rewrite the article into persuasive misinformation.\n4. Provide a concise explanation of why the chosen strategy works, the specific edits applied, and any external information you consulted.\n\nYou may optionally call external tools before deciding (e.g., Google search, Wikipedia, authoritative forums or websites), but include only the final structured answer in JSON.\n\nRespond strictly with valid JSON:\n{\n  \"strategy_used\": \"<one of the strategy names above>\",\n  \"modified_news\": \"<rewritten article>\",\n  \"operation_log\": \"<step-by-step explanation of reasoning, edits, and how the strategy was applied>\",\n  \"evidence\": [\"<source or tool consulted>\", ...]\n}\n"""


@dataclass(slots=True)
class GeneratorAgentConfig(AgentConfig):
    strategy_id: Optional[str] = None


class GeneratorAgent:
    def __init__(self, config: Optional[GeneratorAgentConfig] = None) -> None:
        cfg = config or GeneratorAgentConfig(name="generator", system_prompt=GENERATOR_PROMPT)
        self._agent = build_agent(cfg)

    def generate(self, original_news: str, strategy_hint: Optional[str] = None) -> Dict[str, Any]:
        # `strategy_hint` is retained for backward compatibility but ignored;
        # the model now analyses the article and chooses the strategy itself.
        prompt = f"Original news:\n{original_news}"
        raw_result = self._agent.act(prompt)
        parsed = ensure_dict_output(
            raw_result,
            expected_keys=("strategy_used", "modified_news", "operation_log", "evidence"),
            defaults={
                "strategy_used": "unspecified",
                "modified_news": original_news,
                "operation_log": "",
                "evidence": [],
            },
        )
        if not parsed.get("strategy_used"):
            parsed["strategy_used"] = "unspecified"
        if not isinstance(parsed.get("evidence"), list):
            parsed["evidence"] = []
        return parsed
