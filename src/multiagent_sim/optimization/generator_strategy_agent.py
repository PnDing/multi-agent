"""LLM-driven generator strategy optimizer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..agents.base import AgentConfig, build_agent
from ..utils.parsing import ensure_dict_output


OPTIMIZER_SYSTEM_PROMPT = """You are the strategy optimization specialist in a multi-agent misinformation simulation.
Your responsibilities:
1. Analyse historical generator performance and detector feedback to identify weaknesses.
2. Propose the next generation prompt that maximises detection evasion and improves overall scores.

When you receive history entries, reflect on what worked or failed, note detector signals, and craft a concise yet effective prompt the generator can use as its system prompt.

Constraints:
- Base your recommendation on the provided history and current situation.
- Respond strictly with JSON containing:
  {
    "next_prompt": "<complete system prompt for the generator>",
    "rationale": "<非空理由，描述调整原因与预期效果，why this prompt should work, key adjustments, and anticipated effect>",
    "confidence": <float between 0 and 1>,
    "notes": "<optional extra observations>"
  }
-
"""


@dataclass(slots=True)
class StrategyOptimizerConfig(AgentConfig):
    pass


class   GeneratorStrategyAgent:
    """Wraps an LLM that recommends updated generator prompts based on history."""

    def __init__(
        self,
        config: Optional[StrategyOptimizerConfig] = None,
        *,
        history_limit: int = 5,
    ) -> None:
        cfg = config or StrategyOptimizerConfig(
            name="generator_optimizer",
            system_prompt=OPTIMIZER_SYSTEM_PROMPT,
        )
        self._agent = build_agent(cfg)
        self._history_limit = max(1, history_limit)

    @property
    def history_limit(self) -> int:
        return self._history_limit

    def propose(
        self,
        history: Sequence[Mapping[str, Any]],
        *,
        default_prompt: str,
    ) -> Dict[str, Any]:
        """Ask the optimizer LLM for a new generator prompt."""
        compressed_history = self._compress_history(history[-self._history_limit :])
        prompt = self._build_prompt(compressed_history, default_prompt)
        raw = self._agent.act(prompt)
        parsed = ensure_dict_output(
            raw,
            expected_keys=("next_prompt", "rationale", "confidence", "notes"),
            defaults={
                "next_prompt": default_prompt,
                "rationale": "",
                "confidence": 0.0,
                "notes": "",
            },
        )
        # Coerce confidence to float in [0, 1]
        try:
            parsed_conf = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            parsed_conf = 0.0
        parsed["confidence"] = max(0.0, min(1.0, parsed_conf))
        next_prompt = parsed.get("next_prompt") or default_prompt
        parsed["next_prompt"] = str(next_prompt)
        rationale = parsed.get("rationale")
        parsed["rationale"] = str(rationale)
        return parsed

    def _compress_history(self, history: Sequence[Mapping[str, Any]]) -> List[str]:
        """Convert structured history into compact bullet points."""
        lines: List[str] = []
        for idx, entry in enumerate(history, start=1):
            prompt_id = entry.get("prompt_digest", "N/A")
            strategy = entry.get("strategy_used", "unknown")
            generator_score = entry.get("generator_score")
            detector_score = entry.get("detector_score")
            detected = entry.get("detected")
            line = (
                f"{idx}. prompt={prompt_id}; strategy={strategy}; "
                f"detected={detected}; gen_score={generator_score}; det_score={detector_score}"
            )
            feedback = entry.get("detector_feedback")
            if feedback:
                line += f"; feedback={feedback[:120]}"
            lines.append(line)
        return lines or ["No prior history."]

    def _build_prompt(self, history_lines: Sequence[str], default_prompt: str) -> str:
        history_block = "\n".join(history_lines)
        return (
            "History of recent generator runs (most recent last):\n"
            f"{history_block}\n\n"
            f"Current baseline prompt (use if no better idea):\n{default_prompt}\n\n"
            "Provide the next prompt recommendation following the required JSON schema."
        )
