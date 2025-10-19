"""LLM-driven detector strategy optimizer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..agents.base import AgentConfig, build_agent
from ..utils.parsing import ensure_dict_output


OPTIMIZER_SYSTEM_PROMPT = """您是多智能体假新闻对抗系统中的检测优化器，核心任务是根据检测器的表现和生成器的攻击策略，动态优化检测器的提示词。您的每次决策需实现：
1. 漏洞修复：解析检测失败案例，定位策略弱点；
2. 误报规避：减少对真实新闻的误判；
3. 评分进化：推动检测策略的评分持续提升。

请综合历史记录与当前基线提示词，生成新的检测器提示词，并给出理由与置信度。严格以 JSON 回复：
{
  "next_prompt": "<新的 system prompt 文本>",
  "rationale": "<概述为何这样调整，可引用历史案例>",
  "confidence": <0 到 1 之间的数值>,
  "notes": "<可选补充>"
}
"""


@dataclass(slots=True)
class DetectorStrategyAgentConfig(AgentConfig):
    pass


class DetectorStrategyAgent:
    """Wraps an LLM that proposes updated detector prompts."""

    def __init__(
        self,
        config: Optional[DetectorStrategyAgentConfig] = None,
        *,
        history_limit: int = 5,
    ) -> None:
        cfg = config or DetectorStrategyAgentConfig(
            name="detector_optimizer",
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
        compressed = self._compress_history(history[-self._history_limit :])
        prompt = self._build_prompt(compressed, default_prompt)
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
        try:
            parsed_conf = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            parsed_conf = 0.0
        parsed["confidence"] = max(0.0, min(1.0, parsed_conf))
        next_prompt = parsed.get("next_prompt") or default_prompt
        parsed["next_prompt"] = str(next_prompt)
        return parsed

    def _compress_history(self, history: Sequence[Mapping[str, Any]]) -> List[str]:
        lines: List[str] = []
        for idx, entry in enumerate(history, start=1):
            prompt_digest = entry.get("prompt_digest", "N/A")
            detected_correctly = entry.get("detector_success")
            ground_truth = entry.get("ground_truth")
            auth = entry.get("authenticity")
            generator_strategy = entry.get("generator_strategy")
            line = (
                f"{idx}. prompt={prompt_digest}; success={detected_correctly}; "
                f"gt={ground_truth}; detector_pred={auth}; generator_strategy={generator_strategy}"
            )
            feedback = entry.get("operation_log")
            if feedback:
                line += f"; log={feedback[:120]}"
            evidence = entry.get("evidence")
            if evidence:
                line += f"; evidence={evidence[:2]}"
            lines.append(line)
        return lines or ["No detector history available."]

    def _build_prompt(self, history_lines: Sequence[str], default_prompt: str) -> str:
        history_block = "\n".join(history_lines)
        return (
            "Recent detector history (oldest first):\n"
            f"{history_block}\n\n"
            f"Current baseline detector prompt:\n{default_prompt}\n\n"
            "Provide the next prompt recommendation following the required JSON schema."
        )
