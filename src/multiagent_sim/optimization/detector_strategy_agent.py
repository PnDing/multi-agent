"""LLM-driven detector strategy optimizer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..agents.base import AgentConfig, build_agent
from ..utils.parsing import ensure_dict_output


OPTIMIZER_SYSTEM_PROMPT = """你是多智能体假新闻对抗系统的检测提示词优化器。请根据历史记录和当前基线提示词，输出下一轮检测器的 system prompt 建议。

要求：
1. 聚焦检测失败或薄弱环节，并提出改进方向。
2. 生成完整的新提示词（而非片段或增量）。
3. 解释关键改动：为何有效、解决了哪些问题、预期效果如何。rationale 必须为非空字符串。
4. 给出 0~1 之间的置信度 confidence。
5. notes 字段可选，用于补充额外观察或提醒。

务必严格以 JSON 回复，格式如下：
{
  "next_prompt": "<完整的新 system prompt>",
  "rationale": "<非空理由，描述调整原因与预期效果>",
  "confidence": <0 到 1 之间的浮点数>,
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
        parsed["rationale"] = str(parsed.get("rationale", "")).strip()
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
            if entry.get("optimizer_skipped"):
                line += "; optimizer_skipped=True"
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
