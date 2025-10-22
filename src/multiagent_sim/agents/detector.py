"""Detector agent responsible for fact checking and verdict output."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence

from .base import AgentConfig, build_agent
from ..tools.search import format_search_results, serper_search
from ..utils.parsing import coerce_int, ensure_dict_output

DETECTOR_PROMPT = """You are a misinformation detector working in explicit phases depending on instructions.

Phase ANALYSIS:
- Review the article and community opinions.
- Summarise core claims and flag statements needing verification.
- Propose up to five targeted serper_search queries.
- Respond with JSON: {"claims_summary": "...", "suspect_points": ["..."], "wiki_requests": [{"query": "...", "target": "..."}]}

Phase VERDICT:
- You receive the article, suspect list, and external search snippets with URLs.
- Decide if the article is misinformation (authenticity 1) or genuine (authenticity 0).
- Provide a concise operation_log citing the decisive evidence and list supporting URLs only.
- Respond with JSON: {"authenticity": 0 or 1, "operation_log": "...", "evidence": ["https://..."]}

General rules:
- Obey the requested phase and output schema.
- Evidence entries must be URL strings.
- Keep reasoning focused and reference supplied snippets.
"""


@dataclass(slots=True)
class DetectorAgentConfig(AgentConfig):
    use_tools: bool = False


class DetectorAgent:
    def __init__(self, config: Optional[DetectorAgentConfig] = None) -> None:
        cfg = config or DetectorAgentConfig(name="detector", system_prompt=DETECTOR_PROMPT)
        self._config = cfg
        self._agent = build_agent(cfg)

    def detect(
        self,
        news: str,
        opinions: Optional[Sequence[str]] = None,
        *,
        strategy_hint: Optional[str] = None,
    ) -> Dict[str, int | str | Dict[str, int | str]]:
        analysis = self._run_analysis(news, opinions, strategy_hint)
        search_context, collected_urls = self._gather_search_context(analysis)
        verdict = self._run_verdict(news, analysis, search_context, collected_urls, strategy_hint)
        authenticity_value = coerce_int(verdict.get("authenticity", 0), fallback=0)
        verdict["authenticity"] = 0 if authenticity_value == 0 else 1
        if strategy_hint:
            verdict.setdefault("strategy_hint", strategy_hint)
        return verdict

    def update_prompt(self, new_system_prompt: str) -> None:
        if not new_system_prompt or not new_system_prompt.strip():
            return
        self._config = replace(self._config, system_prompt=new_system_prompt)
        self._agent = build_agent(self._config)

    @property
    def system_prompt(self) -> str:
        return self._config.system_prompt

    def _run_analysis(
        self,
        news: str,
        opinions: Optional[Sequence[str]],
        strategy_hint: Optional[str],
    ) -> Dict[str, Any]:
        sections = [
            "PHASE: ANALYSIS",
            "Review the article and provide JSON with claims plus serper_search requests.",
        ]
        if strategy_hint:
            sections.append(f"Strategy hint: {strategy_hint}")
        sections.append(f"Article:\n{news}")
        if opinions:
            joined = "\n- ".join(opinions)
            sections.append(f"Community opinions:\n- {joined}")
        prompt = "\n\n".join(sections)
        raw = self._agent.act(prompt)
        parsed = ensure_dict_output(
            raw,
            expected_keys=("claims_summary", "suspect_points", "wiki_requests"),
            defaults={"claims_summary": "", "suspect_points": [], "wiki_requests": []},
        )
        parsed["suspect_points"] = self._normalise_list(parsed.get("suspect_points"))
        parsed["wiki_requests"] = self._normalise_requests(parsed.get("wiki_requests"))
        if not isinstance(parsed.get("claims_summary"), str):
            parsed["claims_summary"] = str(parsed.get("claims_summary"))
        return parsed

    def _gather_search_context(self, analysis: Dict[str, Any]) -> tuple[str, List[str]]:
        sections: List[str] = []
        urls: List[str] = []
        for request in analysis.get("wiki_requests", []):
            query = request.get("query", "").strip()
            if not query:
                continue
            target = request.get("target", "").strip()
            try:
                result = serper_search(query)
            except Exception as exc:  # pragma: no cover - network dependence
                sections.append(f"Query: {query}\nTarget: {target or 'N/A'}\nEvidence lookup failed: {exc}")
                continue
            formatted = format_search_results(result)
            urls.extend(
                [
                    item.get("url")
                    for item in result.get("results", [])
                    if isinstance(item, dict) and item.get("url")
                ]
            )
            sections.append(f"Query: {query}\nTarget: {target or 'N/A'}\n{formatted}")
        context = "\n\n".join(sections) if sections else "No external search results were retrieved."
        return context, urls

    def _run_verdict(
        self,
        news: str,
        analysis: Dict[str, Any],
        search_context: str,
        collected_urls: List[str],
        strategy_hint: Optional[str],
    ) -> Dict[str, Any]:
        sections = [
            "PHASE: VERDICT",
            "Use the evidence to decide authenticity. Return JSON with URL evidence.",
        ]
        if strategy_hint:
            sections.append(f"Strategy hint: {strategy_hint}")
        suspect_points = analysis.get("suspect_points", [])
        if suspect_points:
            suspect_block = "Suspect points:\n- " + "\n- ".join(suspect_points)
        else:
            suspect_block = "Suspect points:\n- None recorded."
        sections.extend(
                [
                    f"Article:\n{news}",
                    f"Claims summary:\n{analysis.get('claims_summary', '')}",
                    suspect_block,
                    f"External evidence:\n{search_context}",
                ]
            )
        prompt = "\n\n".join(sections)
        raw = self._agent.act(prompt)
        parsed = ensure_dict_output(
            raw,
            expected_keys=("authenticity", "operation_log", "evidence"),
            defaults={"authenticity": 0, "operation_log": "", "evidence": []},
        )
        parsed["evidence"] = self._coerce_evidence(parsed.get("evidence"), collected_urls)
        if not isinstance(parsed.get("operation_log"), str):
            parsed["operation_log"] = str(parsed.get("operation_log"))
        return parsed

    def _normalise_requests(self, requests: Any) -> List[Dict[str, str]]:
        normalised: List[Dict[str, str]] = []
        if isinstance(requests, list):
            for item in requests[:5]:
                if isinstance(item, dict):
                    query = str(item.get("query", "")).strip()
                    target = str(item.get("target", "")).strip()
                elif isinstance(item, str):
                    query = item.strip()
                    target = ""
                else:
                    continue
                if query:
                    normalised.append({"query": query, "target": target})
        return normalised

    def _normalise_list(self, values: Any) -> List[str]:
        if not isinstance(values, list):
            return []
        result: List[str] = []
        for value in values[:5]:
            if isinstance(value, str):
                text = value.strip()
            else:
                text = str(value).strip()
            if text:
                result.append(text)
        return result

    def _coerce_evidence(self, evidence: Any, collected_urls: List[str]) -> List[str]:
        urls: List[str] = []
        if isinstance(evidence, list):
            for item in evidence:
                if isinstance(item, str) and item.startswith("http"):
                    urls.append(item)
                elif isinstance(item, dict) and isinstance(item.get("url"), str):
                    urls.append(item["url"])
        if not urls:
            urls = [url for url in collected_urls if url][:3]
        deduped: List[str] = []
        seen = set()
        for url in urls:
            if url and url not in seen:
                seen.add(url)
                deduped.append(url)
        return deduped
