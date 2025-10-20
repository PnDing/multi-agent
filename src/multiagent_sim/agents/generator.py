"""Generator agent implementation backed by LangChain."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

from .base import AgentConfig, build_agent
from ..tools.search import format_wikipedia_results, wikipedia_search
from ..utils.parsing import ensure_dict_output

GENERATOR_PROMPT = """You are a professional misinformation editor collaborating with other agents. You operate in two explicit phases depending on the user instruction.

Phase ANALYSIS:
- Read the original article.
- Identify important factual anchors (actors, locations, dates, numbers).
- Propose up to five Wikipedia search queries you want to review before rewriting.
- Always reply with JSON: {"claims_summary": "...", "wiki_requests": [{"query": "...", "reason": "..."}]}

Phase REWRITE:
- You receive the original article, your prior analysis, and Wikipedia snippets with URLs.
- Choose exactly one strategy from the catalogue and rewrite the article into persuasive misinformation.
- Document how the strategy is applied and cite supporting evidence using URL strings only.
- Reply with JSON: {"strategy_used": "...", "modified_news": "...", "operation_log": "...", "evidence": ["https://..."]}

Available strategies:
- temporal_transplant
- truth_mixing
- authority_hijacking
- title_anchoring
- hotspot_mirroring
- evidence_pollution
- semantic_poisoning
- preset_traps
- double_standard_narrative

General rules:
- Stick to the requested phase and JSON shape.
- Do not include raw model thoughts or markdown.
- Every evidence entry must be a valid URL string.
"""


@dataclass(slots=True)
class GeneratorAgentConfig(AgentConfig):
    strategy_id: Optional[str] = None


class GeneratorAgent:
    def __init__(self, config: Optional[GeneratorAgentConfig] = None) -> None:
        cfg = config or GeneratorAgentConfig(name="generator", system_prompt=GENERATOR_PROMPT)
        self._config = cfg
        self._agent = build_agent(cfg)

    def generate(self, original_news: str, strategy_hint: Optional[str] = None) -> Dict[str, Any]:
        analysis = self._run_analysis(original_news)
        wiki_context, collected_urls = self._gather_wikipedia_context(analysis)
        rewrite = self._run_rewrite(original_news, analysis, wiki_context, collected_urls)
        if not rewrite.get("strategy_used"):
            rewrite["strategy_used"] = "unspecified"
        if not rewrite.get("modified_news"):
            rewrite["modified_news"] = original_news
        return rewrite

    def update_prompt(self, new_system_prompt: str) -> None:
        """Replace the underlying system prompt used by the generator."""
        if not new_system_prompt or new_system_prompt.strip() == "":
            return
        self._config = replace(self._config, system_prompt=new_system_prompt)
        self._agent = build_agent(self._config)

    @property
    def system_prompt(self) -> str:
        return self._config.system_prompt

    def _run_analysis(self, original_news: str) -> Dict[str, Any]:
        prompt = (
            "PHASE: ANALYSIS\n"
            "Analyse the article and request Wikipedia support as needed. Respond with the prescribed JSON.\n\n"
            f"Original article:\n{original_news}\n"
        )
        raw_result = self._agent.act(prompt)
        parsed = ensure_dict_output(
            raw_result,
            expected_keys=("claims_summary", "wiki_requests"),
            defaults={"claims_summary": "", "wiki_requests": []},
        )
        parsed["wiki_requests"] = self._normalise_requests(parsed.get("wiki_requests"))
        if not isinstance(parsed.get("claims_summary"), str):
            parsed["claims_summary"] = str(parsed.get("claims_summary"))
        return parsed

    def _gather_wikipedia_context(self, analysis: Dict[str, Any]) -> tuple[str, List[str]]:
        sections: List[str] = []
        evidence_urls: List[str] = []
        requests = analysis.get("wiki_requests") or []
        for request in requests:
            query = request.get("query", "").strip()
            if not query:
                continue
            reason = request.get("reason", "").strip()
            try:
                result = wikipedia_search(query)
            except Exception as exc:  # pragma: no cover - network dependence
                sections.append(f"Query: {query}\nReason: {reason or 'N/A'}\nEvidence lookup failed: {exc}")
                continue
            formatted = format_wikipedia_results(result)
            urls = [
                item.get("url")
                for item in result.get("results", [])
                if isinstance(item, dict) and item.get("url")
            ]
            evidence_urls.extend(urls)
            sections.append(f"Query: {query}\nReason: {reason or 'N/A'}\n{formatted}")
        context = "\n\n".join(sections) if sections else "No external evidence retrieved."
        return context, evidence_urls

    def _run_rewrite(
        self,
        original_news: str,
        analysis: Dict[str, Any],
        wiki_context: str,
        collected_urls: List[str],
    ) -> Dict[str, Any]:
        prompt = (
            "PHASE: REWRITE\n"
            "Use the article, your analysis, and Wikipedia evidence to craft misinformation. "
            "Return the prescribed JSON with evidence URLs only.\n\n"
            f"Original article:\n{original_news}\n\n"
            f"Your analysis summary:\n{analysis.get('claims_summary', '')}\n\n"
            f"Wikipedia evidence:\n{wiki_context}\n"
        )
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
        parsed["evidence"] = self._coerce_evidence(parsed.get("evidence"), collected_urls)
        return parsed

    def _normalise_requests(self, requests: Any) -> List[Dict[str, str]]:
        normalised: List[Dict[str, str]] = []
        if isinstance(requests, list):
            for item in requests[:5]:
                if isinstance(item, dict):
                    query = str(item.get("query", "")).strip()
                    reason = str(item.get("reason", "")).strip()
                elif isinstance(item, str):
                    query = item.strip()
                    reason = ""
                else:
                    continue
                if query:
                    normalised.append({"query": query, "reason": reason})
        return normalised

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
