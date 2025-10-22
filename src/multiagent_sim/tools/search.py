"""External knowledge tool adapters backed by the Serper API."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

try:
    from langchain.tools import tool
except ImportError:  # pragma: no cover - optional dependency
    tool = None  # type: ignore


SERPER_ENDPOINT = "https://google.serper.dev/search"
DEFAULT_SERPER_API_KEY = "fa7b3427e010e14cdba10dee2419ddb2ca5e3656"
SERPER_API_KEY = os.getenv("SERPER_API_KEY", DEFAULT_SERPER_API_KEY)


def _normalise_snippet(text: Optional[str]) -> str:
    if not text:
        return ""
    return " ".join(text.split())


def _normalise_result(entry: Dict[str, Any]) -> Dict[str, str]:
    url = entry.get("link") or entry.get("url") or ""
    return {
        "title": entry.get("title") or "",
        "snippet": _normalise_snippet(entry.get("snippet") or entry.get("description")),
        "url": url,
        "source": entry.get("source") or entry.get("domain") or "",
    }


@dataclass(slots=True)
class SerperClient:
    """Thin client over the Serper Google Search API."""

    api_key: str = SERPER_API_KEY
    timeout: float = 10.0
    default_limit: int = 5
    _session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "Serper API key missing. Set SERPER_API_KEY environment variable or update DEFAULT_SERPER_API_KEY."
            )
        self._session = requests.Session()

    def search(self, query: str, *, limit: Optional[int] = None) -> Dict[str, Any]:
        if not query:
            raise ValueError("query must be provided")
        result_limit = limit or self.default_limit
        payload = {"q": query, "num": result_limit}
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        response = self._session.post(
            SERPER_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        organic_results = data.get("organic") or []
        results: List[Dict[str, str]] = []
        for item in organic_results[:result_limit]:
            normalised = _normalise_result(item)
            if normalised["url"]:
                results.append(normalised)
        return {"query": query, "results": results, "raw": data}


def format_search_results(payload: Dict[str, Any]) -> str:
    items = payload.get("results") or []
    if not items:
        return "No external search results were retrieved."
    lines: List[str] = []
    for idx, item in enumerate(items, start=1):
        snippet = item.get("snippet") or "No summary available."
        url = item.get("url") or "URL unavailable"
        source = item.get("source")
        if source:
            lines.append(f"[{idx}] {snippet} (URL: {url}, Source: {source})")
        else:
            lines.append(f"[{idx}] {snippet} (URL: {url})")
    return "\n".join(lines)


if tool is not None:

    @tool("serper_search", return_direct=False)
    def serper_search(query: str) -> Dict[str, Any]:
        """Search the web via Serper for supporting evidence."""

        client = SerperClient()
        return client.search(query)

else:

    def serper_search(query: str) -> Dict[str, Any]:  # type: ignore
        raise ImportError("langchain is required to register serper_search tool")
