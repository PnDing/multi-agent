"""External knowledge tool adapters (Wikipedia API)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests

try:
    from langchain.tools import tool
except ImportError:  # pragma: no cover - optional dependency
    tool = None  # type: ignore

WIKIPEDIA_ENDPOINT = "https://en.wikipedia.org/w/api.php"


@dataclass(slots=True)
class WikipediaClient:
    language: str = "en"
    timeout: float = 10.0
    _session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._session = requests.Session()

    def search(self, query: str, limit: int = 3) -> Dict[str, Any]:
        if not query:
            raise ValueError("query must be provided")
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "utf8": 1,
            "srlimit": limit,
        }
        response = self._session.get(
            WIKIPEDIA_ENDPOINT,
            params=params,
            headers={"Accept": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        items = payload.get("query", {}).get("search", [])
        results = [
            {
                "title": item.get("title"),
                "snippet": item.get("snippet"),
                "pageid": item.get("pageid"),
                "timestamp": item.get("timestamp"),
            }
            for item in items
        ]
        return {"query": query, "results": results}


if tool is not None:

    @tool("wikipedia_search", return_direct=False)
    def wikipedia_search(query: str) -> Dict[str, Any]:
        """Search Wikipedia for supporting evidence."""

        client = WikipediaClient()
        return client.search(query)
else:

    def wikipedia_search(query: str) -> Dict[str, Any]:  # type: ignore
        raise ImportError("langchain is required to register wikipedia_search tool")
