"""External knowledge tool adapters (Wikipedia API)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import html
import re
import requests

try:
    from langchain.tools import tool
except ImportError:  # pragma: no cover - optional dependency
    tool = None  # type: ignore

WIKIPEDIA_ENDPOINT = "https://en.wikipedia.org/w/api.php"


def _build_wikipedia_url(title: Optional[str], pageid: Optional[int]) -> Optional[str]:
    if title:
        slug = quote(title.replace(" ", "_"))
        return f"https://en.wikipedia.org/wiki/{slug}"
    if pageid:
        return f"https://en.wikipedia.org/?curid={pageid}"
    return None


def _clean_snippet(snippet: Optional[str]) -> str:
    if not snippet:
        return ""
    text = html.unescape(snippet)
    text = re.sub(r"<[^>]+>", "", text)
    return " ".join(text.split())


def _augment_results_with_fullurl(
    session: requests.Session,
    results: List[Dict[str, Any]],
    timeout: float,
) -> Dict[Any, Dict[str, Any]]:
    pageids = [str(item.get("pageid")) for item in results if item.get("pageid")]
    if not pageids:
        return {}
    params = {
        "action": "query",
        "pageids": "|".join(pageids),
        "prop": "info|pageprops",
        "inprop": "url",
        "redirects": 1,
        "format": "json",
        "utf8": 1,
        "formatversion": 2,
    }
    try:
        response = session.get(
            WIKIPEDIA_ENDPOINT,
            params=params,
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
    except Exception:
        return {}
    payload = response.json()
    pages = payload.get("query", {}).get("pages", []) or []
    page_lookup: Dict[Any, Dict[str, Any]] = {}
    title_lookup: Dict[str, Dict[str, Any]] = {}
    for page in pages:
        pid = page.get("pageid")
        if pid is not None:
            page_lookup[pid] = page
        title = page.get("title")
        if isinstance(title, str):
            title_lookup[title] = page
    redirects = payload.get("query", {}).get("redirects", []) or []
    redirect_id_map: Dict[Any, Any] = {}
    redirect_title_map: Dict[str, str] = {}
    for red in redirects:
        fromid = red.get("fromid")
        toid = red.get("toid")
        if fromid is not None and toid is not None:
            redirect_id_map[fromid] = toid
        from_title = red.get("from")
        to_title = red.get("to")
        if isinstance(from_title, str) and isinstance(to_title, str):
            redirect_title_map[from_title] = to_title
    extra_info: Dict[Any, Dict[str, Any]] = {}
    for entry in results:
        pid = entry.get("pageid")
        page = page_lookup.get(pid)
        if not page and pid in redirect_id_map:
            page = page_lookup.get(redirect_id_map[pid])
        if not page:
            to_title = redirect_title_map.get(entry.get("title"))
            if to_title:
                page = title_lookup.get(to_title)
        if not page:
            continue
        entry["canonical_title"] = page.get("title") or entry.get("title")
        entry["pageid"] = page.get("pageid") or entry.get("pageid")
        entry["url"] = (
            page.get("fullurl")
            or _build_wikipedia_url(page.get("title"), page.get("pageid"))
            or entry.get("url")
        )
        entry["pageprops"] = page.get("pageprops") or {}
        if page.get("missing") == "":
            entry["missing"] = ""
        extra_info[entry.get("pageid")] = page
    return extra_info


def _is_valid_entry(entry: Dict[str, Any]) -> bool:
    url = entry.get("url") or ""
    if not url or url.startswith("https://en.wikipedia.org/w/index.php"):
        return False
    pageprops = entry.get("pageprops") or {}
    if isinstance(pageprops, dict) and pageprops.get("disambiguation") == "":
        return False
    if entry.get("missing") == "":
        return False
    return True


def _select_primary_result(
    results: List[Dict[str, Any]],
    session: requests.Session,
    timeout: float,
) -> List[Dict[str, Any]]:
    for entry in results:
        if _is_valid_entry(entry):
            return [entry]

    for entry in results:
        pageprops = entry.get("pageprops") or {}
        if isinstance(pageprops, dict) and pageprops.get("disambiguation") == "":
            disambiguation_links = _extract_links_from_disambiguation(entry, session, timeout)
            if disambiguation_links:
                return disambiguation_links

    return results[:1] if results else []


def _extract_links_from_disambiguation(
    entry: Dict[str, Any],
    session: requests.Session,
    timeout: float,
) -> List[Dict[str, Any]]:
    title = entry.get("canonical_title") or entry.get("title")
    if not title:
        return []
    params = {
        "action": "query",
        "list": "search",
        "srsearch": title,
        "format": "json",
        "utf8": 1,
        "formatversion": 2,
        "srlimit": 5,
    }
    try:
        response = session.get(
            WIKIPEDIA_ENDPOINT,
            params=params,
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
    except Exception:
        return []
    payload = response.json()
    items = payload.get("query", {}).get("search", []) or []
    candidates: List[Dict[str, Any]] = []
    for item in items:
        candidate_title = item.get("title")
        if not candidate_title or candidate_title == title:
            continue
        candidates.append(
            {
                "title": candidate_title,
                "snippet": item.get("snippet"),
                "pageid": item.get("pageid"),
                "timestamp": item.get("timestamp"),
                "url": _build_wikipedia_url(candidate_title, item.get("pageid")),
                "clean_snippet": _clean_snippet(item.get("snippet")),
            }
        )
    if not candidates:
        return []
    _augment_results_with_fullurl(session, candidates, timeout)
    for candidate in candidates:
        if _is_valid_entry(candidate):
            return [candidate]
    return candidates[:1]


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
            "formatversion": 2,
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
        results: List[Dict[str, Any]] = []
        for item in items:
            url = _build_wikipedia_url(item.get("title"), item.get("pageid"))
            results.append(
                {
                    "title": item.get("title"),
                    "snippet": item.get("snippet"),
                    "pageid": item.get("pageid"),
                    "timestamp": item.get("timestamp"),
                    "url": url,
                    "clean_snippet": _clean_snippet(item.get("snippet")),
                }
            )
        _augment_results_with_fullurl(self._session, results, self.timeout)

        filtered = _select_primary_result(results, self._session, self.timeout)
        suggestion = payload.get("query", {}).get("searchinfo", {}).get("suggestion")
        if (not filtered or not _is_valid_entry(filtered[0])) and suggestion:
            suggested = suggestion.strip()
            if suggested and suggested.lower() != query.lower():
                return self.search(suggested, limit)
        resolved_query = (suggestion or query) if filtered else query
        return {
            "query": resolved_query,
            "results": filtered,
            "suggestion": suggestion,
        }


def format_wikipedia_results(payload: Dict[str, Any]) -> str:
    items = payload.get("results") or []
    if not items:
        return "No relevant Wikipedia entries found."
    lines = []
    for idx, item in enumerate(items, start=1):
        cleaned = item.get("clean_snippet") or ""
        url = item.get("url") or "URL unavailable"
        lines.append(f"[{idx}] {cleaned} (URL: {url})")
    return "\n".join(lines)


if tool is not None:

    @tool("wikipedia_search", return_direct=False)
    def wikipedia_search(query: str) -> Dict[str, Any]:
        """Search Wikipedia for supporting evidence."""

        client = WikipediaClient()
        return client.search(query)
else:

    def wikipedia_search(query: str) -> Dict[str, Any]:  # type: ignore
        raise ImportError("langchain is required to register wikipedia_search tool")
