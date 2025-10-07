"""Lightweight social graph helpers."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, MutableMapping, Set


class CommunityGraph:
    """Undirected social graph linking user ids."""

    def __init__(self) -> None:
        self._edges: MutableMapping[str, Set[str]] = defaultdict(set)

    def add_edge(self, source: str, target: str) -> None:
        self._edges[source].add(target)
        self._edges[target].add(source)

    def neighbours(self, node_id: str) -> List[str]:
        return list(self._edges.get(node_id, ()))

    def nodes(self) -> List[str]:
        return list(self._edges.keys())

    def degree(self, node_id: str) -> int:
        return len(self._edges.get(node_id, ()))

    def ingest_edges(self, edges: Iterable[tuple[str, str]]) -> None:
        for src, dst in edges:
            self.add_edge(src, dst)
