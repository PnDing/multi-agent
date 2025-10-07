"""Identifier helpers."""
from __future__ import annotations

import itertools


class IdGenerator:
    def __init__(self, prefix: str = "uid") -> None:
        self._prefix = prefix
        self._counter = itertools.count()

    def next(self) -> str:
        return f"{self._prefix}_{next(self._counter)}"
