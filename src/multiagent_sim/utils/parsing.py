"""Helper functions for parsing agent outputs."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Mapping, Sequence


def _extract_json_candidate(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None


def ensure_dict_output(
    result: Any,
    *,
    expected_keys: Sequence[str],
    defaults: Mapping[str, Any],
) -> Dict[str, Any]:
    """Ensure a dictionary-like payload with predictable keys.

    Parameters
    ----------
    result: AgentExecutor return value, string, or dict.
    expected_keys: Keys to preserve from parsed payload when available.
    defaults: Fallback values for missing keys.
    """

    payload: Dict[str, Any] = {}
    raw_output: str | None = None

    if isinstance(result, Mapping):
        if all(key in result for key in expected_keys):
            payload.update({key: result[key] for key in expected_keys})
        if "output" in result and isinstance(result["output"], str):
            raw_output = result["output"]
        payload.setdefault("raw_payload", dict(result))
    elif isinstance(result, str):
        raw_output = result
    else:
        raw_output = str(result)

    if raw_output:
        candidate = _extract_json_candidate(raw_output)
        if candidate:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, Mapping):
                    for key in expected_keys:
                        if key in parsed:
                            payload[key] = parsed[key]
            except json.JSONDecodeError:
                pass

    for key in expected_keys:
        if key not in payload:
            payload[key] = raw_output if raw_output is not None else defaults.get(key)

    for key, value in defaults.items():
        payload.setdefault(key, value)

    if raw_output is not None:
        payload.setdefault("raw_output", raw_output)

    return dict(payload)


def coerce_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def flatten_list(items: Iterable[str]) -> str:
    return " | ".join(item for item in items if item)
