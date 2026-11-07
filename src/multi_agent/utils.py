"""Small helpers shared by the multi-agent graph and the attribution judge."""

import json
import re
from typing import Any


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Tolerant JSON extraction: strips code fences, takes the outermost {...}."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
