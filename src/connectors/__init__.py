"""
Connector registry — look up and instantiate connectors by name.

To register a new connector:
1. Create your_connector.py implementing ProbeConnector
2. Add it to CONNECTOR_REGISTRY below
3. Use it: python scripts/evaluate.py --connector your_name
"""

from src.connectors.base import (
    CorpusSchema,
    FieldInfo,
    ProbeConnector,
    ProbeResult,
    SourceDocument,
)
from src.connectors.tmdb import TMDBConnector

# add new connectors here as you build them
CONNECTOR_REGISTRY: dict[str, type[ProbeConnector]] = {
    "tmdb": TMDBConnector,
}


def get_connector(name: str, **kwargs) -> ProbeConnector:
    """
    Look up a connector by name and instantiate it.

    Raises KeyError with a helpful message if the name isn't registered.
    """
    if name not in CONNECTOR_REGISTRY:
        available = ", ".join(sorted(CONNECTOR_REGISTRY.keys()))
        raise KeyError(
            f"Unknown connector {name!r}. "
            f"Available connectors: {available}"
        )

    connector_cls = CONNECTOR_REGISTRY[name]
    return connector_cls(**kwargs)


__all__ = [
    "ProbeConnector",
    "ProbeResult",
    "SourceDocument",
    "CorpusSchema",
    "FieldInfo",
    "get_connector",
    "CONNECTOR_REGISTRY",
]
