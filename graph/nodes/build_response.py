from typing import Any, Dict

from graph.state import GraphState
from sources import extract_sources


def build_response(state: GraphState) -> Dict[str, Any]:
    """
    Build the structured response fields from the final graph state.
    """

    print("---BUILD RESPONSE---")

    documents = state.get(
        "documents",
        [],
    )

    sources = extract_sources(documents)

    route = "web" if any(
        source.get("type") == "web"
        for source in sources
    ) else "local"

    answer = state.get(
        "generation",
        "",
    )

    retry_count = state.get(
        "retry_count",
        0,
    )

    return {
        "answer": answer,
        "generation": answer,
        "sources": sources,
        "route": route,
        "retry_count": retry_count,
    }