from typing import Any, Dict

from graph.state import GraphState
from sources import extract_sources


def build_response(state: GraphState) -> Dict[str, Any]:
    """
    Build the final structured response from the graph state.
    """
    print("---BUILD RESPONSE---")

    documents = state.get("documents", [])
    sources = extract_sources(documents)

    # Preserve the route already stored in the graph state.
    route = state.get("route", "unknown")

    # If no route was explicitly stored, infer it from sources.
    if route == "unknown":
        route = (
            "web"
            if any(source.get("type") == "web" for source in sources)
            else "local"
        )

    answer = state.get("generation", "")
    retry_count = state.get("retry_count", 0)

    return {
        "answer": answer,
        "generation": answer,
        "sources": sources,
        "route": route,
        "retry_count": retry_count,
    }