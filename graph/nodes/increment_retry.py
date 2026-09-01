from typing import Any, Dict

from graph.state import GraphState


def increment_retry(state: GraphState) -> Dict[str, Any]:
    retry_count = state.get("retry_count", 0)

    print(
        f"---INCREMENT GENERATION RETRY: "
        f"{retry_count + 1}---"
    )

    return {
        "retry_count": retry_count + 1,
    }