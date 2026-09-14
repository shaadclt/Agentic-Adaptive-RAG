from typing import Any, Dict

from graph.state import GraphState


def web_search_rejected(
    state: GraphState,
) -> Dict[str, Any]:
    question = state["question"]

    return {
        "question": question,
        "generation": (
            "I couldn't answer this confidently using "
            "the available local documents, and web search "
            "was not approved."
        ),
        "route": "web_rejected",
        "hitl_status": "rejected",
    }