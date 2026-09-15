from typing import Any, Dict

from graph.state import GraphState


REJECTION_MESSAGE = (
    "I couldn't answer this confidently using "
    "the available local documents, and web search "
    "was not approved."
)


def web_search_rejected(
    state: GraphState,
) -> Dict[str, Any]:
    """
    Return a transparent response when the user
    rejects the web-search request.
    """

    return {
        "question": state["question"],
        "generation": REJECTION_MESSAGE,
        "answer": REJECTION_MESSAGE,
        "route": "web_rejected",
        "hitl_status": "rejected",
        "grounded": False,
        "answers_question": False,
        "sources": [],
    }