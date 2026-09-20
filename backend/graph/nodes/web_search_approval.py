from typing import Any, Dict

from langgraph.types import interrupt

from backend.graph.state import GraphState


def request_web_search_approval(
    state: GraphState,
) -> Dict[str, Any]:
    """
    Pause execution and request human approval
    before performing an external web search.
    """

    question = state["question"]

    decision = interrupt(
        {
            "type": "web_search_approval",
            "question": question,
            "title": "Web search required",
            "message": (
                "The local knowledge base does not contain "
                "enough information to confidently answer "
                "this question. A web search is required."
            ),
        }
    )

    approved = bool(decision)

    if approved:
        print("---HUMAN APPROVED WEB SEARCH---")

        return {
            "web_search_approved": True,
            "hitl_status": "approved",
            "hitl_reason": (
                "Human approved web search."
            ),
        }

    print("---HUMAN REJECTED WEB SEARCH---")

    return {
        "web_search_approved": False,
        "hitl_status": "rejected",
        "hitl_reason": (
            "Human rejected web search."
        ),
    }