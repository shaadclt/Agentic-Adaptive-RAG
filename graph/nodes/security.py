from typing import Any, Dict

from graph.state import GraphState
from security.prompt_guard import check_prompt


SECURITY_BLOCK_MESSAGE = (
    "I can't process this request because it contains "
    "patterns associated with prompt-injection or "
    "instruction-override attacks."
)


def security_check(
    state: GraphState,
) -> Dict[str, Any]:
    """
    Validate user input before it reaches the router,
    retrieval pipeline, or external tools.
    """

    print("---SECURITY CHECK---")

    question = state["question"]

    result = check_prompt(question)

    if not result.allowed:
        print("---SECURITY CHECK: BLOCKED---")

        return {
            "question": question,
            "generation": SECURITY_BLOCK_MESSAGE,
            "answer": SECURITY_BLOCK_MESSAGE,
            "route": "security_blocked",
            "security_status": "blocked",
            "security_reason": result.reason,
            "security_event": "prompt_injection_detected",
            "security_redactions": 0,
        }

    print("---SECURITY CHECK: PASSED---")

    return {
        "question": question,
        "security_status": "passed",
        "security_reason": result.reason,
        "security_event": "",
        "security_redactions": 0,
    }


def route_after_security(
    state: GraphState,
) -> str:
    """
    Decide whether the request can enter the RAG pipeline.
    """

    if state.get("security_status") == "blocked":
        return "blocked"

    return "allowed"