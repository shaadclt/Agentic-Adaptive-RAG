from typing import Any, Dict

from backend.graph.state import GraphState
from backend.security.output_guard import sanitize_output


def output_security_check(
    state: GraphState,
) -> Dict[str, Any]:
    """
    Sanitize model output before it reaches the API/frontend.
    """

    print("---OUTPUT SECURITY CHECK---")

    generation = state.get(
        "generation",
        "",
    )

    result = sanitize_output(
        generation,
    )

    if result.safe:
        print("---OUTPUT SECURITY CHECK: PASSED---")

        return {
            "generation": result.output,
            "security_redactions": 0,
        }

    print(
        "---OUTPUT SECURITY CHECK: "
        f"{result.redactions} REDACTIONS---"
    )

    return {
        "generation": result.output,
        "security_status": "output_sanitized",
        "security_reason": result.reason,
        "security_event": "output_secret_redacted",
        "security_redactions": result.redactions,
    }