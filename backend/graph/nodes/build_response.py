from typing import Any, Dict

from graph.state import GraphState
from backend.sources import extract_sources


def build_response(
    state: GraphState,
) -> Dict[str, Any]:
    print("---BUILD RESPONSE---")

    documents = state.get(
        "documents",
        [],
    )

    sources = extract_sources(
        documents,
    )

    route = state.get(
        "route",
        "unknown",
    )

    if route == "unknown":
        route = (
            "web"
            if any(
                source.get("type") == "web"
                for source in sources
            )
            else "local"
        )

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

        "retrieved_documents": state.get(
            "retrieved_documents",
            len(documents),
        ),

        "relevant_documents": state.get(
            "relevant_documents",
            0,
        ),

        "grounded": state.get(
            "grounded",
            False,
        ),

        "answers_question": state.get(
            "answers_question",
            False,
        ),

        # HITL
        "hitl_status": state.get(
            "hitl_status",
            "",
        ),

        "hitl_reason": state.get(
            "hitl_reason",
            "",
        ),

        # Security
        "security_status": state.get(
            "security_status",
            "passed",
        ),

        "security_reason": state.get(
            "security_reason",
            "",
        ),

        "security_event": state.get(
            "security_event",
            "",
        ),

        "security_redactions": state.get(
            "security_redactions",
            0,
        ),
    }