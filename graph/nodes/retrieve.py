import time
from typing import Any, Dict

from graph.state import GraphState
from retrieval import retriever


MAX_RETRIEVAL_RETRIES = 2
RETRIEVAL_RETRY_DELAY_SECONDS = 2


def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve documents for the RAG pipeline.

    Reuses candidate documents produced during the
    evidence-aware routing stage when available.

    A small retry mechanism protects against transient
    embedding/provider failures such as HTTP 502 errors.
    """

    print("---RETRIEVE---")

    question = state["question"]

    candidate_documents = state.get(
        "candidate_documents",
        [],
    )

    if candidate_documents:
        print(
            "---REUSING ROUTER RETRIEVAL: "
            f"{len(candidate_documents)} DOCUMENTS---"
        )

        return {
            "documents": candidate_documents,
            "question": question,
            "retrieved_documents": len(candidate_documents),
        }

    last_error = None

    for attempt in range(
        MAX_RETRIEVAL_RETRIES + 1
    ):

        try:

            documents = retriever.invoke(
                question
            )

            print(
                f"---RETRIEVED "
                f"{len(documents)} DOCUMENTS---"
            )

            return {
                "documents": documents,
                "question": question,
                "retrieved_documents": len(documents),
            }

        except Exception as exc:

            last_error = exc

            if attempt >= MAX_RETRIEVAL_RETRIES:
                break

            print(
                "---RETRIEVAL ERROR: "
                f"{exc}"
            )

            print(
                "---RETRYING RETRIEVAL "
                f"({attempt + 1}/"
                f"{MAX_RETRIEVAL_RETRIES})---"
            )

            time.sleep(
                RETRIEVAL_RETRY_DELAY_SECONDS
            )

    raise last_error