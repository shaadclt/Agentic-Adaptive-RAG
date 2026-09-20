from typing import Any, Dict

from backend.graph.chains.generation import generation_chain
from backend.graph.context import format_documents
from backend.graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")

    question = state["question"]

    documents = state.get(
        "documents",
        [],
    )

    context = format_documents(
        documents
    )

    generation = generation_chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return {
        "generation": generation,
        "question": question,
    }