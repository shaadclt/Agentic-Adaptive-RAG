from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")

    question = state["question"]
    documents = state.get("documents", [])

    generation = generation_chain.invoke(
        {
            "context": documents,
            "question": question,
        }
    )

    retry_count = state.get("retry_count", 0)

    return {
        "generation": generation,
        "question": question,
        "retry_count": retry_count + 1,
    }