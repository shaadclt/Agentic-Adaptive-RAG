from typing import Any, Dict

from graph.state import GraphState
from retrieval import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve relevant documents from the local vector store.
    """

    print("---RETRIEVE---")

    question = state["question"]

    documents = retriever.invoke(question)

    print(
        f"---RETRIEVED {len(documents)} DOCUMENTS---"
    )

    return {
        "documents": documents,
        "question": question,
        "retrieved_documents": len(documents),
    }