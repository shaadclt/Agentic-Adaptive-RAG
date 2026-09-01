from typing import List, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    """
    State passed between nodes in the Agentic Adaptive RAG graph.

    Attributes:
        question: The user's question.
        generation: The latest LLM-generated answer.
        web_search: Whether web search should be performed.
        documents: Retrieved documents from local or web sources.
        retry_count: Number of generation retries performed.
    """

    question: str
    generation: str
    web_search: bool
    documents: List[Document]
    retry_count: int