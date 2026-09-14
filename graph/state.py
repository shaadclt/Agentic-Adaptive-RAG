from typing import Any, Dict, List, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    question: str
    generation: str
    answer: str
    web_search: bool
    route: str
    router_context: str
    candidate_documents: List[Document]
    documents: List[Document]
    sources: List[Dict[str, Any]]
    retry_count: int
    retrieved_documents: int
    relevant_documents: int
    grounded: bool
    answers_question: bool

    # Human-in-the-loop state
    web_search_approved: bool
    hitl_status: str
    hitl_reason: str