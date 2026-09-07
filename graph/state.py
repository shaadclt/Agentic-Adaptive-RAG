from typing import Any, Dict, List, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    question: str
    generation: str
    answer: str
    web_search: bool
    route: str
    documents: List[Document]
    sources: List[Dict[str, Any]]
    retry_count: int