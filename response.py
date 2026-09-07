from typing import Any, Dict, List

from pydantic import BaseModel, Field


class Source(BaseModel):
    """A source used to support an answer."""

    type: str = Field(
        description="Source type, either local or web."
    )

    title: str = Field(
        default="",
        description="Human-readable source title."
    )

    file_name: str = Field(
        default="",
        description="Local document filename."
    )

    url: str = Field(
        default="",
        description="Web source URL."
    )

    source: str = Field(
        default="",
        description="Original local source path."
    )

    document_id: str = Field(
        default="",
        description="Unique local document identifier."
    )


class RAGResponse(BaseModel):
    """Structured response returned by the RAG application."""

    answer: str = Field(
        description="Generated answer to the user's question."
    )

    sources: List[Source] = Field(
        default_factory=list,
        description="Sources used to generate the answer."
    )

    route: str = Field(
        description="Route used to answer the question."
    )

    retry_count: int = Field(
        default=0,
        description="Number of generation retries."
    )


def create_response(
    answer: str,
    sources: List[Dict[str, Any]],
    route: str,
    retry_count: int,
) -> RAGResponse:
    """Create a validated structured RAG response."""

    return RAGResponse(
        answer=answer,
        sources=[
            Source(**source)
            for source in sources
        ],
        route=route,
        retry_count=retry_count,
    )