from typing import Any, Dict, List

from pydantic import BaseModel, Field


class Source(BaseModel):
    type: str = Field(
        description="Source type, either local or web."
    )

    title: str = Field(
        default="",
        description="Human-readable source title.",
    )

    file_name: str = Field(
        default="",
        description="Local document filename.",
    )

    url: str = Field(
        default="",
        description="Web source URL.",
    )

    source: str = Field(
        default="",
        description="Original local source path.",
    )

    document_id: str = Field(
        default="",
        description="Unique local document identifier.",
    )


class RAGResponse(BaseModel):
    answer: str = Field(
        description="Generated answer to the user's question."
    )

    sources: List[Source] = Field(
        default_factory=list,
        description="Sources used to generate the answer.",
    )

    route: str = Field(
        description="Route used to answer the question."
    )

    retry_count: int = Field(
        default=0,
        description="Number of generation retries.",
    )

    security_status: str = Field(
        default="passed",
        description="Security status of the request.",
    )

    security_reason: str = Field(
        default="",
        description="Security decision explanation.",
    )

    security_event: str = Field(
        default="",
        description="Security event generated during the run.",
    )

    security_redactions: int = Field(
        default=0,
        description="Number of output security redactions.",
    )


def create_response(
    answer: str,
    sources: List[Dict[str, Any]],
    route: str,
    retry_count: int,
    security_status: str = "passed",
    security_reason: str = "",
    security_event: str = "",
    security_redactions: int = 0,
) -> RAGResponse:

    return RAGResponse(
        answer=answer,
        sources=[
            Source(**source)
            for source in sources
        ],
        route=route,
        retry_count=retry_count,
        security_status=security_status,
        security_reason=security_reason,
        security_event=security_event,
        security_redactions=security_redactions,
    )