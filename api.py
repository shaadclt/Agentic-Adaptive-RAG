from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from evaluation import EvaluationResult, EvaluationTracker
from graph.graph import app
from ingestion import (
    build_vectorstore,
    delete_document,
    list_documents,
)
from observability import RunObserver
from sources import extract_sources
from fastapi.middleware.cors import CORSMiddleware


api = FastAPI(
    title="Agentic Adaptive RAG API",
    description=(
        "Production-oriented Agentic RAG API "
        "powered by LangGraph."
    ),
    version="1.0.0",
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


evaluation_tracker = EvaluationTracker()


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask the RAG agent.",
    )


class SourceResponse(BaseModel):
    type: str
    title: str = ""
    file_name: str = ""
    url: str = ""
    source: str = ""
    document_id: str = ""


class MetricsResponse(BaseModel):
    retrieved_documents: int
    relevant_documents: int
    grounded: bool
    answers_question: bool
    retry_count: int
    latency_seconds: float


class ChatResponse(BaseModel):
    answer: str
    route: str
    sources: List[SourceResponse]
    metrics: MetricsResponse


class DocumentResponse(BaseModel):
    file_name: str
    file_type: str
    document_id: str
    source: str


@api.get(
    "/health",
)
def health() -> Dict[str, str]:
    return {
        "status": "healthy",
        "service": "agentic-adaptive-rag",
    }


@api.get(
    "/documents",
    response_model=List[DocumentResponse],
)
def get_documents() -> List[Dict[str, Any]]:
    return list_documents()


@api.post(
    "/documents/upload",
)
async def upload_documents(
    files: List[UploadFile] = File(...),
) -> Dict[str, Any]:

    supported_extensions = {
        ".pdf",
        ".docx",
        ".txt",
        ".md",
    }

    upload_directory = (
        "uploads"
    )

    from pathlib import Path

    upload_path = Path(
        upload_directory
    )

    upload_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_files = []
    rejected_files = []

    for file in files:

        if not file.filename:
            rejected_files.append(
                {
                    "file_name": "unknown",
                    "reason": "Missing filename.",
                }
            )
            continue

        suffix = Path(
            file.filename
        ).suffix.lower()

        if suffix not in supported_extensions:
            rejected_files.append(
                {
                    "file_name": file.filename,
                    "reason": (
                        "Unsupported file type."
                    ),
                }
            )
            continue

        destination = (
            upload_path
            / Path(file.filename).name
        )

        content = await file.read()

        destination.write_bytes(
            content
        )

        saved_files.append(
            str(destination)
        )

    if not saved_files:
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "No supported files "
                    "were uploaded."
                ),
                "rejected_files": (
                    rejected_files
                ),
            },
        )

    try:

        build_vectorstore(
            saved_files
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Document ingestion failed: "
                f"{exc}"
            ),
        ) from exc

    return {
        "message": (
            "Documents processed successfully."
        ),
        "uploaded_files": [
            Path(path).name
            for path in saved_files
        ],
        "rejected_files": rejected_files,
    }


@api.delete(
    "/documents/{document_id}",
)
def remove_document(
    document_id: str,
) -> Dict[str, Any]:

    deleted = delete_document(
        document_id
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=(
                "Document not found."
            ),
        )

    return {
        "message": (
            "Document removed successfully."
        ),
        "document_id": document_id,
    }


@api.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
) -> ChatResponse:

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail=(
                "Question cannot be empty."
            ),
        )

    observer = RunObserver(
        question=question
    )

    import time

    start_time = time.perf_counter()

    try:

        result = app.invoke(
            {
                "question": question,
                "retry_count": 0,
            }
        )

        latency = (
            time.perf_counter()
            - start_time
        )

        observer.finish(
            result
        )

    except Exception as exc:

        observer.fail(
            exc
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"RAG execution failed: "
                f"{exc}"
            ),
        ) from exc

    answer = result.get(
        "answer",
        result.get(
            "generation",
            "No answer generated.",
        ),
    )

    sources = result.get(
        "sources"
    )

    if sources is None:

        sources = extract_sources(
            result.get(
                "documents",
                [],
            )
        )

    route = result.get(
        "route",
        "unknown",
    )

    retry_count = result.get(
        "retry_count",
        0,
    )

    retrieved_documents = result.get(
        "retrieved_documents",
        0,
    )

    relevant_documents = result.get(
        "relevant_documents",
        0,
    )

    grounded = result.get(
        "grounded",
        False,
    )

    answers_question = result.get(
        "answers_question",
        False,
    )

    evaluation_result = EvaluationResult(
        question=question,
        answer=answer,
        route=route,
        retrieved_documents=(
            retrieved_documents
        ),
        relevant_documents=(
            relevant_documents
        ),
        grounded=grounded,
        answers_question=(
            answers_question
        ),
        retry_count=retry_count,
        latency_seconds=latency,
        sources=sources,
    )

    evaluation_tracker.add(
        evaluation_result
    )

    metrics = MetricsResponse(
        retrieved_documents=(
            retrieved_documents
        ),
        relevant_documents=(
            relevant_documents
        ),
        grounded=grounded,
        answers_question=(
            answers_question
        ),
        retry_count=retry_count,
        latency_seconds=round(
            latency,
            4,
        ),
    )

    return ChatResponse(
        answer=answer,
        route=route,
        sources=[
            SourceResponse(
                **source
            )
            for source in sources
        ],
        metrics=metrics,
    )


@api.get(
    "/observability",
)
def get_observability() -> Dict[str, Any]:

    summary = (
        evaluation_tracker.summary()
    )

    return {
        "evaluation": summary,
    }