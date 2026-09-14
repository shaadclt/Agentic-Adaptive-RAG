from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command
from pydantic import BaseModel, Field

from evaluation import EvaluationResult, EvaluationTracker
from graph.graph import app
from ingestion import (
    build_vectorstore,
    delete_document,
    list_documents,
)
from observability import (
    OBSERVABILITY_FILE,
    RunObserver,
    load_observations,
    summarize_observations,
)
from sources import extract_sources


api = FastAPI(
    title="Agentic Adaptive RAG API",
    description=(
        "Production-oriented Agentic RAG API "
        "powered by LangGraph."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

evaluation_tracker = EvaluationTracker()


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

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
    status: str = "completed"

    answer: str = ""
    route: str = ""

    sources: List[SourceResponse] = Field(
        default_factory=list
    )

    metrics: MetricsResponse | None = None

    thread_id: str = ""

    approval_required: bool = False
    approval_type: str = ""
    approval_title: str = ""
    approval_message: str = ""


class WebSearchApprovalRequest(BaseModel):
    approved: bool


class DocumentResponse(BaseModel):
    file_name: str
    file_type: str
    document_id: str
    source: str


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@api.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "healthy",
        "service": "agentic-adaptive-rag",
    }


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

@api.get(
    "/documents",
    response_model=List[DocumentResponse],
)
def get_documents() -> List[Dict[str, Any]]:
    return list_documents()


@api.post("/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
) -> Dict[str, Any]:

    supported_extensions = {
        ".pdf",
        ".docx",
        ".txt",
        ".md",
    }

    upload_directory = "uploads"

    upload_path = Path(upload_directory)
    upload_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_files: List[str] = []
    rejected_files: List[Dict[str, str]] = []

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
                    "reason": "Unsupported file type.",
                }
            )
            continue

        destination = (
            upload_path
            / Path(file.filename).name
        )

        content = await file.read()

        destination.write_bytes(content)

        saved_files.append(
            str(destination)
        )

    if not saved_files:
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "No supported files were uploaded."
                ),
                "rejected_files": rejected_files,
            },
        )

    try:
        build_vectorstore(saved_files)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Document ingestion failed: {exc}"
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


@api.delete("/documents/{document_id}")
def remove_document(
    document_id: str,
) -> Dict[str, Any]:

    deleted = delete_document(
        document_id
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return {
        "message": (
            "Document removed successfully."
        ),
        "document_id": document_id,
    }


# ---------------------------------------------------------------------------
# HITL Helpers
# ---------------------------------------------------------------------------

def extract_interrupt_payload(
    result: Dict[str, Any],
) -> Dict[str, Any] | None:
    """
    Extract the payload generated by LangGraph interrupt().
    """

    interrupts = result.get(
        "__interrupt__"
    )

    if not interrupts:
        return None

    first_interrupt = interrupts[0]

    payload = getattr(
        first_interrupt,
        "value",
        first_interrupt,
    )

    if isinstance(payload, dict):
        return payload

    return {
        "type": "approval",
        "title": "Approval required",
        "message": str(payload),
    }


# ---------------------------------------------------------------------------
# Build completed response
# ---------------------------------------------------------------------------

def build_chat_response(
    result: Dict[str, Any],
    thread_id: str,
    latency: float = 0.0,
) -> ChatResponse:

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

    metrics = MetricsResponse(
        retrieved_documents=retrieved_documents,
        relevant_documents=relevant_documents,
        grounded=grounded,
        answers_question=answers_question,
        retry_count=retry_count,
        latency_seconds=round(
            latency,
            4,
        ),
    )

    return ChatResponse(
        status="completed",
        answer=answer,
        route=route,
        sources=[
            SourceResponse(**source)
            for source in sources
        ],
        metrics=metrics,
        thread_id=thread_id,
    )


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

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
            detail="Question cannot be empty.",
        )

    thread_id = str(
        uuid4()
    )

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    observer = RunObserver(
        question=question
    )

    start_time = perf_counter()

    try:

        result = app.invoke(
            {
                "question": question,
                "retry_count": 0,
            },
            config,
        )

        latency = (
            perf_counter()
            - start_time
        )

        # ---------------------------------------------------------------
        # HITL interrupt
        # ---------------------------------------------------------------

        interrupt_payload = (
            extract_interrupt_payload(
                result
            )
        )

        if interrupt_payload:

            print(
                "---HUMAN APPROVAL REQUIRED---"
            )

            return ChatResponse(
                status="approval_required",
                thread_id=thread_id,
                approval_required=True,
                approval_type=(
                    interrupt_payload.get(
                        "type",
                        "approval",
                    )
                ),
                approval_title=(
                    interrupt_payload.get(
                        "title",
                        "Approval required",
                    )
                ),
                approval_message=(
                    interrupt_payload.get(
                        "message",
                        "",
                    )
                ),
            )

        # ---------------------------------------------------------------
        # Normal completed response
        # ---------------------------------------------------------------

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
                f"RAG execution failed: {exc}"
            ),
        ) from exc

    # -------------------------------------------------------------------
    # Build result
    # -------------------------------------------------------------------

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
        latency_seconds=(
            latency
        ),
        sources=sources,
    )

    evaluation_tracker.add(
        evaluation_result
    )

    return build_chat_response(
        result=result,
        thread_id=thread_id,
        latency=latency,
    )


# ---------------------------------------------------------------------------
# HITL Web Search Approval
# ---------------------------------------------------------------------------

@api.post(
    "/chat/{thread_id}/web-search",
    response_model=ChatResponse,
)
def approve_web_search(
    thread_id: str,
    request: WebSearchApprovalRequest,
) -> ChatResponse:

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    start_time = perf_counter()

    try:

        result = app.invoke(
            Command(
                resume=request.approved
            ),
            config,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to resume RAG execution: "
                f"{exc}"
            ),
        ) from exc

    latency = (
        perf_counter()
        - start_time
    )

    # -------------------------------------------------------------------
    # If another interrupt occurs
    # -------------------------------------------------------------------

    interrupt_payload = (
        extract_interrupt_payload(
            result
        )
    )

    if interrupt_payload:

        return ChatResponse(
            status="approval_required",
            thread_id=thread_id,
            approval_required=True,
            approval_type=(
                interrupt_payload.get(
                    "type",
                    "approval",
                )
            ),
            approval_title=(
                interrupt_payload.get(
                    "title",
                    "Approval required",
                )
            ),
            approval_message=(
                interrupt_payload.get(
                    "message",
                    "",
                )
            ),
        )

    # -------------------------------------------------------------------
    # Human rejected web search
    # -------------------------------------------------------------------

    if not request.approved:

        answer = result.get(
            "answer",
            result.get(
                "generation",
                (
                    "I couldn't answer this "
                    "confidently using the available "
                    "local documents, and web search "
                    "was not approved."
                ),
            ),
        )

        result["answer"] = answer
        result["generation"] = answer
        result["route"] = "web_rejected"
        result["sources"] = []

        return build_chat_response(
            result=result,
            thread_id=thread_id,
            latency=latency,
        )

    # -------------------------------------------------------------------
    # Human approved web search
    # -------------------------------------------------------------------

    observer = RunObserver(
        question=result.get(
            "question",
            "",
        )
    )

    observer._start_time = (
        observer._start_time
        - latency
    )

    observer.finish(
        result
    )

    # -------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------

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
        question=result.get(
            "question",
            "",
        ),
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

    return build_chat_response(
        result=result,
        thread_id=thread_id,
        latency=latency,
    )


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------

@api.get("/observability")
def get_observability() -> Dict[str, Any]:

    observations = load_observations()

    return {
        "summary": summarize_observations(
            observations
        ),
        "total_records": len(
            observations
        ),
    }


@api.get("/observability/summary")
def get_observability_summary() -> Dict[str, Any]:

    observations = load_observations()

    return summarize_observations(
        observations
    )


@api.get("/observability/runs")
def get_observability_runs(
    limit: int = 50,
) -> Dict[str, Any]:

    if limit < 1:
        raise HTTPException(
            status_code=400,
            detail="Limit must be at least 1.",
        )

    observations = load_observations()

    recent = observations[-limit:]

    recent.reverse()

    return {
        "runs": recent,
        "count": len(recent),
    }


@api.delete("/observability")
def clear_observability() -> Dict[str, Any]:
    """
    Clear all observability history.

    This does not affect:
    - uploaded documents
    - Chroma vector database
    - evaluation history
    - benchmark results
    """

    deleted = False

    if OBSERVABILITY_FILE.exists():
        OBSERVABILITY_FILE.unlink()
        deleted = True

    return {
        "message": (
            "Observability history cleared successfully."
        ),
        "deleted": deleted,
    }