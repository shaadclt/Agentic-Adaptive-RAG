from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from evaluation import EvaluationResult, EvaluationTracker
from graph.graph import app
from ingestion import (
    delete_document,
    document_exists,
    get_files_from_directory,
    ingest_file,
    list_documents,
)
from observability import (
    RunObserver,
    load_observations,
    summarize_observations,
)
from response import Source
from security.prompt_guard import check_prompt


# ============================================================
# CONFIGURATION
# ============================================================

UPLOAD_DIR = Path("./uploads")

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FASTAPI
# ============================================================

app_api = FastAPI(
    title="Agentic Adaptive RAG API",
    description=(
        "Production-oriented Agentic RAG API with "
        "adaptive routing, HITL web-search approval, "
        "security controls, evaluation, and observability."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app_api.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODELS
# ============================================================

class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
    )


class WebSearchApprovalRequest(BaseModel):
    approved: bool


class SourceResponse(BaseModel):
    type: str = ""
    title: str = ""
    file_name: str = ""
    url: str = ""
    source: str = ""
    document_id: str = ""


class MetricsResponse(BaseModel):
    retrieved_documents: int = 0
    relevant_documents: int = 0
    grounded: bool = False
    answers_question: bool = False
    retry_count: int = 0
    latency_seconds: float = 0.0


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

    # Security
    security_status: str = "passed"

    security_reason: str = ""

    security_event: str = ""

    security_redactions: int = 0


class DocumentResponse(BaseModel):
    document_id: str
    file_name: str
    file_type: str = ""
    source: str = ""
    chunk_count: int = 0


class HealthResponse(BaseModel):
    status: str
    documents: int
    version: str


# ============================================================
# HELPERS
# ============================================================

def get_graph_config(
    thread_id: str,
) -> Dict[str, Any]:
    return {
        "configurable": {
            "thread_id": thread_id,
        }
    }


def extract_interrupt(
    result: Dict[str, Any],
) -> Dict[str, Any] | None:

    interrupts = result.get(
        "__interrupt__"
    )

    if not interrupts:
        return None

    first_interrupt = interrupts[0]

    value = getattr(
        first_interrupt,
        "value",
        None,
    )

    if isinstance(
        value,
        dict,
    ):
        return value

    return {
        "type": "human_approval",
        "title": "Approval required",
        "message": "Human approval is required to continue.",
    }


def build_chat_response(
    result: Dict[str, Any],
    thread_id: str,
    latency_seconds: float,
) -> ChatResponse:

    raw_sources = result.get(
        "sources",
        [],
    )

    sources = []

    for source in raw_sources:
        if isinstance(
            source,
            dict,
        ):
            sources.append(
                SourceResponse(
                    **{
                        key: source.get(
                            key,
                            "",
                        )
                        for key in SourceResponse.model_fields
                    }
                )
            )

    metrics = MetricsResponse(
        retrieved_documents=result.get(
            "retrieved_documents",
            0,
        ),
        relevant_documents=result.get(
            "relevant_documents",
            0,
        ),
        grounded=result.get(
            "grounded",
            False,
        ),
        answers_question=result.get(
            "answers_question",
            False,
        ),
        retry_count=result.get(
            "retry_count",
            0,
        ),
        latency_seconds=round(
            latency_seconds,
            4,
        ),
    )

    return ChatResponse(
        status="completed",

        answer=result.get(
            "answer",
            result.get(
                "generation",
                "",
            ),
        ),

        route=result.get(
            "route",
            "",
        ),

        sources=sources,

        metrics=metrics,

        thread_id=thread_id,

        security_status=result.get(
            "security_status",
            "passed",
        ),

        security_reason=result.get(
            "security_reason",
            "",
        ),

        security_event=result.get(
            "security_event",
            "",
        ),

        security_redactions=result.get(
            "security_redactions",
            0,
        ),

        approval_required=False,
    )


def save_uploaded_file(
    upload: UploadFile,
) -> Path:

    if not upload.filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a filename.",
        )

    original_name = Path(
        upload.filename
    ).name

    extension = Path(
        original_name
    ).suffix.lower()

    allowed_extensions = {
        ".pdf",
        ".docx",
        ".txt",
        ".md",
    }

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Supported types: PDF, DOCX, TXT, MD."
            ),
        )

    safe_name = (
        f"{uuid.uuid4().hex}"
        f"_{original_name}"
    )

    destination = (
        UPLOAD_DIR
        / safe_name
    )

    try:
        with destination.open(
            "wb"
        ) as output:

            shutil.copyfileobj(
                upload.file,
                output,
            )

    except Exception as exc:

        if destination.exists():
            destination.unlink()

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to save uploaded file: {exc}"
            ),
        ) from exc

    return destination


# ============================================================
# HEALTH
# ============================================================

@app_api.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:

    documents = list_documents()

    return HealthResponse(
        status="ok",
        documents=len(documents),
        version="1.0.0",
    )


# ============================================================
# DOCUMENTS
# ============================================================

@app_api.get(
    "/documents",
)
def get_documents():
    """
    Return all documents currently stored in the
    local knowledge base.
    """

    try:
        documents = list_documents()

        return {
            "documents": documents,
            "count": len(documents),
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {exc}",
        ) from exc


@app_api.post(
    "/documents/upload",
)
def upload_documents(
    files: List[UploadFile] = File(...),
):
    """
    Upload and ingest documents into the local
    vector knowledge base.
    """

    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files were uploaded.",
        )

    if len(files) > 5:
        raise HTTPException(
            status_code=400,
            detail="Maximum 5 files per upload.",
        )

    results = []

    for upload in files:

        path = save_uploaded_file(
            upload
        )

        try:

            result = ingest_file(
                path
            )

            results.append(
                {
                    "file_name": upload.filename,
                    "status": "ingested",
                    "result": result,
                }
            )

        except ValueError as exc:

            if path.exists():
                path.unlink()

            results.append(
                {
                    "file_name": upload.filename,
                    "status": "skipped",
                    "message": str(exc),
                }
            )

        except Exception as exc:

            if path.exists():
                path.unlink()

            results.append(
                {
                    "file_name": upload.filename,
                    "status": "failed",
                    "message": str(exc),
                }
            )

    successful = sum(
        item["status"] == "ingested"
        for item in results
    )

    return {
        "message": "Upload processing completed.",
        "uploaded": successful,
        "results": results,
        "documents": list_documents(),
    }


@app_api.delete(
    "/documents/{document_id}",
)
def remove_document(
    document_id: str,
):
    """
    Delete a document and all of its chunks
    from the vector knowledge base.
    """

    try:

        if not document_exists(
            document_id
        ):
            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        delete_document(
            document_id
        )

        return {
            "message": "Document deleted successfully.",
            "document_id": document_id,
        }

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {exc}",
        ) from exc


# ============================================================
# CHAT
# ============================================================

@app_api.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    # --------------------------------------------------------
    # Early security validation
    # --------------------------------------------------------

    security_result = check_prompt(
        question
    )

    if not security_result.allowed:

        return ChatResponse(
            status="blocked",

            answer=(
                "I can't process this request because "
                "it contains patterns associated with "
                "prompt-injection or instruction-override attacks."
            ),

            route="security_blocked",

            sources=[],

            thread_id="",

            approval_required=False,

            security_status="blocked",

            security_reason=security_result.reason,

            security_event="prompt_injection_detected",

            security_redactions=0,
        )

    # --------------------------------------------------------
    # Create HITL thread
    # --------------------------------------------------------

    thread_id = str(
        uuid.uuid4()
    )

    config = get_graph_config(
        thread_id
    )

    observer = RunObserver(
        question=question
    )

    try:

        result = app.invoke(
            {
                "question": question,
                "retry_count": 0,
            },
            config=config,
        )

        # ----------------------------------------------------
        # HITL interrupt
        # ----------------------------------------------------

        interrupt_data = extract_interrupt(
            result
        )

        if interrupt_data:

            return ChatResponse(
                status="approval_required",

                answer="",

                route="web",

                sources=[],

                metrics=None,

                thread_id=thread_id,

                approval_required=True,

                approval_type=interrupt_data.get(
                    "type",
                    "human_approval",
                ),

                approval_title=interrupt_data.get(
                    "title",
                    "Approval required",
                ),

                approval_message=interrupt_data.get(
                    "message",
                    "Human approval is required to continue.",
                ),

                security_status=result.get(
                    "security_status",
                    "passed",
                ),

                security_reason=result.get(
                    "security_reason",
                    "",
                ),

                security_event=result.get(
                    "security_event",
                    "",
                ),

                security_redactions=result.get(
                    "security_redactions",
                    0,
                ),
            )

        # ----------------------------------------------------
        # Completed normally
        # ----------------------------------------------------

        observation = observer.finish(
            result
        )

        return build_chat_response(
            result=result,
            thread_id=thread_id,
            latency_seconds=observation.get(
                "latency_seconds",
                0.0,
            ),
        )

    except Exception as exc:

        observer.fail(
            exc
        )

        raise HTTPException(
            status_code=500,
            detail=f"RAG execution failed: {exc}",
        ) from exc


# ============================================================
# HITL WEB SEARCH APPROVAL
# ============================================================

@app_api.post(
    "/chat/{thread_id}/web-search",
    response_model=ChatResponse,
)
def web_search_decision(
    thread_id: str,
    request: WebSearchApprovalRequest,
):

    from langgraph.types import Command

    config = get_graph_config(
        thread_id
    )

    observer = RunObserver(
        question=(
            f"HITL web-search decision "
            f"for thread {thread_id}"
        )
    )

    try:

        result = app.invoke(
            Command(
                resume=request.approved
            ),
            config=config,
        )

        # ----------------------------------------------------
        # Another interrupt should not normally occur, but
        # handle it safely if the graph asks again.
        # ----------------------------------------------------

        interrupt_data = extract_interrupt(
            result
        )

        if interrupt_data:

            return ChatResponse(
                status="approval_required",

                answer="",

                route="web",

                thread_id=thread_id,

                approval_required=True,

                approval_type=interrupt_data.get(
                    "type",
                    "human_approval",
                ),

                approval_title=interrupt_data.get(
                    "title",
                    "Approval required",
                ),

                approval_message=interrupt_data.get(
                    "message",
                    "Human approval is required.",
                ),
            )

        observation = observer.finish(
            result
        )

        return build_chat_response(
            result=result,
            thread_id=thread_id,
            latency_seconds=observation.get(
                "latency_seconds",
                0.0,
            ),
        )

    except Exception as exc:

        observer.fail(
            exc
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to resume RAG execution: {exc}"
            ),
        ) from exc


# ============================================================
# OBSERVABILITY
# ============================================================

@app_api.get(
    "/observability",
)
def get_observability():

    observations = load_observations()

    return {
        "summary": summarize_observations(
            observations
        ),
        "runs": observations,
    }


@app_api.get(
    "/observability/summary",
)
def get_observability_summary():

    observations = load_observations()

    return summarize_observations(
        observations
    )


@app_api.get(
    "/observability/runs",
)
def get_observability_runs():

    return {
        "runs": load_observations()
    }


@app_api.delete(
    "/observability",
)
def clear_observability():

    from observability import OBSERVABILITY_FILE

    if OBSERVABILITY_FILE.exists():
        OBSERVABILITY_FILE.unlink()

    return {
        "message": "Observability history cleared.",
    }


# ============================================================
# ROOT
# ============================================================

@app_api.get("/")
def root():

    return {
        "name": "Agentic Adaptive RAG API",
        "status": "running",
        "docs": "/docs",
    }


# ============================================================
# ASGI APPLICATION
# ============================================================

app = app_api