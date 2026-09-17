from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from evaluation import EvaluationResult, EvaluationTracker
from ingestion import (
    SUPPORTED_EXTENSIONS,
    build_vectorstore,
    delete_document,
    list_documents,
)
from observability import (
    RunObserver,
    clear_observability_history,
    get_observability_summary,
    load_observability_history,
)

from graph.graph import app as rag_app


load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "http://localhost:3000",
)

MAX_UPLOAD_SIZE_MB = int(
    os.getenv("MAX_UPLOAD_SIZE_MB", "25")
)

MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

api = FastAPI(
    title="Agentic Adaptive RAG API",
    description=(
        "Agentic RAG backend with adaptive routing, "
        "document ingestion, HITL web-search approval, "
        "observability, security and evaluation."
    ),
    version="1.0.0",
)


# Keep both variable names available.
# This prevents problems if the frontend/backend tooling expects either.
app = api


api.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_URL,
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None


class HITLRequest(BaseModel):
    thread_id: str
    approved: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_str(value: Any) -> str:
    if value is None:
        return ""

    return str(value)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower() in {
            "true",
            "1",
            "yes",
            "y",
        }

    return bool(value)


def _normalise_route(result: Dict[str, Any]) -> str:
    route = result.get("route")

    if route:
        route = str(route)

        if route == "vectorstore":
            return "local"

        if route == "websearch":
            return "web"

        return route

    # Some graph implementations expose web_search separately.
    if result.get("web_search") is True:
        return "web"

    return "unknown"


def _build_evaluation_result(
    question: str,
    result: Dict[str, Any],
    latency_seconds: float,
    error: str = "",
) -> EvaluationResult:

    answer = _safe_str(
        result.get(
            "answer",
            result.get("generation", ""),
        )
    )

    retrieved_documents = _safe_int(
        result.get(
            "retrieved_documents",
            len(result.get("documents", []) or []),
        )
    )

    relevant_documents = _safe_int(
        result.get(
            "relevant_documents",
            0,
        )
    )

    return EvaluationResult(
        question=question,
        answer=answer,
        route=_normalise_route(result),
        retrieved_documents=retrieved_documents,
        relevant_documents=relevant_documents,
        grounded=_safe_bool(
            result.get("grounded", False)
        ),
        answers_question=_safe_bool(
            result.get("answers_question", False)
        ),
        retry_count=_safe_int(
            result.get("retry_count", 0)
        ),
        latency_seconds=round(
            latency_seconds,
            4,
        ),
        sources=result.get(
            "sources",
            [],
        ) or [],
        error=error,
    )


def _record_evaluation(
    question: str,
    result: Dict[str, Any],
    latency_seconds: float,
    error: str = "",
) -> None:

    try:
        tracker = EvaluationTracker()

        evaluation_result = _build_evaluation_result(
            question=question,
            result=result,
            latency_seconds=latency_seconds,
            error=error,
        )

        tracker.add(evaluation_result)

    except Exception as exc:
        # Evaluation must never break the actual RAG request.
        print(
            "WARNING: Failed to record evaluation:",
            str(exc),
        )


def _get_thread_id(request_thread_id: Optional[str]) -> str:
    if request_thread_id:
        return request_thread_id

    # Simple deterministic fallback for clients that don't
    # explicitly provide a thread ID.
    return "default"


# ---------------------------------------------------------------------------
# Root / health
# ---------------------------------------------------------------------------


@api.get("/")
def root() -> Dict[str, str]:
    return {
        "name": "Agentic Adaptive RAG API",
        "status": "running",
    }


@api.get("/health")
def health() -> Dict[str, Any]:
    try:
        documents = list_documents()

        return {
            "status": "healthy",
            "documents": len(documents),
            "evaluation_enabled": True,
            "observability_enabled": True,
        }

    except Exception as exc:
        return {
            "status": "degraded",
            "error": str(exc),
            "evaluation_enabled": True,
            "observability_enabled": True,
        }


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


@api.get("/documents")
def get_documents() -> Dict[str, Any]:
    try:
        documents = list_documents()

        return {
            "documents": documents,
            "count": len(documents),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load documents: {exc}",
        )


@api.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
) -> Dict[str, Any]:

    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files were provided.",
        )

    uploaded: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    temp_paths: List[str] = []

    try:
        for uploaded_file in files:

            filename = uploaded_file.filename or ""

            if not filename:
                rejected.append(
                    {
                        "file_name": filename,
                        "reason": "Missing filename.",
                    }
                )
                continue

            extension = Path(
                filename
            ).suffix.lower()

            if extension not in SUPPORTED_EXTENSIONS:
                rejected.append(
                    {
                        "file_name": filename,
                        "reason": (
                            "Unsupported file type. "
                            f"Supported types: "
                            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                        ),
                    }
                )
                continue

            contents = await uploaded_file.read()

            if len(contents) > MAX_UPLOAD_SIZE_BYTES:
                rejected.append(
                    {
                        "file_name": filename,
                        "reason": (
                            f"File exceeds the "
                            f"{MAX_UPLOAD_SIZE_MB} MB limit."
                        ),
                    }
                )
                continue

            temp_dir = tempfile.mkdtemp(
                prefix="agentic_rag_"
            )

            temp_path = os.path.join(
                temp_dir,
                Path(filename).name,
            )

            with open(
                temp_path,
                "wb",
            ) as output_file:
                output_file.write(contents)

            temp_paths.append(temp_path)

            uploaded.append(
                {
                    "file_name": filename,
                    "temp_path": temp_path,
                    "size_bytes": len(contents),
                    "extension": extension,
                }
            )

        if not uploaded:
            return {
                "message": "No files were uploaded.",
                "uploaded": [],
                "rejected": rejected,
            }

        build_vectorstore(
            [
                item["temp_path"]
                for item in uploaded
            ]
        )

        return {
            "message": (
                f"Successfully processed "
                f"{len(uploaded)} file(s)."
            ),
            "uploaded": [
                {
                    "file_name": item["file_name"],
                    "size_bytes": item["size_bytes"],
                    "extension": item["extension"],
                }
                for item in uploaded
            ],
            "rejected": rejected,
            "documents": list_documents(),
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {exc}",
        )

    finally:

        # Remove temporary files after Chroma ingestion.
        for temp_path in temp_paths:

            try:
                temp_dir = os.path.dirname(
                    temp_path
                )

                if os.path.exists(temp_path):
                    os.remove(temp_path)

                if os.path.isdir(temp_dir):
                    shutil.rmtree(
                        temp_dir,
                        ignore_errors=True,
                    )

            except Exception:
                pass


@api.delete("/documents/{document_id}")
def remove_document(
    document_id: str,
) -> Dict[str, Any]:

    try:

        deleted = delete_document(
            document_id
        )

        if not deleted:
            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        return {
            "message": "Document deleted.",
            "document_id": document_id,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {exc}",
        )


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


@api.post("/chat")
def chat(
    request: ChatRequest,
) -> Dict[str, Any]:

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    thread_id = _get_thread_id(
        request.thread_id
    )

    observer = RunObserver(
        question=question
    )

    start_time = perf_counter()

    try:

        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        result = rag_app.invoke(
            {
                "question": question,
            },
            config=config,
        )

        latency = (
            perf_counter() - start_time
        )

        result = dict(result or {})

        # ---------------------------------------------------------------
        # HITL interrupt handling
        # ---------------------------------------------------------------

        if "__interrupt__" in result:

            interrupts = result.get(
                "__interrupt__"
            ) or []

            interrupt_value: Any = None

            if interrupts:
                first_interrupt = interrupts[0]

                interrupt_value = getattr(
                    first_interrupt,
                    "value",
                    first_interrupt,
                )

            observer.finish(
                result=result,
                latency_seconds=latency,
            )

            # We deliberately don't treat an interrupted request as
            # a completed evaluation yet.
            return {
                "status": "approval_required",
                "thread_id": thread_id,
                "question": question,
                "interrupt": interrupt_value,
                "hitl_status": "pending",
                "message": (
                    "The agent wants to use "
                    "web search. Approval required."
                ),
            }

        # ---------------------------------------------------------------
        # Normal completed request
        # ---------------------------------------------------------------

        observer.finish(
            result=result,
            latency_seconds=latency,
        )

        _record_evaluation(
            question=question,
            result=result,
            latency_seconds=latency,
        )

        return {
            "status": "completed",
            "thread_id": thread_id,
            "question": question,
            "answer": _safe_str(
                result.get(
                    "answer",
                    result.get(
                        "generation",
                        "",
                    ),
                )
            ),
            "generation": _safe_str(
                result.get(
                    "generation",
                    "",
                )
            ),
            "route": _normalise_route(
                result
            ),
            "sources": result.get(
                "sources",
                [],
            ) or [],
            "retrieved_documents": _safe_int(
                result.get(
                    "retrieved_documents",
                    len(
                        result.get(
                            "documents",
                            [],
                        )
                        or []
                    ),
                )
            ),
            "relevant_documents": _safe_int(
                result.get(
                    "relevant_documents",
                    0,
                )
            ),
            "grounded": _safe_bool(
                result.get(
                    "grounded",
                    False,
                )
            ),
            "answers_question": _safe_bool(
                result.get(
                    "answers_question",
                    False,
                )
            ),
            "retry_count": _safe_int(
                result.get(
                    "retry_count",
                    0,
                )
            ),
            "latency_seconds": round(
                latency,
                4,
            ),
            "hitl_status": result.get(
                "hitl_status"
            ),
            "hitl_reason": result.get(
                "hitl_reason"
            ),
            "security_status": result.get(
                "security_status"
            ),
            "security_reason": result.get(
                "security_reason"
            ),
            "security_redactions": result.get(
                "security_redactions",
                [],
            ),
        }

    except Exception as exc:

        latency = (
            perf_counter() - start_time
        )

        error_message = str(exc)

        # Record failed requests as evaluation results.
        _record_evaluation(
            question=question,
            result={},
            latency_seconds=latency,
            error=error_message,
        )

        try:
            observer.fail(
                error_message,
                latency_seconds=latency,
            )
        except Exception:
            pass

        raise HTTPException(
            status_code=500,
            detail=error_message,
        )


# ---------------------------------------------------------------------------
# HITL approval
# ---------------------------------------------------------------------------


@api.post("/chat/approval")
def chat_approval(
    request: HITLRequest,
) -> Dict[str, Any]:

    thread_id = request.thread_id

    if not thread_id:
        raise HTTPException(
            status_code=400,
            detail="thread_id is required.",
        )

    start_time = perf_counter()

    try:

        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        result = rag_app.invoke(
            {
                "web_search_approved": request.approved,
                "hitl_status": (
                    "approved"
                    if request.approved
                    else "rejected"
                ),
            },
            config=config,
        )

        latency = (
            perf_counter() - start_time
        )

        result = dict(result or {})

        question = _safe_str(
            result.get(
                "question",
                "",
            )
        )

        observer = RunObserver(
            question=question
        )

        observer.finish(
            result=result,
            latency_seconds=latency,
        )

        # Only record a completed answer.
        _record_evaluation(
            question=question,
            result=result,
            latency_seconds=latency,
        )

        return {
            "status": "completed",
            "thread_id": thread_id,
            "question": question,
            "answer": _safe_str(
                result.get(
                    "answer",
                    result.get(
                        "generation",
                        "",
                    ),
                )
            ),
            "generation": _safe_str(
                result.get(
                    "generation",
                    "",
                )
            ),
            "route": _normalise_route(
                result
            ),
            "sources": result.get(
                "sources",
                [],
            ) or [],
            "retrieved_documents": _safe_int(
                result.get(
                    "retrieved_documents",
                    len(
                        result.get(
                            "documents",
                            [],
                        )
                        or []
                    ),
                )
            ),
            "relevant_documents": _safe_int(
                result.get(
                    "relevant_documents",
                    0,
                )
            ),
            "grounded": _safe_bool(
                result.get(
                    "grounded",
                    False,
                )
            ),
            "answers_question": _safe_bool(
                result.get(
                    "answers_question",
                    False,
                )
            ),
            "retry_count": _safe_int(
                result.get(
                    "retry_count",
                    0,
                )
            ),
            "latency_seconds": round(
                latency,
                4,
            ),
            "hitl_status": result.get(
                "hitl_status"
            ),
            "hitl_reason": result.get(
                "hitl_reason"
            ),
            "security_status": result.get(
                "security_status"
            ),
            "security_reason": result.get(
                "security_reason"
            ),
            "security_redactions": result.get(
                "security_redactions",
                [],
            ),
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@api.get("/evaluation")
def get_evaluation() -> Dict[str, Any]:

    try:

        tracker = EvaluationTracker()

        return {
            "results": [
                result.to_dict()
                for result in tracker.results
            ],
            "summary": tracker.summary(),
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load evaluation history: "
                f"{exc}"
            ),
        )


@api.delete("/evaluation")
def clear_evaluation() -> Dict[str, str]:

    try:

        tracker = EvaluationTracker()
        tracker.clear()

        return {
            "message": (
                "Evaluation history cleared."
            )
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to clear evaluation history: "
                f"{exc}"
            ),
        )


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


@api.get("/observability")
def get_observability() -> Dict[str, Any]:

    try:

        history = load_observability_history()

        return {
            "runs": history,
            "summary": get_observability_summary(),
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load observability history: "
                f"{exc}"
            ),
        )


@api.delete("/observability")
def clear_observability() -> Dict[str, str]:

    try:

        clear_observability_history()

        return {
            "message": (
                "Observability history cleared."
            )
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to clear observability history: "
                f"{exc}"
            ),
        )