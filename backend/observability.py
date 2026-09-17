import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


OBSERVABILITY_FILE = Path("./observability_history.jsonl")


class RunObserver:
    """
    Records one RAG execution for observability.
    """

    def __init__(
        self,
        question: str,
        storage_path: Path = OBSERVABILITY_FILE,
    ):
        self.question = question
        self.storage_path = Path(storage_path)

    def _write(self, record: Dict[str, Any]) -> None:
        self.storage_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.storage_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            json.dump(
                record,
                file,
                ensure_ascii=False,
            )
            file.write("\n")

    def record(
        self,
        route: str = "unknown",
        retrieved_documents: int = 0,
        relevant_documents: int = 0,
        grounded: bool = False,
        answers_question: bool = False,
        retry_count: int = 0,
        latency_seconds: float = 0.0,
        sources: Optional[List[Dict[str, Any]]] = None,
        success: bool = True,
        error: str = "",
        hitl_status: str = "",
        hitl_reason: str = "",
    ) -> None:
        """
        Persist one observability record.
        """

        sources = sources or []

        local_source_count = sum(
            1
            for source in sources
            if source.get("type") == "local"
        )

        web_source_count = sum(
            1
            for source in sources
            if source.get("type") == "web"
        )

        record = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),

            "question": self.question,

            "route": route,

            "retrieved_documents": int(
                retrieved_documents or 0
            ),

            "relevant_documents": int(
                relevant_documents or 0
            ),

            "grounded": bool(grounded),

            "answers_question": bool(
                answers_question
            ),

            "retry_count": int(
                retry_count or 0
            ),

            "latency_seconds": round(
                float(latency_seconds or 0.0),
                4,
            ),

            "source_count": len(sources),

            "local_source_count": local_source_count,

            "web_source_count": web_source_count,

            "success": bool(success),

            "error": str(error or ""),

            "hitl_status": str(
                hitl_status or ""
            ),

            "hitl_reason": str(
                hitl_reason or ""
            ),
        }

        self._write(record)


def load_observability_history(
    storage_path: Path = OBSERVABILITY_FILE,
) -> List[Dict[str, Any]]:
    """
    Load observability records from JSONL storage.
    """

    storage_path = Path(storage_path)

    if not storage_path.exists():
        return []

    records: List[Dict[str, Any]] = []

    with storage_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

                if isinstance(record, dict):
                    records.append(record)

            except json.JSONDecodeError:
                # Ignore malformed individual records
                # instead of breaking the whole dashboard.
                continue

    return records


def calculate_observability_summary(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Calculate dashboard-level observability metrics.
    """

    total_runs = len(records)

    if total_runs == 0:
        return {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "success_rate": 0.0,
            "average_latency_seconds": 0.0,
            "average_retries": 0.0,
            "grounded_rate": 0.0,
            "answer_quality_rate": 0.0,
            "local_route_count": 0,
            "web_route_count": 0,
            "hitl_approved_count": 0,
            "hitl_rejected_count": 0,
        }

    successful_runs = sum(
        1
        for record in records
        if record.get("success", False)
    )

    failed_runs = total_runs - successful_runs

    total_latency = sum(
        float(
            record.get(
                "latency_seconds",
                0.0,
            )
            or 0.0
        )
        for record in records
    )

    total_retries = sum(
        int(
            record.get(
                "retry_count",
                0,
            )
            or 0
        )
        for record in records
    )

    grounded_count = sum(
        1
        for record in records
        if record.get("grounded", False)
    )

    answer_quality_count = sum(
        1
        for record in records
        if record.get(
            "answers_question",
            False,
        )
    )

    local_route_count = sum(
        1
        for record in records
        if record.get("route") == "vectorstore"
    )

    web_route_count = sum(
        1
        for record in records
        if record.get("route") == "websearch"
    )

    hitl_approved_count = sum(
        1
        for record in records
        if record.get("hitl_status") == "approved"
    )

    hitl_rejected_count = sum(
        1
        for record in records
        if record.get("hitl_status") == "rejected"
    )

    return {
        "total_runs": total_runs,

        "successful_runs": successful_runs,

        "failed_runs": failed_runs,

        "success_rate": round(
            successful_runs / total_runs,
            4,
        ),

        "average_latency_seconds": round(
            total_latency / total_runs,
            4,
        ),

        "average_retries": round(
            total_retries / total_runs,
            4,
        ),

        "grounded_rate": round(
            grounded_count / total_runs,
            4,
        ),

        "answer_quality_rate": round(
            answer_quality_count / total_runs,
            4,
        ),

        "local_route_count": local_route_count,

        "web_route_count": web_route_count,

        "hitl_approved_count": hitl_approved_count,

        "hitl_rejected_count": hitl_rejected_count,
    }


def clear_observability_history(
    storage_path: Path = OBSERVABILITY_FILE,
) -> None:
    """
    Delete all observability history.
    """

    storage_path = Path(storage_path)

    if storage_path.exists():
        storage_path.unlink()