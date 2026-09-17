import json

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional


OBSERVABILITY_FILE = Path(
    "./observability_history.jsonl"
)


class RunObserver:
    """
    Collect and persist observability and security
    information for a single Agentic RAG execution.
    """

    def __init__(
        self,
        question: str,
        storage_path: Path | str = OBSERVABILITY_FILE,
    ) -> None:

        self.question = question

        self.storage_path = Path(
            storage_path
        )

        self.started_at = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        self._start_time = perf_counter()

    def finish(
        self,
        result: Dict[str, Any],
        error: Optional[str] = None,
    ) -> Dict[str, Any]:

        latency_seconds = (
            perf_counter()
            - self._start_time
        )

        sources = result.get(
            "sources",
            [],
        )

        observation = {
            "timestamp": self.started_at,

            "question": self.question,

            "route": result.get(
                "route",
                "unknown",
            ),

            "retrieved_documents": result.get(
                "retrieved_documents",
                0,
            ),

            "relevant_documents": result.get(
                "relevant_documents",
                0,
            ),

            "grounded": result.get(
                "grounded",
                False,
            ),

            "answers_question": result.get(
                "answers_question",
                False,
            ),

            "retry_count": result.get(
                "retry_count",
                0,
            ),

            "latency_seconds": round(
                latency_seconds,
                4,
            ),

            "source_count": len(
                sources
            ),

            "local_source_count": sum(
                source.get("type") == "local"
                for source in sources
                if isinstance(
                    source,
                    dict,
                )
            ),

            "web_source_count": sum(
                source.get("type") == "web"
                for source in sources
                if isinstance(
                    source,
                    dict,
                )
            ),

            "success": error is None,

            "error": error or "",

            # HITL
            "hitl_status": result.get(
                "hitl_status",
                "",
            ),

            "hitl_reason": result.get(
                "hitl_reason",
                "",
            ),

            # Security
            "security_status": result.get(
                "security_status",
                "passed",
            ),

            "security_reason": result.get(
                "security_reason",
                "",
            ),

            "security_event": result.get(
                "security_event",
                "",
            ),

            "security_redactions": result.get(
                "security_redactions",
                0,
            ),
        }

        self._save(
            observation
        )

        return observation

    def fail(
        self,
        error: Exception,
    ) -> Dict[str, Any]:

        return self.finish(
            {},
            error=str(error),
        )

    def _save(
        self,
        observation: Dict[str, Any],
    ) -> None:

        self.storage_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.storage_path.open(
            "a",
            encoding="utf-8",
        ) as file:

            json.dump(
                observation,
                file,
                ensure_ascii=False,
            )

            file.write("\n")


def load_observations(
    storage_path: Path | str = OBSERVABILITY_FILE,
) -> List[Dict[str, Any]]:

    path = Path(
        storage_path
    )

    if not path.exists():
        return []

    observations = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(
                    line
                )

                if isinstance(
                    record,
                    dict,
                ):
                    observations.append(
                        record
                    )

            except json.JSONDecodeError:
                continue

    return observations


def summarize_observations(
    observations: List[Dict[str, Any]],
) -> Dict[str, Any]:

    if not observations:

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

            "security_blocked_count": 0,
            "security_sanitized_count": 0,
            "security_event_count": 0,
        }

    total = len(
        observations
    )

    successful = sum(
        bool(
            item.get(
                "success",
                False,
            )
        )
        for item in observations
    )

    grounded = sum(
        bool(
            item.get(
                "grounded",
                False,
            )
        )
        for item in observations
    )

    answers_question = sum(
        bool(
            item.get(
                "answers_question",
                False,
            )
        )
        for item in observations
    )

    local_routes = sum(
        item.get("route") == "local"
        for item in observations
    )

    web_routes = sum(
        item.get("route") == "web"
        for item in observations
    )

    hitl_approved = sum(
        item.get("hitl_status")
        == "approved"
        for item in observations
    )

    hitl_rejected = sum(
        item.get("hitl_status")
        == "rejected"
        for item in observations
    )

    security_blocked = sum(
        item.get(
            "security_status"
        )
        == "blocked"
        for item in observations
    )

    security_sanitized = sum(
        item.get(
            "security_status"
        )
        == "output_sanitized"
        for item in observations
    )

    security_events = sum(
        bool(
            item.get(
                "security_event",
                "",
            )
        )
        for item in observations
    )

    total_latency = sum(
        float(
            item.get(
                "latency_seconds",
                0.0,
            )
        )
        for item in observations
    )

    total_retries = sum(
        int(
            item.get(
                "retry_count",
                0,
            )
        )
        for item in observations
    )

    return {
        "total_runs": total,

        "successful_runs": successful,

        "failed_runs": total - successful,

        "success_rate": successful / total,

        "average_latency_seconds":
            total_latency / total,

        "average_retries":
            total_retries / total,

        "grounded_rate":
            grounded / total,

        "answer_quality_rate":
            answers_question / total,

        "local_route_count":
            local_routes,

        "web_route_count":
            web_routes,

        "hitl_approved_count":
            hitl_approved,

        "hitl_rejected_count":
            hitl_rejected,

        "security_blocked_count":
            security_blocked,

        "security_sanitized_count":
            security_sanitized,

        "security_event_count":
            security_events,
    }


def run_with_observability(
    app,
    question: str,
) -> Dict[str, Any]:

    observer = RunObserver(
        question=question
    )

    try:

        result = app.invoke(
            {
                "question": question
            }
        )

        observer.finish(
            result
        )

        return result

    except Exception as exc:

        observer.fail(
            exc
        )

        raise