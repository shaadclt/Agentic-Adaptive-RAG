import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional


OBSERVABILITY_FILE = Path(
    "./observability_history.jsonl"
)


class RunObserver:
    """
    Collect and persist observability data for a single
    Agentic RAG execution.
    """

    def __init__(
        self,
        question: str,
        storage_path: Path | str = OBSERVABILITY_FILE,
    ) -> None:
        self.question = question
        self.storage_path = Path(storage_path)

        self.started_at = datetime.now(
            timezone.utc
        ).isoformat()

        self._start_time = perf_counter()

    def finish(
        self,
        result: Dict[str, Any],
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Finalize and persist a run observation.
        """

        latency_seconds = (
            perf_counter() - self._start_time
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

            "source_count": len(sources),

            "local_source_count": sum(
                source.get("type") == "local"
                for source in sources
                if isinstance(source, dict)
            ),

            "web_source_count": sum(
                source.get("type") == "web"
                for source in sources
                if isinstance(source, dict)
            ),

            "success": error is None,

            "error": error or "",
        }

        self._save(
            observation
        )

        return observation

    def fail(
        self,
        error: Exception,
    ) -> Dict[str, Any]:
        """
        Record a failed execution.
        """

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


def run_with_observability(
    app,
    question: str,
) -> Dict[str, Any]:
    """
    Execute the RAG application while recording
    production-style observability information.
    """

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