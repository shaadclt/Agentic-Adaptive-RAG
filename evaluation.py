import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List


EVALUATION_FILE = Path("./evaluation_history.jsonl")


@dataclass
class EvaluationResult:
    question: str
    answer: str
    route: str = "unknown"
    retrieved_documents: int = 0
    relevant_documents: int = 0
    grounded: bool = False
    answers_question: bool = False
    retry_count: int = 0
    latency_seconds: float = 0.0
    sources: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def retrieval_relevance_rate(self) -> float:
        if self.retrieved_documents == 0:
            return 0.0

        return self.relevant_documents / self.retrieved_documents

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["retrieval_relevance_rate"] = (
            self.retrieval_relevance_rate
        )
        return data


class EvaluationTracker:
    def __init__(
        self,
        storage_path: Path | str = EVALUATION_FILE,
    ) -> None:
        self.storage_path = Path(storage_path)
        self.results: List[EvaluationResult] = []
        self.load()

    def add(self, result: EvaluationResult) -> None:
        self.results.append(result)
        self.save_result(result)

    def save_result(self, result: EvaluationResult) -> None:
        self.storage_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.storage_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            json.dump(
                result.to_dict(),
                file,
                ensure_ascii=False,
            )
            file.write("\n")

    def load(self) -> None:
        if not self.storage_path.exists():
            return

        self.results = []

        with self.storage_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                try:
                    data = json.loads(line)

                    data.pop(
                        "retrieval_relevance_rate",
                        None,
                    )

                    self.results.append(
                        EvaluationResult(**data)
                    )

                except (
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                ):
                    continue

    def clear(self) -> None:
        self.results = []

        if self.storage_path.exists():
            self.storage_path.unlink()

    def summary(self) -> Dict[str, Any]:
        if not self.results:
            return {
                "total_questions": 0,
                "average_latency_seconds": 0.0,
                "grounded_rate": 0.0,
                "answer_quality_rate": 0.0,
                "average_retries": 0.0,
            }

        total = len(self.results)

        grounded_count = sum(
            result.grounded
            for result in self.results
        )

        useful_count = sum(
            result.answers_question
            for result in self.results
        )

        total_latency = sum(
            result.latency_seconds
            for result in self.results
        )

        total_retries = sum(
            result.retry_count
            for result in self.results
        )

        return {
            "total_questions": total,
            "average_latency_seconds": (
                total_latency / total
            ),
            "grounded_rate": (
                grounded_count / total
            ),
            "answer_quality_rate": (
                useful_count / total
            ),
            "average_retries": (
                total_retries / total
            ),
        }


def measure_latency(func, *args, **kwargs):
    start = perf_counter()

    result = func(*args, **kwargs)

    latency = perf_counter() - start

    return result, latency