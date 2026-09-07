from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, List


@dataclass
class EvaluationResult:
    """
    Stores evaluation information for a single RAG query.
    """

    question: str
    answer: str

    route: str = "unknown"

    retrieved_documents: int = 0
    relevant_documents: int = 0

    grounded: bool = False
    answers_question: bool = False

    retry_count: int = 0

    latency_seconds: float = 0.0

    sources: List[Dict[str, Any]] = field(
        default_factory=list
    )

    @property
    def retrieval_relevance_rate(self) -> float:
        """
        Percentage of retrieved documents considered relevant.
        """
        if self.retrieved_documents == 0:
            return 0.0

        return (
            self.relevant_documents
            / self.retrieved_documents
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert evaluation result to a dictionary.
        """
        return {
            "question": self.question,
            "answer": self.answer,
            "route": self.route,
            "retrieved_documents": self.retrieved_documents,
            "relevant_documents": self.relevant_documents,
            "retrieval_relevance_rate": (
                self.retrieval_relevance_rate
            ),
            "grounded": self.grounded,
            "answers_question": self.answers_question,
            "retry_count": self.retry_count,
            "latency_seconds": self.latency_seconds,
            "sources": self.sources,
        }


class EvaluationTracker:
    """
    Tracks evaluation results across multiple queries.
    """

    def __init__(self) -> None:
        self.results: List[EvaluationResult] = []

    def add(self, result: EvaluationResult) -> None:
        self.results.append(result)

    def summary(self) -> Dict[str, Any]:
        """
        Calculate aggregate evaluation metrics.
        """

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
    """
    Execute a function and measure its latency.
    """

    start = perf_counter()

    result = func(
        *args,
        **kwargs,
    )

    latency = perf_counter() - start

    return result, latency