import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------
# Make the project root importable when this file is executed directly.
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from graph.graph import app


DATASET_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "benchmark_dataset.json"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "benchmark_results.json"
)

REQUEST_DELAY_SECONDS = 5


def load_dataset() -> List[Dict[str, Any]]:
    with DATASET_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def is_rate_limit_error(error: Exception) -> bool:
    text = str(error).lower()

    return (
        "rate limit" in text
        or "rate_limit" in text
        or "429" in text
        or "tokens per minute" in text
        or "tpm" in text
    )


def run_question(question: str) -> Dict[str, Any]:
    start = time.perf_counter()

    result = app.invoke(
        {
            "question": question,
            "retry_count": 0,
        }
    )

    latency = time.perf_counter() - start

    result["latency_seconds"] = latency

    return result


def evaluate_question(item: Dict[str, Any]) -> Dict[str, Any]:
    question = item["question"]

    expected_route = item.get(
        "expected_route",
        "unknown",
    )

    expected_terms = item.get(
        "expected_answer_contains",
        [],
    )

    try:
        result = run_question(question)

        answer = str(
            result.get(
                "answer",
                result.get("generation", ""),
            )
        )

        actual_route = result.get(
            "route",
            "unknown",
        )

        routing_correct = (
            actual_route == expected_route
        )

        answer_lower = answer.lower()

        answer_contains_expected = all(
            str(term).lower() in answer_lower
            for term in expected_terms
        )

        answers_question = bool(
            result.get(
                "answers_question",
                False,
            )
        )

        grounded = bool(
            result.get(
                "grounded",
                False,
            )
        )

        passed = (
            routing_correct
            and grounded
            and answers_question
            and answer_contains_expected
        )

        return {
            "question": question,
            "expected_route": expected_route,
            "actual_route": actual_route,
            "routing_correct": routing_correct,
            "retrieved_documents": result.get(
                "retrieved_documents",
                0,
            ),
            "relevant_documents": result.get(
                "relevant_documents",
                0,
            ),
            "grounded": grounded,
            "answers_question": answers_question,
            "answer_contains_expected": answer_contains_expected,
            "retry_count": result.get(
                "retry_count",
                0,
            ),
            "latency_seconds": result.get(
                "latency_seconds",
                0.0,
            ),
            "answer": answer,
            "passed": passed,
            "error": None,
        }

    except Exception as exc:
        return {
            "question": question,
            "expected_route": expected_route,
            "actual_route": "error",
            "routing_correct": False,
            "retrieved_documents": 0,
            "relevant_documents": 0,
            "grounded": False,
            "answers_question": False,
            "answer_contains_expected": False,
            "retry_count": 0,
            "latency_seconds": 0.0,
            "answer": "",
            "passed": False,
            "error": str(exc),
            "rate_limited": is_rate_limit_error(exc),
        }


def calculate_summary(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:

    total = len(results)

    if total == 0:
        return {
            "questions_evaluated": 0,
            "successful_questions": 0,
            "benchmark_failures": 0,
            "errors": 0,
            "rate_limited": 0,
            "routing_accuracy": 0.0,
            "grounded_answer_rate": 0.0,
            "answer_quality_rate": 0.0,
            "average_local_retrieval_rate": 0.0,
            "average_latency_seconds": 0.0,
            "average_retries": 0.0,
            "overall_pass_rate": 0.0,
        }

    error_results = [
        result
        for result in results
        if result.get("error")
    ]

    successful_results = [
        result
        for result in results
        if not result.get("error")
    ]

    local_results = [
        result
        for result in successful_results
        if result.get("expected_route") == "local"
    ]

    routing_correct = sum(
        result.get(
            "routing_correct",
            False,
        )
        for result in successful_results
    )

    grounded_count = sum(
        result.get(
            "grounded",
            False,
        )
        for result in successful_results
    )

    answer_quality_count = sum(
        result.get(
            "answers_question",
            False,
        )
        for result in successful_results
    )

    passed = sum(
        result.get(
            "passed",
            False,
        )
        for result in successful_results
    )

    if local_results:

        retrieval_rates = []

        for result in local_results:

            retrieved = result.get(
                "retrieved_documents",
                0,
            )

            relevant = result.get(
                "relevant_documents",
                0,
            )

            if retrieved:
                retrieval_rates.append(
                    relevant / retrieved
                )

        average_local_retrieval_rate = (
            sum(retrieval_rates)
            / len(retrieval_rates)
            if retrieval_rates
            else 0.0
        )

    else:
        average_local_retrieval_rate = 0.0

    total_latency = sum(
        result.get(
            "latency_seconds",
            0.0,
        )
        for result in successful_results
    )

    total_retries = sum(
        result.get(
            "retry_count",
            0,
        )
        for result in successful_results
    )

    total_successful = len(
        successful_results
    )

    return {
        "questions_evaluated": total,
        "successful_questions": total_successful,

        "benchmark_failures": sum(
            not result.get(
                "passed",
                False,
            )
            for result in successful_results
        ),

        "errors": len(error_results),

        "rate_limited": sum(
            result.get(
                "rate_limited",
                False,
            )
            for result in error_results
        ),

        "routing_accuracy": (
            routing_correct / total_successful
            if total_successful
            else 0.0
        ),

        "grounded_answer_rate": (
            grounded_count / total_successful
            if total_successful
            else 0.0
        ),

        "answer_quality_rate": (
            answer_quality_count / total_successful
            if total_successful
            else 0.0
        ),

        "average_local_retrieval_rate": (
            average_local_retrieval_rate
        ),

        "average_latency_seconds": (
            total_latency / total_successful
            if total_successful
            else 0.0
        ),

        "average_retries": (
            total_retries / total_successful
            if total_successful
            else 0.0
        ),

        "overall_pass_rate": (
            passed / total_successful
            if total_successful
            else 0.0
        ),
    }


def save_results(
    results: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "summary": summary,
        "results": results,
    }

    with RESULTS_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )


def print_summary(
    summary: Dict[str, Any],
) -> None:

    print()
    print("=" * 70)
    print("AGENTIC ADAPTIVE RAG — BENCHMARK")
    print("=" * 70)

    print(
        f"Questions evaluated:        "
        f"{summary['questions_evaluated']}"
    )

    print(
        f"Successful questions:       "
        f"{summary['successful_questions']}"
    )

    print(
        f"Benchmark failures:         "
        f"{summary['benchmark_failures']}"
    )

    print(
        f"Errors:                     "
        f"{summary['errors']}"
    )

    print(
        f"Rate-limited:               "
        f"{summary['rate_limited']}"
    )

    print(
        f"Routing accuracy:           "
        f"{summary['routing_accuracy']:.1%}"
    )

    print(
        f"Grounded answer rate:       "
        f"{summary['grounded_answer_rate']:.1%}"
    )

    print(
        f"Answer quality rate:        "
        f"{summary['answer_quality_rate']:.1%}"
    )

    print(
        f"Local retrieval relevance:  "
        f"{summary['average_local_retrieval_rate']:.1%}"
    )

    print(
        f"Average latency:             "
        f"{summary['average_latency_seconds']:.2f}s"
    )

    print(
        f"Average retries:             "
        f"{summary['average_retries']:.2f}"
    )

    print(
        f"Overall pass rate:           "
        f"{summary['overall_pass_rate']:.1%}"
    )

    print("=" * 70)


def main() -> None:

    dataset = load_dataset()

    results = []

    print(
        f"Running benchmark with "
        f"{len(dataset)} questions..."
    )

    for index, item in enumerate(dataset):

        question = item["question"]

        print()
        print(
            f"[{index + 1}/{len(dataset)}] "
            f"{question}"
        )

        result = evaluate_question(item)

        results.append(result)

        if result.get("error"):

            print(
                f"ERROR: {result['error']}"
            )

        else:

            print(
                f"Route: {result['actual_route']}"
            )

            print(
                f"Expected: "
                f"{result['expected_route']}"
            )

            print(
                f"Passed: "
                f"{result['passed']}"
            )

            print(
                f"Latency: "
                f"{result['latency_seconds']:.2f}s"
            )

        if index < len(dataset) - 1:
            time.sleep(
                REQUEST_DELAY_SECONDS
            )

    summary = calculate_summary(
        results
    )

    save_results(
        results,
        summary,
    )

    print_summary(summary)

    print(
        f"Results saved to: "
        f"{RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()