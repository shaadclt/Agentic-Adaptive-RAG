import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List


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


def load_dataset() -> List[Dict[str, Any]]:
    with DATASET_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def contains_expected_answer(
    answer: str,
    expected_terms: List[str],
) -> bool:
    answer_lower = answer.lower()

    return all(
        term.lower() in answer_lower
        for term in expected_terms
    )


def evaluate_question(
    item: Dict[str, Any],
) -> Dict[str, Any]:

    question = item["question"]

    expected_route = item["expected_route"]

    expected_terms = item.get(
        "expected_answer_contains",
        [],
    )

    start = perf_counter()

    result: Dict[str, Any] = {}

    try:
        result = app.invoke(
            {
                "question": question,
                "retry_count": 0,
            }
        )

        latency = perf_counter() - start

        actual_route = result.get(
            "route",
            "unknown",
        )

        grounded = result.get(
            "grounded",
            False,
        )

        answers_question = result.get(
            "answers_question",
            False,
        )

        retry_count = result.get(
            "retry_count",
            0,
        )

        answer = result.get(
            "generation",
            result.get(
                "answer",
                "",
            ),
        )

        retrieved_documents = result.get(
            "retrieved_documents",
            0,
        )

        relevant_documents = result.get(
            "relevant_documents",
            0,
        )

        answer_contains_expected = (
            contains_expected_answer(
                answer,
                expected_terms,
            )
        )

        routing_correct = (
            actual_route == expected_route
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
            "retrieved_documents": retrieved_documents,
            "relevant_documents": relevant_documents,
            "grounded": grounded,
            "answers_question": answers_question,
            "answer_contains_expected": (
                answer_contains_expected
            ),
            "retry_count": retry_count,
            "latency_seconds": round(
                latency,
                2,
            ),
            "answer": answer,
            "passed": passed,
            "error": None,
        }

    except Exception as exc:
        latency = perf_counter() - start

        retry_count = result.get(
            "retry_count",
            0,
        )

        actual_route = result.get(
            "route",
            "error",
        )

        error_message = str(exc)

        rate_limited = (
            "rate_limit_exceeded" in error_message
            or "Rate limit" in error_message
            or "429" in error_message
        )

        return {
            "question": question,
            "expected_route": expected_route,
            "actual_route": actual_route,
            "routing_correct": False,
            "retrieved_documents": result.get(
                "retrieved_documents",
                0,
            ),
            "relevant_documents": result.get(
                "relevant_documents",
                0,
            ),
            "grounded": False,
            "answers_question": False,
            "answer_contains_expected": False,
            "retry_count": retry_count,
            "latency_seconds": round(
                latency,
                2,
            ),
            "answer": result.get(
                "generation",
                "",
            ),
            "passed": False,
            "error": error_message,
            "rate_limited": rate_limited,
        }


def build_summary(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:

    total = len(results)

    if total == 0:
        return {
            "questions_evaluated": 0,
            "successful_questions": 0,
            "failed_questions": 0,
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

    successful_results = [
        result
        for result in results
        if not result.get("error")
    ]

    error_results = [
        result
        for result in results
        if result.get("error")
    ]

    rate_limited_results = [
        result
        for result in error_results
        if result.get("rate_limited", False)
    ]

    total_successful = len(
        successful_results
    )

    routing_correct = sum(
        result["routing_correct"]
        for result in successful_results
    )

    grounded = sum(
        result["grounded"]
        for result in successful_results
    )

    answers_question = sum(
        result["answers_question"]
        for result in successful_results
    )

    passed = sum(
        result["passed"]
        for result in successful_results
    )

    # Retrieval quality should only measure questions
    # expected to use the local knowledge base.
    local_results = [
        result
        for result in successful_results
        if result["expected_route"] == "local"
    ]

    local_retrieval_rates = []

    for result in local_results:
        retrieved = result[
            "retrieved_documents"
        ]

        relevant = result[
            "relevant_documents"
        ]

        if retrieved:
            local_retrieval_rates.append(
                relevant / retrieved
            )

    average_local_retrieval_rate = (
        sum(local_retrieval_rates)
        / len(local_retrieval_rates)
        if local_retrieval_rates
        else 0.0
    )

    average_latency = (
        sum(
            result["latency_seconds"]
            for result in successful_results
        )
        / total_successful
        if total_successful
        else 0.0
    )

    average_retries = (
        sum(
            result["retry_count"]
            for result in results
        )
        / total
    )

    return {
        "questions_evaluated": total,
        "successful_questions": total_successful,
        "failed_questions": sum(
            not result["passed"]
            for result in successful_results
        ),
        "errors": len(error_results),
        "rate_limited": len(
            rate_limited_results
        ),
        "routing_accuracy": (
            routing_correct / total_successful
            if total_successful
            else 0.0
        ),
        "grounded_answer_rate": (
            grounded / total_successful
            if total_successful
            else 0.0
        ),
        "answer_quality_rate": (
            answers_question / total_successful
            if total_successful
            else 0.0
        ),
        "average_local_retrieval_rate": (
            average_local_retrieval_rate
        ),
        "average_latency_seconds": (
            average_latency
        ),
        "average_retries": (
            average_retries
        ),
        "overall_pass_rate": (
            passed / total_successful
            if total_successful
            else 0.0
        ),
    }


def print_summary(
    results: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:

    print()
    print("=" * 70)
    print("EVALUATION BENCHMARK")
    print("=" * 70)

    for index, result in enumerate(
        results,
        start=1,
    ):
        print()
        print(
            f"{index}. {result['question']}"
        )

        print(
            f"   Route: "
            f"{result['actual_route']} "
            f"(expected: "
            f"{result['expected_route']})"
        )

        print(
            f"   Routing correct: "
            f"{result['routing_correct']}"
        )

        print(
            f"   Retrieved documents: "
            f"{result['retrieved_documents']}"
        )

        print(
            f"   Relevant documents: "
            f"{result['relevant_documents']}"
        )

        print(
            f"   Grounded: "
            f"{result['grounded']}"
        )

        print(
            f"   Answers question: "
            f"{result['answers_question']}"
        )

        print(
            f"   Expected terms found: "
            f"{result['answer_contains_expected']}"
        )

        print(
            f"   Retries: "
            f"{result['retry_count']}"
        )

        print(
            f"   Latency: "
            f"{result['latency_seconds']}s"
        )

        print(
            f"   PASS: "
            f"{result['passed']}"
        )

        if result.get("rate_limited"):
            print(
                "   ERROR TYPE: RATE LIMITED"
            )

        if result.get("error"):
            print(
                f"   ERROR: "
                f"{result['error']}"
            )

    print()
    print("-" * 70)

    print(
        f"Questions evaluated:          "
        f"{summary['questions_evaluated']}"
    )

    print(
        f"Successful questions:         "
        f"{summary['successful_questions']}"
    )

    print(
        f"Failed questions:             "
        f"{summary['failed_questions']}"
    )

    print(
        f"Errors:                       "
        f"{summary['errors']}"
    )

    print(
        f"Rate-limited:                 "
        f"{summary['rate_limited']}"
    )

    print(
        f"Routing accuracy:             "
        f"{summary['routing_accuracy']:.1%}"
    )

    print(
        f"Grounded answer rate:        "
        f"{summary['grounded_answer_rate']:.1%}"
    )

    print(
        f"Answer quality rate:         "
        f"{summary['answer_quality_rate']:.1%}"
    )

    print(
        f"Local retrieval relevance:   "
        f"{summary['average_local_retrieval_rate']:.1%}"
    )

    print(
        f"Average latency:              "
        f"{summary['average_latency_seconds']:.2f}s"
    )

    print(
        f"Average retries:              "
        f"{summary['average_retries']:.2f}"
    )

    print(
        f"Overall pass rate:            "
        f"{summary['overall_pass_rate']:.1%}"
    )

    print("=" * 70)


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


def main() -> None:

    dataset = load_dataset()

    print(
        f"Loaded {len(dataset)} "
        "benchmark questions."
    )

    results = []

    for item in dataset:

        print()
        print("=" * 70)
        print(
            f"QUESTION: {item['question']}"
        )
        print("=" * 70)

        result = evaluate_question(item)

        results.append(result)

        if result.get("error"):
            print(
                f"ERROR evaluating question: "
                f"{result['error']}"
            )
        else:
            print(
                f"Route: "
                f"{result['actual_route']}"
            )

            print(
                f"Expected route: "
                f"{result['expected_route']}"
            )

            print(
                f"Routing correct: "
                f"{result['routing_correct']}"
            )

            print(
                f"Grounded: "
                f"{result['grounded']}"
            )

            print(
                f"Answers question: "
                f"{result['answers_question']}"
            )

            print(
                f"Expected terms found: "
                f"{result['answer_contains_expected']}"
            )

            print(
                f"Retrieved documents: "
                f"{result['retrieved_documents']}"
            )

            print(
                f"Relevant documents: "
                f"{result['relevant_documents']}"
            )

            retrieved = result[
                "retrieved_documents"
            ]

            relevant = result[
                "relevant_documents"
            ]

            retrieval_rate = (
                relevant / retrieved
                if retrieved
                else 0.0
            )

            print(
                f"Retrieval relevance: "
                f"{retrieval_rate:.1%}"
            )

            print(
                f"Retries: "
                f"{result['retry_count']}"
            )

            print(
                f"Latency: "
                f"{result['latency_seconds']}s"
            )

            print(
                f"PASS: "
                f"{result['passed']}"
            )

    summary = build_summary(results)

    save_results(
        results,
        summary,
    )

    print_summary(
        results,
        summary,
    )

    print()
    print(
        f"Detailed results saved to: "
        f"{RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()