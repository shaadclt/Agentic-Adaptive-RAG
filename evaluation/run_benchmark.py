import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

from graph.graph import app


DATASET_FILE = Path("evaluation/benchmark_dataset.json")
RESULTS_FILE = Path("evaluation/benchmark_results.json")


def load_dataset() -> List[Dict[str, Any]]:
    with DATASET_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def contains_expected_answer(answer: str, expected_terms: List[str]) -> bool:
    if not expected_terms:
        return True

    normalized_answer = answer.lower()

    return all(
        term.lower() in normalized_answer
        for term in expected_terms
    )


def evaluate_question(item: Dict[str, Any]) -> Dict[str, Any]:
    question = item["question"]
    expected_route = item["expected_route"]
    expected_terms = item.get("expected_answer_contains", [])

    print()
    print("=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    start = perf_counter()

    result = app.invoke(
        {
            "question": question,
            "retry_count": 0,
        }
    )

    latency = perf_counter() - start

    answer = result.get("answer", result.get("generation", ""))
    actual_route = result.get("route", "unknown")

    grounded = bool(result.get("grounded", False))
    answers_question = bool(result.get("answers_question", False))

    routing_correct = actual_route == expected_route

    answer_contains_expected = contains_expected_answer(
        answer,
        expected_terms,
    )

    passed = (
        routing_correct
        and grounded
        and answers_question
        and answer_contains_expected
    )

    evaluation = {
        "question": question,
        "expected_route": expected_route,
        "actual_route": actual_route,
        "routing_correct": routing_correct,
        "answer": answer,
        "expected_answer_contains": expected_terms,
        "answer_contains_expected": answer_contains_expected,
        "grounded": grounded,
        "answers_question": answers_question,
        "retrieved_documents": result.get("retrieved_documents", 0),
        "relevant_documents": result.get("relevant_documents", 0),
        "retrieval_relevance_rate": (
            result.get("relevant_documents", 0)
            / result.get("retrieved_documents", 1)
            if result.get("retrieved_documents", 0) > 0
            else 0.0
        ),
        "retry_count": result.get("retry_count", 0),
        "latency_seconds": latency,
        "passed": passed,
    }

    print(f"Route: {actual_route}")
    print(f"Expected route: {expected_route}")
    print(f"Routing correct: {routing_correct}")
    print(f"Grounded: {grounded}")
    print(f"Answers question: {answers_question}")
    print(f"Expected terms found: {answer_contains_expected}")
    print(f"Retries: {evaluation['retry_count']}")
    print(f"Latency: {latency:.2f}s")
    print(f"PASS: {passed}")

    return evaluation


def build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "questions_evaluated": 0,
            "routing_accuracy": 0.0,
            "grounded_answer_rate": 0.0,
            "answer_quality_rate": 0.0,
            "average_retrieval_relevance": 0.0,
            "average_latency_seconds": 0.0,
            "average_retries": 0.0,
            "overall_pass_rate": 0.0,
        }

    total = len(results)

    routing_correct = sum(
        result["routing_correct"]
        for result in results
    )

    grounded = sum(
        result["grounded"]
        for result in results
    )

    answer_quality = sum(
        result["answers_question"]
        for result in results
    )

    retrieval_relevance = sum(
        result["retrieval_relevance_rate"]
        for result in results
    )

    latency = sum(
        result["latency_seconds"]
        for result in results
    )

    retries = sum(
        result["retry_count"]
        for result in results
    )

    passed = sum(
        result["passed"]
        for result in results
    )

    return {
        "questions_evaluated": total,
        "routing_accuracy": routing_correct / total,
        "grounded_answer_rate": grounded / total,
        "answer_quality_rate": answer_quality / total,
        "average_retrieval_relevance": retrieval_relevance / total,
        "average_latency_seconds": latency / total,
        "average_retries": retries / total,
        "overall_pass_rate": passed / total,
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print()
    print()
    print("Evaluation Benchmark")
    print("--------------------------------------------")
    print(
        f"Questions evaluated:       "
        f"{summary['questions_evaluated']}"
    )
    print(
        f"Routing accuracy:          "
        f"{summary['routing_accuracy']:.1%}"
    )
    print(
        f"Grounded answer rate:     "
        f"{summary['grounded_answer_rate']:.1%}"
    )
    print(
        f"Answer quality rate:      "
        f"{summary['answer_quality_rate']:.1%}"
    )
    print(
        f"Average retrieval rate:   "
        f"{summary['average_retrieval_relevance']:.1%}"
    )
    print(
        f"Average latency:           "
        f"{summary['average_latency_seconds']:.2f}s"
    )
    print(
        f"Average retries:           "
        f"{summary['average_retries']:.2f}"
    )
    print(
        f"Overall pass rate:        "
        f"{summary['overall_pass_rate']:.1%}"
    )
    print("--------------------------------------------")


def save_results(
    results: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "summary": summary,
        "results": results,
    }

    with RESULTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:
    dataset = load_dataset()

    print()
    print(f"Loaded {len(dataset)} benchmark questions.")

    results = []

    for item in dataset:
        try:
            result = evaluate_question(item)
            results.append(result)
        except Exception as exc:
            print()
            print(f"ERROR evaluating question: {exc}")

            results.append(
                {
                    "question": item["question"],
                    "expected_route": item["expected_route"],
                    "actual_route": "error",
                    "routing_correct": False,
                    "answer": "",
                    "expected_answer_contains": item.get(
                        "expected_answer_contains",
                        [],
                    ),
                    "answer_contains_expected": False,
                    "grounded": False,
                    "answers_question": False,
                    "retrieved_documents": 0,
                    "relevant_documents": 0,
                    "retrieval_relevance_rate": 0.0,
                    "retry_count": 0,
                    "latency_seconds": 0.0,
                    "passed": False,
                    "error": str(exc),
                }
            )

    summary = build_summary(results)

    save_results(results, summary)

    print_summary(summary)

    print()
    print(f"Detailed results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
