import json
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
)

from graph.graph import app


DATASET_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "benchmark_dataset.json"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "ragas_results.json"
)


def load_benchmark() -> List[Dict[str, Any]]:
    with DATASET_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def run_application(
    question: str,
) -> Dict[str, Any]:

    return app.invoke(
        {
            "question": question,
            "retry_count": 0,
        }
    )


def build_samples(
    benchmark: List[Dict[str, Any]],
) -> List[SingleTurnSample]:

    samples = []

    for item in benchmark:

        question = item["question"]

        # RAGAS evaluation is focused on questions
        # that are expected to use the local knowledge base.
        if item.get("expected_route") != "local":
            continue

        print()
        print(
            f"Evaluating: {question}"
        )

        result = run_application(
            question
        )

        documents = result.get(
            "documents",
            [],
        )

        contexts = [
            document.page_content
            for document in documents
            if getattr(
                document,
                "page_content",
                "",
            )
        ]

        answer = result.get(
            "generation",
            result.get(
                "answer",
                "",
            ),
        )

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )

        samples.append(sample)

        print(
            f"Route: "
            f"{result.get('route', 'unknown')}"
        )

        print(
            f"Retrieved contexts: "
            f"{len(contexts)}"
        )

        print(
            f"Retries: "
            f"{result.get('retry_count', 0)}"
        )

    return samples


def save_results(
    result: Any,
) -> None:

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if hasattr(
        result,
        "to_pandas",
    ):
        dataframe = result.to_pandas()

        records = dataframe.to_dict(
            orient="records"
        )

        summary = {}

        for column in dataframe.columns:
            try:
                summary[column] = float(
                    dataframe[column].mean()
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

    else:
        records = []

        summary = {}

    output = {
        "summary": summary,
        "results": records,
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

    benchmark = load_benchmark()

    samples = build_samples(
        benchmark
    )

    if not samples:
        raise RuntimeError(
            "No local benchmark samples "
            "were available for RAGAS evaluation."
        )

    dataset = EvaluationDataset(
        samples=samples
    )

    print()
    print("=" * 60)
    print("RAGAS EVALUATION")
    print("=" * 60)

    # The existing LangChain LLM/embedding objects are
    # intentionally reused so evaluation uses the same
    # model configuration as the application.
    from model import llm_model

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
        ],
        llm=llm_model,
    )

    print()
    print(result)

    save_results(
        result
    )

    print()
    print(
        f"RAGAS results saved to: "
        f"{RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()