import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics.collections import (
    answer_relevancy,
    faithfulness,
)

from graph.chains.generation import generation_chain
from retrieval import retriever
from model import embed_model, llm_model


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


# Groq has an 8,000 TPM limit in the current project configuration.
# A small delay between questions reduces the chance of hitting
# the rolling token-per-minute limit.
REQUEST_DELAY_SECONDS = 8


def load_benchmark() -> List[Dict[str, Any]]:
    with DATASET_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def run_rag(question: str) -> Dict[str, Any]:
    """
    Run only the core RAG pipeline required for RAGAS evaluation.

    This intentionally bypasses:
    - LLM routing
    - retrieval grading
    - hallucination grading
    - answer grading
    - retry loop

    RAGAS will evaluate faithfulness and answer relevancy separately.
    """

    documents = retriever.invoke(question)

    generation = generation_chain.invoke(
        {
            "context": documents,
            "question": question,
        }
    )

    return {
        "documents": documents,
        "generation": generation,
    }


def build_samples(
    benchmark: List[Dict[str, Any]],
) -> List[SingleTurnSample]:

    samples: List[SingleTurnSample] = []

    local_items = [
        item
        for item in benchmark
        if item.get("expected_route") == "local"
    ]

    print(
        f"Preparing {len(local_items)} local RAG "
        "samples for RAGAS..."
    )

    for index, item in enumerate(local_items):
        question = item["question"]

        print()
        print("=" * 60)
        print(
            f"RAG sample {index + 1}/{len(local_items)}"
        )
        print(f"Question: {question}")
        print("=" * 60)

        try:
            result = run_rag(question)

            documents = result["documents"]

            contexts = [
                document.page_content
                for document in documents
                if getattr(
                    document,
                    "page_content",
                    "",
                )
            ]

            answer = result["generation"]

            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
            )

            samples.append(sample)

            print(
                f"Retrieved contexts: {len(contexts)}"
            )
            print(
                f"Answer length: {len(answer)} characters"
            )

        except Exception as exc:
            print(
                f"ERROR evaluating question: {exc}"
            )

        # Avoid immediately consuming the remaining TPM window.
        if index < len(local_items) - 1:
            print(
                f"Waiting {REQUEST_DELAY_SECONDS}s "
                "before next RAG request..."
            )
            time.sleep(REQUEST_DELAY_SECONDS)

    return samples


def save_results(result: Any) -> None:
    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if hasattr(result, "to_pandas"):
        dataframe = result.to_pandas()

        records = dataframe.to_dict(
            orient="records"
        )

        summary: Dict[str, float] = {}

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

    samples = build_samples(benchmark)

    if not samples:
        raise RuntimeError(
            "No RAGAS samples were successfully created."
        )

    dataset = EvaluationDataset(
        samples=samples
    )

    print()
    print("=" * 60)
    print("RAGAS EVALUATION")
    print("=" * 60)
    print(
        f"Samples: {len(samples)}"
    )
    print(
        "Metrics: faithfulness, answer_relevancy"
    )
    print("=" * 60)

    print()
    print(
        "Running RAGAS evaluation..."
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
        ],
        llm=llm_model,
        embeddings=embed_model,
    )

    print()
    print("=" * 60)
    print("RAGAS RESULTS")
    print("=" * 60)

    print(result)

    save_results(result)

    print()
    print(
        f"RAGAS results saved to: {RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()