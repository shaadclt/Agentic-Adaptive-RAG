from evaluation import (
    EvaluationResult,
    EvaluationTracker,
)


def test_evaluation_result():

    result = EvaluationResult(
        question="What is Chroma?",
        answer="Chroma is a vector database.",
        route="local",
        retrieved_documents=4,
        relevant_documents=3,
        grounded=True,
        answers_question=True,
        retry_count=0,
        latency_seconds=1.5,
    )

    assert result.question == "What is Chroma?"

    assert result.route == "local"

    assert result.retrieved_documents == 4

    assert result.relevant_documents == 3

    assert result.retrieval_relevance_rate == 0.75

    assert result.grounded is True

    assert result.answers_question is True


def test_evaluation_result_zero_documents():

    result = EvaluationResult(
        question="Test question",
        answer="Test answer",
    )

    assert result.retrieval_relevance_rate == 0.0


def test_evaluation_tracker():

    tracker = EvaluationTracker()

    tracker.add(
        EvaluationResult(
            question="Question 1",
            answer="Answer 1",
            route="local",
            grounded=True,
            answers_question=True,
            retry_count=0,
            latency_seconds=1.0,
        )
    )

    tracker.add(
        EvaluationResult(
            question="Question 2",
            answer="Answer 2",
            route="web",
            grounded=True,
            answers_question=False,
            retry_count=1,
            latency_seconds=3.0,
        )
    )

    summary = tracker.summary()

    assert summary["total_questions"] == 2

    assert summary["average_latency_seconds"] == 2.0

    assert summary["grounded_rate"] == 1.0

    assert summary["answer_quality_rate"] == 0.5

    assert summary["average_retries"] == 0.5


def test_tracker_empty():

    tracker = EvaluationTracker()

    summary = tracker.summary()

    assert summary["total_questions"] == 0

    assert summary["grounded_rate"] == 0.0