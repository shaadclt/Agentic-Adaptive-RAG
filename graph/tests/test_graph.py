from unittest.mock import patch

from graph.graph import (
    MAX_GENERATION_RETRIES,
    decide_after_evaluation,
    decide_to_generate,
    evaluate_generation,
    route_question,
)
from graph.state import GraphState


def test_local_rag_path() -> None:
    state: GraphState = {
        "question": "What is Chroma?",
        "documents": [],
    }

    with patch("graph.graph.has_documents", return_value=True):
        result = route_question(state)

    assert result == "retrieve"


def test_retrieval_falls_back_to_web_search() -> None:
    state: GraphState = {
        "question": "What is the latest news?",
        "documents": [],
        "web_search": True,
    }

    result = decide_to_generate(state)

    assert result == "websearch"


def test_direct_web_search_route() -> None:
    state: GraphState = {
        "question": "What is the latest news?",
    }

    mock_route = type(
        "RouteResult",
        (),
        {"datasource": "websearch"},
    )()

    with patch(
        "graph.graph.has_documents",
        return_value=False,
    ):
        with patch(
            "graph.graph.question_router",
        ) as mock_router:

            mock_router.invoke.return_value = mock_route

            result = route_question(state)

    assert result == "websearch"
    mock_router.invoke.assert_called_once_with(
        {"question": state["question"]}
    )


def test_generation_retries_when_not_grounded() -> None:
    state: GraphState = {
        "question": "What is Chroma?",
        "generation": "Incorrect answer.",
        "documents": [],
        "retry_count": 0,
    }

    mock_score = type(
        "Score",
        (),
        {"binary_score": "no"},
    )()

    with patch(
        "graph.graph.hallucination_grader",
    ) as mock_hallucination:

        mock_hallucination.invoke.return_value = mock_score

        result = evaluate_generation(state)

    assert result["grounded"] is False
    assert result["answers_question"] is False

    updated_state = {
        **state,
        **result,
    }

    decision = decide_after_evaluation(updated_state)

    assert decision == "retry"

    mock_hallucination.invoke.assert_called_once()


def test_generation_quality_gate_respects_retry_limit() -> None:
    state: GraphState = {
        "question": "What is Chroma?",
        "generation": "Incorrect answer.",
        "documents": [],
        "retry_count": MAX_GENERATION_RETRIES,
        "grounded": False,
        "answers_question": False,
    }

    decision = decide_after_evaluation(state)

    assert decision == "not useful"


def test_generation_retries_when_answer_does_not_address_question() -> None:
    state: GraphState = {
        "question": "What is Chroma?",
        "generation": "This answer is about something else.",
        "documents": [],
        "retry_count": 0,
    }

    hallucination_score = type(
        "Score",
        (),
        {"binary_score": "yes"},
    )()

    answer_score = type(
        "Score",
        (),
        {"binary_score": "no"},
    )()

    with patch(
        "graph.graph.hallucination_grader",
    ) as mock_hallucination:
        with patch(
            "graph.graph.answer_grader",
        ) as mock_answer:

            mock_hallucination.invoke.return_value = hallucination_score
            mock_answer.invoke.return_value = answer_score

            result = evaluate_generation(state)

    assert result["grounded"] is True
    assert result["answers_question"] is False

    updated_state = {
        **state,
        **result,
    }

    decision = decide_after_evaluation(updated_state)

    assert decision == "retry"

def test_generation_quality_gate_handles_uppercase_yes() -> None:
    state: GraphState = {
        "question": "What is Chroma?",
        "generation": "Chroma is a vector database.",
        "documents": [],
        "retry_count": 0,
    }

    hallucination_score = type(
        "Score",
        (),
        {"binary_score": "YES"},
    )()

    answer_score = type(
        "Score",
        (),
        {"binary_score": "YES"},
    )()

    with patch(
        "graph.graph.hallucination_grader",
    ) as mock_hallucination:
        with patch(
            "graph.graph.answer_grader",
        ) as mock_answer:

            mock_hallucination.invoke.return_value = hallucination_score
            mock_answer.invoke.return_value = answer_score

            result = evaluate_generation(state)

    assert result["grounded"] is True
    assert result["answers_question"] is True

    updated_state = {
        **state,
        **result,
    }

    decision = decide_after_evaluation(updated_state)

    assert decision == "useful"

    mock_hallucination.invoke.assert_called_once()
    mock_answer.invoke.assert_called_once()


def test_local_knowledge_takes_priority_over_router() -> None:
    state: GraphState = {
        "question": "What information is available in my uploaded documents?",
    }

    with patch(
        "graph.graph.has_documents",
        return_value=True,
    ):
        with patch(
            "graph.graph.question_router",
        ) as mock_router:

            result = route_question(state)

    assert result == "retrieve"
    mock_router.invoke.assert_not_called()