from unittest.mock import patch

from graph.graph import (
    MAX_GENERATION_RETRIES,
    app,
    decide_after_evaluation,
    decide_to_generate,
    route_question,
)
from graph.state import GraphState


def test_local_rag_path() -> None:
    state: GraphState = {
        "question": "What is Chroma?",
        "documents": [],
        "router_context": (
            "Chroma is used as the vector database for the local "
            "knowledge base."
        ),
    }

    mock_route = type(
        "RouteResult",
        (),
        {"datasource": "vectorstore"},
    )()

    with patch(
        "graph.graph.question_router",
    ) as mock_router:

        mock_router.invoke.return_value = mock_route

        decision = route_question(state)

    assert decision == "retrieve"

    mock_router.invoke.assert_called_once_with(
        {
            "question": "What is Chroma?",
            "context": (
                "Chroma is used as the vector database for the local "
                "knowledge base."
            ),
        }
    )


def test_retrieval_falls_back_to_web_search() -> None:
    state: GraphState = {
        "question": "What is the capital of France?",
        "documents": [],
        "web_search": True,
    }

    decision = decide_to_generate(state)

    assert decision == "websearch"


def test_direct_web_search_route() -> None:
    state: GraphState = {
        "question": "What is the latest news?",
        "router_context": (
            "The local knowledge base contains information about "
            "the RAG project's architecture."
        ),
    }

    mock_route = type(
        "RouteResult",
        (),
        {"datasource": "websearch"},
    )()

    with patch(
        "graph.graph.question_router",
    ) as mock_router:

        mock_router.invoke.return_value = mock_route

        decision = route_question(state)

    assert decision == "websearch"

    mock_router.invoke.assert_called_once_with(
        {
            "question": "What is the latest news?",
            "context": (
                "The local knowledge base contains information about "
                "the RAG project's architecture."
            ),
        }
    )


def test_generation_retries_when_not_grounded() -> None:
    state: GraphState = {
        "question": "What is Chroma?",
        "grounded": False,
        "answers_question": False,
        "retry_count": 0,
    }

    decision = decide_after_evaluation(state)

    assert decision == "retry"


def test_generation_quality_gate_respects_retry_limit() -> None:
    state: GraphState = {
        "question": "What is Chroma?",
        "grounded": False,
        "answers_question": False,
        "retry_count": MAX_GENERATION_RETRIES,
    }

    decision = decide_after_evaluation(state)

    assert decision == "not useful"


def test_generation_retries_when_answer_does_not_address_question() -> None:
    state: GraphState = {
        "question": "What is Chroma?",
        "grounded": True,
        "answers_question": False,
        "retry_count": 0,
    }

    decision = decide_after_evaluation(state)

    assert decision == "retry"


def test_generation_quality_gate_handles_uppercase_yes() -> None:
    with patch(
        "graph.graph.hallucination_grader",
    ) as mock_hallucination, patch(
        "graph.graph.answer_grader",
    ) as mock_answer:

        mock_hallucination.invoke.return_value = type(
            "Score",
            (),
            {"binary_score": "YES"},
        )()

        mock_answer.invoke.return_value = type(
            "Score",
            (),
            {"binary_score": "YES"},
        )()

        from graph.graph import evaluate_generation

        state: GraphState = {
            "question": "What is Chroma?",
            "documents": [],
            "generation": "Chroma is a vector database.",
        }

        result = evaluate_generation(state)

    assert result["grounded"] is True
    assert result["answers_question"] is True


def test_local_knowledge_router_decides_local() -> None:
    state: GraphState = {
        "question": "What does this project use Chroma for?",
        "router_context": (
            "Chroma is used as the vector database for the local "
            "knowledge base. Uploaded documents are stored as "
            "vector embeddings in Chroma."
        ),
    }

    mock_route = type(
        "RouteResult",
        (),
        {"datasource": "vectorstore"},
    )()

    with patch(
        "graph.graph.question_router",
    ) as mock_router:

        mock_router.invoke.return_value = mock_route

        decision = route_question(state)

    assert decision == "retrieve"

    mock_router.invoke.assert_called_once_with(
        {
            "question": "What does this project use Chroma for?",
            "context": (
                "Chroma is used as the vector database for the local "
                "knowledge base. Uploaded documents are stored as "
                "vector embeddings in Chroma."
            ),
        }
    )