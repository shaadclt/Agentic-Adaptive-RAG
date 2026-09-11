from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.retrieval_grader import retrieval_grader
from graph.chains.router import RouteQuery, question_router


def normalize_binary_score(score) -> str:
    """Normalize boolean or string grader output."""
    if isinstance(score, bool):
        return "yes" if score else "no"
    return str(score).strip().lower()


def test_retrieval_grader_answer_yes() -> None:
    res = retrieval_grader.invoke(
        {
            "question": "What is the capital of France?",
            "document": "Paris is the capital of France.",
        }
    )

    assert normalize_binary_score(res.binary_score) == "yes"


def test_retrieval_grader_answer_no() -> None:
    res = retrieval_grader.invoke(
        {
            "question": "What is the capital of France?",
            "document": "Berlin is the capital of Germany.",
        }
    )

    assert normalize_binary_score(res.binary_score) == "no"


def test_hallucination_grader_grounded() -> None:
    res = hallucination_grader.invoke(
        {
            "documents": [
                "LangGraph coordinates retrieval and generation."
            ],
            "generation": (
                "LangGraph coordinates retrieval and generation."
            ),
        }
    )

    assert normalize_binary_score(res.binary_score) == "yes"


def test_hallucination_grader_not_grounded() -> None:
    res = hallucination_grader.invoke(
        {
            "documents": [
                "LangGraph coordinates retrieval and generation."
            ],
            "generation": (
                "LangGraph is a database for storing images."
            ),
        }
    )

    assert normalize_binary_score(res.binary_score) == "no"


def test_answer_grader_answers_question() -> None:
    res = answer_grader.invoke(
        {
            "question": "What is LangGraph used for?",
            "generation": (
                "LangGraph is used to coordinate retrieval and generation."
            ),
        }
    )

    assert normalize_binary_score(res.binary_score) == "yes"


def test_answer_grader_does_not_answer_question() -> None:
    res = answer_grader.invoke(
        {
            "question": "What is LangGraph used for?",
            "generation": "Chroma is a vector database.",
        }
    )

    assert normalize_binary_score(res.binary_score) == "no"


def test_router_to_vectorstore() -> None:
    question = "What is LangGraph used for?"

    res: RouteQuery = question_router.invoke(
        {
            "question": question,
            "context": (
                "LangGraph is used to coordinate the different steps "
                "of the RAG workflow, including retrieval, generation, "
                "document grading, and web search."
            ),
        }
    )

    assert res.datasource == "vectorstore"


def test_router_to_websearch() -> None:
    question = "What is the latest news?"

    res: RouteQuery = question_router.invoke(
        {
            "question": question,
            "context": (
                "The local knowledge base contains information about "
                "the RAG project's architecture, retrieval, and generation."
            ),
        }
    )

    assert res.datasource == "websearch"