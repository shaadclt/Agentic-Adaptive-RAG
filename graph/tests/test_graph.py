from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from graph.graph import (
    MAX_GENERATION_RETRIES,
    app,
    grade_generation_grounded_in_documents_and_question,
)


def make_document(
    content: str = "Retrieval-Augmented Generation combines retrieval with generation.",
    source: str = "test",
) -> Document:
    """Create a test document."""
    return Document(
        page_content=content,
        metadata={"source": source},
    )


# ---------------------------------------------------------------------------
# 1. Local RAG path
# ---------------------------------------------------------------------------

def test_local_rag_path():
    """Question routed to vectorstore should follow the local RAG path."""

    mock_router = MagicMock()
    mock_router.invoke.return_value = SimpleNamespace(
        datasource="vectorstore"
    )

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        make_document(
            "RAG retrieves relevant documents and uses them to generate an answer."
        )
    ]

    mock_retrieval_grader = MagicMock()
    mock_retrieval_grader.invoke.return_value = SimpleNamespace(
        binary_score="yes"
    )

    mock_generation_chain = MagicMock()
    mock_generation_chain.invoke.return_value = (
        "RAG retrieves relevant documents before generating an answer."
    )

    mock_hallucination_grader = MagicMock()
    mock_hallucination_grader.invoke.return_value = SimpleNamespace(
        binary_score="yes"
    )

    mock_answer_grader = MagicMock()
    mock_answer_grader.invoke.return_value = SimpleNamespace(
        binary_score="yes"
    )

    with (
        patch("graph.graph.question_router", mock_router),
        patch("graph.nodes.retrieve.retriever", mock_retriever),
        patch(
            "graph.nodes.grade_documents.retrieval_grader",
            mock_retrieval_grader,
        ),
        patch(
            "graph.nodes.generate.generation_chain",
            mock_generation_chain,
        ),
        patch(
            "graph.graph.hallucination_grader",
            mock_hallucination_grader,
        ),
        patch("graph.graph.answer_grader", mock_answer_grader),
    ):
        result = app.invoke(
            {
                "question": "What is RAG?",
                "retry_count": 0,
            }
        )

    assert result["generation"] == (
        "RAG retrieves relevant documents before generating an answer."
    )

    assert result["web_search"] is False
    assert len(result["documents"]) == 1

    mock_router.invoke.assert_called_once()
    mock_retriever.invoke.assert_called_once()
    mock_retrieval_grader.invoke.assert_called_once()
    mock_generation_chain.invoke.assert_called_once()
    mock_hallucination_grader.invoke.assert_called_once()
    mock_answer_grader.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# 2. Retrieval falls back to web search
# ---------------------------------------------------------------------------

def test_retrieval_falls_back_to_web_search():
    """
    If retrieved documents are not relevant, the graph should fall back
    to web search before generating an answer.
    """

    mock_router = MagicMock()
    mock_router.invoke.return_value = SimpleNamespace(
        datasource="vectorstore"
    )

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        make_document(
            "This document is about cooking recipes.",
            source="local",
        )
    ]

    mock_retrieval_grader = MagicMock()
    mock_retrieval_grader.invoke.return_value = SimpleNamespace(
        binary_score="no"
    )

    mock_web_search = MagicMock()
    mock_web_search.invoke.return_value = {
        "results": [
            {
                "content": (
                    "Retrieval-Augmented Generation retrieves external "
                    "documents and uses them as context for generation."
                ),
                "url": "https://example.com/rag",
                "title": "RAG Overview",
            }
        ]
    }

    mock_generation_chain = MagicMock()
    mock_generation_chain.invoke.return_value = (
        "RAG uses retrieved documents as context for generation."
    )

    mock_hallucination_grader = MagicMock()
    mock_hallucination_grader.invoke.return_value = SimpleNamespace(
        binary_score="yes"
    )

    mock_answer_grader = MagicMock()
    mock_answer_grader.invoke.return_value = SimpleNamespace(
        binary_score="yes"
    )

    with (
        patch("graph.graph.question_router", mock_router),
        patch("graph.nodes.retrieve.retriever", mock_retriever),
        patch(
            "graph.nodes.grade_documents.retrieval_grader",
            mock_retrieval_grader,
        ),
        patch(
            "graph.nodes.web_search.web_search_tool",
            mock_web_search,
        ),
        patch(
            "graph.nodes.generate.generation_chain",
            mock_generation_chain,
        ),
        patch(
            "graph.graph.hallucination_grader",
            mock_hallucination_grader,
        ),
        patch("graph.graph.answer_grader", mock_answer_grader),
    ):
        result = app.invoke(
            {
                "question": "What is Retrieval-Augmented Generation?",
                "retry_count": 0,
            }
        )

    assert result["generation"] == (
        "RAG uses retrieved documents as context for generation."
    )

    assert result["web_search"] is False

    # The original local document was filtered out.
    # The web document should be present.
    assert len(result["documents"]) == 1
    assert result["documents"][0].metadata["source"] == "web"
    assert result["documents"][0].metadata["url"] == (
        "https://example.com/rag"
    )

    mock_retrieval_grader.invoke.assert_called_once()
    mock_web_search.invoke.assert_called_once()
    mock_generation_chain.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Direct web-search route
# ---------------------------------------------------------------------------

def test_direct_web_search_route():
    """Questions routed directly to web search should skip retrieval."""

    mock_router = MagicMock()
    mock_router.invoke.return_value = SimpleNamespace(
        datasource="websearch"
    )

    mock_web_search = MagicMock()
    mock_web_search.invoke.return_value = {
        "results": [
            {
                "content": (
                    "Python is a high-level programming language "
                    "widely used for data science and AI."
                ),
                "url": "https://example.com/python",
                "title": "Python Overview",
            }
        ]
    }

    mock_generation_chain = MagicMock()
    mock_generation_chain.invoke.return_value = (
        "Python is a high-level programming language."
    )

    mock_hallucination_grader = MagicMock()
    mock_hallucination_grader.invoke.return_value = SimpleNamespace(
        binary_score="yes"
    )

    mock_answer_grader = MagicMock()
    mock_answer_grader.invoke.return_value = SimpleNamespace(
        binary_score="yes"
    )

    with (
        patch("graph.graph.question_router", mock_router),
        patch(
            "graph.nodes.web_search.web_search_tool",
            mock_web_search,
        ),
        patch(
            "graph.nodes.generate.generation_chain",
            mock_generation_chain,
        ),
        patch(
            "graph.graph.hallucination_grader",
            mock_hallucination_grader,
        ),
        patch("graph.graph.answer_grader", mock_answer_grader),
    ):
        result = app.invoke(
            {
                "question": "What is Python?",
                "retry_count": 0,
            }
        )

    assert result["generation"] == (
        "Python is a high-level programming language."
    )

    assert len(result["documents"]) == 1
    assert result["documents"][0].metadata["source"] == "web"

    mock_router.invoke.assert_called_once()
    mock_web_search.invoke.assert_called_once()

    # Retrieval should not be involved in a direct web-search route.
    # Since it is not patched here, this also ensures the route doesn't
    # unexpectedly execute the local retriever.


# ---------------------------------------------------------------------------
# 4. Generation retries when answer is not grounded
# ---------------------------------------------------------------------------

def test_generation_retries_when_not_grounded():
    """
    If the first generation is not grounded in the documents, the graph
    should increment retry_count and generate again.
    """

    mock_router = MagicMock()
    mock_router.invoke.return_value = SimpleNamespace(
        datasource="vectorstore"
    )

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        make_document(
            "RAG retrieves documents and provides them as context."
        )
    ]

    mock_retrieval_grader = MagicMock()
    mock_retrieval_grader.invoke.return_value = SimpleNamespace(
        binary_score="yes"
    )

    mock_generation_chain = MagicMock()
    mock_generation_chain.invoke.side_effect = [
        "This first answer contains unsupported information.",
        "RAG retrieves documents and uses them as context.",
    ]

    mock_hallucination_grader = MagicMock()
    mock_hallucination_grader.invoke.side_effect = [
        SimpleNamespace(binary_score="no"),
        SimpleNamespace(binary_score="yes"),
    ]

    mock_answer_grader = MagicMock()
    mock_answer_grader.invoke.return_value = SimpleNamespace(
        binary_score="yes"
    )

    with (
        patch("graph.graph.question_router", mock_router),
        patch("graph.nodes.retrieve.retriever", mock_retriever),
        patch(
            "graph.nodes.grade_documents.retrieval_grader",
            mock_retrieval_grader,
        ),
        patch(
            "graph.nodes.generate.generation_chain",
            mock_generation_chain,
        ),
        patch(
            "graph.graph.hallucination_grader",
            mock_hallucination_grader,
        ),
        patch("graph.graph.answer_grader", mock_answer_grader),
    ):
        result = app.invoke(
            {
                "question": "What is RAG?",
                "retry_count": 0,
            }
        )

    assert result["generation"] == (
        "RAG retrieves documents and uses them as context."
    )

    assert result["retry_count"] == 1

    assert mock_generation_chain.invoke.call_count == 2
    assert mock_hallucination_grader.invoke.call_count == 2
    assert mock_answer_grader.invoke.call_count == 1


# ---------------------------------------------------------------------------
# 5. Generation quality gate respects retry limit
# ---------------------------------------------------------------------------

def test_generation_quality_gate_respects_retry_limit():
    """
    Once MAX_GENERATION_RETRIES has been reached, another hallucination
    failure should not trigger another retry.
    """

    mock_hallucination_grader = MagicMock()
    mock_hallucination_grader.invoke.return_value = SimpleNamespace(
        binary_score="no"
    )

    with patch(
        "graph.graph.hallucination_grader",
        mock_hallucination_grader,
    ):
        decision = grade_generation_grounded_in_documents_and_question(
            {
                "question": "What is RAG?",
                "generation": "Unsupported answer",
                "documents": [make_document()],
                "retry_count": MAX_GENERATION_RETRIES,
            }
        )

    assert decision == "not useful"

    mock_hallucination_grader.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# 6. Generation not useful when answer doesn't address question
# ---------------------------------------------------------------------------

def test_generation_not_useful_when_answer_does_not_address_question():
    """
    A grounded generation that does not answer the question should be
    classified as not useful.
    """

    mock_hallucination_grader = MagicMock()
    mock_hallucination_grader.invoke.return_value = SimpleNamespace(
        binary_score="yes"
    )

    mock_answer_grader = MagicMock()
    mock_answer_grader.invoke.return_value = SimpleNamespace(
        binary_score="no"
    )

    with (
        patch(
            "graph.graph.hallucination_grader",
            mock_hallucination_grader,
        ),
        patch(
            "graph.graph.answer_grader",
            mock_answer_grader,
        ),
    ):
        decision = grade_generation_grounded_in_documents_and_question(
            {
                "question": "What is RAG?",
                "generation": "The sky is blue.",
                "documents": [make_document()],
                "retry_count": 0,
            }
        )

    assert decision == "not useful"

    mock_hallucination_grader.invoke.assert_called_once()
    mock_answer_grader.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# 7. Quality gate handles uppercase YES
# ---------------------------------------------------------------------------

def test_generation_quality_gate_handles_uppercase_yes():
    """
    Grader responses should be treated case-insensitively.
    """

    mock_hallucination_grader = MagicMock()
    mock_hallucination_grader.invoke.return_value = SimpleNamespace(
        binary_score="YES"
    )

    mock_answer_grader = MagicMock()
    mock_answer_grader.invoke.return_value = SimpleNamespace(
        binary_score="YES"
    )

    with (
        patch(
            "graph.graph.hallucination_grader",
            mock_hallucination_grader,
        ),
        patch(
            "graph.graph.answer_grader",
            mock_answer_grader,
        ),
    ):
        decision = grade_generation_grounded_in_documents_and_question(
            {
                "question": "What is RAG?",
                "generation": "RAG retrieves documents before generation.",
                "documents": [make_document()],
                "retry_count": 0,
            }
        )

    assert decision == "useful"

    mock_hallucination_grader.invoke.assert_called_once()
    mock_answer_grader.invoke.assert_called_once()