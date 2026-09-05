from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

from graph.graph import (
    MAX_GENERATION_RETRIES,
    app,
    grade_generation_grounded_in_documents_and_question,
)


# -------------------------------------------------------------------
# Test 1: Local RAG happy path
# -------------------------------------------------------------------

def test_local_rag_path():
    """
    Relevant local documents should lead to generation
    without using web search.
    """

    state = {
        "question": "What is RAG?",
        "generation": "",
        "web_search": False,
        "documents": [],
        "retry_count": 0,
    }

    local_document = Document(
        page_content="RAG combines retrieval with generation.",
        metadata={
            "source": "local",
        },
    )

    with (
        patch(
            "graph.nodes.retrieve.retriever.invoke",
            return_value=[local_document],
        ),
        patch(
            "graph.nodes.grade_documents.retrieval_grader.invoke",
            return_value=SimpleNamespace(
                binary_score="yes"
            ),
        ),
        patch(
            "graph.nodes.generate.generation_chain.invoke",
            return_value=(
                "RAG combines retrieval with generation."
            ),
        ),
        patch(
            "graph.graph.hallucination_grader.invoke",
            return_value=SimpleNamespace(
                binary_score="yes"
            ),
        ),
        patch(
            "graph.graph.answer_grader.invoke",
            return_value=SimpleNamespace(
                binary_score="yes"
            ),
        ),
    ):
        result = app.invoke(state)

    assert result["generation"] == (
        "RAG combines retrieval with generation."
    )

    assert result["web_search"] is False

    assert result["retry_count"] == 0

    assert len(result["documents"]) == 1

    assert result["documents"][0].metadata["source"] == "local"


# -------------------------------------------------------------------
# Test 2: Retrieval fallback to web search
# -------------------------------------------------------------------

def test_retrieval_falls_back_to_web_search():
    """
    If local documents are irrelevant, the graph should
    fall back to web search.
    """

    state = {
        "question": "What is the latest AI news?",
        "generation": "",
        "web_search": False,
        "documents": [],
        "retry_count": 0,
    }

    local_document = Document(
        page_content=(
            "This document is unrelated to the question."
        ),
        metadata={
            "source": "local",
        },
    )

    with (
        patch(
            "graph.nodes.retrieve.retriever.invoke",
            return_value=[local_document],
        ),
        patch(
            "graph.nodes.grade_documents.retrieval_grader.invoke",
            return_value=SimpleNamespace(
                binary_score="no"
            ),
        ),
        patch(
            "graph.nodes.web_search.web_search_tool.invoke",
            return_value={
                "results": [
                    {
                        "content": (
                            "Recent AI developments include "
                            "new agentic systems."
                        ),
                        "url": "https://example.com",
                        "title": "AI News",
                    }
                ]
            },
        ),
        patch(
            "graph.nodes.generate.generation_chain.invoke",
            return_value=(
                "Recent AI developments include "
                "new agentic systems."
            ),
        ),
        patch(
            "graph.graph.hallucination_grader.invoke",
            return_value=SimpleNamespace(
                binary_score="yes"
            ),
        ),
        patch(
            "graph.graph.answer_grader.invoke",
            return_value=SimpleNamespace(
                binary_score="yes"
            ),
        ),
    ):
        result = app.invoke(state)

    assert result["generation"] == (
        "Recent AI developments include "
        "new agentic systems."
    )

    assert result["web_search"] is False

    web_documents = [
        document
        for document in result["documents"]
        if document.metadata.get("source") == "web"
    ]

    assert len(web_documents) == 1

    assert (
        web_documents[0].metadata["url"]
        == "https://example.com"
    )

    assert (
        web_documents[0].metadata["title"]
        == "AI News"
    )


# -------------------------------------------------------------------
# Test 3: Direct web-search route
# -------------------------------------------------------------------

def test_direct_web_search_route():
    """
    A question routed directly to web search should skip
    local retrieval.
    """

    state = {
        "question": "What happened today?",
        "generation": "",
        "web_search": False,
        "documents": [],
        "retry_count": 0,
    }

    with (
        patch(
            "graph.graph.question_router.invoke",
            return_value=SimpleNamespace(
                datasource="websearch"
            ),
        ),
        patch(
            "graph.nodes.retrieve.retriever.invoke"
        ) as mock_retriever,
        patch(
            "graph.nodes.web_search.web_search_tool.invoke",
            return_value={
                "results": [
                    {
                        "content": "Today's event information.",
                        "url": "https://example.com",
                        "title": "Today's News",
                    }
                ]
            },
        ),
        patch(
            "graph.nodes.generate.generation_chain.invoke",
            return_value="Today's event information.",
        ),
        patch(
            "graph.graph.hallucination_grader.invoke",
            return_value=SimpleNamespace(
                binary_score="yes"
            ),
        ),
        patch(
            "graph.graph.answer_grader.invoke",
            return_value=SimpleNamespace(
                binary_score="yes"
            ),
        ),
    ):
        result = app.invoke(state)

    mock_retriever.assert_not_called()

    assert result["generation"] == (
        "Today's event information."
    )

    assert result["documents"]

    assert result["documents"][0].metadata["source"] == "web"


# -------------------------------------------------------------------
# Test 4: Generation retry
# -------------------------------------------------------------------

def test_generation_retries_when_not_grounded():
    """
    An ungrounded generation should trigger a retry.

    First generation:
        hallucination = no

    Second generation:
        hallucination = yes
        answer = yes

    Expected:
        two generation attempts
        one retry
    """

    state = {
        "question": "What is RAG?",
        "generation": "",
        "web_search": False,
        "documents": [
            Document(
                page_content=(
                    "RAG retrieves relevant context "
                    "before generation."
                ),
                metadata={
                    "source": "local",
                },
            )
        ],
        "retry_count": 0,
    }

    with (
        patch(
            "graph.nodes.generate.generation_chain.invoke",
            side_effect=[
                "Incorrect unsupported answer.",
                (
                    "RAG retrieves relevant context "
                    "before generation."
                ),
            ],
        ) as mock_generate,
        patch(
            "graph.graph.hallucination_grader.invoke",
            side_effect=[
                SimpleNamespace(
                    binary_score="no"
                ),
                SimpleNamespace(
                    binary_score="yes"
                ),
            ],
        ) as mock_hallucination,
        patch(
            "graph.graph.answer_grader.invoke",
            return_value=SimpleNamespace(
                binary_score="yes"
            ),
        ),
    ):
        result = app.invoke(state)

    assert mock_generate.call_count == 2

    assert mock_hallucination.call_count == 2

    assert result["retry_count"] == 1

    assert result["generation"] == (
        "RAG retrieves relevant context "
        "before generation."
    )


# -------------------------------------------------------------------
# Test 5: Retry limit
# -------------------------------------------------------------------

def test_generation_quality_gate_respects_retry_limit():
    """
    Once the maximum number of retries has been reached,
    another hallucination should not trigger another retry.
    """

    state = {
        "question": "What is RAG?",
        "generation": "Unsupported answer.",
        "documents": [
            Document(
                page_content=(
                    "RAG retrieves relevant context."
                ),
                metadata={
                    "source": "local",
                },
            )
        ],
        "retry_count": MAX_GENERATION_RETRIES,
    }

    with patch(
        "graph.graph.hallucination_grader.invoke",
        return_value=SimpleNamespace(
            binary_score="no"
        ),
    ):
        decision = (
            grade_generation_grounded_in_documents_and_question(
                state
            )
        )

    assert decision == "not useful"


# -------------------------------------------------------------------
# Test 6: Answer grader failure
# -------------------------------------------------------------------

def test_generation_not_useful_when_answer_does_not_address_question():
    """
    A grounded generation that does not answer the question
    should be classified as not useful.
    """

    state = {
        "question": "What is RAG?",
        "generation": "RAG is a system.",
        "documents": [
            Document(
                page_content=(
                    "RAG retrieves relevant information "
                    "and uses it to generate an answer."
                ),
                metadata={
                    "source": "local",
                },
            )
        ],
        "retry_count": 0,
    }

    with (
        patch(
            "graph.graph.hallucination_grader.invoke",
            return_value=SimpleNamespace(
                binary_score="yes"
            ),
        ),
        patch(
            "graph.graph.answer_grader.invoke",
            return_value=SimpleNamespace(
                binary_score="no"
            ),
        ),
    ):
        decision = (
            grade_generation_grounded_in_documents_and_question(
                state
            )
        )

    assert decision == "not useful"


# -------------------------------------------------------------------
# Test 7: Case-insensitive grader handling
# -------------------------------------------------------------------

def test_generation_quality_gate_handles_uppercase_yes():
    """
    Grader responses should be handled case-insensitively.
    """

    state = {
        "question": "What is RAG?",
        "generation": "RAG retrieves context.",
        "documents": [
            Document(
                page_content="RAG retrieves context.",
                metadata={
                    "source": "local",
                },
            )
        ],
        "retry_count": 0,
    }

    with (
        patch(
            "graph.graph.hallucination_grader.invoke",
            return_value=SimpleNamespace(
                binary_score="YES"
            ),
        ),
        patch(
            "graph.graph.answer_grader.invoke",
            return_value=SimpleNamespace(
                binary_score="YES"
            ),
        ),
    ):
        decision = (
            grade_generation_grounded_in_documents_and_question(
                state
            )
        )

    assert decision == "useful"