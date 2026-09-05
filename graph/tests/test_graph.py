from unittest.mock import patch

from graph.graph import workflow


def test_local_rag_path():
    """Relevant local documents should lead to generation without web search."""

    initial_state = {
        "question": "What is retrieval augmented generation?",
        "generation": "",
        "web_search": False,
        "documents": [],
        "retry_count": 0,
    }

    with (
        patch("graph.nodes.retrieve.retriever.invoke") as mock_retriever,
        patch("graph.nodes.grade_documents.retrieval_grader.invoke") as mock_grade,
        patch("graph.nodes.generate.generation_chain.invoke") as mock_generate,
        patch("graph.nodes.generate.hallucination_grader.invoke") as mock_hallucination,
        patch("graph.nodes.generate.answer_grader.invoke") as mock_answer,
    ):
        mock_retriever.return_value = [
            type(
                "Document",
                (),
                {"page_content": "RAG combines retrieval with generation."},
            )()
        ]

        mock_grade.return_value.binary_score = "yes"
        mock_generate.return_value = "RAG combines retrieval with generation."
        mock_hallucination.return_value.binary_score = "yes"
        mock_answer.return_value.binary_score = "yes"

        result = workflow.invoke(initial_state)

    assert result["generation"] == "RAG combines retrieval with generation."
    assert result["web_search"] is False
    assert result["retry_count"] == 0


def test_web_search_route():
    """Questions routed directly to web search should use web search before generation."""

    initial_state = {
        "question": "What happened in the latest AI news?",
        "generation": "",
        "web_search": True,
        "documents": [],
        "retry_count": 0,
    }

    with (
        patch("graph.nodes.web_search.TavilyClient") as mock_tavily,
        patch("graph.nodes.generate.generation_chain.invoke") as mock_generate,
        patch("graph.nodes.generate.hallucination_grader.invoke") as mock_hallucination,
        patch("graph.nodes.generate.answer_grader.invoke") as mock_answer,
    ):
        mock_tavily.return_value.search.return_value = [
            {
                "content": "Recent AI developments include new agentic systems.",
                "url": "https://example.com",
                "title": "AI News",
            }
        ]

        mock_generate.return_value = "Recent AI developments include new agentic systems."
        mock_hallucination.return_value.binary_score = "yes"
        mock_answer.return_value.binary_score = "yes"

        result = workflow.invoke(initial_state)

    assert result["generation"]
    assert result["web_search"] is True