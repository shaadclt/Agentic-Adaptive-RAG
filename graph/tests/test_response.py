from langchain_core.documents import Document

from graph.nodes.build_response import build_response
from response import RAGResponse


def test_build_local_response():
    documents = [
        Document(
            page_content="Chroma is a vector database.",
            metadata={
                "source": "data/chroma.txt",
                "file_name": "chroma.txt",
                "document_id": "abc123",
                "file_type": ".txt",
            },
        )
    ]

    result = build_response(
        {
            "generation": "Chroma is a vector database.",
            "documents": documents,
            "retry_count": 0,
        }
    )

    assert result["answer"] == (
        "Chroma is a vector database."
    )

    assert result["route"] == "local"

    assert result["retry_count"] == 0

    assert len(result["sources"]) == 1

    assert result["sources"][0]["type"] == "local"

    assert result["sources"][0]["file_name"] == (
        "chroma.txt"
    )


def test_build_web_response():
    documents = [
        Document(
            page_content="Python is a programming language.",
            metadata={
                "source": "web",
                "title": "Python",
                "url": "https://example.com/python",
            },
        )
    ]

    result = build_response(
        {
            "generation": "Python is a programming language.",
            "documents": documents,
            "retry_count": 1,
        }
    )

    assert result["answer"] == (
        "Python is a programming language."
    )

    assert result["route"] == "web"

    assert result["retry_count"] == 1

    assert len(result["sources"]) == 1

    assert result["sources"][0]["type"] == "web"

    assert result["sources"][0]["url"] == (
        "https://example.com/python"
    )


def test_rag_response_validation():
    response = RAGResponse(
        answer="Test answer",
        route="local",
        retry_count=0,
        sources=[],
    )

    assert response.answer == "Test answer"
    assert response.route == "local"
    assert response.retry_count == 0
    assert response.sources == []