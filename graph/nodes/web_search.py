from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState


load_dotenv()


web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")

    question = state["question"]

    existing_documents = state.get("documents", [])

    response = web_search_tool.invoke(
        {"query": question}
    )

    tavily_results = response.get("results", [])

    web_documents = []

    for result in tavily_results:
        content = result.get("content", "")
        url = result.get("url", "")
        title = result.get("title", "")

        if not content:
            continue

        web_documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": "web",
                    "url": url,
                    "title": title,
                },
            )
        )

    documents = [
        *existing_documents,
        *web_documents,
    ]

    return {
        "documents": documents,
        "question": question,
        "web_search": False,
    }