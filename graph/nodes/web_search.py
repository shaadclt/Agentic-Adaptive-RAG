import time
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState


load_dotenv()


MAX_WEB_SEARCH_RETRIES = 2
WEB_SEARCH_RETRY_DELAY_SECONDS = 1

web_search_tool = TavilySearch(max_results=3)


def _extract_web_documents(response: Any) -> list[Document]:
    """
    Convert Tavily results into LangChain Documents.

    Handles normal dictionary responses as well as defensive
    handling for list/string result formats.
    """
    if isinstance(response, dict):
        tavily_results = response.get("results", [])
    elif isinstance(response, list):
        tavily_results = response
    else:
        tavily_results = []

    web_documents: list[Document] = []

    for result in tavily_results:
        if isinstance(result, dict):
            content = result.get("content", "")
            url = result.get("url", "")
            title = result.get("title", "")

        elif isinstance(result, str):
            content = result
            url = ""
            title = ""

        else:
            continue

        if not content:
            continue

        web_documents.append(
            Document(
                page_content=str(content),
                metadata={
                    "source": "web",
                    "url": str(url),
                    "title": str(title),
                },
            )
        )

    return web_documents


def _build_retry_query(question: str) -> str:
    """
    Create a slightly more explicit query for the retry attempt.
    """
    return f"{question} reliable current information"


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")

    question = state["question"]
    existing_documents = state.get("documents", [])

    web_documents: list[Document] = []

    for attempt in range(MAX_WEB_SEARCH_RETRIES + 1):
        query = question

        if attempt > 0:
            query = _build_retry_query(question)

            print(
                "---RETRYING WEB SEARCH "
                f"({attempt}/{MAX_WEB_SEARCH_RETRIES})---"
            )

        try:
            response = web_search_tool.invoke(
                {"query": query}
            )

            web_documents = _extract_web_documents(response)

            print(
                f"---WEB SEARCH RESULTS: "
                f"{len(web_documents)} DOCUMENTS---"
            )

            if web_documents:
                break

            if attempt < MAX_WEB_SEARCH_RETRIES:
                time.sleep(WEB_SEARCH_RETRY_DELAY_SECONDS)

        except Exception as exc:
            print(f"---WEB SEARCH ERROR: {exc}---")

            if attempt >= MAX_WEB_SEARCH_RETRIES:
                break

            print(
                "---RETRYING WEB SEARCH "
                f"({attempt + 1}/{MAX_WEB_SEARCH_RETRIES})---"
            )

            time.sleep(WEB_SEARCH_RETRY_DELAY_SECONDS)

    documents = [
        *existing_documents,
        *web_documents,
    ]

    return {
        "documents": documents,
        "question": question,
        "web_search": False,
    }