from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")

    question = state["question"]
    documents = state.get("documents", [])

    filtered_docs = []

    for document in documents:
        score = retrieval_grader.invoke(
            {
                "question": question,
                "document": document.page_content,
            }
        )

        if score.binary_score.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(document)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")

    web_search = len(filtered_docs) == 0

    if web_search:
        print("---NO RELEVANT DOCUMENTS FOUND---")
        print("---WEB SEARCH REQUIRED---")

    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search,
    }