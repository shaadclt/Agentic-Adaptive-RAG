from typing import Any

from langgraph.graph import END, START, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import question_router
from graph.nodes.build_response import build_response
from graph.nodes.generate import generate
from graph.nodes.grade_documents import grade_documents
from graph.nodes.increment_retry import increment_retry
from graph.nodes.retrieve import retrieve
from graph.nodes.web_search import web_search
from graph.state import GraphState
from retrieval import has_documents


MAX_GENERATION_RETRIES = 2


def normalize_binary_score(score: Any) -> str:
    if isinstance(score, bool):
        return "yes" if score else "no"

    return str(score).strip().lower()


def decide_to_generate(state: GraphState) -> str:
    print("---ASSESS GRADED DOCUMENTS---")

    web_search_required = state.get("web_search", False)

    if web_search_required:
        print(
            "---DECISION: LOCAL DOCUMENTS INSUFFICIENT, "
            "INCLUDE WEB SEARCH---"
        )
        return "websearch"

    print("---DECISION: GENERATE---")
    return "generate"


def grade_generation_grounded_in_documents_and_question(
    state: GraphState,
) -> str:

    print("---CHECK HALLUCINATIONS---")

    question = state["question"]
    documents = state.get("documents", [])
    generation = state.get("generation", "")
    retry_count = state.get("retry_count", 0)

    hallucination_score = hallucination_grader.invoke(
        {
            "documents": documents,
            "generation": generation,
        }
    )

    grounded = normalize_binary_score(
        hallucination_score.binary_score
    )

    if grounded == "yes":

        print(
            "---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---"
        )

        print("---GRADE GENERATION VS QUESTION---")

        answer_score = answer_grader.invoke(
            {
                "question": question,
                "generation": generation,
            }
        )

        answers_question = normalize_binary_score(
            answer_score.binary_score
        )

        if answers_question == "yes":
            print(
                "---DECISION: GENERATION ADDRESSES QUESTION---"
            )
            return "useful"

        print(
            "---DECISION: GENERATION DOES NOT ADDRESS QUESTION---"
        )

        return "not useful"

    print(
        "---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---"
    )

    if retry_count < MAX_GENERATION_RETRIES:

        print(
            f"---RETRY GENERATION "
            f"({retry_count + 1}/{MAX_GENERATION_RETRIES})---"
        )

        return "retry"

    print(
        "---DECISION: MAXIMUM RETRIES REACHED, "
        "USE WEB SEARCH---"
    )

    return "not useful"


def route_question(state: GraphState) -> str:

    print("---ROUTE QUESTION---")

    if has_documents():

        print("---LOCAL KNOWLEDGE BASE AVAILABLE---")
        print("---ROUTE QUESTION TO RAG---")

        return "retrieve"

    print("---NO LOCAL KNOWLEDGE BASE---")

    question = state["question"]

    route = question_router.invoke(
        {
            "question": question
        }
    )

    if route.datasource == "websearch":

        print("---ROUTE QUESTION TO WEB SEARCH---")

        return "websearch"

    print("---ROUTE QUESTION TO RAG---")

    return "retrieve"


def mark_local_route(state: GraphState) -> dict:
    """
    Mark the response route as local RAG.
    """
    return {
        "route": "local"
    }


def mark_web_route(state: GraphState) -> dict:
    """
    Mark the response route as web search.
    """
    return {
        "route": "web"
    }


workflow = StateGraph(GraphState)


# ---------------------------------------------------------
# NODES
# ---------------------------------------------------------

workflow.add_node(
    "retrieve",
    retrieve,
)

workflow.add_node(
    "grade_documents",
    grade_documents,
)

workflow.add_node(
    "websearch",
    web_search,
)

workflow.add_node(
    "generate",
    generate,
)

workflow.add_node(
    "increment_retry",
    increment_retry,
)

workflow.add_node(
    "mark_local_route",
    mark_local_route,
)

workflow.add_node(
    "mark_web_route",
    mark_web_route,
)

workflow.add_node(
    "build_response",
    build_response,
)


# ---------------------------------------------------------
# START → ROUTER
# ---------------------------------------------------------

workflow.add_conditional_edges(
    START,
    route_question,
    {
        "retrieve": "mark_local_route",
        "websearch": "mark_web_route",
    },
)


# ---------------------------------------------------------
# LOCAL RAG PATH
# ---------------------------------------------------------

workflow.add_edge(
    "mark_local_route",
    "retrieve",
)

workflow.add_edge(
    "retrieve",
    "grade_documents",
)


# ---------------------------------------------------------
# DOCUMENT GRADING
# ---------------------------------------------------------

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "websearch": "mark_web_route",
    },
)


# ---------------------------------------------------------
# WEB SEARCH
# ---------------------------------------------------------

workflow.add_edge(
    "mark_web_route",
    "websearch",
)

workflow.add_edge(
    "websearch",
    "generate",
)


# ---------------------------------------------------------
# GENERATION QUALITY CONTROL
# ---------------------------------------------------------

workflow.add_conditional_edges(
    "generate",
    grade_generation_grounded_in_documents_and_question,
    {
        "useful": "build_response",
        "retry": "increment_retry",
        "not useful": "mark_web_route",
    },
)


# ---------------------------------------------------------
# RETRY
# ---------------------------------------------------------

workflow.add_edge(
    "increment_retry",
    "generate",
)


# ---------------------------------------------------------
# FINAL RESPONSE
# ---------------------------------------------------------

workflow.add_edge(
    "build_response",
    END,
)


app = workflow.compile()