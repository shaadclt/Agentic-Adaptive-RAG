from typing import Any

from langgraph.graph import END, START, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import (
    hallucination_grader,
)
from graph.chains.router import question_router

from graph.nodes.build_response import build_response
from graph.nodes.generate import generate
from graph.nodes.grade_documents import (
    grade_documents,
)
from graph.nodes.increment_retry import (
    increment_retry,
)
from graph.nodes.retrieve import retrieve
from graph.nodes.web_search import web_search

from graph.state import GraphState

from retrieval import retriever


MAX_GENERATION_RETRIES = 2

ROUTER_CONTEXT_DOCUMENTS = 4

ROUTER_CONTEXT_CHARS = 6000


def normalize_binary_score(
    score: Any,
) -> str:

    if isinstance(score, bool):
        return (
            "yes"
            if score
            else "no"
        )

    return str(
        score
    ).strip().lower()


def retrieve_router_context(
    state: GraphState,
) -> dict:
    """
    Retrieve local evidence before the initial
    routing decision.

    This gives the router actual knowledge-base
    evidence instead of asking it to route using
    the question alone.
    """

    print(
        "---RETRIEVE ROUTER CONTEXT---"
    )

    question = state["question"]

    documents = retriever.invoke(
        question
    )

    context_documents = documents[
        :ROUTER_CONTEXT_DOCUMENTS
    ]

    context_parts = []

    total_chars = 0

    for document in context_documents:

        content = getattr(
            document,
            "page_content",
            "",
        )

        if not content:
            continue

        remaining = (
            ROUTER_CONTEXT_CHARS
            - total_chars
        )

        if remaining <= 0:
            break

        content = content[
            :remaining
        ]

        context_parts.append(
            content
        )

        total_chars += len(
            content
        )

    context = (
        "\n\n---\n\n".join(
            context_parts
        )
    )

    print(
        "---ROUTER CONTEXT: "
        f"{len(context_parts)} DOCUMENTS---"
    )

    return {
        "router_context": context,
    }


def route_question(
    state: GraphState,
) -> str:
    """
    Route the question using retrieved local
    evidence.

    The router sees both the question and the
    top local retrieval results.
    """

    print(
        "---ROUTE QUESTION---"
    )

    question = state[
        "question"
    ]

    context = state.get(
        "router_context",
        "",
    )

    route = question_router.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    if (
        route.datasource
        == "websearch"
    ):
        print(
            "---ROUTER DECISION: "
            "WEB SEARCH---"
        )

        return "websearch"

    print(
        "---ROUTER DECISION: "
        "LOCAL RAG---"
    )

    return "retrieve"


def mark_local_route(
    state: GraphState,
) -> dict:

    return {
        "route": "local"
    }


def mark_web_route(
    state: GraphState,
) -> dict:

    return {
        "route": "web"
    }


def decide_to_generate(
    state: GraphState,
) -> str:

    print(
        "---ASSESS GRADED DOCUMENTS---"
    )

    web_search_required = state.get(
        "web_search",
        False,
    )

    if web_search_required:

        print(
            "---DECISION: LOCAL DOCUMENTS "
            "INSUFFICIENT, INCLUDE WEB SEARCH---"
        )

        return "websearch"

    print(
        "---DECISION: GENERATE---"
    )

    return "generate"


def evaluate_generation(
    state: GraphState,
) -> dict:

    print(
        "---CHECK HALLUCINATIONS---"
    )

    question = state[
        "question"
    ]

    documents = state.get(
        "documents",
        [],
    )

    generation = state.get(
        "generation",
        "",
    )

    hallucination_score = (
        hallucination_grader.invoke(
            {
                "documents": documents,
                "generation": generation,
            }
        )
    )

    grounded = (
        normalize_binary_score(
            hallucination_score.binary_score
        )
        == "yes"
    )

    if not grounded:

        print(
            "---DECISION: GENERATION IS NOT "
            "GROUNDED IN DOCUMENTS---"
        )

        return {
            "grounded": False,
            "answers_question": False,
        }

    print(
        "---DECISION: GENERATION IS GROUNDED "
        "IN DOCUMENTS---"
    )

    print(
        "---GRADE GENERATION VS QUESTION---"
    )

    answer_score = (
        answer_grader.invoke(
            {
                "question": question,
                "generation": generation,
            }
        )
    )

    answers_question = (
        normalize_binary_score(
            answer_score.binary_score
        )
        == "yes"
    )

    if answers_question:

        print(
            "---DECISION: GENERATION "
            "ADDRESSES QUESTION---"
        )

    else:

        print(
            "---DECISION: GENERATION DOES "
            "NOT ADDRESS QUESTION---"
        )

    return {
        "grounded": grounded,
        "answers_question": answers_question,
    }


def decide_after_evaluation(
    state: GraphState,
) -> str:

    grounded = state.get(
        "grounded",
        False,
    )

    answers_question = state.get(
        "answers_question",
        False,
    )

    retry_count = state.get(
        "retry_count",
        0,
    )

    if (
        grounded
        and answers_question
    ):
        return "useful"

    if (
        retry_count
        < MAX_GENERATION_RETRIES
    ):

        print(
            f"---RETRY GENERATION "
            f"({retry_count + 1}/"
            f"{MAX_GENERATION_RETRIES})---"
        )

        return "retry"

    print(
        "---DECISION: MAXIMUM RETRIES "
        "REACHED, RETURN BEST "
        "AVAILABLE ANSWER---"
    )

    return "not useful"


workflow = StateGraph(
    GraphState
)


workflow.add_node(
    "retrieve_router_context",
    retrieve_router_context,
)

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
    "evaluate_generation",
    evaluate_generation,
)

workflow.add_node(
    "build_response",
    build_response,
)


workflow.add_edge(
    START,
    "retrieve_router_context",
)


workflow.add_conditional_edges(
    "retrieve_router_context",
    route_question,
    {
        "retrieve": "mark_local_route",
        "websearch": "mark_web_route",
    },
)


workflow.add_edge(
    "mark_local_route",
    "retrieve",
)


workflow.add_edge(
    "retrieve",
    "grade_documents",
)


workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "websearch": "mark_web_route",
    },
)


workflow.add_edge(
    "mark_web_route",
    "websearch",
)


workflow.add_edge(
    "websearch",
    "generate",
)


workflow.add_edge(
    "generate",
    "evaluate_generation",
)


workflow.add_conditional_edges(
    "evaluate_generation",
    decide_after_evaluation,
    {
        "useful": "build_response",
        "retry": "increment_retry",
        "not useful": "build_response",
    },
)


workflow.add_edge(
    "increment_retry",
    "generate",
)


workflow.add_edge(
    "build_response",
    END,
)


app = workflow.compile()