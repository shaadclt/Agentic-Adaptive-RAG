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


# =========================================================
# HELPERS
# =========================================================

def normalize_binary_score(score: Any) -> str:
    """
    Normalize grader output to a lowercase yes/no string.
    """

    if isinstance(score, bool):
        return "yes" if score else "no"

    return str(score).strip().lower()


# =========================================================
# INITIAL ROUTING
# =========================================================

def route_question(state: GraphState) -> str:
    """
    Decide whether to use the local knowledge base or
    web search.

    Local knowledge takes priority whenever documents
    are available.
    """

    print("---ROUTE QUESTION---")

    # -----------------------------------------------------
    # LOCAL KNOWLEDGE BASE AVAILABLE
    # -----------------------------------------------------

    if has_documents():

        print(
            "---LOCAL KNOWLEDGE BASE AVAILABLE---"
        )

        print(
            "---ROUTE QUESTION TO RAG---"
        )

        return "retrieve"

    # -----------------------------------------------------
    # NO LOCAL KNOWLEDGE BASE
    # -----------------------------------------------------

    print(
        "---NO LOCAL KNOWLEDGE BASE---"
    )

    question = state["question"]

    route = question_router.invoke(
        {
            "question": question
        }
    )

    if route.datasource == "websearch":

        print(
            "---ROUTE QUESTION TO WEB SEARCH---"
        )

        return "websearch"

    print(
        "---ROUTE QUESTION TO RAG---"
    )

    return "retrieve"


# =========================================================
# ROUTE MARKERS
# =========================================================

def mark_local_route(
    state: GraphState,
) -> dict:
    """
    Store local RAG as the selected route.
    """

    return {
        "route": "local"
    }


def mark_web_route(
    state: GraphState,
) -> dict:
    """
    Store web search as the selected route.
    """

    return {
        "route": "web"
    }


# =========================================================
# DOCUMENT DECISION
# =========================================================

def decide_to_generate(
    state: GraphState,
) -> str:
    """
    Decide whether the retrieved local documents are
    sufficient for generation.
    """

    print(
        "---ASSESS GRADED DOCUMENTS---"
    )

    web_search_required = state.get(
        "web_search",
        False,
    )

    if web_search_required:

        print(
            "---DECISION: LOCAL DOCUMENTS INSUFFICIENT, "
            "INCLUDE WEB SEARCH---"
        )

        return "websearch"

    print(
        "---DECISION: GENERATE---"
    )

    return "generate"


# =========================================================
# GENERATION EVALUATION
# =========================================================

def evaluate_generation(
    state: GraphState,
) -> dict:
    """
    Evaluate the generated answer for:

    1. Grounding in retrieved documents
    2. Ability to answer the question

    The evaluation metadata is persisted in GraphState.
    """

    print(
        "---CHECK HALLUCINATIONS---"
    )

    question = state["question"]

    documents = state.get(
        "documents",
        [],
    )

    generation = state.get(
        "generation",
        "",
    )

    # -----------------------------------------------------
    # HALLUCINATION / GROUNDING CHECK
    # -----------------------------------------------------

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
            "---DECISION: GENERATION IS NOT GROUNDED "
            "IN DOCUMENTS---"
        )

        return {
            "grounded": False,
            "answers_question": False,
        }

    print(
        "---DECISION: GENERATION IS GROUNDED "
        "IN DOCUMENTS---"
    )

    # -----------------------------------------------------
    # ANSWER QUALITY CHECK
    # -----------------------------------------------------

    print(
        "---GRADE GENERATION VS QUESTION---"
    )

    answer_score = answer_grader.invoke(
        {
            "question": question,
            "generation": generation,
        }
    )

    answers_question = (
        normalize_binary_score(
            answer_score.binary_score
        )
        == "yes"
    )

    if answers_question:

        print(
            "---DECISION: GENERATION ADDRESSES "
            "QUESTION---"
        )

    else:

        print(
            "---DECISION: GENERATION DOES NOT "
            "ADDRESS QUESTION---"
        )

    return {
        "grounded": grounded,
        "answers_question": answers_question,
    }


# =========================================================
# POST-EVALUATION DECISION
# =========================================================

def decide_after_evaluation(
    state: GraphState,
) -> str:
    """
    Decide whether to:

    - accept the answer
    - retry generation
    - fall back to web search
    """

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

    # -----------------------------------------------------
    # SUCCESS
    # -----------------------------------------------------

    if grounded and answers_question:

        return "useful"

    # -----------------------------------------------------
    # NOT GROUNDED
    # -----------------------------------------------------

    if not grounded:

        if retry_count < MAX_GENERATION_RETRIES:

            print(
                f"---RETRY GENERATION "
                f"({retry_count + 1}/"
                f"{MAX_GENERATION_RETRIES})---"
            )

            return "retry"

        print(
            "---DECISION: MAXIMUM RETRIES REACHED, "
            "USE WEB SEARCH---"
        )

        return "not useful"

    # -----------------------------------------------------
    # GROUNDED BUT DOES NOT ANSWER QUESTION
    # -----------------------------------------------------

    return "not useful"


# =========================================================
# GRAPH
# =========================================================

workflow = StateGraph(
    GraphState
)


# =========================================================
# NODES
# =========================================================

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


# =========================================================
# START → INITIAL ROUTING
# =========================================================

workflow.add_conditional_edges(
    START,
    route_question,
    {
        "retrieve": "mark_local_route",
        "websearch": "mark_web_route",
    },
)


# =========================================================
# LOCAL RAG PATH
# =========================================================

workflow.add_edge(
    "mark_local_route",
    "retrieve",
)

workflow.add_edge(
    "retrieve",
    "grade_documents",
)


# =========================================================
# DOCUMENT GRADING → GENERATION / WEB
# =========================================================

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "websearch": "mark_web_route",
    },
)


# =========================================================
# WEB SEARCH PATH
# =========================================================

workflow.add_edge(
    "mark_web_route",
    "websearch",
)

workflow.add_edge(
    "websearch",
    "generate",
)


# =========================================================
# GENERATION → EVALUATION
# =========================================================

workflow.add_edge(
    "generate",
    "evaluate_generation",
)


# =========================================================
# EVALUATION → FINAL / RETRY / WEB
# =========================================================

workflow.add_conditional_edges(
    "evaluate_generation",
    decide_after_evaluation,
    {
        "useful": "build_response",
        "retry": "increment_retry",
        "not useful": "mark_web_route",
    },
)


# =========================================================
# RETRY
# =========================================================

workflow.add_edge(
    "increment_retry",
    "generate",
)


# =========================================================
# BUILD FINAL RESPONSE
# =========================================================

workflow.add_edge(
    "build_response",
    END,
)


# =========================================================
# COMPILE
# =========================================================

app = workflow.compile()