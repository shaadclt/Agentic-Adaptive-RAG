from typing import Any, Dict

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
    """
    Normalize grader output to lowercase yes/no text.
    """

    if isinstance(score, bool):
        return "yes" if score else "no"

    return str(score).strip().lower()


def decide_to_generate(state: GraphState) -> str:
    """
    Decide whether to generate an answer or perform web search
    after grading retrieved documents.
    """

    print("---ASSESS GRADED DOCUMENTS---")

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

    print("---DECISION: GENERATE---")
    return "generate"


def grade_generation_grounded_in_documents_and_question(
    state: GraphState,
) -> str:
    """
    Evaluate whether the generated answer is grounded in the
    available documents and answers the user's question.
    """

    print("---CHECK HALLUCINATIONS---")

    question = state["question"]
    documents = state.get(
        "documents",
        [],
    )
    generation = state.get(
        "generation",
        "",
    )
    retry_count = state.get(
        "retry_count",
        0,
    )

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
            "---DECISION: GENERATION IS GROUNDED "
            "IN DOCUMENTS---"
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
                "---DECISION: GENERATION ADDRESSES "
                "QUESTION---"
            )
            return "useful"

        print(
            "---DECISION: GENERATION DOES NOT "
            "ADDRESS QUESTION---"
        )

        return "not useful"

    print(
        "---DECISION: GENERATION IS NOT GROUNDED "
        "IN DOCUMENTS---"
    )

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


def route_question(state: GraphState) -> str:
    """
    Determine the initial route.

    When the local knowledge base contains documents, local
    retrieval always takes priority over the LLM router.

    When the local knowledge base is empty, the LLM router
    decides between vectorstore and web search.
    """

    print("---ROUTE QUESTION---")

    if has_documents():
        print(
            "---LOCAL KNOWLEDGE BASE AVAILABLE---"
        )
        print(
            "---ROUTE QUESTION TO RAG---"
        )
        return "retrieve"

    print(
        "---NO LOCAL KNOWLEDGE BASE---"
    )

    question = state["question"]

    route = question_router.invoke(
        {
            "question": question,
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


# ---------------------------------------------------------------------------
# Build LangGraph workflow
# ---------------------------------------------------------------------------

workflow = StateGraph(GraphState)


# Nodes
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
    "build_response",
    build_response,
)


# Entry routing
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "retrieve": "retrieve",
        "websearch": "websearch",
    },
)


# Local retrieval → document grading
workflow.add_edge(
    "retrieve",
    "grade_documents",
)


# Document grading → generation or web search
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "websearch": "websearch",
    },
)


# Web search → generation
workflow.add_edge(
    "websearch",
    "generate",
)


# Generation → quality gate
workflow.add_conditional_edges(
    "generate",
    grade_generation_grounded_in_documents_and_question,
    {
        "useful": "build_response",
        "retry": "increment_retry",
        "not useful": "websearch",
    },
)


# Retry → generation
workflow.add_edge(
    "increment_retry",
    "generate",
)


# Final structured response → END
workflow.add_edge(
    "build_response",
    END,
)


# Compile application
app = workflow.compile()