from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import RouteQuery, question_router
from graph.consts import (
    GENERATE,
    GRADE_DOCUMENTS,
    RETRIEVE,
    WEBSEARCH,
    INCREMENT_RETRY,
)
from graph.nodes.generate import generate
from graph.nodes.grade_documents import grade_documents
from graph.nodes.increment_retry import increment_retry
from graph.nodes.retrieve import retrieve
from graph.nodes.web_search import web_search
from graph.state import GraphState


load_dotenv()


MAX_GENERATION_RETRIES = 2


def normalize_binary_score(value) -> bool:
    """
    Normalize grader outputs to a Python boolean.

    Graders may return:
        True / False
        "yes" / "no"
        "true" / "false"
    """

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {
            "yes",
            "true",
            "1",
        }

    return bool(value)


def decide_to_generate(state: GraphState) -> str:
    print("---ASSESS GRADED DOCUMENTS---")

    if state.get("web_search", False):
        print(
            "---DECISION: LOCAL DOCUMENTS INSUFFICIENT, "
            "INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH

    print("---DECISION: GENERATE---")
    return GENERATE


def grade_generation_grounded_in_documents_and_question(
    state: GraphState,
) -> str:

    print("---CHECK HALLUCINATIONS---")

    question = state["question"]
    documents = state.get("documents", [])
    generation = state.get("generation", "")

    # -----------------------------------
    # Hallucination / grounding check
    # -----------------------------------

    score = hallucination_grader.invoke(
        {
            "documents": documents,
            "generation": generation,
        }
    )

    grounded = normalize_binary_score(
        score.binary_score
    )

    if grounded:
        print(
            "---DECISION: GENERATION IS GROUNDED "
            "IN DOCUMENTS---"
        )

        # -----------------------------------
        # Answer quality check
        # -----------------------------------

        print("---GRADE GENERATION VS QUESTION---")

        score = answer_grader.invoke(
            {
                "question": question,
                "generation": generation,
            }
        )

        useful = normalize_binary_score(
            score.binary_score
        )

        if useful:
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

    # -----------------------------------
    # Generation is not grounded
    # -----------------------------------

    print(
        "---DECISION: GENERATION IS NOT "
        "GROUNDED IN DOCUMENTS---"
    )

    retry_count = state.get("retry_count", 0)

    if retry_count < MAX_GENERATION_RETRIES:
        print(
            f"---DECISION: RETRY GENERATION "
            f"({retry_count + 1}/{MAX_GENERATION_RETRIES})---"
        )
        return "retry"

    print("---DECISION: MAX RETRIES REACHED---")

    return "not useful"


def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")

    question = state["question"]

    source: RouteQuery = question_router.invoke(
        {
            "question": question,
        }
    )

    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH

    if source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE

    raise ValueError(
        f"Unsupported datasource returned by router: "
        f"{source.datasource}"
    )


# -----------------------------------
# Build Graph
# -----------------------------------

workflow = StateGraph(GraphState)


workflow.add_node(
    RETRIEVE,
    retrieve,
)

workflow.add_node(
    GRADE_DOCUMENTS,
    grade_documents,
)

workflow.add_node(
    GENERATE,
    generate,
)

workflow.add_node(
    WEBSEARCH,
    web_search,
)

workflow.add_node(
    INCREMENT_RETRY,
    increment_retry,
)


workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)


workflow.add_edge(
    RETRIEVE,
    GRADE_DOCUMENTS,
)


workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)


workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "retry": INCREMENT_RETRY,
        "useful": END,
        "not useful": WEBSEARCH,
    },
)


workflow.add_edge(
    WEBSEARCH,
    GENERATE,
)


workflow.add_edge(
    INCREMENT_RETRY,
    GENERATE,
)


app = workflow.compile()