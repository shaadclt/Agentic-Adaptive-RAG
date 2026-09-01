from dotenv import load_dotenv

from graph.chains.answer_grader import GradeAnswer, answer_grader
from graph.chains.hallucination_grader import (
    GradeHallucinations,
    hallucination_grader,
)
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.router import RouteQuery, question_router


load_dotenv()


def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"

    document = """
    Agent memory allows an AI agent to retain information from previous
    interactions and use that information when making future decisions.
    """

    res: GradeDocuments = retrieval_grader.invoke(
        {
            "question": question,
            "document": document,
        }
    )

    assert res.binary_score.lower() == "yes"


def test_retrieval_grader_answer_no() -> None:
    question = "agent memory"

    document = """
    Pizza dough is commonly made using flour, water, yeast, salt,
    and sometimes olive oil.
    """

    res: GradeDocuments = retrieval_grader.invoke(
        {
            "question": question,
            "document": document,
        }
    )

    assert res.binary_score.lower() == "no"


def test_hallucination_grader_grounded() -> None:
    documents = [
        """
        Agent memory allows an agent to store information from previous
        interactions and use it later.
        """
    ]

    generation = """
    Agent memory allows an agent to retain information from previous
    interactions.
    """

    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": documents,
            "generation": generation,
        }
    )

    assert res.binary_score is True


def test_hallucination_grader_not_grounded() -> None:
    documents = [
        """
        Agent memory allows an agent to store information from previous
        interactions.
        """
    ]

    generation = """
    Pizza was invented in Italy and is made using dough and tomato sauce.
    """

    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": documents,
            "generation": generation,
        }
    )

    assert res.binary_score is False


def test_answer_grader_answers_question() -> None:
    question = "What is agent memory?"

    generation = """
    Agent memory is a mechanism that allows an agent to retain
    information from previous interactions.
    """

    res: GradeAnswer = answer_grader.invoke(
        {
            "question": question,
            "generation": generation,
        }
    )

    assert res.binary_score is True


def test_answer_grader_does_not_answer_question() -> None:
    question = "What is agent memory?"

    generation = """
    Pizza dough is usually made with flour, water, yeast, and salt.
    """

    res: GradeAnswer = answer_grader.invoke(
        {
            "question": question,
            "generation": generation,
        }
    )

    assert res.binary_score is False


def test_router_to_vectorstore() -> None:
    question = "What is agent memory?"

    res: RouteQuery = question_router.invoke(
        {"question": question}
    )

    assert res.datasource == "vectorstore"


def test_router_to_websearch() -> None:
    question = "How do I make pizza?"

    res: RouteQuery = question_router.invoke(
        {"question": question}
    )

    assert res.datasource == "websearch"