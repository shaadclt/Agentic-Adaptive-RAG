from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field

from model import llm_model


llm = llm_model


class GradeAnswer(BaseModel):
    binary_score: bool = Field(
        description="Whether the answer addresses the user's question."
    )


structured_llm_grader = llm.with_structured_output(GradeAnswer)


system = """
You are a grader assessing whether an answer addresses and resolves a user's question.

Return a binary score:
- true: the answer directly addresses and resolves the question.
- false: the answer does not adequately address the question.

Do not judge whether the answer is stylistically good.
Focus only on whether the answer actually answers the user's question.
"""


answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            """
User question:

{question}

LLM generation:

{generation}
""",
        ),
    ]
)


answer_grader: RunnableSequence = answer_prompt | structured_llm_grader