from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from model import llm_model


class RouteQuery(BaseModel):
    datasource: Literal[
        "vectorstore",
        "websearch",
    ] = Field(
        ...,
        description=(
            "Choose vectorstore when the retrieved local "
            "knowledge contains enough information to answer "
            "the question. Choose websearch when the local "
            "knowledge is insufficient or current external "
            "information is required."
        ),
    )


llm = llm_model

structured_llm_router = (
    llm.with_structured_output(
        RouteQuery
    )
)


system = """
You are an expert router for an adaptive
Retrieval-Augmented Generation system.

Your task is to decide whether a user's question
should be answered using the local vectorstore or
web search.

You are given:

1. The user's question.
2. Retrieved excerpts from the local knowledge base.

Choose "vectorstore" when the retrieved local
excerpts contain enough relevant information to
answer the question.

Choose "websearch" when:

- The local excerpts do not contain enough information.
- The question requires current or changing information.
- The question clearly requires external information.

Important rules:

- Base the decision primarily on the provided local evidence.
- Do not choose websearch merely because the question is general.
- If the local excerpts clearly discuss the subject of the
  question, prefer vectorstore.
- Questions about the user's project should normally use
  vectorstore when the retrieved excerpts contain relevant
  project information.
- Current facts such as current office holders, latest versions,
  recent events, prices, or current news should use websearch.
- Do not assume information exists in the local knowledge base
  if the retrieved excerpts do not support it.
"""


route_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system,
        ),
        (
            "human",
            """
User question:

{question}


Retrieved local knowledge:

{context}
""",
        ),
    ]
)


question_router = (
    route_prompt
    | structured_llm_router
)