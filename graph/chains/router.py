from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from model import llm_model


class RouteQuery(BaseModel):
    """
    Route a user query to the most relevant datasource.
    """

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description=(
            "Given a user question, choose whether the question "
            "should be answered from the local vectorstore or web search."
        ),
    )


llm = llm_model

structured_llm_router = llm.with_structured_output(RouteQuery)


system = """
You are an expert at routing user questions.

There are two possible data sources:

1. vectorstore
   - Contains documents uploaded by the user.
   - Prefer this source when the question may be answered using
     the user's uploaded documents.

2. websearch
   - Used when the question requires external, current, or general
     information that is unlikely to be available in the user's
     uploaded documents.

Important:
- Prefer the vectorstore when the question relates to the user's
  uploaded knowledge.
- Use websearch for current information, general knowledge, or topics
  clearly outside the user's uploaded knowledge.
- Do not assume that the vectorstore contains only a fixed set of topics.
"""


route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)


question_router = route_prompt | structured_llm_router


"""
The query router is the system's first decision point that determines
the optimal source for answering a user's question.

A RouteQuery Pydantic model constrains the router's output to either
"vectorstore" or "websearch", providing reliable structured parsing.

The router is intentionally designed to prefer user-provided knowledge
when appropriate, while allowing web search for questions requiring
external or current information.

The graph subsequently validates retrieved documents. If the local
documents are not relevant to the question, the workflow falls back
to web search.
"""
