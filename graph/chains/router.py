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
   - Used for current, external, or general information that is
     unlikely to be available in the user's uploaded documents.

Prefer the vectorstore when the question relates to the user's
uploaded knowledge.

Use websearch when the question clearly requires external or
current information.
"""


route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)


question_router = route_prompt | structured_llm_router


"""
The query router is the system's first decision point.

A RouteQuery Pydantic model constrains the router output to either
"vectorstore" or "websearch".

The graph gives priority to user-provided knowledge when a local
knowledge base exists. The router remains available as a fallback
decision mechanism when no local knowledge base is available.

If local retrieval produces documents that are not relevant to the
question, the graph automatically falls back to web search.
"""
