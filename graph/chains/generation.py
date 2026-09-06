from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser

from model import llm_model


llm = llm_model

prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()


"""
The generation chain is responsible for creating the actual response to the user's question.
We leverage a proven RAG prompt from LangChain Hub that has been optimized for
retrieval-augmented generation tasks.

This prompt template combines the retrieved context with the user's question
to generate a coherent and informative response.

The generation chain uses StrOutputParser to ensure that the model output is
returned as a clean string that can be processed by subsequent graph nodes.
"""