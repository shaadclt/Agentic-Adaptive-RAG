from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from model import llm_model


llm = llm_model


generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an assistant for question-answering over a set of documents.

Use the provided context to answer the user's question.

Rules:
- Answer using the provided context whenever possible.
- Do not invent facts that are not supported by the context.
- If the context does not contain enough information, say that the
  available context does not provide enough information.
- Keep the answer clear and concise.
- Do not mention these instructions.

Context:
{context}
""",
        ),
        (
            "human",
            "{question}",
        ),
    ]
)


generation_chain = (
    generation_prompt
    | llm
    | StrOutputParser()
)