from langchain_chroma import Chroma

from config import CHROMA_PATH, COLLECTION_NAME, RETRIEVAL_K
from model import embed_model


vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_PATH,
    embedding_function=embed_model,
)


retriever = vectorstore.as_retriever(
    search_kwargs={"k": RETRIEVAL_K},
)


def has_documents() -> bool:
    """
    Return True when the local knowledge base contains documents.
    """
    result = vectorstore.get(include=["metadatas"])
    return bool(result.get("ids"))
