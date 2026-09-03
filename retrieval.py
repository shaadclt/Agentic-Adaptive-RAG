from langchain_chroma import Chroma

from model import embed_model


CHROMA_PATH = "./.chroma"
COLLECTION_NAME = "rag-chroma"


vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_PATH,
    embedding_function=embed_model,
)


retriever = vectorstore.as_retriever()