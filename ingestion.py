from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader

from model import embed_model


load_dotenv()


URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

CHROMA_PATH = "./.chroma"
COLLECTION_NAME = "rag-chroma"


def load_documents():
    docs = []

    for url in URLS:
        loader = WebBaseLoader(url)
        docs.extend(loader.load())

    return docs


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250,
        chunk_overlap=0,
    )

    return text_splitter.split_documents(documents)


def build_vectorstore():
    print("---LOADING DOCUMENTS---")

    documents = load_documents()

    print(f"Loaded {len(documents)} source documents.")

    print("---SPLITTING DOCUMENTS---")

    document_chunks = split_documents(documents)

    print(f"Created {len(document_chunks)} document chunks.")

    print("---BUILDING CHROMA VECTOR STORE---")

    Chroma.from_documents(
        documents=document_chunks,
        collection_name=COLLECTION_NAME,
        embedding=embed_model,
        persist_directory=CHROMA_PATH,
    )

    print("---VECTOR STORE CREATED---")


if __name__ == "__main__":
    build_vectorstore()