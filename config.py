import os

from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./.chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag-chroma")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "250"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "0"))

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
}

LANGSMITH_TRACING = (
    os.getenv("LANGSMITH_TRACING", "false").lower()
    in {"true", "1", "yes"}
)

LANGSMITH_PROJECT = os.getenv(
    "LANGSMITH_PROJECT",
    "Agentic-Adaptive-RAG",
)