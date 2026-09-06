import os

from dotenv import load_dotenv


load_dotenv()


# -----------------------------
# Vector Store Configuration
# -----------------------------

CHROMA_PATH = os.getenv(
    "CHROMA_PATH",
    "./.chroma",
)

COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "rag-chroma",
)


# -----------------------------
# Document Chunking
# -----------------------------

CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        "250",
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "0",
    )
)


# -----------------------------
# Retrieval
# -----------------------------

RETRIEVAL_K = int(
    os.getenv(
        "RETRIEVAL_K",
        "4",
    )
)


# -----------------------------
# Supported Documents
# -----------------------------

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
}