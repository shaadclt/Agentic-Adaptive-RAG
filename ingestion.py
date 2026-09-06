import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model import embed_model


load_dotenv()


CHROMA_PATH = os.getenv("CHROMA_PATH", "./.chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag-chroma")


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
}


def load_document(file_path: str | Path) -> list[Document]:
    """
    Load a single document based on its file extension.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Document not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected a file, but received: {path}"
        )

    extension = path.suffix.lower()

    if extension == ".pdf":
        loader = PyPDFLoader(str(path))

    elif extension == ".docx":
        loader = Docx2txtLoader(str(path))

    elif extension in {".txt", ".md"}:
        loader = TextLoader(
            str(path),
            encoding="utf-8",
        )

    else:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    documents = loader.load()

    for document in documents:
        document.metadata["source"] = str(path)
        document.metadata["file_name"] = path.name

    return documents


def load_documents(
    file_paths: Iterable[str | Path],
) -> list[Document]:
    """
    Load multiple documents.
    """

    documents = []

    for file_path in file_paths:
        print(f"---LOADING: {file_path}---")

        loaded_documents = load_document(file_path)

        print(
            f"Loaded {len(loaded_documents)} document sections."
        )

        documents.extend(loaded_documents)

    if not documents:
        raise ValueError(
            "No documents were loaded."
        )

    return documents


def split_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Split documents into smaller chunks suitable for retrieval.
    """

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250,
        chunk_overlap=0,
    )

    return text_splitter.split_documents(documents)


def build_vectorstore(
    file_paths: Iterable[str | Path],
) -> Chroma:
    """
    Load, split and add documents to Chroma.
    """

    print("=" * 60)
    print("--- DOCUMENT INGESTION ---")
    print("=" * 60)

    print("\n---LOADING DOCUMENTS---")

    documents = load_documents(file_paths)

    print(
        f"\nLoaded {len(documents)} source documents."
    )

    print("\n---SPLITTING DOCUMENTS---")

    document_chunks = split_documents(documents)

    print(
        f"Created {len(document_chunks)} document chunks."
    )

    print("\n---UPDATING CHROMA VECTOR STORE---")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embed_model,
        persist_directory=CHROMA_PATH,
    )

    vectorstore.add_documents(document_chunks)

    print("---VECTOR STORE UPDATED---")

    print(f"Chroma path: {CHROMA_PATH}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Chunks added: {len(document_chunks)}")

    return vectorstore


def get_files_from_directory(
    directory: str | Path,
) -> list[Path]:
    """
    Return all supported documents from a directory.
    """

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    if not directory.is_dir():
        raise ValueError(
            f"Expected a directory: {directory}"
        )

    files = [
        file
        for file in directory.iterdir()
        if file.is_file()
        and file.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(files)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG vector store."
    )

    parser.add_argument(
        "files",
        nargs="+",
        help=(
            "One or more document paths "
            "(PDF, DOCX, TXT, MD)."
        ),
    )

    args = parser.parse_args()

    build_vectorstore(args.files)