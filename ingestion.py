import hashlib
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
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from config import (
    CHROMA_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    SUPPORTED_EXTENSIONS,
)
from model import embed_model


load_dotenv()


def calculate_document_id(file_path: str | Path) -> str:
    """
    Generate a stable ID from the document contents.

    The same file contents will always produce the
    same document ID, even if the file is renamed.
    """

    path = Path(file_path)

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def calculate_chunk_id(
    document_id: str,
    chunk_index: int,
    content: str,
) -> str:
    """
    Generate a stable ID for an individual chunk.
    """

    raw_id = (
        f"{document_id}:"
        f"{chunk_index}:"
        f"{content}"
    )

    return hashlib.sha256(
        raw_id.encode("utf-8")
    ).hexdigest()


def load_document(
    file_path: str | Path,
) -> list[Document]:
    """
    Load a single document based on its extension.
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
            f"Supported types: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    documents = loader.load()

    document_id = calculate_document_id(path)

    for document in documents:
        document.metadata.update(
            {
                "source": str(path),
                "file_name": path.name,
                "document_id": document_id,
                "file_type": extension,
            }
        )

    return documents


def load_documents(
    file_paths: Iterable[str | Path],
) -> list[Document]:
    """
    Load multiple documents.
    """

    documents = []

    for file_path in file_paths:

        print(
            f"---LOADING: {file_path}---"
        )

        loaded_documents = load_document(
            file_path
        )

        print(
            f"Loaded {len(loaded_documents)} "
            f"document sections."
        )

        documents.extend(
            loaded_documents
        )

    if not documents:
        raise ValueError(
            "No documents were loaded."
        )

    return documents


def split_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Split documents into retrieval chunks.
    """

    text_splitter = (
        RecursiveCharacterTextSplitter
        .from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    )

    return text_splitter.split_documents(
        documents
    )


def get_vectorstore() -> Chroma:
    """
    Return the persistent Chroma vector store.
    """

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embed_model,
        persist_directory=CHROMA_PATH,
    )


def document_exists(
    vectorstore: Chroma,
    document_id: str,
) -> bool:
    """
    Check whether a document already exists
    in the vector store.
    """

    result = vectorstore.get(
        where={
            "document_id": document_id,
        },
        include=["metadatas"],
    )

    return bool(result.get("ids"))


def prepare_chunk_ids(
    document_chunks: list[Document],
) -> list[str]:
    """
    Generate deterministic IDs for document chunks.
    """

    chunk_ids = []

    chunk_counters = {}

    for document in document_chunks:

        document_id = document.metadata[
            "document_id"
        ]

        chunk_index = chunk_counters.get(
            document_id,
            0,
        )

        chunk_id = calculate_chunk_id(
            document_id=document_id,
            chunk_index=chunk_index,
            content=document.page_content,
        )

        document.metadata[
            "chunk_index"
        ] = chunk_index

        chunk_ids.append(chunk_id)

        chunk_counters[document_id] = (
            chunk_index + 1
        )

    return chunk_ids


def build_vectorstore(
    file_paths: Iterable[str | Path],
) -> Chroma:
    """
    Load, split, deduplicate and add documents
    to the persistent Chroma vector store.
    """

    print("=" * 60)
    print("--- DOCUMENT INGESTION ---")
    print("=" * 60)

    print("\n---LOADING DOCUMENTS---")

    documents = load_documents(
        file_paths
    )

    print(
        f"\nLoaded {len(documents)} "
        f"source document sections."
    )

    print("\n---CHECKING DOCUMENTS---")

    vectorstore = get_vectorstore()

    unique_documents = []
    skipped_documents = set()

    for document in documents:

        document_id = document.metadata[
            "document_id"
        ]

        if document_id in skipped_documents:
            continue

        if document_exists(
            vectorstore,
            document_id,
        ):
            print(
                f"---SKIPPING DUPLICATE: "
                f"{document.metadata['file_name']}---"
            )

            skipped_documents.add(
                document_id
            )

        else:
            unique_documents.append(
                document
            )

    if not unique_documents:

        print(
            "\n---NO NEW DOCUMENTS TO INGEST---"
        )

        return vectorstore

    print(
        f"New document sections: "
        f"{len(unique_documents)}"
    )

    print("\n---SPLITTING DOCUMENTS---")

    document_chunks = split_documents(
        unique_documents
    )

    print(
        f"Created {len(document_chunks)} "
        f"document chunks."
    )

    print("\n---GENERATING CHUNK IDS---")

    chunk_ids = prepare_chunk_ids(
        document_chunks
    )

    print(
        f"Generated {len(chunk_ids)} "
        f"stable chunk IDs."
    )

    print(
        "\n---UPDATING CHROMA VECTOR STORE---"
    )

    vectorstore.add_documents(
        documents=document_chunks,
        ids=chunk_ids,
    )

    print("---VECTOR STORE UPDATED---")

    print(
        f"Chroma path: {CHROMA_PATH}"
    )

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    print(
        f"Chunks added: "
        f"{len(document_chunks)}"
    )

    print(
        f"Documents skipped: "
        f"{len(skipped_documents)}"
    )

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
        if (
            file.is_file()
            and file.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ]

    return sorted(files)


def list_documents() -> list[dict]:
    """
    List unique documents currently stored
    in the vector store.
    """

    vectorstore = get_vectorstore()

    result = vectorstore.get(
        include=["metadatas"]
    )

    documents = {}

    for metadata in result.get(
        "metadatas",
        [],
    ):

        if not metadata:
            continue

        document_id = metadata.get(
            "document_id"
        )

        if not document_id:
            continue

        if document_id not in documents:
            documents[document_id] = {
                "file_name": metadata.get(
                    "file_name",
                    "Unknown",
                ),
                "source": metadata.get(
                    "source",
                    "Unknown",
                ),
                "file_type": metadata.get(
                    "file_type",
                    "Unknown",
                ),
            }

    return list(
        documents.values()
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Ingest documents into the "
            "RAG vector store."
        )
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

    build_vectorstore(
        args.files
    )