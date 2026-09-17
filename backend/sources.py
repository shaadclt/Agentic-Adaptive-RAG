from typing import Any, Dict, List

from langchain_core.documents import Document


def extract_sources(
    documents: List[Document],
) -> List[Dict[str, Any]]:
    """
    Extract unique source information from retrieved documents.

    Multiple chunks from the same document are represented
    by a single source.
    """

    sources: List[Dict[str, Any]] = []
    seen = set()

    for document in documents:
        metadata = document.metadata or {}

        if metadata.get("source") == "web":
            url = metadata.get("url", "")

            key = (
                "web",
                url,
            )

            source = {
                "type": "web",
                "title": metadata.get(
                    "title",
                    "Web source",
                ),
                "url": url,
            }

        else:
            document_id = metadata.get(
                "document_id",
                "",
            )

            file_name = metadata.get(
                "file_name",
                "",
            )

            source_path = metadata.get(
                "source",
                "",
            )

            # Prefer document_id when available.
            # Fall back to filename/source for older
            # Chroma records that may not have document_id.
            key = (
                "local",
                document_id
                or file_name
                or source_path,
            )

            source = {
                "type": "local",
                "file_name": (
                    file_name
                    or source_path
                    or "Unknown document"
                ),
                "source": source_path,
                "document_id": document_id,
            }

        if key in seen:
            continue

        seen.add(key)
        sources.append(source)

    return sources