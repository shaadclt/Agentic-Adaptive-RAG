from typing import List, Dict, Any

from langchain_core.documents import Document


def extract_sources(
    documents: List[Document],
) -> List[Dict[str, Any]]:
    """
    Extract unique source information from retrieved documents.
    """

    sources = []
    seen = set()

    for document in documents:
        metadata = document.metadata or {}

        source_type = metadata.get(
            "source",
            "unknown",
        )

        if source_type == "web":
            key = (
                "web",
                metadata.get("url", ""),
            )

            source = {
                "type": "web",
                "title": metadata.get(
                    "title",
                    "Web source",
                ),
                "url": metadata.get(
                    "url",
                    "",
                ),
            }

        else:
            key = (
                "local",
                metadata.get(
                    "document_id",
                    metadata.get(
                        "source",
                        "",
                    ),
                ),
            )

            source = {
                "type": "local",
                "file_name": metadata.get(
                    "file_name",
                    "Unknown document",
                ),
                "source": metadata.get(
                    "source",
                    "",
                ),
                "document_id": metadata.get(
                    "document_id",
                    "",
                ),
            }

        if key in seen:
            continue

        seen.add(key)
        sources.append(source)

    return sources