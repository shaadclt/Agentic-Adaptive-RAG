from typing import List

from langchain_core.documents import Document

from security.content_guard import (
    wrap_untrusted_content,
)


DEFAULT_MAX_CHARS = 6000

DEFAULT_MAX_CHARS_PER_DOCUMENT = 2000


def format_documents(
    documents: List[Document],
    max_chars: int = DEFAULT_MAX_CHARS,
    max_chars_per_document: int = DEFAULT_MAX_CHARS_PER_DOCUMENT,
) -> str:
    """
    Format retrieved documents as explicitly untrusted content.

    Retrieved documents are treated as data/evidence and never
    as executable instructions.
    """

    if not documents:
        return ""

    context_parts = []

    total_chars = 0

    for index, document in enumerate(
        documents,
        start=1,
    ):
        content = document.page_content.strip()

        if not content:
            continue

        content = content[
            :max_chars_per_document
        ]

        remaining_chars = (
            max_chars - total_chars
        )

        if remaining_chars <= 0:
            break

        content = content[
            :remaining_chars
        ]

        metadata = document.metadata or {}

        source_type = (
            "web"
            if metadata.get("source") == "web"
            else "document"
        )

        source_name = (
            metadata.get("title")
            or metadata.get("file_name")
            or metadata.get("source")
            or f"Document {index}"
        )

        wrapped_content = wrap_untrusted_content(
            content=content,
            source_type=source_type,
            source_name=str(source_name),
        )

        context_parts.append(
            f"[Retrieved Evidence {index}]\n"
            f"{wrapped_content}"
        )

        total_chars += len(wrapped_content)

    return "\n\n".join(context_parts)