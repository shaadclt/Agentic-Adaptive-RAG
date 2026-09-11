from typing import List

from langchain_core.documents import Document


DEFAULT_MAX_CHARS = 6000
DEFAULT_MAX_CHARS_PER_DOCUMENT = 2000


def format_documents(
    documents: List[Document],
    max_chars: int = DEFAULT_MAX_CHARS,
    max_chars_per_document: int = DEFAULT_MAX_CHARS_PER_DOCUMENT,
) -> str:
    """
    Convert retrieved documents into a compact text context.

    Limits are applied both per document and across the complete
    context to reduce unnecessary LLM token usage.
    """

    if not documents:
        return ""

    context_parts = []
    total_chars = 0

    for index, document in enumerate(documents, start=1):

        content = document.page_content.strip()

        if not content:
            continue

        content = content[:max_chars_per_document]

        remaining_chars = max_chars - total_chars

        if remaining_chars <= 0:
            break

        content = content[:remaining_chars]

        context_parts.append(
            f"[Document {index}]\n{content}"
        )

        total_chars += len(content)

    return "\n\n".join(context_parts)