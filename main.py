from pathlib import Path

from graph.graph import app
from ingestion import (
    build_vectorstore,
    delete_document,
    list_documents,
)

from sources import extract_sources


def print_menu() -> None:
    print()
    print("=" * 60)
    print("        AGENTIC-ADAPTIVE-RAG")
    print("=" * 60)
    print("1. Add documents")
    print("2. View knowledge base")
    print("3. Remove document")
    print("4. Ask a question")
    print("5. Exit")
    print("=" * 60)


def add_documents() -> None:
    print()
    print("--- ADD DOCUMENTS ---")

    raw_paths = input(
        "Enter file paths separated by commas: "
    ).strip()

    if not raw_paths:
        print("---NO FILES PROVIDED---")
        return

    file_paths = [
        path.strip().strip('"')
        for path in raw_paths.split(",")
        if path.strip()
    ]

    valid_paths = []

    for file_path in file_paths:
        path = Path(file_path)

        if not path.exists():
            print(
                f"---FILE NOT FOUND: {file_path}---"
            )
            continue

        if path.suffix.lower() not in {
            ".pdf",
            ".docx",
            ".txt",
            ".md",
        }:
            print(
                f"---UNSUPPORTED FILE TYPE: {path.name}---"
            )
            continue

        valid_paths.append(str(path))

    if not valid_paths:
        print("---NO VALID DOCUMENTS---")
        return

    print()
    print(
        f"Found {len(valid_paths)} valid document(s)."
    )

    build_vectorstore(valid_paths)


def view_knowledge_base() -> None:
    print()
    print("--- KNOWLEDGE BASE ---")

    documents = list_documents()

    if not documents:
        print("---KNOWLEDGE BASE IS EMPTY---")
        return

    print(
        f"Documents in knowledge base: "
        f"{len(documents)}"
    )
    print()

    for index, document in enumerate(
        documents,
        start=1,
    ):
        print(
            f"[{index}] "
            f"{document['file_name']}"
        )

        print(
            f"    Type: "
            f"{document['file_type']}"
        )

        print(
            f"    ID: "
            f"{document['document_id']}"
        )

        print(
            f"    Source: "
            f"{document['source']}"
        )

        print()


def remove_document() -> None:
    print()
    print("--- REMOVE DOCUMENT ---")

    documents = list_documents()

    if not documents:
        print("---KNOWLEDGE BASE IS EMPTY---")
        return

    for index, document in enumerate(
        documents,
        start=1,
    ):
        print(
            f"{index}. "
            f"{document['file_name']}"
        )

    print()

    choice = input(
        "Enter document number to remove: "
    ).strip()

    if not choice.isdigit():
        print("---INVALID SELECTION---")
        return

    index = int(choice)

    if index < 1 or index > len(documents):
        print("---INVALID DOCUMENT NUMBER---")
        return

    document = documents[index - 1]

    confirm = input(
        f"Remove '{document['file_name']}'? "
        "[y/N]: "
    ).strip().lower()

    if confirm != "y":
        print("---REMOVAL CANCELLED---")
        return

    deleted = delete_document(
        document["document_id"]
    )

    if deleted:
        print(
            f"---REMOVED: "
            f"{document['file_name']}---"
        )
    else:
        print("---DOCUMENT NOT FOUND---")


def ask_question() -> None:
    print()
    print("--- ASK A QUESTION ---")

    question = input(
        "Question: "
    ).strip()

    if not question:
        print("---QUESTION CANNOT BE EMPTY---")
        return

    print()
    print("---PROCESSING QUESTION---")

    result = app.invoke(
        {
            "question": question,
            "retry_count": 0,
        }
    )

    print()
    print("=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(
        result.get(
            "generation",
            "No answer generated.",
        )
    )

    documents = result.get(
        "documents",
        [],
    )

    sources = extract_sources(documents)

    if sources:
        print()
        print("=" * 60)
        print("SOURCES")
        print("=" * 60)

        local_sources = [
            source
            for source in sources
            if source["type"] == "local"
        ]

        web_sources = [
            source
            for source in sources
            if source["type"] == "web"
        ]

        if local_sources:
            print()
            print("Local documents:")

            for source in local_sources:
                print(
                    f"- {source['file_name']}"
                )

        if web_sources:
            print()
            print("Web sources:")

            for source in web_sources:
                title = source.get(
                    "title",
                    "Web source",
                )

                url = source.get(
                    "url",
                    "",
                )

                print(f"- {title}")

                if url:
                    print(f"  {url}")

    else:
        print()
        print("---NO SOURCES AVAILABLE---")

    print("=" * 60)


def main() -> None:
    while True:
        print_menu()

        choice = input(
            "Select an option: "
        ).strip()

        if choice == "1":
            add_documents()

        elif choice == "2":
            view_knowledge_base()

        elif choice == "3":
            remove_document()

        elif choice == "4":
            ask_question()

        elif choice == "5":
            print("---GOODBYE---")
            break

        else:
            print("---INVALID OPTION---")


if __name__ == "__main__":
    main()