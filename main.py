from pathlib import Path

from dotenv import load_dotenv

from graph.graph import app
from ingestion import (
    build_vectorstore,
    list_documents,
)


load_dotenv()


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
}


def format_response(result):
    """
    Format the response from the graph.
    """

    if isinstance(result, dict) and "generation" in result:
        return result["generation"]

    if isinstance(result, dict) and "answer" in result:
        return result["answer"]

    return str(result)


def parse_file_paths(
    user_input: str,
) -> list[Path]:
    """
    Parse comma-separated file paths.
    """

    raw_paths = [
        path.strip().strip('"')
        for path in user_input.split(",")
        if path.strip()
    ]

    return [
        Path(path)
        for path in raw_paths
    ]


def validate_file_paths(
    file_paths: list[Path],
) -> list[Path]:
    """
    Validate supplied document paths.
    """

    valid_files = []

    for file_path in file_paths:

        if not file_path.exists():
            print(
                f"❌ File not found: {file_path}"
            )
            continue

        if not file_path.is_file():
            print(
                f"❌ Not a file: {file_path}"
            )
            continue

        if (
            file_path.suffix.lower()
            not in SUPPORTED_EXTENSIONS
        ):
            print(
                f"❌ Unsupported file type: "
                f"{file_path.suffix}"
            )
            print(
                "   Supported types: "
                "PDF, DOCX, TXT, MD"
            )
            continue

        valid_files.append(
            file_path
        )

    return valid_files


def ingest_documents():
    """
    Ask for document paths and ingest them.
    """

    print("\n" + "=" * 60)
    print("📚 ADD DOCUMENTS")
    print("=" * 60)

    print(
        "\nSupported formats: PDF, DOCX, TXT, MD"
    )

    print(
        "\nFor multiple files, separate paths "
        "with commas."
    )

    user_input = input(
        "\n📄 File path(s): "
    ).strip()

    if not user_input:
        print(
            "\n❌ No file paths provided."
        )
        return

    file_paths = parse_file_paths(
        user_input
    )

    valid_files = validate_file_paths(
        file_paths
    )

    if not valid_files:
        print(
            "\n❌ No valid documents to ingest."
        )
        return

    print(
        f"\nFound {len(valid_files)} "
        f"valid document(s)."
    )

    try:

        build_vectorstore(
            valid_files
        )

        print(
            "\n✅ Document ingestion completed."
        )

    except Exception as error:

        print(
            "\n❌ Document ingestion failed:"
        )

        print(
            f"   {error}"
        )


def show_documents():
    """
    Display documents currently in the
    knowledge base.
    """

    print("\n" + "=" * 60)
    print("📚 KNOWLEDGE BASE")
    print("=" * 60)

    try:

        documents = list_documents()

        if not documents:

            print(
                "\nNo documents found."
            )
            return

        print(
            f"\nKnowledge base contains "
            f"{len(documents)} document(s):\n"
        )

        for index, document in enumerate(
            documents,
            start=1,
        ):

            print(
                f"{index}. "
                f"{document['file_name']}"
            )

            print(
                f"   Type: "
                f"{document['file_type']}"
            )

            print(
                f"   Source: "
                f"{document['source']}"
            )

    except Exception as error:

        print(
            "\n❌ Could not read knowledge base:"
        )

        print(
            f"   {error}"
        )


def ask_question():
    """
    Ask a question and run the RAG graph.
    """

    print("\n" + "=" * 60)
    print("💬 ASK A QUESTION")
    print("=" * 60)

    user_question = input(
        "\n💬 You: "
    ).strip()

    if not user_question:

        print(
            "\n❌ Please enter a question."
        )
        return

    print(
        "\n🤔 Bot: Thinking..."
    )

    try:

        result = app.invoke(
            input={
                "question": user_question,
            }
        )

        response = format_response(
            result
        )

        print(
            f"\n🤖 Bot: {response}"
        )

    except Exception as error:

        print(
            "\n❌ Sorry, I encountered an error:"
        )

        print(
            f"   {error}"
        )


def show_menu():

    print("\n" + "=" * 60)
    print("🤖 AGENTIC ADAPTIVE RAG")
    print("=" * 60)

    print(
        "\n1. Add documents"
    )

    print(
        "2. View knowledge base"
    )

    print(
        "3. Ask a question"
    )

    print(
        "4. Exit"
    )


def main():

    print("=" * 60)
    print("🤖 Agentic Adaptive RAG")
    print("=" * 60)

    print(
        "\nWelcome to the Agentic Adaptive RAG system!"
    )

    print(
        "You can add documents, inspect the "
        "knowledge base, and ask questions."
    )

    while True:

        show_menu()

        try:

            choice = input(
                "\nSelect an option: "
            ).strip()

            if choice == "1":

                ingest_documents()

            elif choice == "2":

                show_documents()

            elif choice == "3":

                ask_question()

            elif choice == "4":

                print(
                    "\n👋 Goodbye!"
                )

                break

            else:

                print(
                    "\n❌ Invalid option."
                )

        except KeyboardInterrupt:

            print(
                "\n\n👋 Goodbye!"
            )

            break

        except Exception as error:

            print(
                f"\n❌ Unexpected error: "
                f"{error}"
            )


if __name__ == "__main__":
    main()