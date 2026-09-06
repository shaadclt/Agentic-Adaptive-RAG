from pathlib import Path

from dotenv import load_dotenv

from graph.graph import app
from ingestion import build_vectorstore


load_dotenv()


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
}


def format_response(result):
    """
    Format the response from the graph for better readability.
    """

    if isinstance(result, dict) and "generation" in result:
        return result["generation"]

    if isinstance(result, dict) and "answer" in result:
        return result["answer"]

    return str(result)


def parse_file_paths(user_input: str) -> list[Path]:
    """
    Parse one or more file paths entered by the user.

    Paths can be separated by commas.
    """

    raw_paths = [
        path.strip().strip('"')
        for path in user_input.split(",")
        if path.strip()
    ]

    return [Path(path) for path in raw_paths]


def validate_file_paths(
    file_paths: list[Path],
) -> list[Path]:
    """
    Validate that all supplied files exist and are supported.
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

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(
                f"❌ Unsupported file type: "
                f"{file_path.suffix}"
            )
            print(
                "   Supported types: "
                "PDF, DOCX, TXT, MD"
            )
            continue

        valid_files.append(file_path)

    return valid_files


def ingest_documents():
    """
    Ask the user for document paths and ingest them.
    """

    print("\n" + "=" * 60)
    print("📚 ADD DOCUMENTS")
    print("=" * 60)

    print(
        "\nSupported formats: PDF, DOCX, TXT, MD"
    )

    print(
        "\nEnter one or more file paths."
    )

    print(
        "For multiple files, separate paths with commas."
    )

    print(
        "\nExample:"
    )

    print(
        r'C:\docs\manual.pdf, C:\docs\policy.docx'
    )

    user_input = input(
        "\n📄 File path(s): "
    ).strip()

    if not user_input:
        print(
            "\n❌ No file paths provided."
        )
        return

    file_paths = parse_file_paths(user_input)

    valid_files = validate_file_paths(
        file_paths
    )

    if not valid_files:
        print(
            "\n❌ No valid documents to ingest."
        )
        return

    print(
        f"\nFound {len(valid_files)} valid "
        f"document(s)."
    )

    try:
        build_vectorstore(valid_files)

        print(
            "\n✅ Document ingestion completed successfully."
        )

    except Exception as error:
        print(
            "\n❌ Document ingestion failed:"
        )
        print(f"   {error}")


def ask_question():
    """
    Ask the user a question and run the RAG graph.
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

        response = format_response(result)

        print(
            f"\n🤖 Bot: {response}"
        )

    except Exception as error:
        print(
            "\n❌ Sorry, I encountered an error:"
        )
        print(f"   {error}")


def show_menu():
    """
    Display the main application menu.
    """

    print("\n" + "=" * 60)
    print("🤖 AGENTIC ADAPTIVE RAG")
    print("=" * 60)

    print(
        "\n1. Add documents"
    )

    print(
        "2. Ask a question"
    )

    print(
        "3. Exit"
    )


def main():

    print("=" * 60)
    print("🤖 Agentic Adaptive RAG")
    print("=" * 60)

    print(
        "\nWelcome to the Agentic Adaptive RAG system!"
    )

    print(
        "You can add your own documents and "
        "ask questions about them."
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

                ask_question()

            elif choice == "3":

                print(
                    "\n👋 Goodbye!"
                )
                break

            else:

                print(
                    "\n❌ Invalid option."
                )

                print(
                    "Please choose 1, 2, or 3."
                )

        except KeyboardInterrupt:

            print(
                "\n\n👋 Goodbye!"
            )
            break

        except Exception as error:

            print(
                f"\n❌ Unexpected error: {error}"
            )


if __name__ == "__main__":
    main()