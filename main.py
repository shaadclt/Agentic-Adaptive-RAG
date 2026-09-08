from pathlib import Path
from time import perf_counter

from evaluation import (
    EvaluationResult,
    EvaluationTracker,
)

from graph.graph import app

from ingestion import (
    build_vectorstore,
    delete_document,
    list_documents,
)

from sources import extract_sources


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
}


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

evaluation_tracker = EvaluationTracker()


# ---------------------------------------------------------
# MENU
# ---------------------------------------------------------

def print_menu() -> None:
    print()
    print("=" * 60)
    print("        AGENTIC-ADAPTIVE-RAG")
    print("=" * 60)
    print("1. Add documents")
    print("2. View knowledge base")
    print("3. Remove document")
    print("4. Ask a question")
    print("5. View evaluation")
    print("6. Exit")
    print("=" * 60)


# ---------------------------------------------------------
# ADD DOCUMENTS
# ---------------------------------------------------------

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

        if not path.is_file():
            print(
                f"---NOT A FILE: {file_path}---"
            )
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(
                f"---UNSUPPORTED FILE TYPE: "
                f"{path.name}---"
            )
            continue

        valid_paths.append(str(path))

    if not valid_paths:
        print("---NO VALID DOCUMENTS---")
        return

    print()
    print(
        f"Found {len(valid_paths)} "
        f"valid document(s)."
    )

    build_vectorstore(valid_paths)


# ---------------------------------------------------------
# VIEW KNOWLEDGE BASE
# ---------------------------------------------------------

def view_knowledge_base() -> None:
    print()
    print("--- KNOWLEDGE BASE ---")

    documents = list_documents()

    if not documents:
        print(
            "---KNOWLEDGE BASE IS EMPTY---"
        )
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


# ---------------------------------------------------------
# REMOVE DOCUMENT
# ---------------------------------------------------------

def remove_document() -> None:
    print()
    print("--- REMOVE DOCUMENT ---")

    documents = list_documents()

    if not documents:
        print(
            "---KNOWLEDGE BASE IS EMPTY---"
        )
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
        print(
            "---INVALID DOCUMENT NUMBER---"
        )
        return

    document = documents[index - 1]

    confirm = input(
        f"Remove "
        f"'{document['file_name']}'? [y/N]: "
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
        print(
            "---DOCUMENT NOT FOUND---"
        )


# ---------------------------------------------------------
# DISPLAY SOURCES
# ---------------------------------------------------------

def display_sources(
    sources: list[dict],
) -> None:

    if not sources:
        print()
        print(
            "---NO SOURCES AVAILABLE---"
        )
        return

    print()
    print("=" * 60)
    print("SOURCES")
    print("=" * 60)

    local_sources = [
        source
        for source in sources
        if source.get("type") == "local"
    ]

    web_sources = [
        source
        for source in sources
        if source.get("type") == "web"
    ]

    # -----------------------------------------------------
    # LOCAL SOURCES
    # -----------------------------------------------------

    if local_sources:

        print()
        print("Local documents:")

        displayed_files = set()

        for source in local_sources:

            file_name = source.get(
                "file_name",
                "Unknown document",
            )

            if file_name in displayed_files:
                continue

            displayed_files.add(
                file_name
            )

            print(
                f"- {file_name}"
            )

    # -----------------------------------------------------
    # WEB SOURCES
    # -----------------------------------------------------

    if web_sources:

        print()
        print("Web sources:")

        displayed_urls = set()

        for source in web_sources:

            title = source.get(
                "title",
                "Web source",
            )

            url = source.get(
                "url",
                "",
            )

            if url in displayed_urls:
                continue

            displayed_urls.add(url)

            print(
                f"- {title}"
            )

            if url:
                print(
                    f"  {url}"
                )


# ---------------------------------------------------------
# ASK QUESTION
# ---------------------------------------------------------

def ask_question() -> None:

    print()
    print("--- ASK A QUESTION ---")

    question = input(
        "Question: "
    ).strip()

    if not question:

        print(
            "---QUESTION CANNOT BE EMPTY---"
        )

        return

    print()
    print("---PROCESSING QUESTION---")

    # -----------------------------------------------------
    # START LATENCY TIMER
    # -----------------------------------------------------

    start_time = perf_counter()

    # -----------------------------------------------------
    # RUN AGENTIC RAG GRAPH
    # -----------------------------------------------------

    result = app.invoke(
        {
            "question": question,
            "retry_count": 0,
        }
    )

    # -----------------------------------------------------
    # CALCULATE LATENCY
    # -----------------------------------------------------

    latency = (
        perf_counter()
        - start_time
    )

    # -----------------------------------------------------
    # EXTRACT ANSWER
    # -----------------------------------------------------

    answer = result.get(
        "answer",
        result.get(
            "generation",
            "No answer generated.",
        ),
    )

    # -----------------------------------------------------
    # EXTRACT SOURCES
    # -----------------------------------------------------

    sources = result.get(
        "sources"
    )

    if sources is None:

        sources = extract_sources(
            result.get(
                "documents",
                [],
            )
        )

    # -----------------------------------------------------
    # EXTRACT EVALUATION DATA
    # -----------------------------------------------------

    route = result.get(
        "route",
        "unknown",
    )

    retry_count = result.get(
        "retry_count",
        0,
    )

    retrieved_documents = result.get(
        "retrieved_documents",
        0,
    )

    relevant_documents = result.get(
        "relevant_documents",
        0,
    )

    grounded = result.get(
        "grounded",
        False,
    )

    answers_question = result.get(
        "answers_question",
        True,
    )

    # -----------------------------------------------------
    # STORE EVALUATION RESULT
    # -----------------------------------------------------

    evaluation_result = EvaluationResult(
        question=question,
        answer=answer,
        route=route,
        retrieved_documents=retrieved_documents,
        relevant_documents=relevant_documents,
        grounded=grounded,
        answers_question=answers_question,
        retry_count=retry_count,
        latency_seconds=latency,
        sources=sources,
    )

    evaluation_tracker.add(
        evaluation_result
    )

    # -----------------------------------------------------
    # DISPLAY ANSWER
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(answer)

    # -----------------------------------------------------
    # DISPLAY SOURCES
    # -----------------------------------------------------

    display_sources(
        sources
    )

    # -----------------------------------------------------
    # DISPLAY METADATA
    # -----------------------------------------------------

    print()

    print(
        f"Route: {route}"
    )

    print(
        f"Generation retries: "
        f"{retry_count}"
    )

    print(
        f"Latency: "
        f"{latency:.2f} seconds"
    )

    if retrieved_documents:

        print(
            f"Retrieved documents: "
            f"{retrieved_documents}"
        )

        print(
            f"Relevant documents: "
            f"{relevant_documents}"
        )

        relevance_rate = (
            relevant_documents
            / retrieved_documents
        )

        print(
            f"Retrieval relevance: "
            f"{relevance_rate * 100:.1f}%"
        )

    print("=" * 60)


# ---------------------------------------------------------
# VIEW EVALUATION
# ---------------------------------------------------------

def view_evaluation() -> None:

    print()
    print("=" * 60)
    print("RAG EVALUATION")
    print("=" * 60)

    summary = evaluation_tracker.summary()

    if summary["total_questions"] == 0:

        print()
        print(
            "---NO QUESTIONS EVALUATED YET---"
        )

        print("=" * 60)

        return

    print()

    print(
        f"Total questions: "
        f"{summary['total_questions']}"
    )

    print(
        f"Average latency: "
        f"{summary['average_latency_seconds']:.2f} "
        f"seconds"
    )

    print(
        f"Grounded answer rate: "
        f"{summary['grounded_rate'] * 100:.1f}%"
    )

    print(
        f"Answer quality rate: "
        f"{summary['answer_quality_rate'] * 100:.1f}%"
    )

    print(
        f"Average generation retries: "
        f"{summary['average_retries']:.2f}"
    )

    print()
    print("=" * 60)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

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

            view_evaluation()

        elif choice == "6":

            print(
                "---GOODBYE---"
            )

            break

        else:

            print(
                "---INVALID OPTION---"
            )


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    main()