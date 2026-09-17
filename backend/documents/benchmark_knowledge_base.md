# Agentic Adaptive RAG — Project Knowledge Base

## Project Overview

Agentic Adaptive RAG is a document question-answering system built with Python, LangGraph, LangChain, Chroma, and large language models.

The system allows users to upload documents and then ask questions about the uploaded knowledge base. Supported document formats include PDF, DOCX, TXT, and Markdown files.

The system is designed around adaptive retrieval. It first attempts to use the user's local knowledge base. When the retrieved documents are not relevant enough to answer the question, the system can fall back to web search.

## LangGraph

LangGraph is used to coordinate the different steps of the RAG workflow.

The graph coordinates retrieval, document grading, generation, hallucination checking, answer grading, retry handling, and web search.

The workflow begins with a user question. Depending on whether a local knowledge base is available, the system either retrieves local documents or uses the question router to determine whether web search is appropriate.

## Chroma

Chroma is used as the vector database for the local knowledge base.

Uploaded documents are split into chunks and stored as vector embeddings in Chroma. Chroma allows the system to perform semantic similarity retrieval against the user's question.

The Chroma database is persisted locally in the .chroma directory.

## Document Ingestion

The ingestion pipeline supports PDF, DOCX, TXT, and Markdown documents.

During ingestion, a document is loaded and split into smaller chunks. Each document receives a deterministic SHA-256 document identifier based on its file contents.

Each chunk also receives a deterministic identifier based on the document identifier, chunk index, and chunk content.

This allows the system to detect duplicate documents and avoid ingesting the same document more than once.

If a document is modified, its content hash changes and it is treated as a new document version.

## Chunking

Documents are divided into smaller chunks before being stored in the vector database.

The default chunk size is 250 characters and the default chunk overlap is 0.

Chunking allows the retriever to work with focused portions of documents instead of retrieving entire large documents.

## Embeddings

The system converts document chunks into vector embeddings before storing them in Chroma.

The embedding model is configured through the EMBEDDING_MODEL environment variable.

The embeddings allow semantic similarity search between a user question and document chunks.

## Retrieval

The retriever uses Chroma to retrieve the most relevant document chunks for a question.

The default retrieval value is four documents, controlled by the RETRIEVAL_K configuration setting.

After retrieval, each document is passed through a retrieval grader.

## Retrieval Grading

Retrieved documents are graded for relevance to the user's question.

Documents judged relevant are kept for generation.

If no retrieved documents are considered relevant, the system activates web search.

This allows the system to avoid generating an answer from irrelevant local documents.

## Web Search Fallback

Web search is used when the local knowledge base does not contain relevant information.

The system uses Tavily for web search.

Web search results are converted into LangChain Document objects and passed into the generation step.

Web search is particularly useful for questions about current information or topics that are not contained in the user's uploaded documents.

## Generation

The generation step uses the retrieved documents as context for answering the question.

The generation prompt instructs the model to use the supplied context and avoid inventing unsupported facts.

If the available context does not contain enough information, the model should say that the context is insufficient.

## Hallucination Grading

After generation, the system checks whether the generated answer is grounded in the supplied documents.

The hallucination grader determines whether the answer is supported by the retrieved context.

If the generated answer is not grounded, the graph can retry generation.

The maximum number of generation retries is currently two.

## Answer Grading

The answer grader checks whether the generated answer actually addresses the user's question.

The answer is evaluated against the original question rather than against the generated answer itself.

An answer must both be grounded in the available context and address the question successfully to be considered useful.

## Retry Handling

The graph tracks generation retries using retry_count.

When a generation is not grounded, the retry counter is incremented and generation is attempted again.

The system has a maximum generation retry limit to prevent endless generation attempts.

## Routing

The system uses a local-first architecture.

When the local knowledge base contains documents, the system initially attempts local retrieval.

If the retrieved local documents are not relevant, the graph falls back to web search.

When there is no local knowledge base, a question router determines whether the question should use the vectorstore or web search.

The router can choose vectorstore for questions related to uploaded knowledge and websearch for questions requiring current or external information.

## Response Sources

The system returns source information together with the answer.

Local sources include the filename, source path, document identifier, and document type information.

Web sources include the title and URL.

Multiple chunks belonging to the same local document are represented as a single source.

## Evaluation

The project contains a persistent evaluation system.

Each evaluated question records the question, answer, route, retrieved document count, relevant document count, groundedness, answer quality, retry count, latency, and sources.

Evaluation results are stored in evaluation_history.jsonl.

The evaluation tracker calculates average latency, grounded answer rate, answer quality rate, and average retries.

## Automated Benchmarking

The project also contains an automated benchmark dataset.

The benchmark evaluates routing correctness, groundedness, answer quality, expected answer terms, retrieval performance, latency, and retries.

Benchmark results are stored separately from persistent interactive evaluation history.

## LangSmith

LangSmith tracing can be enabled through the LANGSMITH_TRACING environment variable.

The project uses the LANGSMITH_PROJECT setting to identify the tracing project.

LangSmith can be used to inspect the execution of the LangGraph workflow and its individual model calls.

## Configuration

Central configuration is stored in config.py and can be overridden through environment variables.

Important configuration values include CHROMA_PATH, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_K, EMBEDDING_MODEL, LLM_MODEL, and LangSmith settings.

## Knowledge Base Management

The command-line application allows users to add documents, view the knowledge base, remove documents, ask questions, view evaluation results, and exit.

Users provide document paths rather than relying on hardcoded application sources.

The knowledge base can therefore be replaced or extended with documents relevant to a particular use case.

## Overall Workflow

The high-level workflow is:

Question
-> determine whether local knowledge exists
-> retrieve local documents when available
-> grade retrieved documents
-> use relevant documents when available
-> perform web search when local documents are insufficient
-> generate an answer
-> check hallucination/groundedness
-> check whether the answer addresses the question
-> retry generation when appropriate
-> return the final answer and its sources.

The purpose of the architecture is to combine retrieval augmented generation with agentic decision-making, evaluation, retries, and web fallback.
