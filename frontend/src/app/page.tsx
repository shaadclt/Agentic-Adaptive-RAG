"use client";

import { ChangeEvent, useEffect, useState } from "react";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

type DocumentItem = {
  file_name: string;
  file_type: string;
  document_id: string;
  source: string;
};

type Source = {
  type: string;
  title?: string;
  file_name?: string;
  url?: string;
  source?: string;
  document_id?: string;
};

type Metrics = {
  retrieved_documents: number;
  relevant_documents: number;
  grounded: boolean;
  answers_question: boolean;
  retry_count: number;
  latency_seconds: number;
};

type ChatResponse = {
  answer: string;
  route: string;
  sources: Source[];
  metrics: Metrics;
};

export default function Home() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState<ChatResponse | null>(null);

  const [loadingDocuments, setLoadingDocuments] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);

  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  async function loadDocuments() {
    setLoadingDocuments(true);
    setError("");

    try {
      const res = await fetch(`${API_URL}/documents`);

      if (!res.ok) {
        throw new Error("Failed to load documents.");
      }

      const data: DocumentItem[] = await res.json();
      setDocuments(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to load documents."
      );
    } finally {
      setLoadingDocuments(false);
    }
  }

  useEffect(() => {
    loadDocuments();
  }, []);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    if (!event.target.files) {
      setSelectedFiles([]);
      return;
    }

    setSelectedFiles(Array.from(event.target.files));
    setMessage("");
    setError("");
  }

  async function uploadFiles() {
    if (selectedFiles.length === 0) {
      setError("Select at least one document.");
      return;
    }

    setUploading(true);
    setMessage("");
    setError("");

    try {
      const formData = new FormData();

      selectedFiles.forEach((file) => {
        formData.append("files", file);
      });

      const res = await fetch(`${API_URL}/documents/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Document upload failed."
        );
      }

      setMessage(
        `Successfully processed ${data.uploaded_files.length} document(s).`
      );

      setSelectedFiles([]);
      await loadDocuments();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Document upload failed."
      );
    } finally {
      setUploading(false);
    }
  }

  async function deleteDocument(documentId: string) {
    setError("");
    setMessage("");

    try {
      const res = await fetch(
        `${API_URL}/documents/${documentId}`,
        {
          method: "DELETE",
        }
      );

      const data = await res.json();

      if (!res.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Failed to delete document."
        );
      }

      setMessage("Document removed successfully.");
      await loadDocuments();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to delete document."
      );
    }
  }

  async function askQuestion() {
    const trimmedQuestion = question.trim();

    if (!trimmedQuestion) {
      setError("Enter a question.");
      return;
    }

    setAsking(true);
    setMessage("");
    setError("");
    setResponse(null);

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: trimmedQuestion,
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "RAG execution failed."
        );
      }

      setResponse(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "RAG execution failed."
      );
    } finally {
      setAsking(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-7xl px-6 py-10">
        <header className="mb-10">
        <div className="mb-6 flex justify-end">
            <a
              href="/observability"
              className="rounded-lg border border-slate-700 px-4 py-2 text-sm text-slate-300 hover:bg-slate-800"
            >
              View Observability →
            </a>
        </div>
          <div className="mb-3 inline-flex rounded-full border border-slate-700 bg-slate-900 px-3 py-1 text-xs font-medium text-slate-300">
            Agentic AI • Adaptive RAG
          </div>

          <h1 className="text-4xl font-bold tracking-tight md:text-5xl">
            Agentic Adaptive RAG
          </h1>

          <p className="mt-4 max-w-3xl text-slate-400">
            Upload your knowledge base, ask questions, and inspect
            how the agent routes, retrieves, evaluates, and
            generates answers.
          </p>
        </header>

        {(message || error) && (
          <div
            className={`mb-6 rounded-xl border px-4 py-3 ${
              error
                ? "border-red-800 bg-red-950/40 text-red-300"
                : "border-emerald-800 bg-emerald-950/40 text-emerald-300"
            }`}
          >
            {error || message}
          </div>
        )}

        <div className="grid gap-6 lg:grid-cols-3">
          {/* Upload */}
          <section className="rounded-2xl border border-slate-800 bg-slate-900 p-6">
            <h2 className="text-xl font-semibold">
              Knowledge Base
            </h2>

            <p className="mt-2 text-sm text-slate-400">
              Upload PDF, DOCX, TXT, or Markdown documents.
            </p>

            <div className="mt-5 rounded-xl border border-dashed border-slate-700 p-5">
              <input
                type="file"
                multiple
                accept=".pdf,.docx,.txt,.md"
                onChange={handleFileChange}
                className="block w-full text-sm text-slate-400 file:mr-4 file:rounded-lg file:border-0 file:bg-slate-700 file:px-4 file:py-2 file:text-sm file:font-medium file:text-white hover:file:bg-slate-600"
              />

              {selectedFiles.length > 0 && (
                <div className="mt-4 space-y-1">
                  {selectedFiles.map((file) => (
                    <p
                      key={`${file.name}-${file.size}`}
                      className="truncate text-sm text-slate-300"
                    >
                      {file.name}
                    </p>
                  ))}
                </div>
              )}

              <button
                onClick={uploadFiles}
                disabled={uploading || selectedFiles.length === 0}
                className="mt-5 w-full rounded-lg bg-blue-600 px-4 py-2.5 font-medium text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {uploading ? "Processing..." : "Upload & Index"}
              </button>
            </div>
          </section>

          {/* Documents */}
          <section className="rounded-2xl border border-slate-800 bg-slate-900 p-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold">
                Documents
              </h2>

              <button
                onClick={loadDocuments}
                disabled={loadingDocuments}
                className="rounded-lg border border-slate-700 px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-800 disabled:opacity-50"
              >
                Refresh
              </button>
            </div>

            <div className="mt-5 space-y-3">
              {documents.length === 0 ? (
                <p className="text-sm text-slate-500">
                  No documents indexed yet.
                </p>
              ) : (
                documents.map((document) => (
                  <div
                    key={document.document_id}
                    className="rounded-xl border border-slate-800 bg-slate-950 p-4"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <p className="truncate font-medium">
                          {document.file_name}
                        </p>

                        <p className="mt-1 text-xs text-slate-500">
                          {document.file_type}
                        </p>

                        <p className="mt-1 truncate text-xs text-slate-600">
                          {document.document_id}
                        </p>
                      </div>

                      <button
                        onClick={() =>
                          deleteDocument(document.document_id)
                        }
                        className="shrink-0 rounded-lg border border-red-900 px-2.5 py-1.5 text-xs text-red-400 hover:bg-red-950"
                      >
                        Remove
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </section>

          {/* Architecture */}
          <section className="rounded-2xl border border-slate-800 bg-slate-900 p-6">
            <h2 className="text-xl font-semibold">
              Agent Pipeline
            </h2>

            <div className="mt-5 space-y-3 text-sm">
              {[
                "Question",
                "Evidence Retrieval",
                "Adaptive Routing",
                "Document Grading",
                "Web Search Fallback",
                "Answer Generation",
                "Hallucination Check",
                "Answer Evaluation",
              ].map((step, index) => (
                <div
                  key={step}
                  className="flex items-center gap-3 rounded-lg border border-slate-800 bg-slate-950 px-3 py-2.5"
                >
                  <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-slate-800 text-xs text-slate-300">
                    {index + 1}
                  </span>

                  <span className="text-slate-300">
                    {step}
                  </span>
                </div>
              ))}
            </div>
          </section>
        </div>

        {/* Chat */}
        <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900 p-6">
          <h2 className="text-xl font-semibold">
            Ask the Agent
          </h2>

          <p className="mt-2 text-sm text-slate-400">
            The system decides whether local knowledge or web
            search is the best source.
          </p>

          <div className="mt-5 flex flex-col gap-3 md:flex-row">
            <input
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !asking) {
                  askQuestion();
                }
              }}
              placeholder="Ask a question..."
              className="flex-1 rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-slate-100 outline-none placeholder:text-slate-600 focus:border-blue-500"
            />

            <button
              onClick={askQuestion}
              disabled={asking}
              className="rounded-lg bg-blue-600 px-6 py-3 font-medium hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {asking ? "Thinking..." : "Ask Agent"}
            </button>
          </div>
        </section>

        {/* Response */}
        {response && (
          <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900 p-6">
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded-full bg-blue-950 px-3 py-1 text-xs font-medium text-blue-300">
                Route: {response.route}
              </span>

              <span
                className={`rounded-full px-3 py-1 text-xs font-medium ${
                  response.metrics.grounded
                    ? "bg-emerald-950 text-emerald-300"
                    : "bg-red-950 text-red-300"
                }`}
              >
                Grounded: {response.metrics.grounded ? "Yes" : "No"}
              </span>

              <span
                className={`rounded-full px-3 py-1 text-xs font-medium ${
                  response.metrics.answers_question
                    ? "bg-emerald-950 text-emerald-300"
                    : "bg-amber-950 text-amber-300"
                }`}
              >
                Answers Question:{" "}
                {response.metrics.answers_question
                  ? "Yes"
                  : "No"}
              </span>
            </div>

            <div className="mt-6">
              <h3 className="text-sm font-medium uppercase tracking-wide text-slate-500">
                Answer
              </h3>

              <div className="mt-2 whitespace-pre-wrap rounded-xl bg-slate-950 p-5 leading-7 text-slate-200">
                {response.answer}
              </div>
            </div>

            {/* Metrics */}
            <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
              <Metric
                label="Retrieved"
                value={response.metrics.retrieved_documents}
              />

              <Metric
                label="Relevant"
                value={response.metrics.relevant_documents}
              />

              <Metric
                label="Retries"
                value={response.metrics.retry_count}
              />

              <Metric
                label="Latency"
                value={`${response.metrics.latency_seconds.toFixed(
                  2
                )}s`}
              />

              <Metric
                label="Sources"
                value={response.sources.length}
              />
            </div>

            {/* Sources */}
            {response.sources.length > 0 && (
              <div className="mt-6">
                <h3 className="text-sm font-medium uppercase tracking-wide text-slate-500">
                  Sources
                </h3>

                <div className="mt-3 space-y-2">
                  {response.sources.map((source, index) => (
                    <div
                      key={`${source.document_id || source.url}-${index}`}
                      className="rounded-lg border border-slate-800 bg-slate-950 px-4 py-3"
                    >
                      {source.type === "web" && source.url ? (
                        <a
                          href={source.url}
                          target="_blank"
                          rel="noreferrer"
                          className="text-sm text-blue-400 hover:text-blue-300"
                        >
                          {source.title || source.url}
                        </a>
                      ) : (
                        <p className="text-sm text-slate-300">
                          {source.file_name ||
                            source.source ||
                            "Local document"}
                        </p>
                      )}

                      <p className="mt-1 text-xs uppercase text-slate-600">
                        {source.type}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>
        )}

        <footer className="mt-10 border-t border-slate-800 pt-6 text-sm text-slate-600">
          Agentic Adaptive RAG • LangGraph • FastAPI • Next.js • TypeScript
        </footer>
      </div>
    </main>
  );
}

function Metric({
  label,
  value,
}: {
  label: string;
  value: string | number;
}) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950 p-4">
      <p className="text-xs uppercase tracking-wide text-slate-500">
        {label}
      </p>

      <p className="mt-1 text-xl font-semibold text-slate-200">
        {value}
      </p>
    </div>
  );
}