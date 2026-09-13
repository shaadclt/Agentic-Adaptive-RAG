"use client";

import {
  ChangeEvent,
  useEffect,
  useRef,
  useState,
} from "react";

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
  const [submittedQuestion, setSubmittedQuestion] = useState("");
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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

  function handleFileChange(
    event: ChangeEvent<HTMLInputElement>
  ) {
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

      const res = await fetch(
        `${API_URL}/documents/upload`,
        {
          method: "POST",
          body: formData,
        }
      );

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

      if (fileInputRef.current) {
        fileInputRef.current.value = "";
        }
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

  async function deleteDocument(
    documentId: string
  ) {
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
      <div className="mx-auto max-w-7xl px-6 py-8">

        {/* Header */}
        <header className="mb-8">
          <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">

            <div>
              <div className="mb-3 inline-flex rounded-full border border-slate-700 bg-slate-900 px-3 py-1 text-xs font-medium text-slate-300">
                Agentic AI • Adaptive RAG
              </div>

              <h1 className="text-4xl font-bold tracking-tight md:text-5xl">
                Agentic Adaptive RAG
              </h1>

              <p className="mt-3 max-w-3xl text-slate-400">
                Upload your knowledge base, ask questions, and let
                the agent decide whether to use local knowledge or
                web search.
              </p>
            </div>

            <a
              href="/observability"
              className="shrink-0 rounded-lg border border-slate-700 px-4 py-2 text-sm text-slate-300 transition hover:bg-slate-800"
            >
              Observability →
            </a>
          </div>
        </header>

        {/* Notifications */}
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

        <div className="grid gap-6 lg:grid-cols-[320px_minmax(0,1fr)]">

          {/* LEFT SIDEBAR */}
          <aside className="space-y-6">

            {/* Knowledge Base */}
            <section className="rounded-2xl border border-slate-800 bg-slate-900 p-5">
              <h2 className="text-lg font-semibold">
                Knowledge Base
              </h2>

              <p className="mt-2 text-sm leading-6 text-slate-400">
                Upload documents to build your private knowledge
                base.
              </p>

              <div className="mt-5 rounded-xl border border-dashed border-slate-700 p-4">

                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept=".pdf,.docx,.txt,.md"
                  onChange={handleFileChange}
                  className="block w-full text-sm text-slate-400 file:mr-3 file:rounded-lg file:border-0 file:bg-slate-700 file:px-3 file:py-2 file:text-sm file:font-medium file:text-white hover:file:bg-slate-600"
                />

                {selectedFiles.length > 0 && (
                  <div className="mt-4 space-y-2">
                    {selectedFiles.map((file) => (
                      <div
                        key={`${file.name}-${file.size}`}
                        className="rounded-lg bg-slate-950 px-3 py-2"
                      >
                        <p className="truncate text-xs text-slate-300">
                          {file.name}
                        </p>
                      </div>
                    ))}
                  </div>
                )}

                <button
                  onClick={uploadFiles}
                  disabled={
                    uploading ||
                    selectedFiles.length === 0
                  }
                  className="mt-4 w-full rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {uploading
                    ? "Processing..."
                    : "Upload & Index"}
                </button>
              </div>
            </section>

            {/* Documents */}
            <section className="rounded-2xl border border-slate-800 bg-slate-900 p-5">

              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold">
                    Documents
                  </h2>

                  <p className="mt-1 text-xs text-slate-500">
                    {documents.length} indexed document
                    {documents.length !== 1 ? "s" : ""}
                  </p>
                </div>

                <button
                  onClick={loadDocuments}
                  disabled={loadingDocuments}
                  className="rounded-lg border border-slate-700 px-3 py-1.5 text-xs text-slate-300 hover:bg-slate-800 disabled:opacity-50"
                >
                  Refresh
                </button>
              </div>

              <div className="mt-4 space-y-3">

                {documents.length === 0 ? (
                  <div className="rounded-xl border border-slate-800 bg-slate-950 p-4 text-center">
                    <p className="text-sm text-slate-500">
                      No documents indexed yet.
                    </p>

                    <p className="mt-1 text-xs text-slate-600">
                      Upload a document above to get started.
                    </p>
                  </div>
                ) : (
                  documents.map((document) => (
                    <div
                      key={document.document_id}
                      className="rounded-xl border border-slate-800 bg-slate-950 p-3"
                    >
                      <div className="flex items-start justify-between gap-2">

                        <div className="min-w-0">
                          <p className="truncate text-sm font-medium text-slate-200">
                            {document.file_name}
                          </p>

                          <p className="mt-1 text-xs uppercase text-slate-600">
                            {document.file_type}
                          </p>
                        </div>

                        <button
                          onClick={() =>
                            deleteDocument(
                              document.document_id
                            )
                          }
                          className="shrink-0 rounded-md border border-red-900 px-2 py-1 text-xs text-red-400 hover:bg-red-950"
                        >
                          Remove
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </section>
          </aside>

          {/* MAIN CONTENT */}
          <section>

            {/* Chat */}
            <section className="rounded-2xl border border-slate-800 bg-slate-900 p-6">

              <h2 className="text-xl font-semibold">
                Ask the Agent
              </h2>

              <p className="mt-2 text-sm text-slate-400">
                Ask questions about your documents or anything
                requiring external information.
              </p>

              <div className="mt-5 flex flex-col gap-3 sm:flex-row">

                <input
                  value={question}
                  onChange={(event) =>
                    setQuestion(event.target.value)
                  }
                  onKeyDown={(event) => {
                    if (
                      event.key === "Enter" &&
                      !asking
                    ) {
                      askQuestion();
                    }
                  }}
                  placeholder="Ask a question..."
                  className="min-w-0 flex-1 rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-slate-100 outline-none placeholder:text-slate-600 focus:border-blue-500"
                />

                <button
                  onClick={askQuestion}
                  disabled={asking}
                  className="rounded-lg bg-blue-600 px-6 py-3 font-medium text-white hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {asking
                    ? "Thinking..."
                    : "Ask Agent"}
                </button>
              </div>
            </section>

            {/* Empty state */}
            {!response && !asking && (
              <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900 p-10 text-center">

                <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-slate-800 text-2xl">
                  ✦
                </div>

                <h3 className="mt-5 text-lg font-semibold">
                  Ready to answer
                </h3>

                <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-slate-500">
                  Upload documents to provide local context,
                  then ask the agent a question.
                </p>
              </section>
            )}

            {/* Loading */}
            {asking && (
              <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900 p-8 text-center">

                <div className="mx-auto h-8 w-8 animate-spin rounded-full border-2 border-slate-700 border-t-blue-500" />

                <p className="mt-4 text-sm text-slate-400">
                  Agent is retrieving, evaluating, and generating
                  your answer...
                </p>
              </section>
            )}

            {/* Response */}
            {response && (
              <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900 p-6">

                {/* Status */}
                <div className="flex flex-wrap gap-2">

                  <StatusBadge
                    label="Route"
                    value={response.route}
                  />

                  <StatusBadge
                    label="Grounded"
                    value={
                      response.metrics.grounded
                        ? "Yes"
                        : "No"
                    }
                    positive={
                      response.metrics.grounded
                    }
                  />

                  <StatusBadge
                    label="Answers Question"
                    value={
                      response.metrics
                        .answers_question
                        ? "Yes"
                        : "No"
                    }
                    positive={
                      response.metrics
                        .answers_question
                    }
                  />
                </div>

                {/* Answer */}
                <div className="mt-6">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Answer
                  </h3>

                  <div className="mt-2 whitespace-pre-wrap rounded-xl border border-slate-800 bg-slate-950 p-5 leading-7 text-slate-200">
                    {response.answer}
                  </div>
                </div>

                {/* Metrics */}
                <div className="mt-6">

                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Run Metrics
                  </h3>

                  <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">

                    <Metric
                      label="Retrieved"
                      value={
                        response.metrics
                          .retrieved_documents
                      }
                    />

                    <Metric
                      label="Relevant"
                      value={
                        response.metrics
                          .relevant_documents
                      }
                    />

                    <Metric
                      label="Retries"
                      value={
                        response.metrics
                          .retry_count
                      }
                    />

                    <Metric
                      label="Latency"
                      value={`${response.metrics.latency_seconds.toFixed(
                        2
                      )}s`}
                    />

                    <Metric
                      label="Sources"
                      value={
                        response.sources.length
                      }
                    />
                  </div>
                </div>

                {/* Sources */}
                {response.sources.length > 0 && (
                  <div className="mt-6">

                    <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                      Sources
                    </h3>

                    <div className="mt-3 space-y-2">

                      {response.sources.map(
                        (source, index) => (
                          <div
                            key={`${source.document_id || source.url}-${index}`}
                            className="rounded-xl border border-slate-800 bg-slate-950 p-4"
                          >

                            {source.type === "web" &&
                            source.url ? (
                              <a
                                href={source.url}
                                target="_blank"
                                rel="noreferrer"
                                className="text-sm text-blue-400 hover:text-blue-300"
                              >
                                {source.title ||
                                  source.url}
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
                        )
                      )}
                    </div>
                  </div>
                )}
              </section>
            )}
          </section>
        </div>

        <footer className="mt-10 border-t border-slate-800 pt-6 text-center text-xs text-slate-600">
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

function StatusBadge({
  label,
  value,
  positive,
}: {
  label: string;
  value: string;
  positive?: boolean;
}) {
  const isPositive = positive ?? true;

  return (
    <span
      className={`rounded-full px-3 py-1 text-xs font-medium ${
        isPositive
          ? "bg-emerald-950 text-emerald-300"
          : "bg-red-950 text-red-300"
      }`}
    >
      {label}: {value}
    </span>
  );
}