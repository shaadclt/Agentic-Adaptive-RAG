"use client";

import {
  ChangeEvent,
  useEffect,
  useRef,
  useState,
} from "react";

import Link from "next/link";


/* ============================================================
   TYPES
============================================================ */

type DocumentItem = {
  document_id: string;
  file_name: string;
  file_type?: string;
  source?: string;
  chunk_count?: number;
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
  status: string;

  answer: string;

  route: string;

  sources: Source[];

  metrics?: Metrics | null;

  thread_id: string;

  approval_required: boolean;

  approval_type: string;

  approval_title: string;

  approval_message: string;

  security_status?: string;

  security_reason?: string;

  security_event?: string;

  security_redactions?: number;
};


/* ============================================================
   CONFIG
============================================================ */

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ||
  "http://127.0.0.1:8000";


/* ============================================================
   COMPONENT
============================================================ */

export default function Home() {
  const [documents, setDocuments] =
    useState<DocumentItem[]>([]);

  const [selectedFiles, setSelectedFiles] =
    useState<File[]>([]);

  const [question, setQuestion] =
    useState("");

  const [submittedQuestion, setSubmittedQuestion] =
    useState("");

  const [response, setResponse] =
    useState<ChatResponse | null>(null);

  const [loadingDocuments, setLoadingDocuments] =
    useState(false);

  const [uploading, setUploading] =
    useState(false);

  const [asking, setAsking] =
    useState(false);

  const [deletingId, setDeletingId] =
    useState<string | null>(null);

  const [error, setError] =
    useState("");

  const [approvalLoading, setApprovalLoading] =
    useState(false);

  const fileInputRef =
    useRef<HTMLInputElement>(null);


  /* ==========================================================
     LOAD DOCUMENTS
  ========================================================== */

  async function loadDocuments() {
    setLoadingDocuments(true);

    try {
      const res = await fetch(
        `${API_BASE}/documents`,
        {
          cache: "no-store",
        }
      );

      if (!res.ok) {
        throw new Error(
          "Failed to load documents."
        );
      }

      const data = await res.json();

      setDocuments(
        Array.isArray(data.documents)
          ? data.documents
          : []
      );

    } catch (err) {
      console.error(err);

      setError(
        "Unable to load the knowledge base."
      );

    } finally {
      setLoadingDocuments(false);
    }
  }


  /* ==========================================================
     INITIAL LOAD
  ========================================================== */

  useEffect(() => {
    loadDocuments();
  }, []);


  /* ==========================================================
     FILE SELECTION
  ========================================================== */

  function handleFileChange(
    event: ChangeEvent<HTMLInputElement>
  ) {
    const files = Array.from(
      event.target.files || []
    );

    setSelectedFiles(files);

    setError("");
  }


  /* ==========================================================
     UPLOAD
  ========================================================== */

  async function uploadFiles() {
    if (!selectedFiles.length) {
      return;
    }

    setUploading(true);

    setError("");

    try {
      const formData = new FormData();

      selectedFiles.forEach(
        (file) => {
          formData.append(
            "files",
            file
          );
        }
      );

      const res = await fetch(
        `${API_BASE}/documents/upload`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await res.json();

      if (!res.ok) {
        throw new Error(
          data.detail ||
          "Upload failed."
        );
      }

      const failed =
        data.results?.filter(
          (item: any) =>
            item.status === "failed"
        ) || [];

      if (failed.length > 0) {
        setError(
          failed
            .map(
              (item: any) =>
                `${item.file_name}: ${
                  item.message || "Upload failed"
                }`
            )
            .join("\n")
        );
      }

      /*
       * Clear selected files after successful
       * upload processing.
       */
      setSelectedFiles([]);

      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }

      await loadDocuments();

    } catch (err) {
      console.error(err);

      setError(
        err instanceof Error
          ? err.message
          : "Upload failed."
      );

    } finally {
      setUploading(false);
    }
  }


  /* ==========================================================
     DELETE DOCUMENT
  ========================================================== */

  async function deleteDocument(
    documentId: string
  ) {
    const confirmed =
      window.confirm(
        "Delete this document from the knowledge base?"
      );

    if (!confirmed) {
      return;
    }

    setDeletingId(documentId);

    setError("");

    try {
      const res = await fetch(
        `${API_BASE}/documents/${encodeURIComponent(
          documentId
        )}`,
        {
          method: "DELETE",
        }
      );

      const data = await res.json();

      if (!res.ok) {
        throw new Error(
          data.detail ||
          "Failed to delete document."
        );
      }

      await loadDocuments();

    } catch (err) {
      console.error(err);

      setError(
        err instanceof Error
          ? err.message
          : "Failed to delete document."
      );

    } finally {
      setDeletingId(null);
    }
  }


  /* ==========================================================
     ASK AGENT
  ========================================================== */

  async function askAgent() {
    const trimmedQuestion =
      question.trim();

    if (!trimmedQuestion) {
      setError(
        "Please enter a question."
      );

      return;
    }

    setAsking(true);

    setError("");

    setResponse(null);

    try {
      const res = await fetch(
        `${API_BASE}/chat`,
        {
          method: "POST",

          headers: {
            "Content-Type":
              "application/json",
          },

          body: JSON.stringify({
            question:
              trimmedQuestion,
          }),
        }
      );

      const data: ChatResponse =
        await res.json();

      if (!res.ok) {
        throw new Error(
          typeof data === "object" &&
          data &&
          "answer" in data
            ? data.answer
            : "Request failed."
        );
      }

      /*
       * Preserve the question that was actually
       * submitted while clearing the input field.
       */
      setSubmittedQuestion(
        trimmedQuestion
      );

      setQuestion("");

      setResponse(data);

    } catch (err) {
      console.error(err);

      setError(
        err instanceof Error
          ? err.message
          : "Unable to process the question."
      );

    } finally {
      setAsking(false);
    }
  }


  /* ==========================================================
     HITL DECISION
  ========================================================== */

  async function handleApproval(
    approved: boolean
  ) {
    if (!response?.thread_id) {
      return;
    }

    setApprovalLoading(true);

    setError("");

    try {
      const res = await fetch(
        `${API_BASE}/chat/${encodeURIComponent(
          response.thread_id
        )}/web-search`,
        {
          method: "POST",

          headers: {
            "Content-Type":
              "application/json",
          },

          body: JSON.stringify({
            approved,
          }),
        }
      );

      const data: ChatResponse =
        await res.json();

      if (!res.ok) {
        throw new Error(
          data.answer ||
          "Unable to process approval."
        );
      }

      setResponse(data);

    } catch (err) {
      console.error(err);

      setError(
        err instanceof Error
          ? err.message
          : "Unable to process approval."
      );

    } finally {
      setApprovalLoading(false);
    }
  }


  /* ==========================================================
     KEYBOARD HANDLER
  ========================================================== */

  function handleQuestionKeyDown(
    event: React.KeyboardEvent<HTMLTextAreaElement>
  ) {
    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {
      event.preventDefault();

      if (!asking) {
        askAgent();
      }
    }
  }


  /* ==========================================================
     RENDER
  ========================================================== */

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">

      <div className="flex min-h-screen">

        {/* ====================================================
            SIDEBAR
        ==================================================== */}

        <aside className="hidden w-80 shrink-0 border-r border-slate-800 bg-slate-950 lg:block">

          <div className="sticky top-0 flex h-screen flex-col">

            {/* Brand */}

            <div className="border-b border-slate-800 px-6 py-6">

              <div className="flex items-center gap-3">

                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-800 text-sm font-bold text-white">
                  AR
                </div>

                <div>
                  <h1 className="font-semibold">
                    Agentic RAG
                  </h1>

                  <p className="text-xs text-slate-500">
                    Adaptive Knowledge Assistant
                  </p>
                </div>

              </div>

            </div>


            {/* Knowledge Base */}

            <div className="flex-1 overflow-y-auto px-4 py-5">

              <div className="mb-3 flex items-center justify-between px-2">

                <div>

                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Knowledge Base
                  </p>

                  <p className="mt-1 text-xs text-slate-600">
                    {documents.length}{" "}
                    {documents.length === 1
                      ? "document"
                      : "documents"}
                  </p>

                </div>

                <button
                  onClick={loadDocuments}
                  disabled={loadingDocuments}
                  className="rounded-lg px-2 py-1 text-xs text-slate-500 transition hover:bg-slate-900 hover:text-slate-300 disabled:opacity-50"
                  title="Refresh"
                >
                  ↻
                </button>

              </div>


              {/* Upload */}

              <div className="mb-5 rounded-xl border border-dashed border-slate-800 bg-slate-900/30 p-4">

                <label className="block cursor-pointer">

                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    accept=".pdf,.docx,.txt,.md"
                    onChange={
                      handleFileChange
                    }
                    className="hidden"
                  />

                  <div className="rounded-lg border border-slate-800 bg-slate-950 px-4 py-4 text-center transition hover:border-slate-700">

                    <div className="text-xl">
                      +
                    </div>

                    <p className="mt-1 text-sm font-medium text-slate-300">
                      Add documents
                    </p>

                    <p className="mt-1 text-xs text-slate-600">
                      PDF, DOCX, TXT, MD
                    </p>

                  </div>

                </label>


                {selectedFiles.length > 0 && (
                  <div className="mt-3">

                    <p className="text-xs text-slate-500">
                      {selectedFiles.length} selected
                    </p>

                    <div className="mt-2 space-y-1">

                      {selectedFiles.map(
                        (file) => (
                          <div
                            key={`${file.name}-${file.size}`}
                            className="truncate rounded-md bg-slate-900 px-2 py-1 text-xs text-slate-400"
                          >
                            {file.name}
                          </div>
                        )
                      )}

                    </div>


                    <button
                      onClick={uploadFiles}
                      disabled={uploading}
                      className="mt-3 w-full rounded-lg bg-slate-100 px-3 py-2 text-sm font-semibold text-slate-950 transition hover:bg-white disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {uploading
                        ? "Uploading..."
                        : "Upload"}
                    </button>

                  </div>
                )}

              </div>


              {/* Documents */}

              <div className="space-y-2">

                {loadingDocuments ? (

                  <div className="rounded-xl border border-slate-900 px-4 py-5 text-center text-sm text-slate-600">
                    Loading documents...
                  </div>

                ) : documents.length === 0 ? (

                  <div className="rounded-xl border border-slate-900 px-4 py-6 text-center">

                    <p className="text-sm text-slate-500">
                      Knowledge base is empty.
                    </p>

                    <p className="mt-1 text-xs text-slate-700">
                      Upload a document to get started.
                    </p>

                  </div>

                ) : (

                  documents.map(
                    (document) => (

                      <div
                        key={
                          document.document_id
                        }
                        className="group rounded-xl border border-slate-900 bg-slate-950 p-3 transition hover:border-slate-800"
                      >

                        <div className="flex items-start gap-3">

                          <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-slate-900 text-xs font-semibold text-slate-500">
                            {(
                              document.file_type ||
                              "DOC"
                            )
                              .replace(".", "")
                              .slice(0, 3)
                              .toUpperCase()}
                          </div>

                          <div className="min-w-0 flex-1">

                            <p
                              className="truncate text-sm font-medium text-slate-300"
                              title={
                                document.file_name
                              }
                            >
                              {
                                document.file_name
                              }
                            </p>

                            <p className="mt-1 text-xs text-slate-600">
                              {document.chunk_count ??
                                0}{" "}
                              chunks
                            </p>

                          </div>

                          <button
                            onClick={() =>
                              deleteDocument(
                                document.document_id
                              )
                            }
                            disabled={
                              deletingId ===
                              document.document_id
                            }
                            className="opacity-0 transition group-hover:opacity-100 rounded-md px-2 py-1 text-xs text-slate-600 hover:bg-red-950/30 hover:text-red-400 disabled:opacity-50"
                            title="Delete document"
                          >
                            {deletingId ===
                            document.document_id
                              ? "..."
                              : "×"}
                          </button>

                        </div>

                      </div>

                    )
                  )

                )}

              </div>

            </div>


            {/* Sidebar bottom */}

            <div className="border-t border-slate-800 p-4">

              <Link
                href="/observability"
                className="flex items-center justify-between rounded-xl px-3 py-3 text-sm text-slate-500 transition hover:bg-slate-900 hover:text-slate-300"
              >
                <span>
                  Observability
                </span>

                <span>
                  →
                </span>

              </Link>

            </div>

          </div>

        </aside>


        {/* ====================================================
            MAIN
        ==================================================== */}

        <section className="min-w-0 flex-1">

          <div className="mx-auto max-w-5xl px-5 py-8 sm:px-8 lg:px-12">


            {/* Mobile header */}

            <div className="mb-8 flex items-center justify-between lg:hidden">

              <div>

                <h1 className="font-semibold">
                  Agentic RAG
                </h1>

                <p className="text-xs text-slate-600">
                  Adaptive Knowledge Assistant
                </p>

              </div>

              <Link
                href="/observability"
                className="rounded-lg border border-slate-800 px-3 py-2 text-xs text-slate-400"
              >
                Observability
              </Link>

            </div>


            {/* Hero */}

            <div className="mb-10">

              <p className="text-sm font-medium text-slate-500">
                Ask the Agent
              </p>

              <h2 className="mt-2 text-3xl font-semibold tracking-tight text-white sm:text-4xl">
                Ask questions across your knowledge base.
              </h2>

              <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-500">
                The agent dynamically chooses local
                retrieval or external web search,
                evaluates evidence, and validates
                generated answers.
              </p>

            </div>


            {/* Error */}

            {error && (
              <div className="mb-6 whitespace-pre-line rounded-xl border border-red-900/50 bg-red-950/20 px-4 py-3 text-sm text-red-300">
                {error}
              </div>
            )}


            {/* =================================================
                ASK BOX
            ================================================= */}

            <div className="rounded-2xl border border-slate-800 bg-slate-900/30 p-4 shadow-2xl shadow-black/10 sm:p-5">

              <textarea
                value={question}
                onChange={(event) =>
                  setQuestion(
                    event.target.value
                  )
                }
                onKeyDown={
                  handleQuestionKeyDown
                }
                disabled={asking}
                placeholder="Ask something about your documents..."
                rows={4}
                className="w-full resize-none bg-transparent text-base leading-7 text-slate-200 outline-none placeholder:text-slate-700 disabled:opacity-50"
              />

              <div className="mt-4 flex items-center justify-between border-t border-slate-800 pt-4">

                <p className="text-xs text-slate-600">
                  Enter to submit · Shift + Enter for new line
                </p>

                <button
                  onClick={askAgent}
                  disabled={
                    asking ||
                    !question.trim()
                  }
                  className="rounded-xl bg-slate-100 px-5 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-white disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {asking
                    ? "Thinking..."
                    : "Ask Agent"}
                </button>

              </div>

            </div>


            {/* =================================================
                EMPTY STATE
            ================================================= */}

            {!response &&
              !asking && (
                <div className="py-20 text-center">

                  <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl border border-slate-800 bg-slate-900 text-lg text-slate-600">
                    ?
                  </div>

                  <p className="mt-4 text-sm text-slate-500">
                    Your answer will appear here.
                  </p>

                </div>
              )}


            {/* =================================================
                LOADING
            ================================================= */}

            {asking && (
              <div className="mt-8 rounded-2xl border border-slate-800 bg-slate-900/20 p-6">

                <div className="flex items-center gap-3">

                  <div className="h-2 w-2 animate-pulse rounded-full bg-slate-400" />

                  <p className="text-sm text-slate-500">
                    Agent is retrieving evidence and evaluating the request...
                  </p>

                </div>

              </div>
            )}


            {/* =================================================
                HITL APPROVAL
            ================================================= */}

            {response?.approval_required && (
              <div className="mt-8 rounded-2xl border border-amber-900/60 bg-amber-950/20 p-6">

                <div className="flex items-start gap-4">

                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-amber-950 text-amber-400">
                    !
                  </div>

                  <div className="min-w-0">

                    <p className="text-sm font-semibold text-amber-300">
                      {response.approval_title ||
                        "Web search approval required"}
                    </p>

                    <p className="mt-2 text-sm leading-6 text-slate-400">
                      {response.approval_message ||
                        "The agent needs permission to search the web."}
                    </p>

                    <div className="mt-5 flex flex-wrap gap-3">

                      <button
                        onClick={() =>
                          handleApproval(
                            true
                          )
                        }
                        disabled={
                          approvalLoading
                        }
                        className="rounded-xl bg-slate-100 px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-white disabled:opacity-50"
                      >
                        {approvalLoading
                          ? "Processing..."
                          : "Allow Web Search"}
                      </button>

                      <button
                        onClick={() =>
                          handleApproval(
                            false
                          )
                        }
                        disabled={
                          approvalLoading
                        }
                        className="rounded-xl border border-slate-700 px-4 py-2.5 text-sm font-semibold text-slate-300 transition hover:bg-slate-900 disabled:opacity-50"
                      >
                        Don't Search
                      </button>

                    </div>

                  </div>

                </div>

              </div>
            )}


            {/* =================================================
                ANSWER
            ================================================= */}

            {response &&
              !response.approval_required && (
                <div className="mt-8 space-y-5">


                  {/* Question */}

                  {submittedQuestion && (
                    <div>

                      <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-600">
                        Question
                      </h3>

                      <div className="mt-2 rounded-xl border border-slate-800 bg-slate-950 px-5 py-4 text-base font-medium leading-6 text-slate-300">
                        {submittedQuestion}
                      </div>

                    </div>
                  )}


                  {/* Answer */}

                  <div>

                    <div className="flex items-center justify-between">

                      <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-600">
                        Answer
                      </h3>

                      {response.route && (
                        <span className="rounded-full border border-slate-800 px-2.5 py-1 text-xs text-slate-500">
                          {response.route}
                        </span>
                      )}

                    </div>

                    <div className="mt-2 whitespace-pre-wrap rounded-xl border border-slate-800 bg-slate-900/40 p-5 text-[15px] leading-7 text-slate-300">
                      {response.answer}
                    </div>

                  </div>


                  {/* Security */}

                  {response.security_status &&
                    response.security_status !==
                      "passed" && (
                      <div className="rounded-xl border border-amber-900/60 bg-amber-950/20 px-4 py-4">

                        <p className="text-xs font-semibold uppercase tracking-wider text-amber-400">
                          Security Notice
                        </p>

                        <p className="mt-2 text-sm leading-6 text-slate-400">
                          {response.security_reason ||
                            "The response was processed by the security layer."}
                        </p>

                        {response.security_redactions &&
                          response.security_redactions >
                            0 && (
                            <p className="mt-2 text-xs text-amber-500">
                              {
                                response.security_redactions
                              }{" "}
                              sensitive value
                              {response.security_redactions ===
                              1
                                ? ""
                                : "s"}{" "}
                              redacted.
                            </p>
                          )}

                      </div>
                    )}


                  {/* Metrics */}

                  {response.metrics && (
                    <div>

                      <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-600">
                        Run Metrics
                      </h3>

                      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">

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
                          label="Grounded"
                          value={
                            response.metrics
                              .grounded
                              ? "Yes"
                              : "No"
                          }
                        />

                        <Metric
                          label="Quality"
                          value={
                            response.metrics
                              .answers_question
                              ? "Yes"
                              : "No"
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

                      </div>

                    </div>
                  )}


                  {/* Sources */}

                  {response.sources &&
                    response.sources.length >
                      0 && (
                      <div>

                        <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-600">
                          Sources
                        </h3>

                        <div className="space-y-2">

                          {response.sources.map(
                            (
                              source,
                              index
                            ) => (

                              <div
                                key={`${source.document_id || source.url || source.file_name}-${index}`}
                                className="rounded-xl border border-slate-800 bg-slate-900/20 px-4 py-3"
                              >

                                <div className="flex items-center gap-3">

                                  <span className="rounded-md bg-slate-900 px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                                    {
                                      source.type
                                    }
                                  </span>

                                  <div className="min-w-0">

                                    {source.type ===
                                      "web" &&
                                    source.url ? (

                                      <a
                                        href={
                                          source.url
                                        }
                                        target="_blank"
                                        rel="noreferrer"
                                        className="truncate text-sm text-slate-400 hover:text-slate-200"
                                      >
                                        {source.title ||
                                          source.url}
                                      </a>

                                    ) : (

                                      <p className="truncate text-sm text-slate-400">
                                        {source.file_name ||
                                          source.source ||
                                          "Local document"}
                                      </p>

                                    )}

                                  </div>

                                </div>

                              </div>

                            )
                          )}

                        </div>

                      </div>
                    )}

                </div>
              )}

          </div>

        </section>

      </div>

    </main>
  );
}


/* ============================================================
   METRIC COMPONENT
============================================================ */

function Metric({
  label,
  value,
}: {
  label: string;
  value: string | number;
}) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/20 px-3 py-3">

      <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-600">
        {label}
      </p>

      <p className="mt-1 text-sm font-semibold text-slate-300">
        {value}
      </p>

    </div>
  );
}