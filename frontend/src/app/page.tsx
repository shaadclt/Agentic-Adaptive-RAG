"use client";

import {
  ChangeEvent,
  FormEvent,
  useEffect,
  useRef,
  useState,
} from "react";
import Link from "next/link";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  "http://localhost:8000";

type DocumentItem = {
  document_id: string;
  file_name: string;
  file_type: string;
  source: string;
  chunk_count: number;
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
};

type UploadResult = {
  file_name: string;
  status: string;
  document_id?: string;
  chunks_added?: number;
  message?: string;
};

export default function HomePage() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);

  const [question, setQuestion] = useState("");
  const [submittedQuestion, setSubmittedQuestion] = useState("");

  const [response, setResponse] =
    useState<ChatResponse | null>(null);

  const [loadingDocuments, setLoadingDocuments] =
    useState(true);

  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [deletingId, setDeletingId] =
    useState<string | null>(null);

  const [error, setError] = useState("");
  const [uploadMessage, setUploadMessage] =
    useState("");

  const [approvalRequired, setApprovalRequired] =
    useState(false);

  const [approvalData, setApprovalData] =
    useState<ChatResponse | null>(null);

  const [approving, setApproving] = useState(false);

  const fileInputRef =
    useRef<HTMLInputElement>(null);

  /*
   * Load existing documents from the backend.
   *
   * Supports both:
   *
   * [
   *   {...}
   * ]
   *
   * and:
   *
   * {
   *   "documents": [...]
   * }
   */
  async function loadDocuments() {
    try {
      setLoadingDocuments(true);
      setError("");

      const result = await fetch(
        `${API_BASE_URL}/documents`,
        {
          cache: "no-store",
        },
      );

      if (!result.ok) {
        throw new Error(
          "Failed to load documents.",
        );
      }

      const data = await result.json();

      const documentList = Array.isArray(data)
        ? data
        : Array.isArray(data.documents)
          ? data.documents
          : [];

      setDocuments(documentList);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to load documents.",
      );
    } finally {
      setLoadingDocuments(false);
    }
  }

  /*
   * Load the existing knowledge base
   * when the page opens.
   */
  useEffect(() => {
    loadDocuments();
  }, []);

  function handleFileChange(
    event: ChangeEvent<HTMLInputElement>,
  ) {
    const files = Array.from(
      event.target.files || [],
    );

    setSelectedFiles(files);
    setUploadMessage("");
    setError("");
  }

  async function handleUpload() {
    if (selectedFiles.length === 0) {
      setError(
        "Please select at least one document.",
      );
      return;
    }

    try {
      setUploading(true);
      setError("");
      setUploadMessage("");

      const formData = new FormData();

      selectedFiles.forEach((file) => {
        formData.append("files", file);
      });

      const result = await fetch(
        `${API_BASE_URL}/documents/upload`,
        {
          method: "POST",
          body: formData,
        },
      );

      const data = await result.json();

      if (!result.ok) {
        throw new Error(
          data.detail ||
            "Document upload failed.",
        );
      }

      const results: UploadResult[] =
        data.results || [];

      const successful = results.filter(
        (item) =>
          item.status === "success" ||
          item.status === "added",
      );

      const duplicates = results.filter(
        (item) =>
          item.status === "duplicate",
      );

      const failed = results.filter(
        (item) =>
          item.status === "error" ||
          item.status === "failed",
      );

      let message = "";

      if (successful.length > 0) {
        message += `${successful.length} document${
          successful.length === 1
            ? ""
            : "s"
        } uploaded successfully.`;
      }

      if (duplicates.length > 0) {
        message += ` ${duplicates.length} duplicate${
          duplicates.length === 1
            ? ""
            : "s"
        } skipped.`;
      }

      if (failed.length > 0) {
        message += ` ${failed.length} upload${
          failed.length === 1
            ? ""
            : "s"
        } failed.`;
      }

      setUploadMessage(
        message || "Upload completed.",
      );

      /*
       * Clear selected files after successful
       * upload.
       */
      setSelectedFiles([]);

      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }

      /*
       * Reload documents so the new files
       * immediately appear in the sidebar.
       */
      await loadDocuments();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Document upload failed.",
      );
    } finally {
      setUploading(false);
    }
  }

  async function handleDelete(
    documentId: string,
  ) {
    const documentToDelete =
      documents.find(
        (document) =>
          document.document_id ===
          documentId,
      );

    const confirmed = window.confirm(
      `Delete "${
        documentToDelete?.file_name ||
        "this document"
      }" from the knowledge base?`,
    );

    if (!confirmed) {
      return;
    }

    try {
      setDeletingId(documentId);
      setError("");

      const result = await fetch(
        `${API_BASE_URL}/documents/${documentId}`,
        {
          method: "DELETE",
        },
      );

      const data = await result.json();

      if (!result.ok) {
        throw new Error(
          data.detail ||
            "Failed to delete document.",
        );
      }

      setDocuments((current) =>
        current.filter(
          (document) =>
            document.document_id !==
            documentId,
        ),
      );

      setUploadMessage(
        data.message ||
          "Document deleted successfully.",
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to delete document.",
      );
    } finally {
      setDeletingId(null);
    }
  }

  async function submitQuestion(
    event?: FormEvent<HTMLFormElement>,
  ) {
    event?.preventDefault();

    const trimmedQuestion =
      question.trim();

    if (!trimmedQuestion) {
      setError(
        "Please enter a question.",
      );
      return;
    }

    try {
      setAsking(true);
      setError("");
      setResponse(null);
      setSubmittedQuestion("");

      const result = await fetch(
        `${API_BASE_URL}/chat`,
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
        },
      );

      const data: ChatResponse =
        await result.json();

      if (!result.ok) {
        throw new Error(
          (
            data as unknown as {
              detail?: string;
            }
          ).detail ||
            "Failed to get an answer.",
        );
      }

      /*
       * The graph paused and is waiting
       * for human approval before web search.
       */
      if (
        data.status ===
          "approval_required" ||
        data.approval_required
      ) {
        setApprovalData(data);
        setApprovalRequired(true);

        /*
         * Store the question so it remains
         * visible while the approval modal
         * is displayed.
         */
        setSubmittedQuestion(
          trimmedQuestion,
        );

        setQuestion("");

        return;
      }

      /*
       * Normal completed response.
       */
      setResponse(data);
      setSubmittedQuestion(
        trimmedQuestion,
      );

      /*
       * Clear question input after answer.
       */
      setQuestion("");
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to get an answer.",
      );
    } finally {
      setAsking(false);
    }
  }

  async function handleWebSearchApproval(
    approved: boolean,
  ) {
    if (!approvalData?.thread_id) {
      setError(
        "The approval session is no longer available. Please submit the question again.",
      );

      setApprovalRequired(false);
      setApprovalData(null);

      return;
    }

    try {
      setApproving(true);
      setError("");

      const result = await fetch(
        `${API_BASE_URL}/chat/${approvalData.thread_id}/web-search`,
        {
          method: "POST",
          headers: {
            "Content-Type":
              "application/json",
          },
          body: JSON.stringify({
            approved,
          }),
        },
      );

      const data: ChatResponse =
        await result.json();

      if (!result.ok) {
        throw new Error(
          (
            data as unknown as {
              detail?: string;
            }
          ).detail ||
            "Failed to process web-search approval.",
        );
      }

      /*
       * Close the approval modal.
       */
      setApprovalRequired(false);
      setApprovalData(null);

      /*
       * Display the final response.
       */
      setResponse(data);

      /*
       * Make sure the question input
       * remains empty after completion.
       */
      setQuestion("");
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to process web-search approval.",
      );
    } finally {
      setApproving(false);
    }
  }

  function getFileExtension(
    fileName: string,
  ) {
    const parts = fileName.split(".");

    return parts.length > 1
      ? parts[
          parts.length - 1
        ].toUpperCase()
      : "FILE";
  }

  function getRouteLabel(
    route: string,
  ) {
    switch (route) {
      case "local":
        return "Local Knowledge Base";

      case "web":
        return "Web Search";

      case "web_rejected":
        return "Web Search Rejected";

      default:
        return route || "Unknown";
    }
  }

  function getRouteClass(
    route: string,
  ) {
    switch (route) {
      case "local":
        return "border-emerald-800 bg-emerald-950/40 text-emerald-300";

      case "web":
        return "border-blue-800 bg-blue-950/40 text-blue-300";

      case "web_rejected":
        return "border-amber-800 bg-amber-950/40 text-amber-300";

      default:
        return "border-slate-700 bg-slate-900 text-slate-300";
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto flex min-h-screen max-w-[1500px]">

        {/* =====================================================
            SIDEBAR
        ====================================================== */}

        <aside className="w-[340px] shrink-0 border-r border-slate-800 bg-slate-950">
          <div className="sticky top-0 flex max-h-screen flex-col">

            {/* Logo */}
            <div className="border-b border-slate-800 px-6 py-6">
              <div className="flex items-center gap-3">

                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white text-sm font-bold text-slate-950">
                  AR
                </div>

                <div>
                  <h1 className="font-semibold tracking-tight">
                    Agentic RAG
                  </h1>

                  <p className="text-xs text-slate-500">
                    Adaptive Knowledge Assistant
                  </p>
                </div>

              </div>
            </div>

            {/* Knowledge Base */}
            <div className="flex-1 overflow-y-auto px-5 py-6">

              <div className="mb-4">
                <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
                  Knowledge Base
                </p>
              </div>

              {/* Upload */}
              <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4">

                <p className="text-sm font-medium text-slate-200">
                  Add documents
                </p>

                <p className="mt-1 text-xs leading-5 text-slate-500">
                  Upload PDF, DOCX, TXT or Markdown files.
                </p>

                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept=".pdf,.docx,.txt,.md"
                  onChange={
                    handleFileChange
                  }
                  className="mt-4 block w-full cursor-pointer text-xs text-slate-400 file:mr-3 file:cursor-pointer file:rounded-lg file:border-0 file:bg-slate-800 file:px-3 file:py-2 file:text-xs file:font-medium file:text-slate-200 hover:file:bg-slate-700"
                />

                {/* Selected files */}
                {selectedFiles.length >
                  0 && (
                  <div className="mt-3 space-y-1">

                    {selectedFiles.map(
                      (file) => (
                        <div
                          key={`${file.name}-${file.size}`}
                          className="truncate rounded-lg bg-slate-950 px-3 py-2 text-xs text-slate-400"
                        >
                          {file.name}
                        </div>
                      ),
                    )}

                  </div>
                )}

                <button
                  onClick={
                    handleUpload
                  }
                  disabled={
                    uploading ||
                    selectedFiles.length ===
                      0
                  }
                  className="mt-4 w-full rounded-xl bg-white px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-slate-200 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {uploading
                    ? "Uploading..."
                    : "Upload documents"}
                </button>

                {uploadMessage && (
                  <p className="mt-3 text-xs leading-5 text-emerald-400">
                    {uploadMessage}
                  </p>
                )}

              </div>

              {/* Documents */}
              <div className="mt-7">

                <div className="mb-3 flex items-center justify-between">

                  <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
                    Documents
                  </p>

                  <span className="rounded-full bg-slate-900 px-2 py-1 text-[11px] text-slate-500">
                    {documents.length}
                  </span>

                </div>

                {loadingDocuments ? (
                  <div className="rounded-xl border border-slate-800 bg-slate-900/40 px-4 py-5 text-xs text-slate-500">
                    Loading documents...
                  </div>
                ) : documents.length ===
                  0 ? (
                  <div className="rounded-xl border border-dashed border-slate-800 px-4 py-6 text-center">

                    <p className="text-sm text-slate-500">
                      No documents yet.
                    </p>

                    <p className="mt-1 text-xs text-slate-600">
                      Upload files to build your knowledge base.
                    </p>

                  </div>
                ) : (
                  <div className="space-y-2">

                    {documents.map(
                      (document) => (
                        <div
                          key={
                            document.document_id
                          }
                          className="group rounded-xl border border-slate-800 bg-slate-900/40 p-3 transition hover:border-slate-700"
                        >

                          <div className="flex gap-3">

                            {/* File icon */}
                            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-slate-800 text-[10px] font-bold text-slate-400">
                              {getFileExtension(
                                document.file_name,
                              )}
                            </div>

                            {/* File details */}
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

                              <p className="mt-1 text-[11px] text-slate-600">
                                {
                                  document.chunk_count
                                }{" "}
                                chunks
                              </p>

                            </div>

                            {/* Delete */}
                            <button
                              onClick={() =>
                                handleDelete(
                                  document.document_id,
                                )
                              }
                              disabled={
                                deletingId ===
                                document.document_id
                              }
                              className="self-start rounded-lg px-2 py-1 text-xs text-slate-600 opacity-0 transition hover:bg-red-950/40 hover:text-red-400 group-hover:opacity-100 disabled:opacity-40"
                              title="Delete document"
                            >
                              {deletingId ===
                              document.document_id
                                ? "..."
                                : "×"}
                            </button>

                          </div>

                        </div>
                      ),
                    )}

                  </div>
                )}

              </div>

            </div>

            {/* Sidebar footer */}
            <div className="border-t border-slate-800 p-5">

              <Link
                href="/observability"
                className="flex items-center justify-between rounded-xl border border-slate-800 bg-slate-900/40 px-4 py-3 text-sm text-slate-400 transition hover:border-slate-700 hover:text-slate-200"
              >
                <span>
                  Observability
                </span>

                <span className="text-slate-600">
                  →
                </span>
              </Link>

            </div>

          </div>
        </aside>

        {/* =====================================================
            MAIN CONTENT
        ====================================================== */}

        <section className="min-w-0 flex-1">

          <div className="mx-auto max-w-5xl px-8 py-10 lg:px-12">

            {/* Header */}
            <div>

              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
                Agentic AI
              </p>

              <h2 className="mt-2 text-3xl font-semibold tracking-tight text-white">
                Ask the Agent
              </h2>

              <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
                Ask questions against your uploaded knowledge base.
                The agent decides whether local retrieval is enough
                or whether external web search is required.
              </p>

            </div>

            {/* Error */}
            {error && (
              <div className="mt-8 rounded-xl border border-red-900/70 bg-red-950/30 px-4 py-3 text-sm text-red-300">
                {error}
              </div>
            )}

            {/* =================================================
                QUESTION INPUT
            ================================================== */}

            <form
              onSubmit={
                submitQuestion
              }
              className="mt-8"
            >

              <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-3 shadow-2xl shadow-black/10">

                <textarea
                  value={question}
                  onChange={(event) =>
                    setQuestion(
                      event.target.value,
                    )
                  }
                  onKeyDown={(event) => {

                    if (
                      event.key ===
                        "Enter" &&
                      !event.shiftKey
                    ) {

                      event.preventDefault();

                      if (
                        question.trim() &&
                        !asking
                      ) {
                        submitQuestion();
                      }

                    }

                  }}
                  placeholder="Ask something about your documents..."
                  rows={4}
                  disabled={asking}
                  className="w-full resize-none bg-transparent px-3 py-2 text-base leading-7 text-slate-200 outline-none placeholder:text-slate-600 disabled:opacity-50"
                />

                <div className="flex items-center justify-between border-t border-slate-800 px-3 pt-3">

                  <span className="text-xs text-slate-600">
                    Enter to submit · Shift + Enter for new line
                  </span>

                  <button
                    type="submit"
                    disabled={
                      asking ||
                      !question.trim()
                    }
                    className="rounded-xl bg-white px-5 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-slate-200 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    {asking
                      ? "Thinking..."
                      : "Ask Agent"}
                  </button>

                </div>

              </div>

            </form>

            {/* Loading */}
            {asking && (
              <div className="mt-8 rounded-2xl border border-slate-800 bg-slate-900/30 p-6">

                <div className="flex items-center gap-3">

                  <div className="h-2 w-2 animate-pulse rounded-full bg-slate-400" />

                  <p className="text-sm text-slate-400">
                    The agent is retrieving evidence and
                    evaluating the best route...
                  </p>

                </div>

              </div>
            )}

            {/* =================================================
                RESPONSE
            ================================================== */}

            {response && (
              <div className="mt-10">

                {/* Question */}
                {submittedQuestion && (
                  <div>

                    <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                      Question
                    </h3>

                    <div className="mt-2 rounded-xl border border-slate-800 bg-slate-950 px-5 py-4 text-base font-medium leading-7 text-slate-200">
                      {
                        submittedQuestion
                      }
                    </div>

                  </div>
                )}

                {/* Answer */}
                <div className="mt-6">

                  <div className="flex items-center justify-between gap-4">

                    <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                      Answer
                    </h3>

                    {response.route && (
                      <span
                        className={`rounded-full border px-3 py-1 text-[11px] font-medium ${getRouteClass(
                          response.route,
                        )}`}
                      >
                        {getRouteLabel(
                          response.route,
                        )}
                      </span>
                    )}

                  </div>

                  <div className="mt-2 whitespace-pre-wrap rounded-xl border border-slate-800 bg-slate-950 p-6 text-[15px] leading-7 text-slate-300">
                    {response.answer ||
                      "No answer was generated."}
                  </div>

                </div>

                {/* Metrics */}
                {response.metrics && (
                  <div className="mt-8">

                    <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                      Run Metrics
                    </h3>

                    <div className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-5">

                      <MetricCard
                        label="Retrieved"
                        value={
                          response.metrics
                            .retrieved_documents
                        }
                      />

                      <MetricCard
                        label="Relevant"
                        value={
                          response.metrics
                            .relevant_documents
                        }
                      />

                      <MetricCard
                        label="Grounded"
                        value={
                          response.metrics
                            .grounded
                            ? "Yes"
                            : "No"
                        }
                      />

                      <MetricCard
                        label="Retries"
                        value={
                          response.metrics
                            .retry_count
                        }
                      />

                      <MetricCard
                        label="Latency"
                        value={`${response.metrics.latency_seconds.toFixed(
                          2,
                        )}s`}
                      />

                    </div>

                  </div>
                )}

                {/* Sources */}
                <div className="mt-8">

                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Sources
                  </h3>

                  {response.sources?.length >
                  0 ? (
                    <div className="mt-3 space-y-2">

                      {response.sources.map(
                        (
                          source,
                          index,
                        ) => (
                          <div
                            key={`${source.document_id || source.url || source.file_name}-${index}`}
                            className="rounded-xl border border-slate-800 bg-slate-900/30 px-4 py-3"
                          >

                            {source.type ===
                            "web" ? (
                              <div>

                                <p className="text-sm font-medium text-slate-300">
                                  {
                                    source.title ||
                                    "Web source"
                                  }
                                </p>

                                {source.url && (
                                  <a
                                    href={
                                      source.url
                                    }
                                    target="_blank"
                                    rel="noreferrer"
                                    className="mt-1 block truncate text-xs text-blue-400 hover:text-blue-300"
                                  >
                                    {
                                      source.url
                                    }
                                  </a>
                                )}

                              </div>
                            ) : (
                              <div>

                                <p className="text-sm font-medium text-slate-300">
                                  {
                                    source.file_name ||
                                    "Local document"
                                  }
                                </p>

                                {source.source && (
                                  <p className="mt-1 truncate text-xs text-slate-600">
                                    {
                                      source.source
                                    }
                                  </p>
                                )}

                              </div>
                            )}

                          </div>
                        ),
                      )}

                    </div>
                  ) : (
                    <div className="mt-3 rounded-xl border border-dashed border-slate-800 px-4 py-5 text-sm text-slate-600">
                      No sources were returned.
                    </div>
                  )}

                </div>

              </div>
            )}

            {/* Empty state */}
            {!response &&
              !asking &&
              !approvalRequired && (
                <div className="mt-12 rounded-2xl border border-dashed border-slate-800 bg-slate-900/20 px-8 py-16 text-center">

                  <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl border border-slate-800 bg-slate-900 text-lg text-slate-500">
                    ?
                  </div>

                  <h3 className="mt-5 text-base font-medium text-slate-300">
                    Your answer will appear here
                  </h3>

                  <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-slate-600">
                    Upload documents and ask the agent a
                    question. The system will retrieve,
                    grade, route and validate the answer.
                  </p>

                </div>
              )}

          </div>

        </section>
      </div>

      {/* =======================================================
          HUMAN-IN-THE-LOOP WEB SEARCH APPROVAL MODAL
      ======================================================== */}

      {approvalRequired &&
        approvalData && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4 backdrop-blur-sm">

            <div className="w-full max-w-lg rounded-2xl border border-slate-700 bg-slate-900 p-6 shadow-2xl">

              {/* Header */}
              <div className="flex items-start gap-4">

                <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-amber-800 bg-amber-950/50 text-lg text-amber-400">
                  !
                </div>

                <div className="min-w-0">

                  <h2 className="text-lg font-semibold text-white">
                    {
                      approvalData.approval_title ||
                      "Web search required"
                    }
                  </h2>

                  <p className="mt-2 text-sm leading-6 text-slate-400">
                    {
                      approvalData.approval_message ||
                      "The agent needs permission to search the web before continuing."
                    }
                  </p>

                </div>

              </div>

              {/* Question */}
              <div className="mt-5 rounded-xl border border-slate-800 bg-slate-950 p-4">

                <p className="text-xs font-semibold uppercase tracking-wider text-slate-600">
                  Question
                </p>

                <p className="mt-2 text-sm leading-6 text-slate-300">
                  {
                    submittedQuestion
                  }
                </p>

              </div>

              {/* HITL explanation */}
              <div className="mt-6 rounded-xl border border-blue-900/50 bg-blue-950/20 p-4">

                <p className="text-xs font-medium text-blue-300">
                  Human-in-the-loop checkpoint
                </p>

                <p className="mt-1 text-xs leading-5 text-slate-500">
                  The agent is paused. No external web search
                  will happen until you approve it.
                </p>

              </div>

              {/* Actions */}
              <div className="mt-6 flex gap-3">

                <button
                  onClick={() =>
                    handleWebSearchApproval(
                      false,
                    )
                  }
                  disabled={approving}
                  className="flex-1 rounded-xl border border-slate-700 px-4 py-3 text-sm font-medium text-slate-300 transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {approving
                    ? "Processing..."
                    : "Reject"}
                </button>

                <button
                  onClick={() =>
                    handleWebSearchApproval(
                      true,
                    )
                  }
                  disabled={approving}
                  className="flex-1 rounded-xl bg-white px-4 py-3 text-sm font-semibold text-slate-950 transition hover:bg-slate-200 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {approving
                    ? "Processing..."
                    : "Allow Web Search"}
                </button>

              </div>

              <p className="mt-4 text-center text-[11px] text-slate-600">
                You control whether the agent can access
                external information.
              </p>

            </div>

          </div>
        )}

    </main>
  );
}

/* =============================================================
   METRIC CARD
============================================================= */

function MetricCard({
  label,
  value,
}: {
  label: string;
  value: string | number;
}) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/30 px-4 py-3">

      <p className="text-[11px] uppercase tracking-wider text-slate-600">
        {label}
      </p>

      <p className="mt-1 text-sm font-semibold text-slate-300">
        {value}
      </p>

    </div>
  );
}