"use client";

import {
  ChangeEvent,
  FormEvent,
  KeyboardEvent,
  useEffect,
  useRef,
  useState,
} from "react";

type DocumentItem = {
  document_id: string;
  file_name: string;
  file_type?: string;
  source?: string;
  chunk_count?: number;
};

type Source = {
  type: string;
  file_name?: string;
  source?: string;
  document_id?: string;
  title?: string;
  url?: string;
};

type ChatResponse = {
  answer?: string;
  generation?: string;
  response?: string;
  route?: string;
  sources?: Source[];
  retrieved_documents?: number;
  relevant_documents?: number;
  grounded?: boolean;
  answers_question?: boolean;
  retry_count?: number;
  latency_seconds?: number;
  hitl_status?: string;
  hitl_reason?: string;
  security_status?: string;
  security_reason?: string;
  security_redactions?: number;
  web_search?: boolean;
  [key: string]: unknown;
};

const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ||
  "http://localhost:8000";

export default function Home() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [question, setQuestion] = useState("");
  const [submittedQuestion, setSubmittedQuestion] = useState("");
  const [response, setResponse] = useState<ChatResponse | null>(null);

  const [loadingDocuments, setLoadingDocuments] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const [pendingApproval, setPendingApproval] = useState(false);
  const [approvalReason, setApprovalReason] = useState("");

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadDocuments();
  }, []);

  async function loadDocuments() {
    try {
      setLoadingDocuments(true);
      setError("");

      const res = await fetch(`${API_URL}/documents`, {
        method: "GET",
        cache: "no-store",
      });

      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        throw new Error(data.detail || "Failed to load documents.");
      }

      setDocuments(Array.isArray(data) ? data : data.documents || []);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to connect to the backend."
      );
    } finally {
      setLoadingDocuments(false);
    }
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files || []);

    if (!files.length) {
      return;
    }

    setSelectedFiles(files);
    setError("");
    setSuccess("");
  }

  function removeSelectedFile(index: number) {
    setSelectedFiles((current) =>
      current.filter((_, fileIndex) => fileIndex !== index)
    );
  }

  function clearFileSelection() {
    setSelectedFiles([]);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  async function uploadFiles() {
    if (!selectedFiles.length) {
      setError("Please select at least one document.");
      return;
    }

    try {
      setUploading(true);
      setError("");
      setSuccess("");

      const formData = new FormData();

      selectedFiles.forEach((file) => {
        formData.append("files", file);
      });

      /*
       * IMPORTANT:
       * Do NOT manually set Content-Type here.
       * The browser automatically creates:
       * multipart/form-data; boundary=...
       */
      const res = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        throw new Error(
          data.detail ||
            data.message ||
            `Upload failed with status ${res.status}.`
        );
      }

      setSuccess(
        data.message ||
          `${selectedFiles.length} document${
            selectedFiles.length > 1 ? "s" : ""
          } uploaded successfully.`
      );

      // Clear the file picker after successful upload.
      clearFileSelection();

      await loadDocuments();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Unable to upload documents."
      );
    } finally {
      setUploading(false);
    }
  }

  async function deleteDocument(documentId: string) {
    try {
      setDeletingId(documentId);
      setError("");
      setSuccess("");

      const res = await fetch(`${API_URL}/documents/${documentId}`, {
        method: "DELETE",
      });

      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        throw new Error(data.detail || "Failed to delete document.");
      }

      setSuccess(data.message || "Document deleted successfully.");

      await loadDocuments();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Unable to delete document."
      );
    } finally {
      setDeletingId(null);
    }
  }

  async function askQuestion(event?: FormEvent) {
    event?.preventDefault();

    const trimmedQuestion = question.trim();

    if (!trimmedQuestion) {
      setError("Please enter a question.");
      return;
    }

    try {
      setAsking(true);
      setError("");
      setSuccess("");
      setResponse(null);
      setPendingApproval(false);
      setApprovalReason("");

      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: trimmedQuestion,
        }),
      });

      const data: ChatResponse = await res.json().catch(() => ({}));

      if (!res.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Failed to process the question."
        );
      }

      setSubmittedQuestion(trimmedQuestion);
      setQuestion("");

      setResponse(data);

      /*
       * Human-in-the-loop:
       * The backend can request approval before web search.
       */
      const hitlStatus =
        typeof data.hitl_status === "string"
          ? data.hitl_status.toLowerCase()
          : "";

      if (
        hitlStatus === "pending" ||
        hitlStatus === "awaiting_approval" ||
        hitlStatus === "approval_required"
      ) {
        setPendingApproval(true);
        setApprovalReason(
          typeof data.hitl_reason === "string"
            ? data.hitl_reason
            : "Web search approval is required."
        );
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Unable to process your question."
      );
    } finally {
      setAsking(false);
    }
  }

  async function submitApproval(approved: boolean) {
    try {
      setAsking(true);
      setError("");
      setSuccess("");

      const res = await fetch(`${API_URL}/chat/approval`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          approved,
        }),
      });

      const data: ChatResponse = await res.json().catch(() => ({}));

      if (!res.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Failed to submit approval."
        );
      }

      setResponse(data);
      setPendingApproval(false);
      setApprovalReason("");
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to submit approval decision."
      );
    } finally {
      setAsking(false);
    }
  }

  function handleQuestionKeyDown(
    event: KeyboardEvent<HTMLTextAreaElement>
  ) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      askQuestion();
    }
  }

  function formatSourceName(source: Source) {
    if (source.type === "web") {
      return source.title || source.url || "Web source";
    }

    return source.file_name || source.source || "Local document";
  }

  function getAnswer() {
    if (!response) {
      return "";
    }

    return (
      response.answer ||
      response.generation ||
      response.response ||
      "No answer was returned."
    );
  }

  return (
    <main className="min-h-screen bg-[#09090b] text-zinc-100">
      <div className="flex min-h-screen">
        {/* SIDEBAR */}
        <aside className="hidden w-[320px] shrink-0 border-r border-zinc-800 bg-[#0d0d0f] lg:flex lg:flex-col">
          <div className="border-b border-zinc-800 px-6 py-5">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white text-black">
                <span className="text-lg font-bold">A</span>
              </div>

              <div>
                <h1 className="font-semibold">Agentic Adaptive RAG</h1>
                <p className="text-xs text-zinc-500">
                  Knowledge Intelligence
                </p>
              </div>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-5">
            <div className="mb-5">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium">Knowledge Base</p>
                  <p className="mt-1 text-xs text-zinc-500">
                    Your uploaded documents
                  </p>
                </div>

                <span className="rounded-full bg-zinc-800 px-2.5 py-1 text-xs text-zinc-400">
                  {documents.length}
                </span>
              </div>
            </div>

            {/* UPLOAD */}
            <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-4">
              <label className="mb-3 block text-sm font-medium">
                Add documents
              </label>

              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".pdf,.docx,.txt,.md"
                onChange={handleFileChange}
                className="block w-full cursor-pointer rounded-xl border border-zinc-700 bg-zinc-950 p-2 text-xs text-zinc-400 file:mr-3 file:rounded-lg file:border-0 file:bg-zinc-800 file:px-3 file:py-2 file:text-xs file:font-medium file:text-zinc-200 hover:file:bg-zinc-700"
              />

              <p className="mt-2 text-[11px] leading-4 text-zinc-600">
                Supported: PDF, DOCX, TXT, Markdown
              </p>

              {selectedFiles.length > 0 && (
                <div className="mt-4 space-y-2">
                  {selectedFiles.map((file, index) => (
                    <div
                      key={`${file.name}-${index}`}
                      className="flex items-center justify-between rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-2"
                    >
                      <div className="min-w-0 pr-3">
                        <p className="truncate text-xs text-zinc-300">
                          {file.name}
                        </p>
                        <p className="text-[10px] text-zinc-600">
                          {(file.size / 1024).toFixed(1)} KB
                        </p>
                      </div>

                      <button
                        type="button"
                        onClick={() => removeSelectedFile(index)}
                        className="text-xs text-zinc-500 hover:text-red-400"
                      >
                        Remove
                      </button>
                    </div>
                  ))}

                  <button
                    type="button"
                    onClick={uploadFiles}
                    disabled={uploading}
                    className="mt-2 w-full rounded-xl bg-white px-4 py-2.5 text-sm font-medium text-black transition hover:bg-zinc-200 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {uploading ? "Uploading..." : "Upload documents"}
                  </button>
                </div>
              )}
            </div>

            {/* DOCUMENTS */}
            <div className="mt-6">
              <div className="mb-3 flex items-center justify-between">
                <p className="text-xs font-medium uppercase tracking-wider text-zinc-500">
                  Documents
                </p>

                <button
                  type="button"
                  onClick={loadDocuments}
                  disabled={loadingDocuments}
                  className="text-xs text-zinc-500 hover:text-zinc-300"
                >
                  Refresh
                </button>
              </div>

              {loadingDocuments ? (
                <div className="rounded-xl border border-zinc-800 p-4 text-center text-xs text-zinc-500">
                  Loading documents...
                </div>
              ) : documents.length === 0 ? (
                <div className="rounded-xl border border-dashed border-zinc-800 p-5 text-center">
                  <p className="text-sm text-zinc-500">
                    No documents uploaded
                  </p>
                  <p className="mt-1 text-xs text-zinc-700">
                    Upload files to build your knowledge base.
                  </p>
                </div>
              ) : (
                <div className="space-y-2">
                  {documents.map((document) => (
                    <div
                      key={document.document_id}
                      className="group rounded-xl border border-zinc-800 bg-zinc-900/40 p-3"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <p
                            className="truncate text-sm text-zinc-300"
                            title={document.file_name}
                          >
                            {document.file_name}
                          </p>

                          <p className="mt-1 text-[10px] uppercase text-zinc-600">
                            {document.file_type ||
                              document.file_name
                                ?.split(".")
                                .pop()
                                ?.toUpperCase() ||
                              "DOCUMENT"}
                          </p>
                        </div>

                        <button
                          type="button"
                          onClick={() =>
                            deleteDocument(document.document_id)
                          }
                          disabled={deletingId === document.document_id}
                          className="text-xs text-zinc-600 opacity-0 transition group-hover:opacity-100 hover:text-red-400 disabled:opacity-50"
                        >
                          {deletingId === document.document_id
                            ? "..."
                            : "Delete"}
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* SIDEBAR LINKS */}
          <div className="border-t border-zinc-800 p-4">
            <a
              href="/evaluation"
              className="mb-2 block rounded-xl px-3 py-2.5 text-sm text-zinc-400 transition hover:bg-zinc-800 hover:text-white"
            >
              Evaluation
            </a>

            <a
              href="/observability"
              className="block rounded-xl px-3 py-2.5 text-sm text-zinc-400 transition hover:bg-zinc-800 hover:text-white"
            >
              Observability
            </a>
          </div>
        </aside>

        {/* MAIN */}
        <section className="flex min-w-0 flex-1 flex-col">
          {/* MOBILE HEADER */}
          <header className="border-b border-zinc-800 bg-[#0d0d0f] px-5 py-4 lg:hidden">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="font-semibold">Agentic Adaptive RAG</h1>
                <p className="text-xs text-zinc-500">
                  Knowledge Intelligence
                </p>
              </div>

              <div className="flex gap-2">
                <a
                  href="/evaluation"
                  className="rounded-lg border border-zinc-800 px-3 py-2 text-xs text-zinc-400"
                >
                  Evaluation
                </a>

                <a
                  href="/observability"
                  className="rounded-lg border border-zinc-800 px-3 py-2 text-xs text-zinc-400"
                >
                  Observability
                </a>
              </div>
            </div>
          </header>

          <div className="mx-auto w-full max-w-5xl flex-1 px-5 py-8 md:px-8 lg:py-12">
            {/* HEADER */}
            <div className="mb-10">
              <div className="mb-3 inline-flex rounded-full border border-zinc-800 bg-zinc-900 px-3 py-1 text-xs text-zinc-500">
                Agentic Retrieval • Adaptive Routing • Grounded Generation
              </div>

              <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
                Ask your knowledge base
              </h2>

              <p className="mt-3 max-w-2xl text-sm leading-6 text-zinc-500 md:text-base">
                Ask questions about your uploaded documents. The system
                dynamically decides whether to use local retrieval or web
                search based on available evidence.
              </p>
            </div>

            {/* ALERTS */}
            {error && (
              <div className="mb-5 rounded-xl border border-red-900/50 bg-red-950/30 px-4 py-3 text-sm text-red-300">
                {error}
              </div>
            )}

            {success && (
              <div className="mb-5 rounded-xl border border-emerald-900/50 bg-emerald-950/30 px-4 py-3 text-sm text-emerald-300">
                {success}
              </div>
            )}

            {/* SECURITY NOTICE */}
            <div className="mb-6 rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4">
              <div className="flex gap-3">
                <div className="mt-0.5 text-zinc-500">✓</div>

                <div>
                  <p className="text-sm font-medium text-zinc-300">
                    Security-aware RAG
                  </p>
                  <p className="mt-1 text-xs leading-5 text-zinc-600">
                    Uploaded and retrieved content is treated as untrusted
                    data. Prompt injection and sensitive output patterns are
                    monitored.
                  </p>
                </div>
              </div>
            </div>

            {/* ASK FORM */}
            <form onSubmit={askQuestion}>
              <div className="rounded-2xl border border-zinc-800 bg-zinc-950 shadow-2xl shadow-black/20">
                <textarea
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  onKeyDown={handleQuestionKeyDown}
                  placeholder="Ask something about your documents..."
                  rows={5}
                  disabled={asking}
                  className="w-full resize-none bg-transparent px-5 py-5 text-sm leading-6 text-zinc-100 outline-none placeholder:text-zinc-700 disabled:opacity-50"
                />

                <div className="flex items-center justify-between border-t border-zinc-800 px-4 py-3">
                  <p className="text-[11px] text-zinc-700">
                    Press Enter to ask · Shift + Enter for a new line
                  </p>

                  <button
                    type="submit"
                    disabled={asking || !question.trim()}
                    className="rounded-xl bg-white px-5 py-2.5 text-sm font-medium text-black transition hover:bg-zinc-200 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    {asking ? "Thinking..." : "Ask Agent"}
                  </button>
                </div>
              </div>
            </form>

            {/* HITL APPROVAL */}
            {pendingApproval && (
              <div className="mt-6 rounded-2xl border border-amber-900/60 bg-amber-950/20 p-5">
                <div className="flex items-start gap-3">
                  <div className="mt-0.5 text-amber-400">!</div>

                  <div className="flex-1">
                    <h3 className="font-medium text-amber-200">
                      Web search approval required
                    </h3>

                    <p className="mt-2 text-sm leading-6 text-amber-300/70">
                      {approvalReason ||
                        "The agent wants to use external web search because local evidence may be insufficient."}
                    </p>

                    <div className="mt-4 flex gap-3">
                      <button
                        type="button"
                        onClick={() => submitApproval(true)}
                        disabled={asking}
                        className="rounded-xl bg-amber-400 px-4 py-2 text-sm font-medium text-black hover:bg-amber-300 disabled:opacity-50"
                      >
                        Approve web search
                      </button>

                      <button
                        type="button"
                        onClick={() => submitApproval(false)}
                        disabled={asking}
                        className="rounded-xl border border-amber-800 px-4 py-2 text-sm font-medium text-amber-300 hover:bg-amber-950/50 disabled:opacity-50"
                      >
                        Reject
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* RESPONSE */}
            {response && (
              <div className="mt-8 space-y-5">
                {/* SUBMITTED QUESTION */}
                {submittedQuestion && (
                  <div className="rounded-2xl border border-zinc-800 bg-zinc-900/30 p-5">
                    <p className="mb-2 text-[10px] font-medium uppercase tracking-wider text-zinc-600">
                      Submitted question
                    </p>

                    <p className="text-sm leading-6 text-zinc-300">
                      {submittedQuestion}
                    </p>
                  </div>
                )}

                {/* ANSWER */}
                <div className="rounded-2xl border border-zinc-800 bg-zinc-950 p-6">
                  <div className="mb-5 flex items-center justify-between">
                    <h3 className="text-sm font-medium text-zinc-300">
                      Answer
                    </h3>

                    {response.route && (
                      <span className="rounded-full border border-zinc-800 bg-zinc-900 px-3 py-1 text-[10px] uppercase tracking-wider text-zinc-500">
                        {String(response.route)}
                      </span>
                    )}
                  </div>

                  <div className="whitespace-pre-wrap text-sm leading-7 text-zinc-300">
                    {getAnswer()}
                  </div>
                </div>

                {/* RUN METRICS */}
                <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                  <Metric
                    label="Retrieved"
                    value={response.retrieved_documents ?? 0}
                  />

                  <Metric
                    label="Relevant"
                    value={response.relevant_documents ?? 0}
                  />

                  <Metric
                    label="Retries"
                    value={response.retry_count ?? 0}
                  />

                  <Metric
                    label="Latency"
                    value={
                      typeof response.latency_seconds === "number"
                        ? `${response.latency_seconds.toFixed(2)}s`
                        : "—"
                    }
                  />
                </div>

                {/* QUALITY */}
                <div className="grid gap-3 md:grid-cols-3">
                  <StatusCard
                    label="Grounded"
                    value={response.grounded}
                  />

                  <StatusCard
                    label="Answers Question"
                    value={response.answers_question}
                  />

                  <StatusCard
                    label="Route"
                    value={response.route || "unknown"}
                  />
                </div>

                {/* SECURITY */}
                {response.security_status && (
                  <div className="rounded-2xl border border-zinc-800 bg-zinc-900/30 p-5">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-zinc-300">
                          Security status
                        </p>

                        <p className="mt-1 text-xs text-zinc-600">
                          {response.security_reason ||
                            "Security checks completed."}
                        </p>
                      </div>

                      <span className="rounded-full border border-zinc-700 px-3 py-1 text-xs text-zinc-400">
                        {response.security_status}
                      </span>
                    </div>

                    {typeof response.security_redactions === "number" &&
                      response.security_redactions > 0 && (
                        <p className="mt-3 text-xs text-amber-400">
                          {response.security_redactions} sensitive item
                          {response.security_redactions > 1 ? "s" : ""}{" "}
                          redacted.
                        </p>
                      )}
                  </div>
                )}

                {/* SOURCES */}
                {Array.isArray(response.sources) &&
                  response.sources.length > 0 && (
                    <div className="rounded-2xl border border-zinc-800 bg-zinc-950 p-5">
                      <div className="mb-4">
                        <h3 className="text-sm font-medium text-zinc-300">
                          Sources
                        </h3>

                        <p className="mt-1 text-xs text-zinc-600">
                          Evidence used to produce this response
                        </p>
                      </div>

                      <div className="space-y-2">
                        {response.sources.map((source, index) => (
                          <div
                            key={`${source.url || source.source || index}-${index}`}
                            className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-3"
                          >
                            <div className="flex items-start gap-3">
                              <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-lg bg-zinc-800 text-[10px] text-zinc-500">
                                {index + 1}
                              </span>

                              <div className="min-w-0">
                                <p className="text-sm text-zinc-300">
                                  {formatSourceName(source)}
                                </p>

                                {source.type === "web" && source.url && (
                                  <a
                                    href={source.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="mt-1 block truncate text-xs text-zinc-600 hover:text-zinc-400"
                                  >
                                    {source.url}
                                  </a>
                                )}

                                {source.type === "local" &&
                                  source.document_id && (
                                    <p className="mt-1 truncate text-xs text-zinc-700">
                                      Document ID: {source.document_id}
                                    </p>
                                  )}
                              </div>
                            </div>
                          </div>
                        ))}
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

function Metric({
  label,
  value,
}: {
  label: string;
  value: string | number;
}) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950 p-4">
      <p className="text-[10px] uppercase tracking-wider text-zinc-600">
        {label}
      </p>

      <p className="mt-2 text-xl font-semibold text-zinc-300">{value}</p>
    </div>
  );
}

function StatusCard({
  label,
  value,
}: {
  label: string;
  value: boolean | string | number | undefined;
}) {
  let displayValue = "—";

  if (typeof value === "boolean") {
    displayValue = value ? "Yes" : "No";
  } else if (value !== undefined && value !== null) {
    displayValue = String(value);
  }

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950 p-4">
      <p className="text-[10px] uppercase tracking-wider text-zinc-600">
        {label}
      </p>

      <p className="mt-2 text-sm font-medium text-zinc-300">{displayValue}</p>
    </div>
  );
}