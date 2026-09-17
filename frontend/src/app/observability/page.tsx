"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  "http://localhost:8000";

type Summary = {
  total_runs: number;
  successful_runs: number;
  failed_runs: number;
  success_rate: number;
  average_latency_seconds: number;
  average_retries: number;
  grounded_rate: number;
  answer_quality_rate: number;
  local_route_count: number;
  web_route_count: number;
  hitl_approved_count: number;
  hitl_rejected_count: number;
};

type Observation = {
  timestamp: string;
  question: string;
  route: string;
  retrieved_documents: number;
  relevant_documents: number;
  grounded: boolean;
  answers_question: boolean;
  retry_count: number;
  latency_seconds: number;
  source_count: number;
  local_source_count: number;
  web_source_count: number;
  success: boolean;
  error: string;
  hitl_status?: string;
  hitl_reason?: string;
};

function MetricCard({
  label,
  value,
  description,
}: {
  label: string;
  value: string | number;
  description?: string;
}) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-5">
      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-600">
        {label}
      </p>

      <p className="mt-2 text-2xl font-semibold tracking-tight text-white">
        {value}
      </p>

      {description && (
        <p className="mt-1 text-xs text-slate-600">
          {description}
        </p>
      )}
    </div>
  );
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatDate(value: string) {
  if (!value) {
    return "—";
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString();
}

function routeLabel(route: string) {
  switch (route) {
    case "local":
      return "Local";
    case "web":
      return "Web";
    case "web_rejected":
      return "Web Rejected";
    default:
      return route || "Unknown";
  }
}

function routeClass(route: string) {
  switch (route) {
    case "local":
      return "border-emerald-800 bg-emerald-950/40 text-emerald-300";
    case "web":
      return "border-blue-800 bg-blue-950/40 text-blue-300";
    case "web_rejected":
      return "border-amber-800 bg-amber-950/40 text-amber-300";
    default:
      return "border-slate-700 bg-slate-900 text-slate-400";
  }
}

function hitlLabel(status?: string) {
  switch (status) {
    case "approved":
      return "Approved";
    case "rejected":
      return "Rejected";
    default:
      return "—";
  }
}

function hitlClass(status?: string) {
  switch (status) {
    case "approved":
      return "border-emerald-800 bg-emerald-950/40 text-emerald-300";
    case "rejected":
      return "border-amber-800 bg-amber-950/40 text-amber-300";
    default:
      return "border-slate-800 bg-slate-900 text-slate-600";
  }
}

export default function ObservabilityPage() {
  const [summary, setSummary] = useState<Summary | null>(
    null,
  );

  const [observations, setObservations] = useState<
    Observation[]
  >([]);

  const [loading, setLoading] = useState(true);
  const [clearing, setClearing] = useState(false);
  const [showClearModal, setShowClearModal] =
    useState(false);

  const [error, setError] = useState("");

  async function loadObservability() {
    try {
      setLoading(true);
      setError("");

      const [summaryResponse, runsResponse] =
        await Promise.all([
          fetch(`${API_BASE_URL}/observability/summary`),
          fetch(`${API_BASE_URL}/observability/runs`),
        ]);

      if (!summaryResponse.ok) {
        throw new Error(
          "Failed to load observability summary.",
        );
      }

      if (!runsResponse.ok) {
        throw new Error(
          "Failed to load observability runs.",
        );
      }

      const summaryData = await summaryResponse.json();
      const runsData = await runsResponse.json();

      setSummary(summaryData);
      setObservations(
        runsData.observations ||
          runsData.runs ||
          [],
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to load observability data.",
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadObservability();
  }, []);

  async function clearHistory() {
    try {
      setClearing(true);
      setError("");

      const result = await fetch(
        `${API_BASE_URL}/observability`,
        {
          method: "DELETE",
        },
      );

      const data = await result.json();

      if (!result.ok) {
        throw new Error(
          data.detail ||
            "Failed to clear observability history.",
        );
      }

      setSummary(null);
      setObservations([]);
      setShowClearModal(false);

      await loadObservability();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to clear observability history.",
      );
    } finally {
      setClearing(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-[1500px] px-8 py-10 lg:px-12">
        {/* Header */}
        <div className="flex flex-col gap-6 border-b border-slate-800 pb-8 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
              Agentic RAG
            </p>

            <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white">
              Observability
            </h1>

            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
              Inspect routing decisions, retrieval quality,
              answer validation, retries, latency and
              human-in-the-loop web-search decisions.
            </p>
          </div>

          <div className="flex gap-3">
            <button
              onClick={loadObservability}
              disabled={loading}
              className="rounded-xl border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm font-medium text-slate-300 transition hover:bg-slate-800 disabled:opacity-40"
            >
              {loading ? "Refreshing..." : "Refresh"}
            </button>

            <Link
              href="/"
              className="rounded-xl bg-white px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-slate-200"
            >
              Back to RAG
            </Link>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="mt-6 rounded-xl border border-red-900/70 bg-red-950/30 px-4 py-3 text-sm text-red-300">
            {error}
          </div>
        )}

        {/* Loading */}
        {loading && !summary ? (
          <div className="mt-10 rounded-2xl border border-slate-800 bg-slate-900/30 p-10 text-center text-sm text-slate-500">
            Loading observability data...
          </div>
        ) : (
          <>
            {/* Summary */}
            <section className="mt-8">
              <div className="mb-4">
                <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
                  System Overview
                </p>
              </div>

              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <MetricCard
                  label="Total Runs"
                  value={
                    summary?.total_runs ?? 0
                  }
                />

                <MetricCard
                  label="Success Rate"
                  value={formatPercent(
                    summary?.success_rate ?? 0,
                  )}
                />

                <MetricCard
                  label="Avg Latency"
                  value={`${(
                    summary?.average_latency_seconds ??
                    0
                  ).toFixed(2)}s`}
                />

                <MetricCard
                  label="Avg Retries"
                  value={(
                    summary?.average_retries ?? 0
                  ).toFixed(2)}
                />

                <MetricCard
                  label="Grounded Rate"
                  value={formatPercent(
                    summary?.grounded_rate ?? 0,
                  )}
                />

                <MetricCard
                  label="Answer Quality"
                  value={formatPercent(
                    summary?.answer_quality_rate ??
                      0,
                  )}
                />

                <MetricCard
                  label="Local Routes"
                  value={
                    summary?.local_route_count ?? 0
                  }
                  description="Answered from local knowledge"
                />

                <MetricCard
                  label="Web Routes"
                  value={
                    summary?.web_route_count ?? 0
                  }
                  description="Required external retrieval"
                />

                <MetricCard
                  label="HITL Approved"
                  value={
                    summary?.hitl_approved_count ?? 0
                  }
                  description="Web searches approved"
                />

                <MetricCard
                  label="HITL Rejected"
                  value={
                    summary?.hitl_rejected_count ?? 0
                  }
                  description="Web searches rejected"
                />

                <MetricCard
                  label="Successful Runs"
                  value={
                    summary?.successful_runs ?? 0
                  }
                />

                <MetricCard
                  label="Failed Runs"
                  value={
                    summary?.failed_runs ?? 0
                  }
                />
              </div>
            </section>

            {/* Architecture signal */}
            <section className="mt-10">
              <div className="rounded-2xl border border-slate-800 bg-slate-900/30 p-6">
                <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
                  <div>
                    <p className="text-sm font-semibold text-slate-300">
                      Agent execution pipeline
                    </p>

                    <p className="mt-1 text-xs leading-5 text-slate-600">
                      Retrieval → grading → routing → optional
                      human approval → web search → generation
                      → validation
                    </p>
                  </div>

                  <div className="flex flex-wrap gap-2">
                    {[
                      "Retrieve",
                      "Grade",
                      "Route",
                      "HITL",
                      "Web",
                      "Generate",
                      "Validate",
                    ].map((step) => (
                      <span
                        key={step}
                        className="rounded-lg border border-slate-800 bg-slate-950 px-3 py-1.5 text-[11px] text-slate-500"
                      >
                        {step}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </section>

            {/* Runs */}
            <section className="mt-10">
              <div className="flex items-end justify-between">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
                    Recent Runs
                  </p>

                  <p className="mt-1 text-xs text-slate-600">
                    Latest agent executions recorded by the
                    backend.
                  </p>
                </div>

                <span className="text-xs text-slate-600">
                  {observations.length} runs
                </span>
              </div>

              {observations.length === 0 ? (
                <div className="mt-4 rounded-2xl border border-dashed border-slate-800 bg-slate-900/20 px-6 py-12 text-center">
                  <p className="text-sm text-slate-500">
                    No observability data yet.
                  </p>

                  <p className="mt-1 text-xs text-slate-600">
                    Ask the agent a question to create the
                    first run.
                  </p>
                </div>
              ) : (
                <div className="mt-4 overflow-hidden rounded-2xl border border-slate-800">
                  <div className="overflow-x-auto">
                    <table className="w-full min-w-[1100px] border-collapse">
                      <thead>
                        <tr className="border-b border-slate-800 bg-slate-900/70 text-left">
                          <th className="px-5 py-4 text-[11px] font-semibold uppercase tracking-wider text-slate-600">
                            Time
                          </th>

                          <th className="px-5 py-4 text-[11px] font-semibold uppercase tracking-wider text-slate-600">
                            Question
                          </th>

                          <th className="px-5 py-4 text-[11px] font-semibold uppercase tracking-wider text-slate-600">
                            Route
                          </th>

                          <th className="px-5 py-4 text-[11px] font-semibold uppercase tracking-wider text-slate-600">
                            HITL
                          </th>

                          <th className="px-5 py-4 text-[11px] font-semibold uppercase tracking-wider text-slate-600">
                            Retrieved
                          </th>

                          <th className="px-5 py-4 text-[11px] font-semibold uppercase tracking-wider text-slate-600">
                            Relevant
                          </th>

                          <th className="px-5 py-4 text-[11px] font-semibold uppercase tracking-wider text-slate-600">
                            Grounded
                          </th>

                          <th className="px-5 py-4 text-[11px] font-semibold uppercase tracking-wider text-slate-600">
                            Retries
                          </th>

                          <th className="px-5 py-4 text-[11px] font-semibold uppercase tracking-wider text-slate-600">
                            Latency
                          </th>
                        </tr>
                      </thead>

                      <tbody>
                        {observations.map(
                          (observation, index) => (
                            <tr
                              key={`${observation.timestamp}-${index}`}
                              className="border-b border-slate-800/70 last:border-b-0 hover:bg-slate-900/40"
                            >
                              <td className="whitespace-nowrap px-5 py-4 text-xs text-slate-600">
                                {formatDate(
                                  observation.timestamp,
                                )}
                              </td>

                              <td
                                className="max-w-[360px] px-5 py-4 text-sm text-slate-300"
                                title={
                                  observation.question
                                }
                              >
                                <div className="line-clamp-2">
                                  {observation.question ||
                                    "—"}
                                </div>
                              </td>

                              <td className="px-5 py-4">
                                <span
                                  className={`inline-flex rounded-full border px-2.5 py-1 text-[10px] font-medium ${routeClass(
                                    observation.route,
                                  )}`}
                                >
                                  {routeLabel(
                                    observation.route,
                                  )}
                                </span>
                              </td>

                              <td className="px-5 py-4">
                                <span
                                  className={`inline-flex rounded-full border px-2.5 py-1 text-[10px] font-medium ${hitlClass(
                                    observation.hitl_status,
                                  )}`}
                                >
                                  {hitlLabel(
                                    observation.hitl_status,
                                  )}
                                </span>
                              </td>

                              <td className="px-5 py-4 text-sm text-slate-400">
                                {
                                  observation.retrieved_documents
                                }
                              </td>

                              <td className="px-5 py-4 text-sm text-slate-400">
                                {
                                  observation.relevant_documents
                                }
                              </td>

                              <td className="px-5 py-4">
                                <span
                                  className={
                                    observation.grounded
                                      ? "text-emerald-400"
                                      : "text-red-400"
                                  }
                                >
                                  {observation.grounded
                                    ? "Yes"
                                    : "No"}
                                </span>
                              </td>

                              <td className="px-5 py-4 text-sm text-slate-400">
                                {
                                  observation.retry_count
                                }
                              </td>

                              <td className="whitespace-nowrap px-5 py-4 text-sm text-slate-400">
                                {observation.latency_seconds.toFixed(
                                  2,
                                )}
                                s
                              </td>
                            </tr>
                          ),
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </section>

            {/* HITL details */}
            {observations.some(
              (item) =>
                item.hitl_status === "approved" ||
                item.hitl_status === "rejected",
            ) && (
              <section className="mt-10">
                <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
                  Human-in-the-Loop Decisions
                </p>

                <div className="mt-4 space-y-3">
                  {observations
                    .filter(
                      (item) =>
                        item.hitl_status ===
                          "approved" ||
                        item.hitl_status ===
                          "rejected",
                    )
                    .slice(0, 10)
                    .map((observation, index) => (
                      <div
                        key={`hitl-${observation.timestamp}-${index}`}
                        className="rounded-2xl border border-slate-800 bg-slate-900/30 p-5"
                      >
                        <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                          <div className="min-w-0">
                            <p className="text-sm font-medium leading-6 text-slate-300">
                              {observation.question}
                            </p>

                            {observation.hitl_reason && (
                              <p className="mt-2 text-xs leading-5 text-slate-600">
                                {observation.hitl_reason}
                              </p>
                            )}
                          </div>

                          <span
                            className={`shrink-0 rounded-full border px-3 py-1 text-[10px] font-medium ${hitlClass(
                              observation.hitl_status,
                            )}`}
                          >
                            {hitlLabel(
                              observation.hitl_status,
                            )}
                          </span>
                        </div>
                      </div>
                    ))}
                </div>
              </section>
            )}

            {/* Clear history */}
            <section className="mt-12 border-t border-slate-800 pt-8">
              <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-400">
                    Observability history
                  </p>

                  <p className="mt-1 text-xs leading-5 text-slate-600">
                    Clearing history only removes recorded
                    telemetry. Your uploaded documents and
                    vector database are not affected.
                  </p>
                </div>

                <button
                  onClick={() =>
                    setShowClearModal(true)
                  }
                  disabled={
                    clearing ||
                    observations.length === 0
                  }
                  className="rounded-xl border border-red-900/70 px-4 py-2.5 text-sm font-medium text-red-400 transition hover:bg-red-950/30 disabled:cursor-not-allowed disabled:opacity-30"
                >
                  Clear History
                </button>
              </div>
            </section>
          </>
        )}
      </div>

      {/* Clear History Confirmation Modal */}
      {showClearModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4 backdrop-blur-sm">
          <div className="w-full max-w-md rounded-2xl border border-slate-700 bg-slate-900 p-6 shadow-2xl">
            <h2 className="text-lg font-semibold text-white">
              Clear observability history?
            </h2>

            <p className="mt-3 text-sm leading-6 text-slate-400">
              This will permanently delete the recorded
              agent execution history.
            </p>

            <div className="mt-4 rounded-xl border border-slate-800 bg-slate-950 p-4">
              <p className="text-xs leading-5 text-slate-600">
                This does not delete:
              </p>

              <ul className="mt-2 space-y-1 text-xs text-slate-500">
                <li>• Uploaded documents</li>
                <li>• Chroma vector database</li>
                <li>• Evaluation history</li>
                <li>• Benchmark results</li>
              </ul>
            </div>

            <div className="mt-6 flex gap-3">
              <button
                onClick={() =>
                  setShowClearModal(false)
                }
                disabled={clearing}
                className="flex-1 rounded-xl border border-slate-700 px-4 py-3 text-sm font-medium text-slate-300 transition hover:bg-slate-800 disabled:opacity-40"
              >
                Cancel
              </button>

              <button
                onClick={clearHistory}
                disabled={clearing}
                className="flex-1 rounded-xl bg-red-500 px-4 py-3 text-sm font-semibold text-white transition hover:bg-red-400 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {clearing
                  ? "Clearing..."
                  : "Clear History"}
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}