"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type ObservabilitySummary = {
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

type ObservabilityRun = {
  timestamp?: string;
  question?: string;
  route?: string;
  retrieved_documents?: number;
  relevant_documents?: number;
  grounded?: boolean;
  answers_question?: boolean;
  retry_count?: number;
  latency_seconds?: number;
  source_count?: number;
  local_source_count?: number;
  web_source_count?: number;
  success?: boolean;
  error?: string;
  hitl_status?: string;
  hitl_reason?: string;
};

type ObservabilityResponse = {
  runs: ObservabilityRun[];
  summary: ObservabilitySummary;
};

const EMPTY_SUMMARY: ObservabilitySummary = {
  total_runs: 0,
  successful_runs: 0,
  failed_runs: 0,
  success_rate: 0,
  average_latency_seconds: 0,
  average_retries: 0,
  grounded_rate: 0,
  answer_quality_rate: 0,
  local_route_count: 0,
  web_route_count: 0,
  hitl_approved_count: 0,
  hitl_rejected_count: 0,
};

function formatPercent(value: number): string {
  return `${(Number(value) || 0).toFixed(1)}%`;
}

function formatSeconds(value: number): string {
  return `${(Number(value) || 0).toFixed(2)}s`;
}

function formatDate(value?: string): string {
  if (!value) return "—";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString();
}

function normalizeSummary(
  summary?: Partial<ObservabilitySummary>
): ObservabilitySummary {
  return {
    total_runs: Number(summary?.total_runs) || 0,
    successful_runs: Number(summary?.successful_runs) || 0,
    failed_runs: Number(summary?.failed_runs) || 0,
    success_rate: Number(summary?.success_rate) || 0,
    average_latency_seconds:
      Number(summary?.average_latency_seconds) || 0,
    average_retries: Number(summary?.average_retries) || 0,
    grounded_rate: Number(summary?.grounded_rate) || 0,
    answer_quality_rate:
      Number(summary?.answer_quality_rate) || 0,
    local_route_count:
      Number(summary?.local_route_count) || 0,
    web_route_count:
      Number(summary?.web_route_count) || 0,
    hitl_approved_count:
      Number(summary?.hitl_approved_count) || 0,
    hitl_rejected_count:
      Number(summary?.hitl_rejected_count) || 0,
  };
}

export default function ObservabilityPage() {
  const router = useRouter();

  const [summary, setSummary] =
    useState<ObservabilitySummary>(EMPTY_SUMMARY);

  const [runs, setRuns] = useState<ObservabilityRun[]>([]);

  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [clearing, setClearing] = useState(false);

  const [error, setError] = useState("");
  const [showClearConfirm, setShowClearConfirm] =
    useState(false);

  const loadObservability = useCallback(async () => {
    try {
      setError("");

      const response = await fetch(
        `${API_URL}/observability`,
        {
          method: "GET",
          cache: "no-store",
        }
      );

      if (!response.ok) {
        throw new Error(
          `Observability API returned ${response.status}`
        );
      }

      const data: ObservabilityResponse =
        await response.json();

      setSummary(normalizeSummary(data.summary));
      setRuns(Array.isArray(data.runs) ? data.runs : []);
    } catch (err) {
      console.error(
        "Failed to load observability:",
        err
      );

      setError(
        "Failed to load observability data. Make sure the FastAPI backend is running."
      );
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadObservability();
  }, [loadObservability]);

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadObservability();
  };

  const handleClearHistory = async () => {
    try {
      setClearing(true);
      setError("");

      const response = await fetch(
        `${API_URL}/observability`,
        {
          method: "DELETE",
        }
      );

      if (!response.ok) {
        throw new Error(
          `Delete request returned ${response.status}`
        );
      }

      setRuns([]);
      setSummary(EMPTY_SUMMARY);
      setShowClearConfirm(false);
    } catch (err) {
      console.error(
        "Failed to clear observability history:",
        err
      );

      setError(
        "Failed to clear observability history."
      );
    } finally {
      setClearing(false);
    }
  };

  return (
    <main className="min-h-screen bg-[#020617] text-white">
      <div className="mx-auto max-w-[1800px] px-5 py-8 sm:px-8 lg:px-10">
        {/* Header */}
        <header className="border-b border-slate-800 pb-10">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <p className="mb-3 text-sm font-semibold uppercase tracking-[0.25em] text-slate-400">
                Agentic RAG
              </p>

              <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
                Observability
              </h1>

              <p className="mt-4 max-w-3xl text-base leading-7 text-slate-400">
                Inspect routing decisions, retrieval quality,
                answer validation, retries, latency and
                human-in-the-loop web-search decisions.
              </p>
            </div>

            <div className="flex flex-wrap gap-3">
              <button
                onClick={handleRefresh}
                disabled={refreshing}
                className="rounded-2xl border border-slate-700 bg-slate-900 px-5 py-3 text-sm font-medium text-slate-200 transition hover:border-slate-600 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {refreshing ? "Refreshing..." : "Refresh"}
              </button>

              <button
                onClick={() => router.push("/")}
                className="rounded-2xl bg-white px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-slate-200"
              >
                Back to RAG
              </button>
            </div>
          </div>
        </header>

        {/* Error */}
        {error && (
          <div className="mt-8 rounded-2xl border border-red-900/70 bg-red-950/30 px-5 py-4 text-sm text-red-300">
            {error}
          </div>
        )}

        {/* Loading */}
        {loading ? (
          <div className="py-24 text-center">
            <div className="mx-auto h-8 w-8 animate-spin rounded-full border-2 border-slate-700 border-t-white" />

            <p className="mt-5 text-sm text-slate-400">
              Loading observability data...
            </p>
          </div>
        ) : (
          <>
            {/* System Overview */}
            <section className="mt-12">
              <div className="mb-6">
                <p className="text-xs font-semibold uppercase tracking-[0.25em] text-slate-500">
                  System Overview
                </p>
              </div>

              <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
                <MetricCard
                  label="Total Runs"
                  value={summary.total_runs}
                />

                <MetricCard
                  label="Success Rate"
                  value={formatPercent(
                    summary.success_rate
                  )}
                />

                <MetricCard
                  label="Avg Latency"
                  value={formatSeconds(
                    summary.average_latency_seconds
                  )}
                />

                <MetricCard
                  label="Avg Retries"
                  value={summary.average_retries.toFixed(2)}
                />

                <MetricCard
                  label="Grounded Rate"
                  value={formatPercent(
                    summary.grounded_rate
                  )}
                />

                <MetricCard
                  label="Answer Quality"
                  value={formatPercent(
                    summary.answer_quality_rate
                  )}
                />

                <MetricCard
                  label="Local Routes"
                  value={summary.local_route_count}
                  description="Answered from local knowledge"
                />

                <MetricCard
                  label="Web Routes"
                  value={summary.web_route_count}
                  description="Required external retrieval"
                />
              </div>
            </section>

            {/* Additional Metrics */}
            <section className="mt-12">
              <div className="mb-6">
                <p className="text-xs font-semibold uppercase tracking-[0.25em] text-slate-500">
                  Human-in-the-Loop
                </p>
              </div>

              <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
                <MetricCard
                  label="Approved Searches"
                  value={summary.hitl_approved_count}
                  description="Web searches approved by user"
                />

                <MetricCard
                  label="Rejected Searches"
                  value={summary.hitl_rejected_count}
                  description="Web searches rejected by user"
                />

                <MetricCard
                  label="Successful Runs"
                  value={summary.successful_runs}
                />

                <MetricCard
                  label="Failed Runs"
                  value={summary.failed_runs}
                />
              </div>
            </section>

            {/* Explanation */}
            <section className="mt-12">
              <div className="grid gap-5 lg:grid-cols-3">
                <InfoCard
                  title="Routing"
                  description="Tracks whether the agent answered using the uploaded knowledge base or routed the question to web search."
                />

                <InfoCard
                  title="Retrieval & Validation"
                  description="Tracks retrieved documents, relevant documents, grounding and whether the generated answer addresses the question."
                />

                <InfoCard
                  title="Performance"
                  description="Tracks execution latency and retry behavior so changes to the RAG pipeline can be measured over time."
                />
              </div>
            </section>

            {/* Recent Runs */}
            <section className="mt-12">
              <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.25em] text-slate-500">
                    Execution History
                  </p>

                  <h2 className="mt-2 text-2xl font-semibold">
                    Recent Runs
                  </h2>
                </div>

                <p className="text-sm text-slate-500">
                  {runs.length} recorded{" "}
                  {runs.length === 1 ? "run" : "runs"}
                </p>
              </div>

              {runs.length === 0 ? (
                <div className="rounded-2xl border border-slate-800 bg-slate-950/60 px-6 py-16 text-center">
                  <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full border border-slate-800 bg-slate-900">
                    <span className="text-xl text-slate-500">
                      ∅
                    </span>
                  </div>

                  <h3 className="mt-5 text-lg font-semibold text-slate-200">
                    No observability runs yet
                  </h3>

                  <p className="mx-auto mt-2 max-w-lg text-sm leading-6 text-slate-500">
                    Ask a question through the RAG interface.
                    Completed executions will appear here with
                    routing, retrieval, validation, latency and
                    HITL information.
                  </p>

                  <button
                    onClick={() => router.push("/")}
                    className="mt-6 rounded-xl bg-white px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-slate-200"
                  >
                    Ask the Agent
                  </button>
                </div>
              ) : (
                <div className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-950/60">
                  <div className="overflow-x-auto">
                    <table className="min-w-[1100px] w-full text-left">
                      <thead className="border-b border-slate-800 bg-slate-900/70">
                        <tr>
                          <th className="px-5 py-4 text-xs font-semibold uppercase tracking-wider text-slate-500">
                            Question
                          </th>

                          <th className="px-5 py-4 text-xs font-semibold uppercase tracking-wider text-slate-500">
                            Route
                          </th>

                          <th className="px-5 py-4 text-xs font-semibold uppercase tracking-wider text-slate-500">
                            Retrieval
                          </th>

                          <th className="px-5 py-4 text-xs font-semibold uppercase tracking-wider text-slate-500">
                            Grounded
                          </th>

                          <th className="px-5 py-4 text-xs font-semibold uppercase tracking-wider text-slate-500">
                            Quality
                          </th>

                          <th className="px-5 py-4 text-xs font-semibold uppercase tracking-wider text-slate-500">
                            Latency
                          </th>

                          <th className="px-5 py-4 text-xs font-semibold uppercase tracking-wider text-slate-500">
                            Retries
                          </th>

                          <th className="px-5 py-4 text-xs font-semibold uppercase tracking-wider text-slate-500">
                            Status
                          </th>

                          <th className="px-5 py-4 text-xs font-semibold uppercase tracking-wider text-slate-500">
                            Time
                          </th>
                        </tr>
                      </thead>

                      <tbody className="divide-y divide-slate-800">
                        {runs.map((run, index) => (
                          <tr
                            key={`${run.timestamp || "run"}-${index}`}
                            className="transition hover:bg-slate-900/50"
                          >
                            <td className="max-w-[320px] px-5 py-5">
                              <div
                                className="truncate text-sm font-medium text-slate-200"
                                title={
                                  run.question || ""
                                }
                              >
                                {run.question || "—"}
                              </div>

                              {run.error && (
                                <div
                                  className="mt-1 max-w-[300px] truncate text-xs text-red-400"
                                  title={run.error}
                                >
                                  {run.error}
                                </div>
                              )}
                            </td>

                            <td className="px-5 py-5">
                              <RouteBadge
                                route={run.route}
                              />
                            </td>

                            <td className="px-5 py-5 text-sm text-slate-400">
                              <div>
                                {run.relevant_documents ??
                                  0}{" "}
                                /{" "}
                                {run.retrieved_documents ??
                                  0}
                              </div>

                              <div className="mt-1 text-xs text-slate-600">
                                relevant / retrieved
                              </div>
                            </td>

                            <td className="px-5 py-5">
                              <BooleanBadge
                                value={run.grounded}
                              />
                            </td>

                            <td className="px-5 py-5">
                              <BooleanBadge
                                value={
                                  run.answers_question
                                }
                              />
                            </td>

                            <td className="px-5 py-5 text-sm text-slate-400">
                              {formatSeconds(
                                run.latency_seconds || 0
                              )}
                            </td>

                            <td className="px-5 py-5 text-sm text-slate-400">
                              {run.retry_count ?? 0}
                            </td>

                            <td className="px-5 py-5">
                              <StatusBadge
                                success={run.success}
                              />
                            </td>

                            <td className="whitespace-nowrap px-5 py-5 text-sm text-slate-500">
                              {formatDate(
                                run.timestamp
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </section>

            {/* Source / HITL breakdown */}
            {runs.length > 0 && (
              <section className="mt-12">
                <div className="grid gap-5 lg:grid-cols-2">
                  <BreakdownCard
                    title="Source Breakdown"
                    items={[
                      {
                        label: "Local knowledge",
                        value: summary.local_route_count,
                      },
                      {
                        label: "Web search",
                        value: summary.web_route_count,
                      },
                    ]}
                  />

                  <BreakdownCard
                    title="Human-in-the-Loop Decisions"
                    items={[
                      {
                        label: "Approved",
                        value:
                          summary.hitl_approved_count,
                      },
                      {
                        label: "Rejected",
                        value:
                          summary.hitl_rejected_count,
                      },
                    ]}
                  />
                </div>
              </section>
            )}

            {/* Clear history */}
            <section className="mt-12 border-t border-slate-800 pt-8">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-slate-300">
                    Observability History
                  </h3>

                  <p className="mt-1 text-sm text-slate-500">
                    Clear the locally stored execution history.
                  </p>
                </div>

                <button
                  onClick={() =>
                    setShowClearConfirm(true)
                  }
                  disabled={runs.length === 0}
                  className="rounded-xl border border-red-900/70 bg-red-950/20 px-4 py-2.5 text-sm font-medium text-red-400 transition hover:bg-red-950/40 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Clear History
                </button>
              </div>
            </section>
          </>
        )}
      </div>

      {/* Clear Confirmation Modal */}
      {showClearConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-5 backdrop-blur-sm">
          <div className="w-full max-w-md rounded-2xl border border-slate-800 bg-slate-950 p-6 shadow-2xl">
            <h2 className="text-xl font-semibold">
              Clear observability history?
            </h2>

            <p className="mt-3 text-sm leading-6 text-slate-400">
              This will permanently remove the locally stored
              observability records. This action cannot be
              undone.
            </p>

            <div className="mt-6 flex justify-end gap-3">
              <button
                onClick={() =>
                  setShowClearConfirm(false)
                }
                disabled={clearing}
                className="rounded-xl border border-slate-700 px-4 py-2.5 text-sm font-medium text-slate-300 transition hover:bg-slate-900 disabled:opacity-50"
              >
                Cancel
              </button>

              <button
                onClick={handleClearHistory}
                disabled={clearing}
                className="rounded-xl bg-red-600 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-red-500 disabled:cursor-not-allowed disabled:opacity-50"
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

/* -------------------------------------------------------------------------- */
/* Reusable Components                                                        */
/* -------------------------------------------------------------------------- */

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
    <div className="min-h-[122px] rounded-2xl border border-slate-800 bg-slate-950/60 p-6">
      <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
        {label}
      </p>

      <p className="mt-4 text-3xl font-bold tracking-tight text-white">
        {value}
      </p>

      {description && (
        <p className="mt-2 text-xs text-slate-600">
          {description}
        </p>
      )}
    </div>
  );
}

function InfoCard({
  title,
  description,
}: {
  title: string;
  description: string;
}) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-950/40 p-6">
      <h3 className="text-base font-semibold text-slate-200">
        {title}
      </h3>

      <p className="mt-3 text-sm leading-6 text-slate-500">
        {description}
      </p>
    </div>
  );
}

function RouteBadge({
  route,
}: {
  route?: string;
}) {
  const normalized =
    (route || "unknown").toLowerCase();

  const isWeb =
    normalized.includes("web");

  const isLocal =
    normalized.includes("vector") ||
    normalized.includes("local");

  if (isWeb) {
    return (
      <span className="inline-flex rounded-full border border-blue-900/70 bg-blue-950/30 px-3 py-1 text-xs font-medium text-blue-400">
        Web Search
      </span>
    );
  }

  if (isLocal) {
    return (
      <span className="inline-flex rounded-full border border-slate-700 bg-slate-900 px-3 py-1 text-xs font-medium text-slate-300">
        Local RAG
      </span>
    );
  }

  return (
    <span className="inline-flex rounded-full border border-slate-800 bg-slate-950 px-3 py-1 text-xs font-medium text-slate-500">
      {route || "Unknown"}
    </span>
  );
}

function BooleanBadge({
  value,
}: {
  value?: boolean;
}) {
  if (value === true) {
    return (
      <span className="inline-flex rounded-full border border-emerald-900/70 bg-emerald-950/30 px-3 py-1 text-xs font-medium text-emerald-400">
        Yes
      </span>
    );
  }

  if (value === false) {
    return (
      <span className="inline-flex rounded-full border border-red-900/70 bg-red-950/30 px-3 py-1 text-xs font-medium text-red-400">
        No
      </span>
    );
  }

  return (
    <span className="inline-flex rounded-full border border-slate-800 bg-slate-950 px-3 py-1 text-xs font-medium text-slate-500">
      —
    </span>
  );
}

function StatusBadge({
  success,
}: {
  success?: boolean;
}) {
  if (success === true) {
    return (
      <span className="inline-flex rounded-full border border-emerald-900/70 bg-emerald-950/30 px-3 py-1 text-xs font-medium text-emerald-400">
        Success
      </span>
    );
  }

  if (success === false) {
    return (
      <span className="inline-flex rounded-full border border-red-900/70 bg-red-950/30 px-3 py-1 text-xs font-medium text-red-400">
        Failed
      </span>
    );
  }

  return (
    <span className="inline-flex rounded-full border border-amber-900/70 bg-amber-950/30 px-3 py-1 text-xs font-medium text-amber-400">
      Unknown
    </span>
  );
}

function BreakdownCard({
  title,
  items,
}: {
  title: string;
  items: {
    label: string;
    value: number;
  }[];
}) {
  const total = items.reduce(
    (sum, item) => sum + item.value,
    0
  );

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-950/50 p-6">
      <h3 className="text-base font-semibold text-slate-200">
        {title}
      </h3>

      <div className="mt-6 space-y-5">
        {items.map((item) => {
          const percentage =
            total > 0
              ? (item.value / total) * 100
              : 0;

          return (
            <div key={item.label}>
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-400">
                  {item.label}
                </span>

                <span className="font-medium text-slate-200">
                  {item.value}
                </span>
              </div>

              <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-900">
                <div
                  className="h-full rounded-full bg-slate-500 transition-all"
                  style={{
                    width: `${percentage}%`,
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}