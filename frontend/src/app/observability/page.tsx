"use client";

import { useEffect, useState } from "react";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  "http://127.0.0.1:8000";

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
};

type Run = {
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
};

export default function ObservabilityPage() {
  const [summary, setSummary] =
    useState<Summary | null>(null);

  const [runs, setRuns] = useState<Run[]>([]);

  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState("");

  const [showClearConfirmation, setShowClearConfirmation] =
    useState(false);

  const [clearing, setClearing] =
    useState(false);

  async function loadData() {
    setLoading(true);
    setError("");

    try {
      const [
        summaryResponse,
        runsResponse,
      ] = await Promise.all([
        fetch(
          `${API_URL}/observability/summary`
        ),
        fetch(
          `${API_URL}/observability/runs?limit=50`
        ),
      ]);

      if (
        !summaryResponse.ok ||
        !runsResponse.ok
      ) {
        throw new Error(
          "Failed to load observability data."
        );
      }

      const summaryData =
        await summaryResponse.json();

      const runsData =
        await runsResponse.json();

      setSummary(summaryData);
      setRuns(runsData.runs || []);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to load observability data."
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadData();
  }, []);

  async function clearHistory() {
    setClearing(true);
    setError("");

    try {
      const response = await fetch(
        `${API_URL}/observability`,
        {
          method: "DELETE",
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Failed to clear observability history."
        );
      }

      setShowClearConfirmation(false);

      await loadData();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to clear observability history."
      );
    } finally {
      setClearing(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-7xl px-6 py-10">

        <div className="flex flex-col justify-between gap-4 md:flex-row md:items-center">
          <div>
            <p className="text-sm font-medium text-blue-400">
              Agentic Adaptive RAG
            </p>

            <h1 className="mt-2 text-4xl font-bold">
              Observability
            </h1>

            <p className="mt-3 text-slate-400">
              Operational metrics and recent agent executions.
            </p>
          </div>

          <a
            href="/"
            className="rounded-lg border border-slate-700 px-4 py-2 text-sm text-slate-300 hover:bg-slate-900"
          >
            ← Back to RAG
          </a>
        </div>

        {error && (
          <div className="mt-6 rounded-xl border border-red-800 bg-red-950/40 px-4 py-3 text-red-300">
            {error}
          </div>
        )}

        {loading ? (
          <div className="mt-10 text-slate-500">
            Loading observability data...
          </div>
        ) : (
          <>
            {summary && (
              <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">

                <MetricCard
                  label="Total Runs"
                  value={summary.total_runs}
                />

                <MetricCard
                  label="Success Rate"
                  value={`${(
                    summary.success_rate * 100
                  ).toFixed(1)}%`}
                />

                <MetricCard
                  label="Avg Latency"
                  value={`${summary.average_latency_seconds.toFixed(
                    2
                  )}s`}
                />

                <MetricCard
                  label="Avg Retries"
                  value={summary.average_retries.toFixed(
                    2
                  )}
                />

                <MetricCard
                  label="Grounded Rate"
                  value={`${(
                    summary.grounded_rate * 100
                  ).toFixed(1)}%`}
                />

                <MetricCard
                  label="Answer Quality"
                  value={`${(
                    summary.answer_quality_rate * 100
                  ).toFixed(1)}%`}
                />

                <MetricCard
                  label="Local Routes"
                  value={summary.local_route_count}
                />

                <MetricCard
                  label="Web Routes"
                  value={summary.web_route_count}
                />
              </div>
            )}

            <section className="mt-8 rounded-2xl border border-slate-800 bg-slate-900 p-6">

              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
  <div>
    <h2 className="text-xl font-semibold">
      Recent Runs
    </h2>

    <p className="mt-1 text-sm text-slate-500">
      Latest Agentic RAG executions.
    </p>
  </div>

  <div className="flex gap-2">
    <button
      onClick={loadData}
      disabled={loading || clearing}
      className="rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-300 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
    >
      Refresh
    </button>

    <button
      onClick={() => setShowClearConfirmation(true)}
      disabled={
        loading ||
        clearing ||
        runs.length === 0
      }
      className="rounded-lg border border-red-900 px-3 py-2 text-sm text-red-400 hover:bg-red-950 disabled:cursor-not-allowed disabled:opacity-50"
    >
      Clear History
    </button>
  </div>
</div>

              <div className="mt-6 overflow-x-auto">
                <table className="w-full min-w-[1000px] text-left text-sm">
                  <thead>
                    <tr className="border-b border-slate-800 text-xs uppercase tracking-wide text-slate-500">
                      <th className="px-3 py-3">
                        Question
                      </th>

                      <th className="px-3 py-3">
                        Route
                      </th>

                      <th className="px-3 py-3">
                        Retrieved
                      </th>

                      <th className="px-3 py-3">
                        Relevant
                      </th>

                      <th className="px-3 py-3">
                        Grounded
                      </th>

                      <th className="px-3 py-3">
                        Answer
                      </th>

                      <th className="px-3 py-3">
                        Retries
                      </th>

                      <th className="px-3 py-3">
                        Latency
                      </th>

                      <th className="px-3 py-3">
                        Status
                      </th>
                    </tr>
                  </thead>

                  <tbody>
                    {runs.length === 0 ? (
                      <tr>
                        <td
                          colSpan={9}
                          className="px-3 py-8 text-center text-slate-600"
                        >
                          No runs recorded yet.
                        </td>
                      </tr>
                    ) : (
                      runs.map((run, index) => (
                        <tr
                          key={`${run.timestamp}-${index}`}
                          className="border-b border-slate-800/70"
                        >
                          <td className="max-w-[320px] px-3 py-4">
                            <div className="truncate text-slate-300">
                              {run.question}
                            </div>

                            <div className="mt-1 text-xs text-slate-600">
                              {formatTimestamp(
                                run.timestamp
                              )}
                            </div>
                          </td>

                          <td className="px-3 py-4">
                            <Badge>
                              {run.route}
                            </Badge>
                          </td>

                          <td className="px-3 py-4 text-slate-400">
                            {run.retrieved_documents}
                          </td>

                          <td className="px-3 py-4 text-slate-400">
                            {run.relevant_documents}
                          </td>

                          <td className="px-3 py-4">
                            <Status
                              value={run.grounded}
                            />
                          </td>

                          <td className="px-3 py-4">
                            <Status
                              value={
                                run.answers_question
                              }
                            />
                          </td>

                          <td className="px-3 py-4 text-slate-400">
                            {run.retry_count}
                          </td>

                          <td className="px-3 py-4 text-slate-400">
                            {run.latency_seconds.toFixed(
                              2
                            )}
                            s
                          </td>

                          <td className="px-3 py-4">
                            <span
                              className={
                                run.success
                                  ? "text-emerald-400"
                                  : "text-red-400"
                              }
                            >
                              {run.success
                                ? "Success"
                                : "Failed"}
                            </span>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        )}
        {showClearConfirmation && (
  <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4">
    <div className="w-full max-w-md rounded-2xl border border-slate-700 bg-slate-900 p-6 shadow-2xl">

      <h2 className="text-xl font-semibold text-slate-100">
        Clear observability history?
      </h2>

      <p className="mt-3 text-sm leading-6 text-slate-400">
        This will permanently remove all recorded RAG
        execution history. Your documents, vector database,
        and evaluation history will not be affected.
      </p>

      <div className="mt-6 flex justify-end gap-3">

        <button
          onClick={() =>
            setShowClearConfirmation(false)
          }
          disabled={clearing}
          className="rounded-lg border border-slate-700 px-4 py-2 text-sm text-slate-300 hover:bg-slate-800 disabled:opacity-50"
        >
          Cancel
        </button>

        <button
          onClick={clearHistory}
          disabled={clearing}
          className="rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-500 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {clearing
            ? "Clearing..."
            : "Clear History"}
        </button>

      </div>
    </div>
  </div>
)}
      </div>
    </main>
  );
}

function MetricCard({
  label,
  value,
}: {
  label: string;
  value: string | number;
}) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5">
      <p className="text-xs uppercase tracking-wide text-slate-500">
        {label}
      </p>

      <p className="mt-2 text-2xl font-bold text-slate-200">
        {value}
      </p>
    </div>
  );
}

function Badge({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <span className="rounded-full bg-slate-800 px-2.5 py-1 text-xs text-slate-300">
      {children}
    </span>
  );
}

function Status({
  value,
}: {
  value: boolean;
}) {
  return (
    <span
      className={
        value
          ? "text-emerald-400"
          : "text-red-400"
      }
    >
      {value ? "Yes" : "No"}
    </span>
  );
}

function formatTimestamp(
  timestamp: string
) {
  try {
    return new Date(
      timestamp
    ).toLocaleString();
  } catch {
    return timestamp;
  }
}