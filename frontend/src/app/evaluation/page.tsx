"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ||
  "http://127.0.0.1:8000";

type EvaluationResult = {
  question: string;
  answer: string;
  route: string;
  retrieved_documents: number;
  relevant_documents: number;
  grounded: boolean;
  answers_question: boolean;
  retry_count: number;
  latency_seconds: number;
  sources: Array<Record<string, unknown>>;
  timestamp: string;
  error: string;
  retrieval_relevance_rate: number;
  success: boolean;
};

type EvaluationSummary = {
  total_questions: number;
  successful_questions: number;
  failed_questions: number;
  average_latency_seconds: number;
  grounded_rate: number;
  answer_quality_rate: number;
  average_retries: number;
  average_retrieval_relevance: number;
};

type EvaluationResponse = {
  results: EvaluationResult[];
  summary: EvaluationSummary;
};

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatLatency(value: number) {
  return `${value.toFixed(2)}s`;
}

function formatDate(value: string) {
  if (!value) {
    return "—";
  }

  try {
    return new Date(value).toLocaleString();
  } catch {
    return value;
  }
}

function truncate(
  value: string,
  length = 100
) {
  if (!value) {
    return "—";
  }

  if (value.length <= length) {
    return value;
  }

  return `${value.slice(0, length)}…`;
}

function MetricCard({
  label,
  value,
  description,
}: {
  label: string;
  value: string;
  description: string;
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-5">
      <div className="text-xs font-medium uppercase tracking-wider text-zinc-500">
        {label}
      </div>

      <div className="mt-2 text-2xl font-semibold text-white">
        {value}
      </div>

      <div className="mt-1 text-xs text-zinc-500">
        {description}
      </div>
    </div>
  );
}

function StatusBadge({
  success,
}: {
  success: boolean;
}) {
  return (
    <span
      className={`inline-flex rounded-full px-2.5 py-1 text-xs font-medium ${
        success
          ? "bg-emerald-400/10 text-emerald-300"
          : "bg-red-400/10 text-red-300"
      }`}
    >
      {success ? "Success" : "Failed"}
    </span>
  );
}

function RouteBadge({
  route,
}: {
  route: string;
}) {
  const normalized = route.toLowerCase();

  const isWeb =
    normalized.includes("web");

  return (
    <span
      className={`inline-flex rounded-full px-2.5 py-1 text-xs font-medium ${
        isWeb
          ? "bg-blue-400/10 text-blue-300"
          : "bg-violet-400/10 text-violet-300"
      }`}
    >
      {route || "unknown"}
    </span>
  );
}

export default function EvaluationPage() {
  const [data, setData] =
    useState<EvaluationResponse | null>(null);

  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState("");

  const [clearing, setClearing] =
    useState(false);

  async function loadEvaluation() {
    try {
      setLoading(true);
      setError("");

      const response = await fetch(
        `${API_BASE}/evaluation`,
        {
          cache: "no-store",
        }
      );

      if (!response.ok) {
        throw new Error(
          `Evaluation request failed (${response.status})`
        );
      }

      const json =
        (await response.json()) as EvaluationResponse;

      setData(json);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to load evaluation data."
      );
    } finally {
      setLoading(false);
    }
  }

  async function clearEvaluation() {
    const confirmed = window.confirm(
      "Clear all evaluation history?"
    );

    if (!confirmed) {
      return;
    }

    try {
      setClearing(true);
      setError("");

      const response = await fetch(
        `${API_BASE}/evaluation`,
        {
          method: "DELETE",
        }
      );

      if (!response.ok) {
        throw new Error(
          `Failed to clear evaluation history (${response.status})`
        );
      }

      await loadEvaluation();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to clear evaluation history."
      );
    } finally {
      setClearing(false);
    }
  }

  useEffect(() => {
    loadEvaluation();
  }, []);

  const summary =
    data?.summary || {
      total_questions: 0,
      successful_questions: 0,
      failed_questions: 0,
      average_latency_seconds: 0,
      grounded_rate: 0,
      answer_quality_rate: 0,
      average_retries: 0,
      average_retrieval_relevance: 0,
    };

  return (
    <main className="min-h-screen bg-[#09090b] text-white">
      <div className="mx-auto max-w-7xl px-6 py-8 lg:px-8">
        {/* Header */}
        <div className="flex flex-col gap-5 border-b border-white/10 pb-7 md:flex-row md:items-center md:justify-between">
          <div>
            <Link
              href="/"
              className="text-sm text-zinc-500 transition hover:text-white"
            >
              ← Back to RAG
            </Link>

            <h1 className="mt-3 text-3xl font-semibold tracking-tight">
              Evaluation
            </h1>

            <p className="mt-2 max-w-2xl text-sm leading-6 text-zinc-500">
              Track answer quality, grounding,
              retrieval relevance, retries and
              latency across live RAG requests.
            </p>
          </div>

          <div className="flex gap-3">
            <button
              onClick={loadEvaluation}
              disabled={loading}
              className="rounded-xl border border-white/10 bg-white/[0.05] px-4 py-2.5 text-sm font-medium text-zinc-200 transition hover:bg-white/[0.09] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading
                ? "Refreshing..."
                : "Refresh"}
            </button>

            <button
              onClick={clearEvaluation}
              disabled={
                clearing ||
                summary.total_questions === 0
              }
              className="rounded-xl border border-red-400/20 bg-red-400/5 px-4 py-2.5 text-sm font-medium text-red-300 transition hover:bg-red-400/10 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {clearing
                ? "Clearing..."
                : "Clear history"}
            </button>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="mt-6 rounded-2xl border border-red-400/20 bg-red-400/5 p-4 text-sm text-red-300">
            {error}
          </div>
        )}

        {/* Loading */}
        {loading && !data && (
          <div className="mt-8 rounded-2xl border border-white/10 bg-white/[0.03] p-10 text-center text-sm text-zinc-500">
            Loading evaluation data...
          </div>
        )}

        {/* Metrics */}
        <section className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            label="Total questions"
            value={String(
              summary.total_questions
            )}
            description="Requests evaluated"
          />

          <MetricCard
            label="Successful"
            value={String(
              summary.successful_questions
            )}
            description="Requests without errors"
          />

          <MetricCard
            label="Grounded rate"
            value={formatPercent(
              summary.grounded_rate
            )}
            description="Answers supported by evidence"
          />

          <MetricCard
            label="Answer quality"
            value={formatPercent(
              summary.answer_quality_rate
            )}
            description="Answers passing the quality gate"
          />

          <MetricCard
            label="Retrieval relevance"
            value={formatPercent(
              summary.average_retrieval_relevance
            )}
            description="Relevant retrieved documents"
          />

          <MetricCard
            label="Average latency"
            value={formatLatency(
              summary.average_latency_seconds
            )}
            description="End-to-end request time"
          />

          <MetricCard
            label="Average retries"
            value={summary.average_retries.toFixed(
              2
            )}
            description="Self-correction iterations"
          />

          <MetricCard
            label="Failed"
            value={String(
              summary.failed_questions
            )}
            description="Requests that returned errors"
          />
        </section>

        {/* Evaluation explanation */}
        <section className="mt-8 grid gap-4 md:grid-cols-3">
          <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-5">
            <div className="text-sm font-semibold text-white">
              Groundedness
            </div>

            <p className="mt-2 text-sm leading-6 text-zinc-500">
              Measures whether the generated
              response was judged to be supported
              by the retrieved evidence.
            </p>
          </div>

          <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-5">
            <div className="text-sm font-semibold text-white">
              Answer quality
            </div>

            <p className="mt-2 text-sm leading-6 text-zinc-500">
              Tracks whether the final answer passed
              the question-answer quality gate in
              the agentic workflow.
            </p>
          </div>

          <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-5">
            <div className="text-sm font-semibold text-white">
              Retrieval relevance
            </div>

            <p className="mt-2 text-sm leading-6 text-zinc-500">
              Compares relevant retrieved documents
              against the total number of retrieved
              documents.
            </p>
          </div>
        </section>

        {/* History */}
        <section className="mt-8">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold">
                Evaluation history
              </h2>

              <p className="mt-1 text-sm text-zinc-500">
                Individual live RAG evaluations.
              </p>
            </div>

            <div className="text-xs text-zinc-600">
              {summary.total_questions} records
            </div>
          </div>

          <div className="overflow-hidden rounded-2xl border border-white/10 bg-white/[0.02]">
            {data?.results?.length ? (
              <div className="overflow-x-auto">
                <table className="w-full min-w-[1100px] text-left">
                  <thead className="border-b border-white/10 bg-white/[0.03]">
                    <tr>
                      <th className="px-5 py-4 text-xs font-medium uppercase tracking-wider text-zinc-500">
                        Question
                      </th>

                      <th className="px-5 py-4 text-xs font-medium uppercase tracking-wider text-zinc-500">
                        Route
                      </th>

                      <th className="px-5 py-4 text-xs font-medium uppercase tracking-wider text-zinc-500">
                        Retrieval
                      </th>

                      <th className="px-5 py-4 text-xs font-medium uppercase tracking-wider text-zinc-500">
                        Grounded
                      </th>

                      <th className="px-5 py-4 text-xs font-medium uppercase tracking-wider text-zinc-500">
                        Quality
                      </th>

                      <th className="px-5 py-4 text-xs font-medium uppercase tracking-wider text-zinc-500">
                        Latency
                      </th>

                      <th className="px-5 py-4 text-xs font-medium uppercase tracking-wider text-zinc-500">
                        Retries
                      </th>

                      <th className="px-5 py-4 text-xs font-medium uppercase tracking-wider text-zinc-500">
                        Status
                      </th>

                      <th className="px-5 py-4 text-xs font-medium uppercase tracking-wider text-zinc-500">
                        Time
                      </th>
                    </tr>
                  </thead>

                  <tbody className="divide-y divide-white/5">
                    {data.results
                      .slice()
                      .reverse()
                      .map(
                        (
                          result,
                          index
                        ) => (
                          <tr
                            key={`${result.timestamp}-${index}`}
                            className="transition hover:bg-white/[0.025]"
                          >
                            <td className="max-w-[300px] px-5 py-4">
                              <div
                                className="text-sm font-medium text-zinc-200"
                                title={
                                  result.question
                                }
                              >
                                {truncate(
                                  result.question,
                                  90
                                )}
                              </div>

                              {result.error && (
                                <div className="mt-1 text-xs text-red-400">
                                  {truncate(
                                    result.error,
                                    80
                                  )}
                                </div>
                              )}
                            </td>

                            <td className="px-5 py-4">
                              <RouteBadge
                                route={
                                  result.route
                                }
                              />
                            </td>

                            <td className="px-5 py-4 text-sm text-zinc-400">
                              {result.relevant_documents}
                              /
                              {
                                result.retrieved_documents
                              }

                              <div className="mt-1 text-xs text-zinc-600">
                                {formatPercent(
                                  result.retrieval_relevance_rate
                                )}
                              </div>
                            </td>

                            <td className="px-5 py-4">
                              <span
                                className={`text-sm ${
                                  result.grounded
                                    ? "text-emerald-300"
                                    : "text-red-300"
                                }`}
                              >
                                {result.grounded
                                  ? "Yes"
                                  : "No"}
                              </span>
                            </td>

                            <td className="px-5 py-4">
                              <span
                                className={`text-sm ${
                                  result.answers_question
                                    ? "text-emerald-300"
                                    : "text-red-300"
                                }`}
                              >
                                {result.answers_question
                                  ? "Pass"
                                  : "Fail"}
                              </span>
                            </td>

                            <td className="px-5 py-4 text-sm text-zinc-400">
                              {formatLatency(
                                result.latency_seconds
                              )}
                            </td>

                            <td className="px-5 py-4 text-sm text-zinc-400">
                              {result.retry_count}
                            </td>

                            <td className="px-5 py-4">
                              <StatusBadge
                                success={
                                  result.success
                                }
                              />
                            </td>

                            <td className="whitespace-nowrap px-5 py-4 text-xs text-zinc-600">
                              {formatDate(
                                result.timestamp
                              )}
                            </td>
                          </tr>
                        )
                      )}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="p-12 text-center">
                <div className="text-sm font-medium text-zinc-300">
                  No evaluations yet
                </div>

                <p className="mt-2 text-sm text-zinc-600">
                  Ask a question in the RAG application
                  and the result will appear here.
                </p>

                <Link
                  href="/"
                  className="mt-5 inline-flex rounded-xl border border-white/10 bg-white/[0.05] px-4 py-2.5 text-sm font-medium text-zinc-200 transition hover:bg-white/[0.09]"
                >
                  Ask a question
                </Link>
              </div>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}