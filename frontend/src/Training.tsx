import { useEffect, useState } from "react";
import { api, post, terminal, type Task } from "./api";

type ModelVersion = {
  id: string;
  status: string;
  selected_l2?: number;
  evaluation?: {
    failed: number;
    independent_holdout: boolean;
    regressions: string[];
    hybrid?: Record<string, number>;
    logistic?: Record<string, number>;
  };
};
type Models = {
  versions: ModelVersion[];
  active: Record<string, string>;
  datasets?: { id: string; queries: number }[];
};

export function Training({
  collection,
  tasks,
  watch,
}: {
  collection: string;
  tasks: Task[];
  watch: (task: Task) => void;
}) {
  const [open, setOpen] = useState(false);
  const [models, setModels] = useState<Models>({ versions: [], active: {} });
  const [review, setReview] = useState("");
  const [dataset, setDataset] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [acknowledge, setAcknowledge] = useState(false);
  const completed = tasks
    .filter(
      (t) =>
        (t.kind === "train" || t.kind === "prepare_training") && terminal(t),
    )
    .map((t) => t.id)
    .join(",");
  async function refresh() {
    setModels(await api<Models>("/training/models"));
  }
  useEffect(() => {
    if (open) void refresh().catch((e) => setError(String(e)));
  }, [open, completed]);
  async function act(operation: () => Promise<void>) {
    setBusy(true);
    setError("");
    try {
      await operation();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  async function snapshot() {
    const value = await post<{ pairs: number; review_template: unknown }>(
      "/training/snapshot",
      {},
    );
    if (!value.pairs)
      throw new Error(
        "No labels yet. Save searches and mark results relevant or irrelevant first.",
      );
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(value.review_template, null, 2)], {
        type: "application/json",
      }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = "feedback-review.json";
    link.click();
    URL.revokeObjectURL(url);
  }
  async function prepare() {
    const task = await post<Task>("/tasks/prepare-training", {
      collection,
      review: JSON.parse(review),
    });
    watch(task);
  }
  async function activate(version: string | null, rollback = false) {
    await post("/training/activation", {
      collection,
      model_version: version,
      rollback,
      acknowledge_experimental: acknowledge,
    });
    await refresh();
  }

  return (
    <details
      className="training-panel"
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary>
        <span className="training-mark" aria-hidden="true">
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
          >
            <path d="m12 3 8 4-8 4-8-4 8-4Z" />
            <path d="m4 12 8 4 8-4M4 17l8 4 8-4" />
          </svg>
        </span>
        <span>Learning &amp; models</span>
        <span className="training-summary-note">Local training</span>
        <svg
          className="training-chevron"
          aria-hidden="true"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
        >
          <path d="m8 10 4 4 4-4" />
        </svg>
      </summary>
      {open && (
        <div className="training-body">
          <p className="training-intro">
            Train from explicitly reviewed feedback. Training never changes the
            active search model automatically.
          </p>
          <p className="training-active">
            <span>Active ranking for {collection || "this collection"}</span>
            <strong>
              {models.active[collection]
                ? `Logistic ${models.active[collection].slice(0, 8)}`
                : "Hybrid"}
            </strong>
          </p>
          {error && (
            <p className="training-error" role="alert">
              {error}
            </p>
          )}
          <div className="training-steps">
            <section className="training-step">
              <h3>
                <span>01</span> Review your feedback
              </h3>
              <button
                className="secondary"
                disabled={busy}
                onClick={() => void act(snapshot)}
              >
                Download review template
              </button>
              <p>
                Review query families and source/duplicate groups, assign
                separate train, validation and test sets, and identify the
                reviewer. Only attest an independent holdout when it has not
                informed earlier tuning.
              </p>
              <label>
                Reviewed feedback file{" "}
                <input
                  type="file"
                  accept=".json,application/json"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) void file.text().then(setReview);
                  }}
                />
              </label>
              <button
                className="secondary"
                disabled={busy || !review || !collection}
                onClick={() => void act(prepare)}
              >
                Prepare reviewed dataset
              </button>
            </section>
            <section className="training-step">
              <h3>
                <span>02</span> Train &amp; evaluate
              </h3>
              <p>
                Choose a reviewed dataset to compare a new model with hybrid
                search.
              </p>
              <label>
                Prepared dataset{" "}
                <select
                  value={dataset}
                  onChange={(event) => setDataset(event.target.value)}
                >
                  <option value="">Select a prepared dataset</option>
                  {(models.datasets || []).map((value) => (
                    <option key={value.id} value={value.id}>
                      {value.id.slice(0, 8)} · {value.queries} queries
                    </option>
                  ))}
                </select>
              </label>
              <button
                className="primary"
                disabled={busy || !dataset || !collection}
                onClick={() =>
                  void act(async () => {
                    watch(
                      await post<Task>("/tasks/train", {
                        collection,
                        dataset_id: dataset,
                      }),
                    );
                  })
                }
              >
                Train and evaluate logistic model
              </button>
            </section>
          </div>
          <div className="training-models-heading">
            <h3>Model versions</h3>
            <span>Activation is always your choice</span>
          </div>
          <label className="checkbox training-consent">
            <input
              type="checkbox"
              checked={acknowledge}
              onChange={(event) => setAcknowledge(event.target.checked)}
            />
            Allow experimental activation with diagnostic judgments or measured
            regressions
          </label>
          {models.active[collection] && (
            <button
              className="secondary"
              disabled={busy}
              onClick={() => void act(() => activate(null))}
            >
              Restore hybrid ranking
            </button>
          )}
          {!models.versions.length && (
            <p className="training-empty">
              Your trained models will appear here, along with evaluation
              results.
            </p>
          )}
          <div className="training-model-grid">
            {models.versions.map((version) => (
              <article className="training-model" key={version.id}>
                <div className="training-model-heading">
                  <strong>{version.id.slice(0, 8)}</strong>
                  <span className="training-status">
                    {version.status.replaceAll("_", " ")}
                  </span>
                </div>
                {version.evaluation && (
                  <p>
                    {version.evaluation.independent_holdout
                      ? "Reviewer-attested independent holdout"
                      : "Diagnostic judgments"}
                    {" · "}
                    {version.evaluation.failed} failed queries ·{" "}
                    {version.evaluation.regressions.length} query regressions.{" "}
                    Known-judgment metrics; unlabeled images remain unknown.
                    {version.evaluation.logistic && (
                      <>
                        {" "}
                        MRR@10:{" "}
                        {version.evaluation.hybrid?.mrr_at_10.toFixed(3)} →{" "}
                        {version.evaluation.logistic.mrr_at_10.toFixed(3)};
                        nDCG@10:{" "}
                        {version.evaluation.hybrid?.ndcg_at_10.toFixed(3)} →{" "}
                        {version.evaluation.logistic.ndcg_at_10.toFixed(3)}.
                      </>
                    )}
                  </p>
                )}
                <div className="training-model-actions">
                  <button
                    className="primary"
                    disabled={
                      busy ||
                      version.status !== "completed" ||
                      !collection ||
                      models.active[collection] === version.id
                    }
                    onClick={() => void act(() => activate(version.id))}
                  >
                    Activate {version.id.slice(0, 8)}
                  </button>
                  <button
                    className="secondary"
                    disabled={
                      busy ||
                      version.status !== "completed" ||
                      !collection ||
                      models.active[collection] === version.id
                    }
                    onClick={() => void act(() => activate(version.id, true))}
                  >
                    Roll back to {version.id.slice(0, 8)}
                  </button>
                </div>
              </article>
            ))}
          </div>
        </div>
      )}
    </details>
  );
}
