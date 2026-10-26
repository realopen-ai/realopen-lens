import { useEffect, useState } from "react";
import { post, terminal, type Task, type SetupReport } from "./api";

export function Setup({
  tasks,
  watch,
  precision,
  languages,
  source,
  onLanguages,
}: {
  tasks: Task[];
  watch: (task: Task) => void;
  precision: string;
  languages: string;
  source?: string;
  onLanguages: (languages: string) => void;
}) {
  const [taskId, setTaskId] = useState("");
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const task = tasks.find((value) => value.id === taskId);
  const busy = pending || Boolean(task && !terminal(task));
  const report = task?.result?.checks ? (task.result as SetupReport) : null;
  const stale =
    report &&
    (report.siglip_precision !== precision ||
      (languages && report.languages !== languages) ||
      report.source !== (source || null));
  const needsAttention = report?.ready === false && !stale;
  useEffect(() => {
    if (error || task?.error || needsAttention) setExpanded(true);
  }, [error, task?.id, task?.revision, task?.error, needsAttention]);
  async function check() {
    setPending(true);
    setError("");
    try {
      const next = await post<Task>("/tasks/setup", {
        collection: "workspace",
        siglip_precision: precision,
        languages,
        source: source || null,
      });
      setTaskId(next.id);
      watch(next);
    } catch (e) {
      setError(String(e));
    } finally {
      setPending(false);
    }
  }
  const installed = report?.ocr?.installed_languages || [];
  const suggested =
    ["eng", "fra", "ara"]
      .filter((code) => installed.includes(code))
      .join("+") ||
    installed[0] ||
    "";
  return (
    <section className="setup-card" aria-label="Device setup">
      <details
        open={expanded}
        onToggle={(event) => setExpanded(event.currentTarget.open)}
      >
        <summary className="setup-summary">
          <span className="setup-title">Device setup</span>
          <span className="muted setup-description">
            Check local models, text recognition and folder access. Existing
            images remain browsable.
          </span>
        </summary>
        <div className="setup-content">
          <button
            type="button"
            className="secondary"
            disabled={busy}
            onClick={() => void check()}
          >
            {busy ? "Checking setup…" : "Check setup"}
          </button>
          <p className="small muted">
            SigLIP2 {precision.toUpperCase()} · E5 INT8 · OCR{" "}
            {languages === "auto"
              ? "automatic script groups"
              : languages || "installed languages"}
            . This check stays offline.
          </p>
          {busy && <p role="status">{task?.phase || "Queued"}</p>}
          {(error || task?.error) && <p role="alert">{error || task?.error}</p>}
          {report && (
            <>
              <p role="status">
                {stale
                  ? "Settings changed. Check setup again."
                  : report.ready
                    ? "Ready to index images."
                    : "Setup needs attention before indexing."}
              </p>
              <ul className="setup-checks">
                {report.checks.map((check) => (
                  <li key={check.id}>
                    <strong>
                      {check.ok ? "✓" : "!"} {check.title}
                    </strong>
                    <p>{check.detail}</p>
                    {!check.ok && <p className="muted">{check.action}</p>}
                  </li>
                ))}
              </ul>
              <p className="small">
                Installed OCR languages:{" "}
                {installed.join(", ") || "None detected"}
              </p>
              {suggested && suggested !== languages && languages !== "auto" && (
                <button
                  type="button"
                  className="secondary"
                  onClick={() => onLanguages(suggested)}
                >
                  Use installed languages ({suggested})
                </button>
              )}
              <p className="small muted">
                Download missing models from the desktop launcher, or run{" "}
                <code>lens models fetch --siglip-precision {precision}</code>.
              </p>
              <a
                href="https://tesseract-ocr.github.io/tessdoc/Installation.html"
                target="_blank"
                rel="noreferrer"
              >
                Tesseract installation and language packs ↗
              </a>
            </>
          )}
        </div>
      </details>
    </section>
  );
}
