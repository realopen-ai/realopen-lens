import { useEffect, useRef, useState } from "react";
import { api, post, terminal, type Task } from "./api";

type Info = {
  directory: string;
  default_directory: string;
  models_directory: string;
  initialized: boolean;
  is_default: boolean;
  workspace_id: string | null;
  build: {
    version: string;
    code_sha256: string;
    gallery_sha256: string;
    package_path: string;
  };
  checkout: { code_matches: boolean; gallery_matches: boolean } | null;
};
type Plan = {
  plan_sha256: string;
  destination: string;
  sources: string[];
  images: number;
  searches: number;
  judgments: number;
  events: number;
  virtual_collections: number;
  duplicate_reviews: number;
  exclusions: string[];
};

export function Workspace({
  active,
  tasks,
  watch,
  onReady,
}: {
  onReady: () => void;
  active: boolean;
  tasks: Task[];
  watch: (task: Task) => void;
}) {
  const announced = useRef<string | null>(null);
  const [info, setInfo] = useState<Info | null>(null);
  const [sources, setSources] = useState("");
  const [destination, setDestination] = useState("");
  const [plan, setPlan] = useState<Plan | null>(null);
  const [operation, setOperation] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [created, setCreated] = useState("");
  const [busy, setBusy] = useState(false);
  const task = tasks.find((t) => t.id === operation);
  const running = busy || Boolean(task && !terminal(task));
  useEffect(() => {
    if (!active) return;
    let valid = true;
    void api<Info>("/workspace")
      .then((value) => {
        if (valid) {
          setInfo(value);
          setDestination((current) => current || value.default_directory);
        }
      })
      .catch((e) => {
        if (valid) setError(String(e));
      });
    return () => {
      valid = false;
    };
  }, [active]);
  useEffect(() => {
    if (!task || !terminal(task)) return;
    if (task.status !== "completed") {
      setError(task.error || `Workspace operation ${task.status}`);
      return;
    }
    if (task.kind === "workspace_preview") {
      if (task.result && "plan_sha256" in task.result)
        setPlan(task.result as unknown as Plan);
      else
        setError(
          "Preview expired after server restart. Preview again before importing.",
        );
    } else if (task.result && "directory" in task.result) {
      const directory = String(task.result.directory);
      setCreated(directory);
      if (directory === info?.directory && announced.current !== task.id) {
        announced.current = task.id;
        onReady();
      }
    }
  }, [task]);
  async function submit(apply = false) {
    setBusy(true);
    setError("");
    try {
      const body =
        apply && plan
          ? {
              sources: plan.sources,
              destination: plan.destination,
              plan_sha256: plan.plan_sha256,
            }
          : {
              sources: sources
                .split("\n")
                .map((s) => s.trim())
                .filter(Boolean),
              destination,
            };
      const value = await post<Task>(
        apply ? "/tasks/workspace-import" : "/tasks/workspace-preview",
        body,
      );
      setOperation(value.id);
      watch(value);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="workspace-panel" aria-label="Workspace settings">
      <h1>Workspace</h1>
      <p className="subtitle">
        Your library, feedback and organization stay together on this device.
      </p>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      {info && (
        <>
          <dl className="workspace-facts">
            <dt>Active workspace</dt>
            <dd>{info.directory}</dd>
            <dt>Storage</dt>
            <dd>{info.is_default ? "Everyday workspace" : "Custom index"}</dd>
            <dt>Models</dt>
            <dd>{info.models_directory}</dd>
            <dt>Application</dt>
            <dd>
              Lens {info.build.version} · build{" "}
              {info.build.code_sha256.slice(0, 12)}
            </dd>
            <dt>Gallery</dt>
            <dd>{info.build.gallery_sha256.slice(0, 12)}</dd>
          </dl>
          {info.checkout &&
            (!info.checkout.code_matches || !info.checkout.gallery_matches) && (
              <p className="error" role="alert">
                The running installation differs from this checkout. Rebuild and
                reinstall, then restart the server.
              </p>
            )}
          <details className="workspace-details">
            <summary>Installation details</summary>
            <p>{info.build.package_path}</p>
            <p>Workspace ID: {info.workspace_id || "Not imported"}</p>
          </details>
        </>
      )}
      <div className="workspace-import">
        <h2>Set up an everyday workspace</h2>
        <p>
          Combine matching saved indexes in a new location. Preview checks
          conflicts before anything is copied. Originals stay where they are.
          Close other sessions editing these sources while importing.
        </p>
        <label>
          Source index directories, one per line
          <textarea
            aria-label="Source index directories"
            value={sources}
            disabled={running}
            onChange={(e) => {
              setSources(e.target.value);
              setPlan(null);
              setCreated("");
            }}
            placeholder="/path/to/existing/index"
            rows={3}
          />
        </label>
        <label>
          New workspace directory
          <input
            aria-label="New workspace directory"
            value={destination}
            disabled={running}
            onChange={(e) => {
              setDestination(e.target.value);
              setPlan(null);
              setCreated("");
            }}
          />
        </label>
        <button
          className="secondary"
          disabled={running || !sources.trim() || !destination.trim()}
          onClick={() => void submit()}
        >
          Preview workspace import
        </button>
        {task && (
          <p role="status">
            {task.phase.replaceAll("_", " ")}
            {task.progress.files
              ? ` · ${task.progress.files} files checked`
              : ""}
          </p>
        )}
        {plan && !created && (
          <section
            className="import-preview"
            aria-label="Workspace import preview"
          >
            <h3>Ready to copy</h3>
            <p>
              {plan.images.toLocaleString()} indexed images · {plan.searches}{" "}
              saved searches · {plan.judgments} current per-search labels ·{" "}
              {plan.duplicate_reviews} duplicate reviews ·{" "}
              {plan.virtual_collections} virtual collections
            </p>
            <p>Destination: {plan.destination}</p>
            <p>
              Model weights, trained model activation and task history stay in
              their original locations. Image originals are referenced, not
              copied.
            </p>
            <button
              className="primary"
              disabled={running}
              onClick={() => void submit(true)}
            >
              Create workspace from this preview
            </button>
          </section>
        )}
        {created && (
          <div className="import-preview" role="status">
            <h3>Workspace created</h3>
            <p>{created}</p>
            <p>
              {created === info?.directory
                ? "This session now uses the imported workspace. Open Image library to browse it."
                : "Stop this server before opening the new workspace. Your current session still uses its original index."}
            </p>
            <code>
              uv run --no-sync lens serve --index-dir {JSON.stringify(created)}
            </code>
          </div>
        )}
      </div>
    </section>
  );
}
