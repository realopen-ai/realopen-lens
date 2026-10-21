import { useEffect, useState } from "react";
import {
  api,
  post,
  media,
  filename,
  terminal,
  type ImageResult,
  type Task,
} from "./api";

type Group = {
  id: string;
  kind: "exact" | "near";
  state: string;
  image_ids: string[];
  results: ImageResult[];
  unavailable: number;
  text_differs?: boolean;
  hamming?: number;
  rgb_mae?: number;
};
type Page = {
  scanned: boolean;
  generation: string;
  total: number;
  groups: Group[];
  summary?: {
    fingerprinted: number;
    images: number;
    errors: unknown[];
    unsupported: unknown[];
  };
};

export function Duplicates({
  collection,
  tasks,
  watch,
}: {
  collection: string;
  tasks: Task[];
  watch: (task: Task) => void;
}) {
  const [open, setOpen] = useState(false);
  const [page, setPage] = useState<Page | null>(null);
  const [offset, setOffset] = useState(0);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const task = tasks.find(
    (t) => t.kind === "duplicates" && t.collection === collection,
  );
  const running = Boolean(task && !terminal(task));
  const indexing = tasks.find(
    (t) => t.kind === "index" && t.collection === collection,
  );
  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    setPage(null);
    setOffset(0);
    setError("");
  }, [collection]);
  useEffect(() => {
    if (!open || !collection) return;
    let alive = true;
    api<Page>(
      `/duplicates?collection=${encodeURIComponent(collection)}&offset=${offset}&limit=12`,
    )
      .then((value) => {
        if (alive) {
          setPage(value);
          setError("");
        }
      })
      .catch((e) => {
        if (alive) setError(String(e));
      });
    return () => {
      alive = false;
    };
  }, [collection, open, offset, refresh, task?.status, indexing?.status]);
  async function scan() {
    setOffset(0);
    setBusy(true);
    setError("");
    try {
      watch(await post<Task>("/tasks/duplicates", { collection }));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  async function review(group: Group, state: string) {
    setBusy(true);
    setError("");
    try {
      await post(`/duplicates/${group.id}/review`, { collection, state });
      setRefresh((v) => v + 1);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <details
      className="training-panel duplicates-panel"
      open={open}
      onToggle={(e) => setOpen(e.currentTarget.open)}
    >
      <summary>
        <strong>Duplicate groups</strong>
        <span className="training-summary-note">Local review</span>
      </summary>
      {open && (
        <div className="training-body">
          <p className="training-intro">
            Scan this source folder for identical pixels and conservative
            near-duplicate suggestions. Originals are never moved or deleted.
            Near matches stay separate in search, even after confirmation.
          </p>
          <div className="duplicate-actions">
            <button
              className="secondary"
              disabled={busy || running || !collection}
              onClick={() => void scan()}
            >
              {running
                ? "Scanning duplicates…"
                : page?.scanned
                  ? "Scan again"
                  : "Scan duplicates"}
            </button>
            <button
              className="text-button"
              disabled={busy}
              onClick={() => setRefresh((v) => v + 1)}
            >
              Refresh groups
            </button>
          </div>
          {running && (
            <p role="status">
              {task?.phase.replaceAll("_", " ")} · {task?.progress.step || 0}/
              {task?.progress.max_steps || 0}
            </p>
          )}
          {task?.status === "failed" && (
            <p role="alert" className="training-error">
              {task.error}
            </p>
          )}
          {error && (
            <p role="alert" className="training-error">
              {error}
            </p>
          )}
          {page && !page.scanned && (
            <p className="training-empty">
              No scan for the current index snapshot. Scan after indexing or
              refreshing this folder to enable exact-pixel collapse in new
              searches.
            </p>
          )}
          {page?.scanned && (
            <>
              <p className="training-active">
                {page.total} groups · {page.summary?.fingerprinted}/
                {page.summary?.images} images fingerprinted ·{" "}
                {page.summary?.errors.length || 0} errors ·{" "}
                {page.summary?.unsupported.length || 0} unsupported images
              </p>
              <p className="muted small">
                Search again to see exact duplicates combined. Folder browsing
                retains each file identity for organization. Selecting a
                combined search result selects all its available exact copies.
              </p>
              {!page.total && (
                <p>No duplicate groups found with the current policy.</p>
              )}
              <div className="duplicate-groups">
                {page.groups.map((group) => (
                  <article className="duplicate-group" key={group.id}>
                    <div className="duplicate-group-heading">
                      <strong>
                        {group.kind === "exact"
                          ? "Identical pixels"
                          : "Possible near-duplicates"}
                      </strong>
                      <span className="pill">{group.state}</span>
                    </div>
                    {group.kind === "near" && (
                      <p className="muted small">
                        Check for changed text, crops and edits before
                        confirming. Detection does not establish equivalent
                        content.
                      </p>
                    )}
                    {group.text_differs && (
                      <p className="organization-notice">
                        OCR text differs between these images. Review the
                        originals for meaningful changes.
                      </p>
                    )}
                    <div className="duplicate-images">
                      {group.results.map((image) => (
                        <figure key={image.image_id}>
                          <a
                            href={media(
                              collection,
                              image,
                              "original",
                              page.generation,
                            )}
                            target="_blank"
                            rel="noreferrer"
                            aria-label={`Open original ${filename(image)}`}
                          >
                            <img
                              src={media(
                                collection,
                                image,
                                "thumbnail",
                                page.generation,
                              )}
                              alt={filename(image)}
                              loading="lazy"
                            />
                          </a>
                          <figcaption>
                            {filename(image)} · {image.width} × {image.height}
                          </figcaption>
                          <details>
                            <summary>{image.paths.length} source paths</summary>
                            {image.paths.map((path) => (
                              <p key={path}>{path}</p>
                            ))}
                          </details>
                        </figure>
                      ))}
                    </div>
                    {group.unavailable > 0 && (
                      <p>
                        {group.unavailable} unavailable members; refresh the
                        source index and scan again.
                      </p>
                    )}
                    {group.kind === "near" && (
                      <div className="duplicate-actions">
                        <button
                          className="secondary"
                          disabled={
                            busy || running || group.state === "confirmed"
                          }
                          onClick={() => void review(group, "confirmed")}
                        >
                          Confirm similarity
                        </button>
                        <button
                          className="secondary"
                          disabled={
                            busy || running || group.state === "dismissed"
                          }
                          onClick={() => void review(group, "dismissed")}
                        >
                          Dismiss suggestion
                        </button>
                        {group.state !== "unreviewed" && (
                          <button
                            className="text-button"
                            disabled={busy || running}
                            onClick={() => void review(group, "unreviewed")}
                          >
                            Reset review
                          </button>
                        )}
                      </div>
                    )}
                  </article>
                ))}
              </div>
              {page.total > 12 && (
                <div className="duplicate-actions">
                  <button
                    className="secondary"
                    disabled={!offset || busy}
                    onClick={() => setOffset((v) => Math.max(0, v - 12))}
                  >
                    Previous groups
                  </button>
                  <span>
                    {offset + 1}–{Math.min(offset + 12, page.total)} of{" "}
                    {page.total}
                  </span>
                  <button
                    className="secondary"
                    disabled={offset + 12 >= page.total || busy}
                    onClick={() => setOffset((v) => v + 12)}
                  >
                    Next groups
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </details>
  );
}
