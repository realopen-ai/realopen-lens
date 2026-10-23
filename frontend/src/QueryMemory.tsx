import { useState } from "react";
import {
  api,
  media,
  filename,
  type ImageResult,
  type MemoryDecision,
} from "./api";

export function QueryMemory({
  collection,
  query,
  scope,
  revision,
  enabled,
  allFolders = false,
}: {
  collection: string;
  query: string;
  scope?: string;
  revision?: number;
  enabled: boolean;
  allFolders?: boolean;
}) {
  const [page, setPage] = useState<{
    results: ImageResult[];
    total: number;
    generation: string;
  } | null>(null);
  const [offset, setOffset] = useState(0);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  async function load(start = 0) {
    setBusy(true);
    setError("");
    const params = new URLSearchParams({
      collection,
      query,
      offset: String(start),
      limit: "12",
    });
    if (scope) params.set("virtual_collection", scope);
    if (revision) params.set("virtual_revision", String(revision));
    try {
      setPage(
        await api(
          `${allFolders ? "/library/hidden" : "/feedback/hidden"}?${params}`,
        ),
      );
      setOffset(start);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  async function restore(image: ImageResult) {
    const decision = image.feedback;
    if (!decision) return;
    setBusy(true);
    setError("");
    try {
      await api<{ memory: MemoryDecision }>(
        `/feedback/runs/${decision.run_id}/${decision.image_id}?expected_event_id=${decision.event_id}`,
        { method: "DELETE" },
      );
      setMessage("Judgment cleared. Run the search again to refresh results.");
      await load(offset);
    } catch (e) {
      setError(String(e));
      setBusy(false);
    }
  }
  return (
    <section className="query-memory" aria-label="Remembered query feedback">
      <div className="memory-heading">
        <span>
          {enabled
            ? "Relevant images come first; irrelevant images are hidden for this exact query and search scope."
            : "Query feedback is disabled for these results. You can still review hidden judgments."}
        </span>
        <button
          className="secondary"
          disabled={busy}
          onClick={() => (page ? setPage(null) : void load())}
        >
          {page ? "Close hidden images" : "Show hidden images"}
        </button>
      </div>
      {message && <p role="status">{message}</p>}
      {error && <p role="alert">{error}</p>}
      {page && (
        <>
          <p>{page.total} hidden images</p>
          <div className="hidden-images">
            {page.results.map((image) => (
              <article key={image.image_id}>
                <img
                  src={media(collection, image, "thumbnail", page.generation)}
                  alt={filename(image)}
                />
                <span>{filename(image)}</span>
                <button
                  className="secondary"
                  disabled={busy}
                  onClick={() => void restore(image)}
                >
                  Clear judgment
                </button>
              </article>
            ))}
          </div>
          <div className="memory-heading">
            <button
              className="secondary"
              disabled={busy || offset === 0}
              onClick={() => void load(Math.max(0, offset - 12))}
            >
              Previous hidden images
            </button>
            <button
              className="secondary"
              disabled={busy || offset + 12 >= page.total}
              onClick={() => void load(offset + 12)}
            >
              Next hidden images
            </button>
          </div>
        </>
      )}
    </section>
  );
}
