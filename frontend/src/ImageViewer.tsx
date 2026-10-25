import { useEffect, useRef, useState } from "react";
import { api, filename, media, post, type ImageResult } from "./api";

interface ViewerGroup {
  id: string;
  name: string;
  collection_id: string;
  revision: number;
  member_image_ids: string[];
}
interface Details {
  image_id: string;
  collection: string;
  collection_id: string;
  generation: string;
  metadata: Record<string, unknown>;
  paths: string[];
  available_paths: string[];
  text: string;
  groups: ViewerGroup[];
}
interface Props {
  image: ImageResult;
  images: ImageResult[];
  collection: string;
  generation: string | null;
  onClose: () => void;
  onNavigate: (image: ImageResult) => void;
  onSimilar: (image: ImageResult) => void;
  onLabel: (image: ImageResult, label: string) => Promise<void>;
  label?: string;
  feedbackEnabled: boolean;
  feedbackBusy: boolean;
  feedbackError?: string;
  onCollectionsChanged: (id: string) => Promise<void>;
}

export function ImageViewer(props: Props) {
  const dialog = useRef<HTMLElement>(null);
  const index = props.images.findIndex(
    (row) => row.image_id === props.image.image_id,
  );
  const callbacks = useRef(props);
  callbacks.current = props;
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    dialog.current?.querySelector<HTMLButtonElement>("button")?.focus();
    return () => {
      document.body.style.overflow = overflow;
      if (previous?.isConnected) previous.focus();
    };
  }, []);
  useEffect(() => {
    const key = (event: KeyboardEvent) => {
      const current = callbacks.current;
      const items = Array.from(
        dialog.current?.querySelectorAll<HTMLElement>(
          'button:not(:disabled), a[href], input, select, textarea, [tabindex="0"]',
        ) || [],
      );
      if (event.key === "Escape") {
        event.preventDefault();
        current.onClose();
      } else if (event.key === "Tab") {
        const first = items[0],
          last = items.at(-1);
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last?.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first?.focus();
        }
      } else if (
        !event.altKey &&
        !event.ctrlKey &&
        !event.metaKey &&
        !(
          event.target instanceof HTMLElement &&
          event.target.closest(
            'input, select, textarea, [contenteditable="true"]',
          )
        ) &&
        !window.getSelection()?.toString()
      ) {
        const position = current.images.findIndex(
          (row) => row.image_id === current.image.image_id,
        );
        const target =
          event.key === "ArrowRight"
            ? position + 1
            : event.key === "ArrowLeft"
              ? position - 1
              : -1;
        if (target >= 0 && target < current.images.length) {
          event.preventDefault();
          current.onNavigate(current.images[target]);
        }
      }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, []);
  return (
    <div className="overlay viewer-overlay" onClick={props.onClose}>
      <section
        className="image-viewer"
        ref={dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby="detail-title"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="viewer-header">
          <button
            className="secondary"
            aria-label="Close details"
            onClick={props.onClose}
          >
            Close <span aria-hidden="true">×</span>
          </button>
          <span className="viewer-position" aria-live="polite">
            {index + 1} / {props.images.length} loaded images
          </span>
          <div className="viewer-navigation">
            <button
              className="secondary"
              aria-label="Previous image"
              disabled={index <= 0}
              onClick={() => props.onNavigate(props.images[index - 1])}
            >
              ←
            </button>
            <button
              className="secondary"
              aria-label="Next image"
              disabled={index < 0 || index >= props.images.length - 1}
              onClick={() => props.onNavigate(props.images[index + 1])}
            >
              →
            </button>
          </div>
        </header>
        <ViewerContent
          key={`${props.image.image_id}:${props.image.sources?.[0]?.generation || props.generation}`}
          {...props}
        />
      </section>
    </div>
  );
}

function ViewerContent(props: Props) {
  const { image, collection, generation } = props;
  const [zoom, setZoom] = useState(0);
  const [details, setDetails] = useState<Details | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [fallback, setFallback] = useState(false);
  const [unavailable, setUnavailable] = useState(false);
  const mounted = useRef(true);
  const source = image.sources?.[0];
  const reference = {
    collection: source?.collection || collection,
    collection_id: source?.collection_id,
    image_id: source?.image_id || image.image_id,
    generation: source?.generation || generation,
  };
  const query = new URLSearchParams(
    Object.entries(reference).filter(([, value]) => value != null) as [
      string,
      string,
    ][],
  ).toString();
  useEffect(() => {
    mounted.current = true;
    const controller = new AbortController();
    api<Details>(`/image-details?${query}`, { signal: controller.signal })
      .then(setDetails)
      .catch((reason) => {
        if (!controller.signal.aborted) setError(String(reason));
      });
    return () => {
      mounted.current = false;
      controller.abort();
    };
  }, [query]);
  async function action(kind: "open" | "reveal") {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await post("/image-actions", { ...reference, action: kind });
      if (mounted.current)
        setNotice(
          kind === "open"
            ? "Opened in your default image app."
            : "Opened the source location.",
        );
    } catch (reason) {
      if (mounted.current) setError(String(reason));
    } finally {
      if (mounted.current) setBusy(false);
    }
  }
  async function membership(group: ViewerGroup) {
    if (!details) return;
    setBusy(true);
    setError("");
    setNotice("");
    const removing = group.member_image_ids.length > 0;
    try {
      await api(`/virtual-collections/${group.id}/members`, {
        method: removing ? "DELETE" : "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          collection: details.collection,
          all_folders: group.collection_id === "library-v1",
          revision: group.revision,
          image_ids: removing ? group.member_image_ids : [reference.image_id],
        }),
      });
      const updated = await api<Details>(`/image-details?${query}`);
      if (mounted.current) {
        setDetails(updated);
        setNotice(`${removing ? "Removed from" : "Added to"} “${group.name}”.`);
      }
      await props.onCollectionsChanged(group.id);
    } catch (reason) {
      if (mounted.current) {
        setError(String(reason));
        // Refresh stale revisions; keep the error visible and require an explicit retry.
        try {
          const updated = await api<Details>(`/image-details?${query}`);
          if (mounted.current) setDetails(updated);
        } catch {
          /* Original error remains visible. */
        }
      }
    } finally {
      if (mounted.current) setBusy(false);
    }
  }
  const zooms = ["Fit", "100%", "150%", "200%"];
  return (
    <div className="viewer-body">
      <div className="viewer-visual">
        <div className="viewer-toolbar" aria-label="Image zoom">
          <button
            className="secondary"
            onClick={() => setZoom(0)}
            aria-pressed={zoom === 0}
          >
            Fit
          </button>
          <button
            className="secondary"
            aria-label="Zoom out"
            disabled={zoom === 0}
            onClick={() => setZoom((value) => value - 1)}
          >
            −
          </button>
          <output aria-live="polite">{zooms[zoom]}</output>
          <button
            className="secondary"
            aria-label="Zoom in"
            disabled={zoom === zooms.length - 1 || unavailable}
            onClick={() => setZoom((value) => value + 1)}
          >
            +
          </button>
        </div>
        <div
          className={`viewer-canvas zoom-${zoom}`}
          tabIndex={0}
          aria-label="Image preview; scroll to pan when zoomed"
        >
          {unavailable ? (
            <p role="status">Image preview is unavailable.</p>
          ) : (
            <img
              className="viewer-picture"
              src={media(
                collection,
                image,
                fallback ? "thumbnail" : "original",
                generation,
              )}
              alt={filename(image)}
              onError={() => {
                if (!fallback) setFallback(true);
                else setUnavailable(true);
              }}
            />
          )}
        </div>
        {fallback && !unavailable && (
          <p className="viewer-fallback" role="status">
            Original unavailable in the browser. Showing the indexed thumbnail.
          </p>
        )}
      </div>
      <aside className="viewer-info">
        <span className="eyebrow">IMAGE DETAILS</span>
        <h2 id="detail-title">{filename(image)}</h2>
        <p className="muted">
          {image.width} × {image.height} · {image.status}
          {details?.metadata.format ? ` · ${details.metadata.format}` : ""}
        </p>
        <p className="viewer-folder">
          Folder: {details?.collection || source?.collection || collection}
        </p>
        <div className="viewer-actions">
          <button
            className="secondary"
            disabled={busy || !details?.available_paths.length}
            onClick={() => void action("open")}
          >
            Open original
          </button>
          <button
            className="secondary"
            disabled={busy || !details?.available_paths.length}
            onClick={() => void action("reveal")}
          >
            Reveal in folder
          </button>
          <button className="secondary" onClick={() => props.onSimilar(image)}>
            Find similar images
          </button>
        </div>
        {props.feedbackError && (
          <p className="issue" role="alert">
            {props.feedbackError}
          </p>
        )}
        {error && (
          <p className="issue" role="alert">
            {error}
          </p>
        )}
        {notice && <p role="status">{notice}</p>}
        {!details && !error && <p role="status">Loading image details…</p>}
        <section aria-label="Image feedback">
          <h3>Search feedback</h3>
          <div className="feedback-actions">
            {(["relevant", "irrelevant"] as const).map((value) => (
              <button
                key={value}
                className={props.label === value ? "chosen" : ""}
                disabled={!props.feedbackEnabled || props.feedbackBusy}
                aria-pressed={props.label === value}
                onClick={() => void props.onLabel(image, value)}
              >
                {value === "relevant" ? "Relevant" : "Irrelevant"}
              </button>
            ))}
          </div>
          {!props.feedbackEnabled && (
            <p className="muted">
              Run a search with “Save for feedback” enabled to rate results.
            </p>
          )}
        </section>
        <section aria-label="Image virtual collections">
          <h3>Virtual collections</h3>
          {details?.groups.map((group) => (
            <label className="viewer-membership" key={group.id}>
              <input
                type="checkbox"
                checked={group.member_image_ids.length > 0}
                disabled={busy}
                onChange={() => void membership(group)}
              />
              <span>
                {group.name}
                <small>
                  {group.collection_id === "library-v1"
                    ? "Across folders"
                    : "This folder"}
                </small>
              </span>
            </label>
          ))}
          {details && !details.groups.length && (
            <p className="muted">
              Create a virtual collection in the sidebar to save this image.
            </p>
          )}
        </section>
        <h3>Source paths</h3>
        {(details?.paths || image.paths).map((path) => (
          <code className="path" key={path}>
            {path}
          </code>
        ))}
        <h3>{details ? "Extracted text" : "Extracted text preview"}</h3>
        <pre className="viewer-text" dir="auto" tabIndex={0}>
          {details?.text ?? (image.text || "No text extracted.")}
          {details && !details.text ? "No text extracted." : ""}
        </pre>
        {image.issues.map((issue, index) => (
          <p className="issue" key={index}>
            {issue.stage}: {issue.error}
          </p>
        ))}
      </aside>
    </div>
  );
}
