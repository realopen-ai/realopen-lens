import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { post, type AutoRefresh, type Collection, type Task } from "./api";

export function FolderActions({
  folder,
  busy,
  watch,
  onChanged,
}: {
  folder: Collection;
  busy: boolean;
  watch: (task: Task) => void;
  onChanged: (folder: Collection, name: string | null) => Promise<void>;
}) {
  const [open, setOpen] = useState(false);
  const [action, setAction] = useState<"rename" | "remove" | null>(null);
  const [name, setName] = useState(folder.name);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [automatic, setAutomatic] = useState(folder.auto_refresh);
  useEffect(() => setAutomatic(folder.auto_refresh), [folder.auto_refresh]);
  async function configureAutomatic(
    enabled: boolean,
    interval = automatic?.interval_seconds || 60,
  ) {
    setPending(true);
    setError("");
    const previous = automatic;
    setAutomatic({
      enabled,
      interval_seconds: interval,
      state: enabled ? "waiting" : "off",
      error: null,
      last_checked_at: null,
      task_id: null,
    });
    try {
      setAutomatic(
        await post<AutoRefresh>("/collections/auto-refresh", {
          collection: folder.name,
          collection_id: folder.id,
          enabled,
          interval_seconds: interval,
        }),
      );
    } catch (e) {
      setAutomatic(previous);
      setError(String(e));
    } finally {
      setPending(false);
    }
  }
  const container = useRef<HTMLDivElement>(null);
  const menu = useRef<HTMLDivElement>(null);
  const dialog = useRef<HTMLElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (event: PointerEvent) => {
      if (
        !container.current?.contains(event.target as Node) &&
        !menu.current?.contains(event.target as Node)
      )
        setOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
        trigger.current?.focus();
      }
    };
    document.addEventListener("pointerdown", close);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", close);
      document.removeEventListener("keydown", escape);
    };
  }, [open]);
  useLayoutEffect(() => {
    if (!open) return;
    const position = () => {
      if (!trigger.current || !menu.current) return;
      const anchor = trigger.current.getBoundingClientRect();
      const popup = menu.current;
      popup.style.maxHeight = `${Math.max(80, window.innerHeight - 24)}px`;
      const bounds = popup.getBoundingClientRect();
      popup.style.left = `${Math.max(12, Math.min(anchor.right - bounds.width, window.innerWidth - bounds.width - 12))}px`;
      popup.style.top = `${Math.max(12, Math.min(anchor.bottom + 6, window.innerHeight - bounds.height - 12))}px`;
    };
    position();
    window.addEventListener("resize", position);
    window.addEventListener("scroll", position, true);
    return () => {
      window.removeEventListener("resize", position);
      window.removeEventListener("scroll", position, true);
    };
  }, [open, automatic?.enabled, error, busy]);
  useEffect(() => {
    if (open) menu.current?.querySelector<HTMLButtonElement>("button")?.focus();
  }, [open]);
  useEffect(() => {
    if (!action) return;
    const items = () =>
      Array.from(
        dialog.current?.querySelectorAll<HTMLElement>(
          "button:not(:disabled), input:not(:disabled)",
        ) || [],
      );
    (
      dialog.current?.querySelector<HTMLInputElement>("input") || items()[0]
    )?.focus();
    const key = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !pending) setAction(null);
      if (event.key === "Tab") {
        const values = items(),
          first = values[0],
          last = values.at(-1);
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last?.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first?.focus();
        }
      }
    };
    document.addEventListener("keydown", key);
    return () => {
      document.removeEventListener("keydown", key);
      trigger.current?.focus();
    };
  }, [action, pending]);
  function choose(value: "rename" | "remove") {
    setOpen(false);
    setError("");
    setName(folder.name);
    setAction(value);
  }
  async function run(kind: "refresh" | "rename" | "remove") {
    setPending(true);
    setError("");
    try {
      const body = { collection: folder.name, collection_id: folder.id };
      if (kind === "refresh") {
        watch(await post<Task>("/collections/refresh", body));
        setOpen(false);
      } else {
        const result = await post<{ name: string | null }>(
          `/collections/${kind}`,
          {
            ...body,
            ...(kind === "rename" ? { name: name.trim() } : {}),
          },
        );
        await onChanged(folder, result.name);
        setAction(null);
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setPending(false);
    }
  }
  return (
    <div className="folder-actions" ref={container}>
      <button
        ref={trigger}
        className="icon-button"
        aria-label={`Manage folder ${folder.name}`}
        aria-expanded={open}
        aria-haspopup="dialog"
        onClick={() => {
          setError("");
          setOpen(!open);
        }}
      >
        <svg
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="currentColor"
          aria-hidden="true"
        >
          <circle cx="5" cy="12" r="1.7" />
          <circle cx="12" cy="12" r="1.7" />
          <circle cx="19" cy="12" r="1.7" />
        </svg>
      </button>
      {open &&
        createPortal(
          <div
            ref={menu}
            className="folder-menu"
            role="dialog"
            aria-label={`Actions for ${folder.name}`}
          >
            <button
              disabled={busy || pending}
              onClick={() => void run("refresh")}
            >
              <ActionIcon kind="refresh" />
              Refresh index
            </button>
            <div className="folder-automatic">
              <label className="inline">
                <input
                  type="checkbox"
                  checked={automatic?.enabled || false}
                  disabled={pending}
                  onChange={(event) =>
                    void configureAutomatic(event.target.checked)
                  }
                />
                <span>Automatic refresh</span>
              </label>
              {automatic?.enabled && (
                <label>
                  Check for changes
                  <select
                    aria-label={`Refresh interval for ${folder.name}`}
                    disabled={pending}
                    value={automatic.interval_seconds}
                    onChange={(event) =>
                      void configureAutomatic(true, Number(event.target.value))
                    }
                  >
                    {![60, 300, 900].includes(automatic.interval_seconds) && (
                      <option value={automatic.interval_seconds}>
                        Every {automatic.interval_seconds} seconds
                      </option>
                    )}
                    <option value={60}>Every minute</option>
                    <option value={300}>Every 5 minutes</option>
                    <option value={900}>Every 15 minutes</option>
                  </select>
                </label>
              )}
              <p className="muted small">
                While Lens is running. Waits for copies to settle.
              </p>
              {automatic?.enabled && (
                <p className="small" role="status">
                  {automatic.state === "settling"
                    ? "Waiting for files to settle…"
                    : automatic.state === "checking"
                      ? "Checking for changes…"
                      : automatic.state === "queued"
                        ? "Refresh queued"
                        : automatic.state === "running"
                          ? "Refreshing index…"
                          : automatic.state === "unavailable"
                            ? "Folder check needs attention"
                            : "Watching for changes"}
                </p>
              )}
              {automatic?.error && (
                <p className="small" role="status">
                  {automatic.error}
                </p>
              )}
            </div>
            <button disabled={busy || pending} onClick={() => choose("rename")}>
              <ActionIcon kind="rename" />
              Rename
            </button>
            <button
              className="folder-remove"
              disabled={busy || pending}
              onClick={() => choose("remove")}
            >
              <ActionIcon kind="remove" />
              Remove from Lens
            </button>
            {busy && (
              <p className="muted small">Wait for active tasks to finish.</p>
            )}
            {error && <p role="alert">{error}</p>}
          </div>,
          document.body,
        )}
      {action && (
        <div className="overlay" onClick={() => !pending && setAction(null)}>
          <section
            ref={dialog}
            className="dialog folder-dialog"
            role="dialog"
            aria-modal="true"
            aria-label={
              action === "rename"
                ? `Rename ${folder.name}`
                : `Remove ${folder.name} from Lens`
            }
            onClick={(event) => event.stopPropagation()}
          >
            <h2>
              {action === "rename"
                ? "Rename folder"
                : "Remove folder from Lens?"}
            </h2>
            <p className="muted folder-source">{folder.root}</p>
            <form
              onSubmit={(event) => {
                event.preventDefault();
                void run(action);
              }}
            >
              {action === "rename" ? (
                <>
                  <label>
                    Folder name
                    <input
                      required
                      maxLength={128}
                      value={name}
                      disabled={pending}
                      onChange={(event) => setName(event.target.value)}
                    />
                  </label>
                  <p className="muted">
                    Changes its name in Lens. Feedback, virtual collections and
                    duplicate reviews stay attached.
                  </p>
                </>
              ) : (
                <>
                  <p>
                    Remove “{folder.name}” and its indexed listing from Lens?
                    Your original files stay untouched.
                  </p>
                  <p className="muted">
                    Shared processing caches and saved history are retained.
                    Adding this folder again creates a new collection.
                  </p>
                </>
              )}
              {error && <p role="alert">{error}</p>}
              <div className="folder-dialog-buttons">
                <button
                  type="button"
                  className="secondary"
                  disabled={pending}
                  onClick={() => setAction(null)}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="primary"
                  disabled={
                    busy || pending || (action === "rename" && !name.trim())
                  }
                >
                  {pending
                    ? "Saving…"
                    : action === "rename"
                      ? "Save name"
                      : "Remove from Lens"}
                </button>
              </div>
            </form>
          </section>
        </div>
      )}
    </div>
  );
}

function ActionIcon({ kind }: { kind: "refresh" | "rename" | "remove" }) {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {kind === "refresh" ? (
        <>
          <path d="M20 7v5h-5" />
          <path d="M4 17v-5h5" />
          <path d="M6.1 7a7 7 0 0 1 11.6-1L20 9M4 15l2.3 3A7 7 0 0 0 18 17" />
        </>
      ) : kind === "rename" ? (
        <>
          <path d="m16 3 5 5-12 12-6 1 1-6Z" />
          <path d="m13 6 5 5" />
        </>
      ) : (
        <>
          <path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7" />
        </>
      )}
    </svg>
  );
}
