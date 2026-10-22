import { useEffect, useRef, useState } from "react";
import { post, type Collection, type Task } from "./api";

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
  const container = useRef<HTMLDivElement>(null);
  const dialog = useRef<HTMLElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (event: PointerEvent) => {
      if (!container.current?.contains(event.target as Node)) setOpen(false);
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
        aria-haspopup="true"
        onClick={() => {
          setError("");
          setOpen(!open);
        }}
      >
        ⋯
      </button>
      {open && (
        <div className="folder-menu" aria-label={`Actions for ${folder.name}`}>
          <button
            disabled={busy || pending}
            onClick={() => void run("refresh")}
          >
            Refresh index
          </button>
          <button disabled={busy || pending} onClick={() => choose("rename")}>
            Rename
          </button>
          <button disabled={busy || pending} onClick={() => choose("remove")}>
            Remove from Lens
          </button>
          {busy && (
            <p className="muted small">Wait for active tasks to finish.</p>
          )}
          {error && <p role="alert">{error}</p>}
        </div>
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
