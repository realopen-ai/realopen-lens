import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { api, post } from "./api";

export interface VirtualCollection {
  id: string;
  name: string;
  collection_id: string;
  revision: number;
  member_count: number;
  available_count: number;
  unavailable_count: number;
}

export function VirtualCollections({
  collection,
  groups,
  activeId,
  creating,
  setCreating,
  onBrowse,
  onChanged,
  onCreated,
}: {
  collection: string;
  groups: VirtualCollection[];
  activeId: string | null;
  creating: boolean;
  setCreating: (value: boolean) => void;
  onBrowse: (group: VirtualCollection) => void;
  onChanged: (id: string) => Promise<void>;
  onCreated: (id: string) => void;
}) {
  const [editing, setEditing] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const dialog = useRef<HTMLDivElement>(null);
  const group = groups.find((g) => g.id === editing);
  const visible = creating || Boolean(editing);
  function close() {
    setCreating(false);
    setEditing(null);
    setDeleting(false);
    setError("");
  }
  useEffect(() => {
    if (!visible) return;
    const previous = document.activeElement as HTMLElement | null;
    const elements = () =>
      Array.from(
        dialog.current?.querySelectorAll<HTMLElement>(
          "button:not(:disabled), input",
        ) || [],
      );
    (
      elements().find((element) => element.tagName === "INPUT") || elements()[0]
    )?.focus();
    const key = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busy) close();
      if (event.key === "Tab") {
        const items = elements();
        if (!items.length) {
          event.preventDefault();
          return;
        }
        const first = items[0],
          last = items.at(-1);
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last?.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first?.focus();
        }
      }
    };
    window.addEventListener("keydown", key);
    return () => {
      window.removeEventListener("keydown", key);
      previous?.focus();
    };
  }, [visible, busy, deleting]);
  useEffect(() => {
    if (creating) {
      setName("");
      setError("");
    }
  }, [creating]);
  async function save(remove = false) {
    if (!collection || (editing && !group)) {
      setError("Collection changed; close and refresh the list.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      let id: string;
      if (creating) {
        const created = await post<{ id: string }>("/virtual-collections", {
          collection,
          name,
        });
        id = created.id;
      } else {
        id = group!.id;
        await api(`/virtual-collections/${id}`, {
          method: remove ? "DELETE" : "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            collection,
            revision: group!.revision,
            ...(remove ? {} : { name }),
          }),
        });
      }
      await onChanged(id);
      if (creating) onCreated(id);
      close();
    } catch (error) {
      setError(String(error));
      await onChanged("").catch(() => {});
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <section className="virtual-nav" aria-label="Virtual collections">
        <div className="collection-heading">
          <span>VIRTUAL COLLECTIONS</span>
          <button
            className="icon-button"
            aria-label="Create virtual collection"
            disabled={!collection}
            onClick={() => setCreating(true)}
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              aria-hidden="true"
            >
              <path d="M12 5v14M5 12h14" />
            </svg>
          </button>
        </div>
        <div className="virtual-list">
          {groups.map((value) => (
            <div
              className={`virtual-item ${activeId === value.id ? "current" : ""}`}
              key={value.id}
            >
              <button
                className="virtual-open"
                aria-label={`Open virtual collection ${value.name}`}
                aria-current={activeId === value.id ? "page" : undefined}
                onClick={() => onBrowse(value)}
              >
                <span>{value.name}</span>
                <span className="virtual-count">{value.available_count}</span>
              </button>
              <button
                className="icon-button"
                aria-label={`Manage virtual collection ${value.name}`}
                onClick={() => {
                  setEditing(value.id);
                  setName(value.name);
                  setDeleting(false);
                  setError("");
                }}
              >
                ···
              </button>
            </div>
          ))}
          {!groups.length && (
            <p className="virtual-help">Group images without moving files.</p>
          )}
        </div>
      </section>
      {visible &&
        createPortal(
          <div
            className="overlay"
            onClick={(event) => {
              if (event.target === event.currentTarget && !busy) close();
            }}
          >
            <div
              className="dialog import-dialog virtual-dialog"
              ref={dialog}
              role="dialog"
              aria-modal="true"
              aria-labelledby="virtual-dialog-title"
            >
              <button
                className="dialog-close icon-button"
                aria-label="Close virtual collection dialog"
                disabled={busy}
                onClick={close}
              >
                ×
              </button>
              <h2 id="virtual-dialog-title">
                {deleting
                  ? "Delete virtual collection?"
                  : creating
                    ? "Create virtual collection"
                    : "Manage virtual collection"}
              </h2>
              <p>
                {deleting
                  ? `Remove “${group?.name || name}” and its memberships. Original image files stay in their source folders.`
                  : `Organize images from ${collection}. Membership is saved separately from the indexed files.`}
              </p>
              {error && (
                <p className="error" role="alert">
                  {error.replace(/^Error: /, "")}
                </p>
              )}
              {deleting ? (
                <div className="virtual-dialog-actions">
                  <button
                    className="secondary"
                    disabled={busy}
                    onClick={() => setDeleting(false)}
                  >
                    Keep collection
                  </button>
                  <button
                    className="primary virtual-danger"
                    disabled={busy}
                    onClick={() => void save(true)}
                  >
                    Delete collection
                  </button>
                </div>
              ) : (
                <form
                  onSubmit={(event) => {
                    event.preventDefault();
                    void save();
                  }}
                >
                  <label>
                    Collection name
                    <input
                      required
                      maxLength={80}
                      value={name}
                      onChange={(event) => setName(event.target.value)}
                    />
                  </label>
                  <div className="virtual-dialog-actions">
                    {!creating && (
                      <button
                        className="secondary virtual-danger"
                        type="button"
                        disabled={busy}
                        onClick={() => setDeleting(true)}
                      >
                        Delete collection…
                      </button>
                    )}
                    <button
                      className="primary"
                      type="submit"
                      disabled={busy || !name.trim()}
                    >
                      {busy
                        ? "Saving…"
                        : creating
                          ? "Create collection"
                          : "Save name"}
                    </button>
                  </div>
                </form>
              )}
            </div>
          </div>,
          document.body,
        )}
    </>
  );
}

export function SelectionToolbar({
  selected,
  groups,
  target,
  setTarget,
  activeId,
  busy,
  onAdd,
  onRemove,
  onClear,
  onCreate,
}: {
  selected: number;
  groups: VirtualCollection[];
  target: string;
  setTarget: (value: string) => void;
  activeId: string | null;
  busy: boolean;
  onAdd: () => void;
  onRemove: () => void;
  onClear: () => void;
  onCreate: () => void;
}) {
  if (!selected) return null;
  return (
    <section
      className="selection-toolbar"
      aria-label="Selected images"
      aria-live="polite"
    >
      <strong>{selected} selected</strong>
      <label>
        <span className="sr-only">Add to virtual collection</span>
        <select
          aria-label="Add to virtual collection"
          value={target}
          onChange={(event) => setTarget(event.target.value)}
          disabled={busy}
        >
          <option value="">Choose a collection</option>
          {groups.map((group) => (
            <option value={group.id} key={group.id}>
              {group.name}
            </option>
          ))}
        </select>
      </label>
      <button className="primary" disabled={busy || !target} onClick={onAdd}>
        Add selected
      </button>
      <button className="secondary" disabled={busy} onClick={onCreate}>
        New collection
      </button>
      {activeId && (
        <button className="secondary" disabled={busy} onClick={onRemove}>
          Remove from collection
        </button>
      )}
      <button className="text-button" disabled={busy} onClick={onClear}>
        Clear selection
      </button>
    </section>
  );
}
