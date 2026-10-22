import { useEffect, useRef, useState } from "react";
import { api } from "./api";

type Listing = {
  path: string;
  parent: string | null;
  roots: string[];
  folders: { name: string; path: string }[];
  next_offset: number | null;
};

export function FolderPicker({
  initial,
  onSelect,
  onClose,
}: {
  initial: string;
  onSelect: (path: string) => void;
  onClose: () => void;
}) {
  const [listing, setListing] = useState<Listing | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const revision = useRef(0);

  async function browse(path?: string, offset = 0) {
    const request = ++revision.current;
    setBusy(true);
    setError("");
    try {
      const params = new URLSearchParams({ offset: String(offset) });
      if (path) params.set("path", path);
      const result = await api<Listing>(`/folders?${params}`);
      if (request !== revision.current) return;
      setListing((previous) =>
        offset && previous
          ? { ...result, folders: [...previous.folders, ...result.folders] }
          : result,
      );
    } catch (failure) {
      if (request === revision.current) setError(String(failure));
    } finally {
      if (request === revision.current) setBusy(false);
    }
  }

  useEffect(() => {
    void browse(initial || undefined);
    return () => {
      revision.current++;
    };
  }, []);

  return (
    <section
      className="folder-picker"
      aria-label="Choose an image folder"
      aria-busy={busy}
    >
      <div className="folder-picker-heading">
        <strong>Choose a folder</strong>
        <button type="button" className="text-button" onClick={onClose}>
          Cancel selection
        </button>
      </div>
      <p className="folder-picker-path">{listing?.path || initial}</p>
      <div className="folder-picker-heading">
        <button
          type="button"
          className="secondary"
          disabled={busy || !listing?.parent}
          onClick={() => void browse(listing?.parent || undefined)}
        >
          Up one folder
        </button>
        <button
          type="button"
          className="secondary"
          disabled={busy}
          onClick={() => void browse()}
        >
          Home / import roots
        </button>
      </div>
      {listing && listing.roots.length > 1 && (
        <select
          aria-label="Import root"
          disabled={busy}
          value={listing.roots.includes(listing.path) ? listing.path : ""}
          onChange={(event) => void browse(event.target.value)}
        >
          <option value="" disabled>
            Choose a root
          </option>
          {listing.roots.map((root) => (
            <option key={root} value={root}>
              {root}
            </option>
          ))}
        </select>
      )}
      {error && <p role="alert">{error}</p>}
      {busy && <p role="status">Loading folders…</p>}
      <div className="folder-picker-list">
        {listing?.folders.map((folder) => (
          <button
            type="button"
            className="secondary"
            key={folder.name}
            disabled={busy}
            onClick={() => void browse(folder.path)}
          >
            <svg
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.7"
              aria-hidden="true"
            >
              <path d="M3 7V5a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z" />
            </svg>
            {folder.name}
          </button>
        ))}
        {listing && !listing.folders.length && !busy && (
          <p>No visible subfolders. You can select this folder.</p>
        )}
      </div>
      {listing?.next_offset != null && (
        <button
          type="button"
          className="secondary"
          disabled={busy}
          onClick={() => void browse(listing.path, listing.next_offset!)}
        >
          Load more folders
        </button>
      )}
      <button
        type="button"
        className="primary full-width"
        disabled={busy || !listing || !!error}
        onClick={() => listing && onSelect(listing.path)}
      >
        Select this folder
      </button>
    </section>
  );
}
