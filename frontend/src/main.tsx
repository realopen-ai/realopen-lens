import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  api,
  uploadImage,
  post,
  media,
  filename,
  terminal,
  type Collection,
  type ImageResult,
  type Task,
  type MemoryDecision,
} from "./api";
import "./style.css";
import { ThemeSwitch } from "./ThemeSwitch";
import { Duplicates } from "./Duplicates";
import { Workspace } from "./Workspace";
import { QueryMemory } from "./QueryMemory";
import { Training } from "./Training";
import {
  VirtualCollections,
  SelectionToolbar,
  type VirtualCollection,
} from "./VirtualCollections";

function Icon({ name, size = 20 }: { name: string; size?: number }) {
  const paths: Record<string, React.ReactNode> = {
    search: (
      <>
        <circle cx="10.8" cy="10.8" r="6.8" />
        <path d="m16 16 5 5" />
      </>
    ),
    grid: (
      <>
        <rect x="3" y="3" width="7" height="7" rx="1.5" />
        <rect x="14" y="3" width="7" height="7" rx="1.5" />
        <rect x="3" y="14" width="7" height="7" rx="1.5" />
        <rect x="14" y="14" width="7" height="7" rx="1.5" />
      </>
    ),
    plus: <path d="M12 5v14M5 12h14" />,
    image: (
      <>
        <rect x="3" y="3" width="18" height="18" rx="3" />
        <circle cx="8" cy="8" r="1.5" />
        <path d="m3 17 5-5 4 4 4-6 5 7" />
      </>
    ),
    close: <path d="m6 6 12 12M6 18 18 6" />,
    check: <path d="m5 12 4 4L19 6" />,
    similar: (
      <>
        <rect x="3" y="7" width="13" height="14" rx="2" />
        <path d="M8 3h11a2 2 0 0 1 2 2v11M7 12h5M7 16h5" />
      </>
    ),
    arrow: <path d="M5 12h14m-6-6 6 6-6 6" />,
    folder: (
      <path d="M3 7V5a2 2 0 0 1 2-2h5l3 4h6a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z" />
    ),
    shield: (
      <>
        <path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3Z" />
        <path d="m8 12 3 3 5-6" />
      </>
    ),
  };
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name] || paths.image}
    </svg>
  );
}

type RepeatSearch =
  | {
      kind: "search" | "similar";
      body: Record<string, unknown>;
      heading: string;
    }
  | { kind: "upload"; file: File };

function App() {
  const [view, setView] = useState<
    "library" | "learning" | "duplicates" | "workspace" | "activity"
  >("library");
  const [collections, setCollections] = useState<Collection[]>([]);
  const [collection, setCollection] = useState("");
  const selectedCollection = useRef("");
  const [virtuals, setVirtuals] = useState<VirtualCollection[]>([]);
  const [virtualId, setVirtualId] = useState<string | null>(null);
  const virtualScope = useRef<VirtualCollection | null>(null);
  const [unavailable, setUnavailable] = useState(0);
  const [selected, setSelected] = useState(new Set<string>());
  const [targetVirtual, setTargetVirtual] = useState("");
  const [creatingVirtual, setCreatingVirtual] = useState(false);
  const [selectionBusy, setSelectionBusy] = useState(false);
  const [organizationNotice, setOrganizationNotice] = useState("");
  const [images, setImages] = useState<ImageResult[]>([]);
  const [query, setQuery] = useState("");
  const [title, setTitle] = useState("All images");
  const [total, setTotal] = useState(0);
  const [generation, setGeneration] = useState<string | null>(null);
  const [resultRanking, setResultRanking] = useState("");
  const [isBrowse, setIsBrowse] = useState(true);
  const [loading, setLoading] = useState(false);
  const [reconnecting, setReconnecting] = useState(false);
  const [error, setError] = useState("");
  const [mode, setMode] = useState("hybrid");
  const [useFeedback, setUseFeedback] = useState(true);
  const [memoryEnabled, setMemoryEnabled] = useState(false);
  const [undo, setUndo] = useState<{
    image: ImageResult;
    decision: MemoryDecision;
    prior: string | null;
    run: string;
  } | null>(null);
  const [saveRun, setSaveRun] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  const [repeatSearch, setRepeatSearch] = useState<RepeatSearch | null>(null);
  const [labels, setLabels] = useState<Record<string, string>>({});
  const [labelsLoading, setLabelsLoading] = useState(false);
  const [uploadBytes, setUploadBytes] = useState<{
    loaded: number;
    total: number;
  } | null>(null);
  const [detail, setDetail] = useState<ImageResult | null>(null);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [importing, setImporting] = useState(false);
  const [roots, setRoots] = useState<string[]>([]);
  const [folder, setFolder] = useState("");
  const [name, setName] = useState("personal");
  const [precision, setPrecision] = useState("int8");
  const [languages, setLanguages] = useState("eng+fra+ara");
  const streams = useRef(new Map<string, EventSource>());
  const foreground = useRef<string | null>(null);
  const requestVersion = useRef(0);
  const fileInput = useRef<HTMLInputElement>(null);

  async function loadCollections() {
    const values = await api<Collection[]>("/collections");
    setCollections(values);
    return values;
  }
  async function loadVirtuals(source: string) {
    const values = await api<VirtualCollection[]>(
      `/virtual-collections?collection=${encodeURIComponent(source)}`,
    );
    if (selectedCollection.current === source) {
      setVirtuals(values);
      setTargetVirtual((current) =>
        values.some((group) => group.id === current) ? current : "",
      );
    }
    return values;
  }
  async function browse(
    value: string,
    offset = 0,
    group: VirtualCollection | null = null,
  ) {
    setView("library");
    const version = ++requestVersion.current;
    foreground.current = null;
    setRepeatSearch(null);
    setUndo(null);
    setMemoryEnabled(false);
    sessionStorage.removeItem("lens-task");
    if (selectedCollection.current !== value) {
      setVirtuals([]);
      setTargetVirtual("");
    }
    selectedCollection.current = value;
    virtualScope.current = group;
    setVirtualId(group?.id || null);
    setCollection(value);
    setLoading(true);
    setError("");
    setDetail(null);
    setRunId(null);
    setLabels({});
    if (!offset) {
      setSelected(new Set());
      setOrganizationNotice("");
    }
    try {
      const url = group
        ? `/virtual-collections/${group.id}/images?collection=${encodeURIComponent(value)}&revision=${group.revision}&offset=${offset}&limit=60${offset && generation ? `&generation=${generation}` : ""}`
        : `/collections/${encodeURIComponent(value)}/images?offset=${offset}&limit=60`;
      const data = await api<{
        results: ImageResult[];
        total: number;
        generation: string;
        unavailable_count?: number;
      }>(url);
      if (version !== requestVersion.current) return;
      setImages((old) => (offset ? [...old, ...data.results] : data.results));
      setTotal(data.total);
      setGeneration(data.generation);
      setUnavailable(data.unavailable_count || 0);
      setIsBrowse(true);
      setTitle(group?.name || "All images");
      if (!offset) await loadVirtuals(value);
    } catch (e) {
      if (version === requestVersion.current) {
        setError(String(e));
        if (!offset) setImages([]);
      }
    } finally {
      if (version === requestVersion.current) setLoading(false);
    }
  }
  async function organizationChanged(id: string, source = collection) {
    const values = await loadVirtuals(source);
    if (source !== selectedCollection.current) return;
    if (virtualScope.current?.id === id) {
      await browse(source, 0, values.find((group) => group.id === id) || null);
    }
  }
  async function refreshVirtual() {
    const source = collection,
      id = virtualId;
    try {
      const values = await loadVirtuals(source);
      if (
        selectedCollection.current === source &&
        virtualScope.current?.id === id
      )
        await browse(
          source,
          0,
          values.find((group) => group.id === id) || null,
        );
    } catch (error) {
      setError(String(error));
    }
  }
  function toggleImage(image: ImageResult) {
    const ids = [image.image_id, ...(image.duplicate_ids || [])];
    setSelected((old) => {
      const next = new Set(old);
      if (ids.every((id) => next.has(id))) ids.forEach((id) => next.delete(id));
      else if (new Set([...next, ...ids]).size <= 500)
        ids.forEach((id) => next.add(id));
      else setError("Select up to 500 images per action.");
      return next;
    });
  }
  async function changeSelection(remove = false) {
    const source = collection;
    const group = virtuals.find(
      (value) => value.id === (remove ? virtualId : targetVirtual),
    );
    if (!selected.size) return;
    if (!group) {
      setError("Choose an available virtual collection.");
      return;
    }
    setSelectionBusy(true);
    setError("");
    try {
      await api(`/virtual-collections/${group.id}/members`, {
        method: remove ? "DELETE" : "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          collection: source,
          revision: group.revision,
          image_ids: [...selected],
        }),
      });
      await organizationChanged(group.id, source);
      if (selectedCollection.current === source) {
        setSelected(new Set());
        setOrganizationNotice(
          remove
            ? "Removed the selection from this collection."
            : `Saved the selection to “${group.name}”.`,
        );
      }
    } catch (e) {
      if (selectedCollection.current === source) setError(String(e));
      await loadVirtuals(source).catch(() => {});
    } finally {
      setSelectionBusy(false);
    }
  }
  function receive(task: Task) {
    setTasks((old) =>
      [task, ...old.filter((t) => t.id !== task.id)].slice(0, 50),
    );
    if (!terminal(task)) return;
    streams.current.get(task.id)?.close();
    streams.current.delete(task.id);
    if (task.kind === "index" && task.status === "failed")
      setError(task.error || "Indexing failed");
    if (
      task.kind === "index" &&
      ["completed", "completed_with_errors"].includes(task.status)
    ) {
      void loadCollections()
        .then(() => {
          if (
            !foreground.current &&
            selectedCollection.current === task.collection
          ) {
            void loadVirtuals(task.collection)
              .then((values) =>
                browse(
                  task.collection,
                  0,
                  values.find(
                    (group) => group.id === virtualScope.current?.id,
                  ) || null,
                ),
              )
              .catch((e) => setError(String(e)));
          }
        })
        .catch((e) => setError(String(e)));
      setImporting(false);
    }
    if (foreground.current !== task.id) return;
    setLoading(false);
    if (task.status === "completed" && !task.result?.results) {
      setImages([]);
      setTotal(0);
      setRunId(null);
      setIsBrowse(false);
      setError(
        "These temporary search results are no longer available. Run the search again.",
      );
      return;
    }
    if (task.status === "completed" && task.result?.results) {
      setResultRanking(
        task.result.ranking
          ? `${task.result.ranking.kind}${task.result.ranking.sha256 ? ` · ${task.result.ranking.sha256.slice(0, 8)}` : ""}`
          : "",
      );
      setMemoryEnabled(Boolean(task.result.feedback?.enabled));
      setImages(task.result.results);
      setTotal(task.result.results.length);
      const savedRun = task.result.run_id || null;
      setRunId(savedRun);
      const remembered = Object.fromEntries(
        task.result.results
          .filter((image) => image.feedback?.label)
          .map((image) => [image.image_id, image.feedback!.label!]),
      );
      setLabels(remembered);
      setLabelsLoading(Boolean(savedRun));
      if (savedRun)
        void api<{ judgments: { image_id: string; label: string }[] }>(
          `/feedback/runs/${savedRun}`,
        )
          .then((value) => {
            if (foreground.current === task.id)
              setLabels(
                Object.fromEntries([
                  ...(value.judgments || []).map((j) => [j.image_id, j.label]),
                  ...Object.entries(remembered),
                ]),
              );
          })
          .catch((e) => setError(String(e)))
          .finally(() => {
            if (foreground.current === task.id) setLabelsLoading(false);
          });
      setGeneration(null);
      setIsBrowse(false);
    } else if (task.status !== "completed")
      setError(task.error || `Search ${task.status}`);
  }
  function watch(task: Task) {
    receive(task);
    if (terminal(task) || streams.current.has(task.id)) return;
    const stream = new EventSource(`/api/tasks/${task.id}/events`);
    streams.current.set(task.id, stream);
    stream.addEventListener("progress", (event) => {
      setReconnecting(false);
      receive(JSON.parse((event as MessageEvent).data));
    });
    stream.onerror = () => {
      setReconnecting(true);
      void api("/session").catch(() => {});
    }; // EventSource reconnects; a disconnect never means completion.
  }
  useEffect(() => {
    let disposed = false;
    void (async () => {
      const session = await api<{ allowed_roots: string[] }>("/session");
      if (disposed) return;
      setRoots(session.allowed_roots);
      setFolder(session.allowed_roots[0] || "");
      const values = await loadCollections();
      if (disposed) return;
      const remembered = sessionStorage.getItem("lens-task");
      const pending = await api<Task[]>("/tasks");
      if (disposed) return;
      const previous = pending.find(
        (t) => t.id === remembered && t.kind !== "index",
      );
      let restore = true;
      if (values.length) {
        const source = previous?.collection || values[0].name;
        const scopes = await loadVirtuals(source);
        const group =
          scopes.find((value) => value.id === previous?.virtual_collection) ||
          null;
        restore = !previous?.virtual_collection || Boolean(group);
        await browse(source, 0, group);
        if (!restore)
          setError(
            "The saved search's virtual collection is unavailable. Choose a collection and search again.",
          );
      }
      setTasks(pending);
      if (previous && restore) {
        foreground.current = previous.id;
        sessionStorage.setItem("lens-task", previous.id);
        setTitle("Search results");
        setLoading(!terminal(previous));
        if (terminal(previous)) receive(previous);
      }
      pending.filter((t) => !terminal(t)).forEach(watch);
    })().catch((e) => {
      if (!disposed) setError(String(e));
    });
    return () => {
      disposed = true;
      streams.current.forEach((s) => s.close());
      streams.current.clear();
    };
  }, []);
  useEffect(() => {
    if (!detail && !importing) return;
    const previous = document.activeElement as HTMLElement | null;
    const dialog = document.querySelector<HTMLElement>('[role="dialog"]');
    const focusable = () =>
      Array.from(
        dialog?.querySelectorAll<HTMLElement>(
          "button:not(:disabled), input, select, summary, a[href]",
        ) || [],
      );
    focusable()[0]?.focus();
    const listener = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setDetail(null);
        setImporting(false);
      }
      if (e.key === "Tab") {
        const items = focusable(),
          first = items[0],
          last = items.at(-1);
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last?.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first?.focus();
        }
      }
    };
    window.addEventListener("keydown", listener);
    return () => {
      window.removeEventListener("keydown", listener);
      previous?.focus();
    };
  }, [detail, importing]);

  async function start(
    kind: "search" | "similar",
    body: Record<string, unknown>,
    heading: string,
  ) {
    setView("library");
    const scope = virtualScope.current;
    body = {
      ...body,
      ...(!("virtual_collection" in body) && scope
        ? { virtual_collection: scope.id, virtual_revision: scope.revision }
        : {}),
    };
    setUndo(null);
    setMemoryEnabled(false);
    setRepeatSearch({ kind, body: { ...body }, heading });
    setSelected(new Set());
    setOrganizationNotice("");
    const version = ++requestVersion.current;
    foreground.current = "submitting";
    setImages([]);
    setTotal(0);
    setRunId(null);
    setLabels({});
    setIsBrowse(false);
    setLoading(true);
    setError("");
    setDetail(null);
    setTitle(heading);
    try {
      const task = await post<Task>(`/tasks/${kind}`, body);
      if (version === requestVersion.current) {
        foreground.current = task.id;
        sessionStorage.setItem("lens-task", task.id);
      }
      watch(task);
    } catch (e) {
      if (version === requestVersion.current) {
        foreground.current = null;
        setLoading(false);
        setError(String(e));
      }
    }
  }
  async function upload(file: File, record = saveRun) {
    setSelected(new Set());
    setOrganizationNotice("");
    const scope = virtualScope.current;
    setUndo(null);
    setMemoryEnabled(false);
    setRepeatSearch({ kind: "upload", file });
    if (file.size > 64 * 1024 * 1024) {
      setError("Select an image no larger than 64 MiB.");
      return;
    }
    setUploadBytes({ loaded: 0, total: file.size });
    const version = ++requestVersion.current;
    foreground.current = "submitting";
    setImages([]);
    setTotal(0);
    setRunId(null);
    setLabels({});
    setIsBrowse(false);
    setLoading(true);
    setError("");
    setTitle(`Similar to ${file.name}`);
    try {
      const task = await uploadImage(
        `/tasks/upload?collection=${encodeURIComponent(collection)}&save_run=${record}${scope ? `&virtual_collection=${scope.id}&virtual_revision=${scope.revision}` : ""}`,
        file,
        (loaded, total) => setUploadBytes({ loaded, total }),
      );
      if (version === requestVersion.current) {
        foreground.current = task.id;
        sessionStorage.setItem("lens-task", task.id);
      }
      watch(task);
    } catch (e) {
      if (version === requestVersion.current) {
        foreground.current = null;
        setLoading(false);
        setError(String(e));
      }
    } finally {
      setUploadBytes(null);
    }
  }
  function enableFeedback() {
    setSaveRun(true);
    if (repeatSearch?.kind === "upload") void upload(repeatSearch.file, true);
    else if (repeatSearch)
      void start(
        repeatSearch.kind,
        { ...repeatSearch.body, save_run: true },
        repeatSearch.heading,
      );
    else
      document
        .querySelector<HTMLInputElement>('input[aria-label="Search images"]')
        ?.focus();
  }
  async function label(image: ImageResult, value: string) {
    if (!runId || labelsLoading) return;
    const taskId = foreground.current;
    setLabelsLoading(true);
    try {
      const old = labels[image.image_id];
      const expected = memoryEnabled
        ? image.feedback?.event_id || 0
        : undefined;
      const suffix =
        expected === undefined ? "" : `?expected_event_id=${expected}`;
      const result = await api<{ memory: MemoryDecision | null }>(
        `/feedback/runs/${runId}/${image.image_id}${suffix}`,
        {
          method: old === value ? "DELETE" : "PUT",
          headers: { "Content-Type": "application/json" },
          body:
            old === value
              ? undefined
              : JSON.stringify({ label: value, expected_event_id: expected }),
        },
      );
      if (foreground.current !== taskId) return;
      setLabels((oldLabels) => {
        const next = { ...oldLabels };
        if (old === value) delete next[image.image_id];
        else next[image.image_id] = value;
        return next;
      });
      if (memoryEnabled && result.memory) {
        const decision = result.memory;
        if (decision.label === "irrelevant") {
          setImages((rows) =>
            rows.filter((row) => row.image_id !== image.image_id),
          );
          setTotal((count) => Math.max(0, count - 1));
          setSelected((current) => {
            const next = new Set(current);
            next.delete(image.image_id);
            return next;
          });
          setUndo({ image, decision, prior: old || null, run: runId });
        } else {
          setImages((rows) =>
            rows
              .map((row) =>
                row.image_id === image.image_id
                  ? { ...row, feedback: decision }
                  : row,
              )
              .sort(
                (a, b) =>
                  Number(b.feedback?.label === "relevant") -
                    Number(a.feedback?.label === "relevant") || a.rank - b.rank,
              ),
          );
        }
      }
    } catch (e) {
      setError(String(e));
    } finally {
      if (foreground.current === taskId) setLabelsLoading(false);
    }
  }
  async function undoFeedback() {
    if (!undo || labelsLoading) return;
    const action = undo;
    const taskId = foreground.current;
    setLabelsLoading(true);
    try {
      const path = `/feedback/runs/${action.run}/${action.image.image_id}`;
      const result = await api<{ memory: MemoryDecision }>(
        action.prior
          ? path
          : `${path}?expected_event_id=${action.decision.event_id}`,
        {
          method: action.prior ? "PUT" : "DELETE",
          headers: { "Content-Type": "application/json" },
          body: action.prior
            ? JSON.stringify({
                label: action.prior,
                expected_event_id: action.decision.event_id,
              })
            : undefined,
        },
      );
      if (foreground.current !== taskId) return;
      setImages((rows) =>
        [...rows, { ...action.image, feedback: result.memory }].sort(
          (a, b) =>
            Number(b.feedback?.label === "relevant") -
              Number(a.feedback?.label === "relevant") || a.rank - b.rank,
        ),
      );
      setTotal((count) => count + 1);
      setLabels((current) => {
        const next = { ...current };
        if (action.prior) next[action.image.image_id] = action.prior;
        else delete next[action.image.image_id];
        return next;
      });
      setUndo(null);
    } catch (e) {
      setError(String(e));
    } finally {
      if (foreground.current === taskId) setLabelsLoading(false);
    }
  }
  async function importFolder(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    try {
      const task = await post<Task>("/tasks/index", {
        root: folder,
        collection: name,
        siglip_precision: precision,
        languages,
      });
      watch(task);
      setImporting(false);
    } catch (e) {
      setError(String(e));
    }
  }
  const active = tasks.filter((t) => !terminal(t));
  const latest =
    active.find((t) => t.status === "running") || active[0] || tasks[0];
  const current = collections.find((c) => c.name === collection);
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="/" aria-label="RealOpen Lens home">
          <span className="brand-mark">
            <Icon name="search" size={24} />
          </span>
          <span>
            Lens<span className="brand-small">by RealOpen</span>
          </span>
        </a>
        <div className="workspace-label">YOUR WORKSPACE</div>
        <button
          className={`nav-item ${view === "library" && !virtualId ? "selected" : ""}`}
          onClick={() => collection && void browse(collection)}
        >
          <Icon name="grid" />
          Image library
        </button>
        <div className="collection-heading">
          <span>FOLDERS</span>
          <button
            className="icon-button"
            aria-label="Add collection"
            onClick={() => setImporting(true)}
          >
            <Icon name="plus" size={16} />
          </button>
        </div>
        <div className="collection-list">
          {collections.map((c) => (
            <button
              key={c.name}
              className={`collection-item ${collection === c.name ? "current" : ""}`}
              onClick={() => void browse(c.name)}
            >
              <Icon name="folder" size={17} />
              <span>{c.name}</span>
              <span className="collection-dot" />
            </button>
          ))}
          {!collections.length && (
            <p className="muted small">Your collections will appear here.</p>
          )}
        </div>
        <VirtualCollections
          collection={collection}
          groups={virtuals}
          activeId={virtualId}
          creating={creatingVirtual}
          setCreating={setCreatingVirtual}
          onBrowse={(group) => void browse(collection, 0, group)}
          onChanged={organizationChanged}
          onCreated={setTargetVirtual}
        />
        <div className="workspace-label tools-label">TOOLS</div>
        <nav className="tools-nav" aria-label="Workspace tools">
          <button
            className="nav-item mobile-library"
            onClick={() => setView("library")}
          >
            <Icon name="grid" />
            Library
          </button>
          {(
            [
              ["learning", "Learning"],
              ["duplicates", "Duplicates"],
              ["activity", "Activity"],
              ["workspace", "Workspace settings"],
            ] as const
          ).map(([key, label]) => (
            <button
              key={key}
              className={`nav-item ${view === key ? "selected" : ""}`}
              aria-current={view === key ? "page" : undefined}
              onClick={() => setView(key)}
            >
              <Icon
                name={
                  key === "duplicates"
                    ? "similar"
                    : key === "learning"
                      ? "grid"
                      : key === "activity"
                        ? "check"
                        : "folder"
                }
              />
              {label}
              {key === "activity" && active.length > 0 && (
                <span className="pill">{active.length}</span>
              )}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <span className="local-badge">
            <span className="status-dot" />
            Running locally
          </span>
          <p>
            Your images. Your machine.
            <br />
            Nothing leaves this device.
          </p>
          <span className="version">REALOPEN LENS · 0.1</span>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <span>
            <span className="muted">Workspace</span>
            <span className="breadcrumb">/</span>
            {view === "library"
              ? "Image library"
              : view === "learning"
                ? "Learning & models"
                : view === "duplicates"
                  ? "Duplicate groups"
                  : view === "activity"
                    ? "Activity"
                    : "Settings"}
          </span>
          <div className="topbar-actions">
            {view === "library" && collections.length > 0 && (
              <button
                className="primary add-folder"
                onClick={() => setImporting(true)}
              >
                <Icon name="plus" size={16} />
                Add folder
              </button>
            )}
            <span className="privacy">
              <Icon name="shield" size={16} />
              Private by design
            </span>
            <ThemeSwitch />
          </div>
        </header>
        <div className={`content view-${view}`}>
          <div hidden={view !== "learning"}>
            <h1>Learning & models</h1>
            <Training
              collection={collection}
              tasks={tasks}
              watch={watch}
              active={view === "learning"}
            />
          </div>
          <div hidden={view !== "duplicates"}>
            <h1>Duplicate groups</h1>
            <Duplicates
              collection={collection}
              tasks={tasks}
              watch={watch}
              active={view === "duplicates"}
            />
          </div>
          <div hidden={view !== "workspace"}>
            <Workspace
              active={view === "workspace"}
              onReady={() => {
                void loadCollections()
                  .then((values) => {
                    if (values.length) void browse(values[0].name);
                  })
                  .catch((e) => setError(String(e)));
              }}
              tasks={tasks}
              watch={watch}
            />
          </div>
          <section hidden={view !== "activity"} aria-label="Task history">
            <h1>Activity</h1>
            <p className="subtitle">
              Recent local tasks. Active work continues when you browse your
              images.
            </p>
            {tasks.length ? (
              tasks.map((task) => (
                <article className="activity-row" key={task.id}>
                  <strong>
                    {task.kind.replaceAll("_", " ")} · {task.collection}
                  </strong>
                  <span>
                    {task.status.replaceAll("_", " ")} ·{" "}
                    {task.phase.replaceAll("_", " ")}
                  </span>
                  {task.error && <p role="alert">{task.error}</p>}
                </article>
              ))
            ) : (
              <p>No tasks yet.</p>
            )}
          </section>
          <section hidden={view !== "library"} className="library-search">
            {collections.length > 0 ? (
              <h1 className="visually-hidden">Image library</h1>
            ) : (
              <div className="page-heading">
                <div>
                  {!collections.length && (
                    <div className="eyebrow">
                      A LITTLE LESS SEARCHING. A LITTLE MORE FINDING.
                    </div>
                  )}
                  <h1>
                    {collections.length
                      ? "Image library"
                      : "Your images, within reach."}
                  </h1>
                  {!collections.length && (
                    <p className="subtitle">
                      Find a moment, a screenshot, or that thing you can almost
                      remember.
                    </p>
                  )}
                </div>
                <button className="primary" onClick={() => setImporting(true)}>
                  <Icon name="plus" size={17} />
                  Add folder
                </button>
              </div>
            )}
            <form
              className="search-form"
              onSubmit={(e) => {
                e.preventDefault();
                if (query.trim() && collection)
                  void start(
                    "search",
                    {
                      query,
                      collection,
                      mode,
                      save_run: saveRun,
                      use_feedback: useFeedback,
                      limit: 50,
                    },
                    `Results for “${query}”`,
                  );
              }}
            >
              <Icon name="search" size={23} />
              <input
                aria-label="Search images"
                placeholder="Describe what you're looking for…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                disabled={!collection}
              />
              <input
                type="file"
                ref={fileInput}
                hidden
                accept="image/jpeg,image/png,image/webp,image/tiff,image/bmp"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) void upload(file);
                  e.target.value = "";
                }}
              />
              <button
                type="button"
                className="image-search"
                aria-label="Search by image"
                title="Search by image"
                disabled={!collection}
                onClick={() => fileInput.current?.click()}
              >
                <Icon name="image" />
              </button>
              <button
                type="submit"
                aria-label="Search"
                className="search-submit"
                disabled={!collection || !query.trim()}
              >
                <span>Search</span>
                <Icon name="arrow" size={18} />
              </button>
            </form>
            <div className="search-options">
              <span>Search by meaning or words inside an image.</span>
              <div>
                <label>
                  Search mode{" "}
                  <select
                    aria-label="Search mode"
                    value={mode}
                    onChange={(e) => setMode(e.target.value)}
                  >
                    <option value="hybrid">Hybrid</option>
                    <option value="visual">Visual</option>
                    <option value="lexical">Exact text</option>
                    <option value="semantic">Text meaning</option>
                  </select>
                </label>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={useFeedback}
                    onChange={(e) => setUseFeedback(e.target.checked)}
                  />
                  Remember query feedback
                </label>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={saveRun}
                    onChange={(e) => setSaveRun(e.target.checked)}
                  />
                  Save for feedback
                </label>
              </div>
            </div>
            {!isBrowse && !loading && repeatSearch?.kind === "search" && (
              <details className="query-options">
                <summary className="memory-status">
                  {memoryEnabled
                    ? "Query feedback enabled"
                    : "Baseline results · query feedback disabled"}
                </summary>
                <QueryMemory
                  enabled={memoryEnabled}
                  key={JSON.stringify(repeatSearch.body)}
                  collection={String(repeatSearch.body.collection)}
                  query={String(repeatSearch.body.query)}
                  scope={
                    repeatSearch.body.virtual_collection as string | undefined
                  }
                  revision={
                    repeatSearch.body.virtual_revision as number | undefined
                  }
                />
              </details>
            )}
            {undo && (
              <div className="memory-heading" role="status">
                <span>
                  Image hidden for this query. Search again to fill the
                  remaining places.
                </span>
                <button
                  className="secondary"
                  disabled={labelsLoading}
                  onClick={() => void undoFeedback()}
                >
                  Undo feedback
                </button>
              </div>
            )}
            {virtualId && (
              <div className="scope-banner">
                <span>
                  Searching within{" "}
                  <strong>
                    {virtuals.find((group) => group.id === virtualId)?.name ||
                      "virtual collection"}
                  </strong>
                </span>
                <button
                  className="text-button"
                  disabled={loading}
                  onClick={() => void refreshVirtual()}
                >
                  Refresh collection
                </button>
                <button
                  className="text-button"
                  onClick={() => void browse(collection)}
                >
                  Search all images
                </button>
              </div>
            )}
          </section>
          {uploadBytes && (
            <section className="task-panel" aria-label="Upload progress">
              <span className="spinner" />
              <div className="task-description">
                <strong>Uploading image</strong>
                <span>
                  {(uploadBytes.loaded / 1048576).toFixed(1)} /{" "}
                  {(uploadBytes.total / 1048576).toFixed(1)} MiB
                </span>
                <progress
                  max={uploadBytes.total || 1}
                  value={uploadBytes.loaded}
                />
              </div>
            </section>
          )}
          {reconnecting && (
            <p className="loading-label" role="status">
              Reconnecting to live progress…
            </p>
          )}
          {error && (
            <div className="error" role="alert">
              <span>{error.replace(/^Error: /, "")}</span>
              <button
                className="icon-button"
                onClick={() => setError("")}
                aria-label="Dismiss error"
              >
                <Icon name="close" size={16} />
              </button>
            </div>
          )}
          {latest && (!terminal(latest) || latest.status !== "completed") && (
            <section
              className={`task-panel ${terminal(latest) ? "finished" : ""}`}
              aria-label="Task progress"
              data-task-id={latest.id}
              aria-live="polite"
            >
              <div className="task-symbol">
                {terminal(latest) ? (
                  <Icon
                    name={
                      latest.status.startsWith("completed") ? "check" : "close"
                    }
                  />
                ) : (
                  <span className="spinner" />
                )}
              </div>
              <div className="task-description">
                <strong>
                  {latest.kind === "index"
                    ? "Indexing"
                    : latest.kind === "search"
                      ? "Searching"
                      : latest.kind === "train"
                        ? "Training and evaluating"
                        : latest.kind === "prepare_training"
                          ? "Preparing reviewed data"
                          : latest.kind === "duplicates"
                            ? "Scanning duplicates"
                            : latest.kind === "workspace_preview"
                              ? "Checking workspace sources"
                              : latest.kind === "workspace_import"
                                ? "Importing workspace"
                                : "Finding similar images"}{" "}
                  · {latest.collection}
                </strong>
                <span>
                  {latest.phase.replaceAll("_", " ")}
                  {latest.progress.step !== undefined
                    ? ` · step ${latest.progress.step}/${latest.progress.max_steps}`
                    : ""}
                  {latest.progress.query
                    ? ` · query ${latest.progress.query}/${latest.progress.queries}`
                    : ""}
                  {latest.progress.unique_images
                    ? ` · ${latest.progress.phase_completed_images || 0}/${latest.progress.unique_images} images`
                    : latest.progress.files
                      ? ` · ${latest.progress.files} files found`
                      : ""}
                  {latest.result?.images !== undefined
                    ? ` · ${latest.result.images} ${latest.kind === "duplicates" ? "indexed images" : "searchable images"}`
                    : ""}
                  {latest.status === "completed_with_errors"
                    ? latest.kind === "duplicates"
                      ? ` · ${latest.result?.errors?.length || 0} fingerprint errors`
                      : ` · ${latest.result?.failed_files || 0} failed, ${latest.result?.degraded_files || 0} degraded files`
                    : ""}
                  {latest.error ? ` · ${latest.error}` : ""}
                </span>
                {latest.progress.unique_images ? (
                  <progress
                    aria-label="Stage progress"
                    max={latest.progress.unique_images}
                    value={latest.progress.phase_completed_images || 0}
                  />
                ) : null}
              </div>
              <span className="elapsed">
                {Math.round(latest.elapsed_seconds)}s
              </span>
              {!terminal(latest) && (
                <button
                  className="text-button"
                  onClick={() =>
                    void post(`/tasks/${latest.id}/cancel`, {}).catch((e) =>
                      setError(String(e)),
                    )
                  }
                >
                  Stop
                </button>
              )}
              {active.length > 1 && (
                <span className="pill">{active.length - 1} queued</span>
              )}
            </section>
          )}
          <section hidden={view !== "library"} className="library-results">
            <div className="gallery-heading">
              <div>
                <h2>{title}</h2>
                <span className="count">
                  {total.toLocaleString()}{" "}
                  {isBrowse ? "images in this collection" : "results"}
                </span>
              </div>
              <div className="gallery-context">
                {current && <span className="pill">{current.name}</span>}
                {!isBrowse && resultRanking && (
                  <span className="ranking-caption">{resultRanking}</span>
                )}
                <Icon name="grid" size={18} />
              </div>
            </div>
            {!isBrowse && !loading && images.length > 0 && (
              <div className="feedback-caption" aria-label="Search feedback">
                {runId ? (
                  <span>
                    Feedback saved · mark images relevant or irrelevant below.
                  </span>
                ) : (
                  <button className="text-button" onClick={enableFeedback}>
                    {repeatSearch
                      ? "Search again with feedback"
                      : "Enable for next search"}
                  </button>
                )}
              </div>
            )}
            {unavailable > 0 && virtualId && isBrowse && (
              <p className="scope-help">
                {unavailable} saved{" "}
                {unavailable === 1 ? "image is" : "images are"} unavailable in
                the current source index. Membership is retained.
              </p>
            )}
            {organizationNotice && (
              <p className="organization-notice" role="status">
                {organizationNotice}
              </p>
            )}
            {images.length > 0 && !loading && (
              <div className="selection-controls">
                <button
                  className="text-button"
                  disabled={
                    new Set(
                      images.flatMap((image) => [
                        image.image_id,
                        ...(image.duplicate_ids || []),
                      ]),
                    ).size > 500 || selectionBusy
                  }
                  onClick={() =>
                    setSelected(
                      new Set(
                        images.flatMap((image) => [
                          image.image_id,
                          ...(image.duplicate_ids || []),
                        ]),
                      ),
                    )
                  }
                >
                  Select visible images
                </button>
                <span>Select up to 500 images per action</span>
              </div>
            )}
            <SelectionToolbar
              selected={selected.size}
              groups={virtuals}
              target={targetVirtual}
              setTarget={setTargetVirtual}
              activeId={virtualId}
              busy={selectionBusy}
              onAdd={() => void changeSelection()}
              onRemove={() => void changeSelection(true)}
              onClear={() => setSelected(new Set())}
              onCreate={() => setCreatingVirtual(true)}
            />
            {loading && (
              <div className="loading-label" role="status">
                <span className="spinner" />
                Working locally. Progress appears above.
              </div>
            )}
            {!images.length && !loading ? (
              <section className="empty">
                <div className="empty-icon">
                  <Icon name={collection ? "search" : "image"} size={35} />
                </div>
                <h2>
                  {collection
                    ? virtualId
                      ? !isBrowse
                        ? "No matching images"
                        : unavailable
                          ? "No saved images are available"
                          : "This collection is empty"
                      : !isBrowse
                        ? "No matching images"
                        : "Nothing here yet"
                    : "Make room for a little discovery."}
                </h2>
                <p>
                  {collection
                    ? virtualId
                      ? !isBrowse
                        ? "Try a different query or search all images in the source folder."
                        : unavailable
                          ? "Restore the source files and refresh their folder index to see them here again."
                          : "Select images from the library or search results, then add them to this collection."
                      : !isBrowse
                        ? "Try another query, review hidden images, or browse the library."
                        : "Try another search or index a folder to start exploring."
                    : "Add a folder of photos, screenshots, documents or diagrams. Lens keeps your original files right where they are."}
                </p>
                {virtualId || (collection && !isBrowse) ? (
                  <button
                    className="primary"
                    onClick={() => void browse(collection)}
                  >
                    Browse source images
                  </button>
                ) : (
                  <button
                    className="primary"
                    onClick={() => setImporting(true)}
                  >
                    <Icon name="plus" size={17} />
                    Add your first folder
                  </button>
                )}
              </section>
            ) : (
              <div className={`gallery ${loading ? "working" : ""}`}>
                {images.map((image) => (
                  <article
                    className={`image-card ${selected.has(image.image_id) ? "is-selected" : ""}`}
                    key={image.image_id}
                  >
                    <label className="image-select">
                      <input
                        type="checkbox"
                        aria-label={`Select ${filename(image)}`}
                        checked={selected.has(image.image_id)}
                        disabled={selectionBusy || loading}
                        onChange={() => toggleImage(image)}
                      />
                    </label>
                    <button
                      className="image-frame"
                      aria-label={`View ${filename(image)}`}
                      onClick={() => setDetail(image)}
                    >
                      <img
                        src={media(collection, image, "thumbnail", generation)}
                        alt={filename(image)}
                        loading="lazy"
                        onLoad={(e) =>
                          e.currentTarget.classList.remove("unavailable")
                        }
                        onError={(e) => {
                          e.currentTarget.classList.add("unavailable");
                        }}
                      />
                      <span className="rank-tag">{image.rank}</span>
                      {image.status === "degraded" && (
                        <span className="degraded">Partial text</span>
                      )}
                    </button>
                    <div className="card-caption">
                      <button
                        onClick={() => setDetail(image)}
                        title={filename(image)}
                      >
                        {filename(image)}
                      </button>
                      <span>
                        {image.width} × {image.height}
                        {image.paths.length > 1
                          ? ` · ${image.paths.length} copies`
                          : ""}
                      </span>
                    </div>
                    <div className="card-actions">
                      <button
                        title="Find similar images"
                        onClick={() =>
                          void start(
                            "similar",
                            {
                              collection,
                              image_id: image.image_id,
                              save_run: saveRun,
                              limit: 50,
                            },
                            `Similar to ${filename(image)}`,
                          )
                        }
                      >
                        <Icon name="similar" size={15} />
                        Similar
                      </button>
                    </div>
                    {!isBrowse && (
                      <div
                        className="feedback-actions"
                        aria-label={`Feedback for ${filename(image)}`}
                      >
                        <button
                          className={
                            labels[image.image_id] === "relevant"
                              ? "chosen"
                              : ""
                          }
                          disabled={!runId || labelsLoading}
                          aria-label={`Mark ${filename(image)} relevant`}
                          aria-pressed={labels[image.image_id] === "relevant"}
                          title={
                            runId
                              ? "Relevant (click again to remove)"
                              : "Run this search with feedback enabled to label results"
                          }
                          onClick={() => void label(image, "relevant")}
                        >
                          <Icon name="check" size={14} />
                          <span>Relevant</span>
                        </button>
                        <button
                          className={
                            labels[image.image_id] === "irrelevant"
                              ? "chosen"
                              : ""
                          }
                          disabled={!runId || labelsLoading}
                          aria-label={`Mark ${filename(image)} irrelevant`}
                          aria-pressed={labels[image.image_id] === "irrelevant"}
                          title={
                            runId
                              ? "Irrelevant (click again to remove)"
                              : "Run this search with feedback enabled to label results"
                          }
                          onClick={() => void label(image, "irrelevant")}
                        >
                          <Icon name="close" size={14} />
                          <span>Irrelevant</span>
                        </button>
                      </div>
                    )}
                  </article>
                ))}
              </div>
            )}
            {isBrowse && images.length < total && (
              <div className="load-more">
                <button
                  className="secondary"
                  disabled={loading}
                  onClick={() =>
                    void browse(collection, images.length, virtualScope.current)
                  }
                >
                  Load more images
                </button>
              </div>
            )}
          </section>
          <footer>
            <span>
              <span className="status-dot" />
              Local search, no cloud required.
            </span>
            <span>Originals stay untouched.</span>
          </footer>
        </div>
      </main>
      {importing && (
        <div className="overlay" onClick={() => setImporting(false)}>
          <section
            className="dialog import-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="import-title"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="dialog-close icon-button"
              aria-label="Close import"
              onClick={() => setImporting(false)}
            >
              <Icon name="close" />
            </button>
            <div className="empty-icon">
              <Icon name="folder" size={28} />
            </div>
            <h2 id="import-title">Add a folder to Lens</h2>
            <p className="muted">
              Your originals stay in place. Indexing runs in the background with
              live progress.
            </p>
            <form onSubmit={(e) => void importFolder(e)}>
              <label>
                Collection name
                <input
                  required
                  maxLength={128}
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </label>
              <label>
                Folder path
                <input
                  required
                  value={folder}
                  onChange={(e) => setFolder(e.target.value)}
                  placeholder="/path/to/images"
                />
              </label>
              <p className="field-help">
                Allowed import roots:{" "}
                {roots.length
                  ? roots.join(", ")
                  : "None configured. Start Lens with --allow-root /path/to/images."}{" "}
                Existing collection roots can also be refreshed.
              </p>
              <details>
                <summary>Indexing options</summary>
                <label>
                  Vision precision
                  <select
                    value={precision}
                    onChange={(e) => setPrecision(e.target.value)}
                  >
                    <option value="int8">INT8 · lower memory</option>
                    <option value="fp32">FP32 · higher memory</option>
                  </select>
                </label>
                <label>
                  Installed OCR languages
                  <input
                    required
                    value={languages}
                    onChange={(e) => setLanguages(e.target.value)}
                  />
                </label>
                <p className="field-help">
                  Refresh an existing collection with its current precision.
                  Models and language packs must already be installed.
                </p>
              </details>
              <button className="primary full-width" type="submit">
                Start indexing
                <Icon name="arrow" size={18} />
              </button>
            </form>
          </section>
        </div>
      )}
      {detail && (
        <div className="overlay" onClick={() => setDetail(null)}>
          <section
            className="dialog detail-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="detail-title"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="dialog-close icon-button"
              aria-label="Close details"
              onClick={() => setDetail(null)}
            >
              <Icon name="close" />
            </button>
            <div className="detail-preview">
              <img
                src={media(collection, detail, "original", generation)}
                onError={(e) => {
                  if (!e.currentTarget.dataset.fallback) {
                    e.currentTarget.dataset.fallback = "true";
                    e.currentTarget.src = media(
                      collection,
                      detail,
                      "thumbnail",
                      generation,
                    );
                  }
                }}
                alt={filename(detail)}
              />
            </div>
            <div className="detail-info">
              <span className="eyebrow">IMAGE DETAILS</span>
              <h2 id="detail-title">{filename(detail)}</h2>
              <p className="muted">
                {detail.width} × {detail.height} · {detail.status}
                {detail.visual_similarity !== null
                  ? ` · similarity ${detail.visual_similarity.toFixed(3)}`
                  : ""}
              </p>
              <button
                className="secondary"
                onClick={() =>
                  void start(
                    "similar",
                    {
                      collection,
                      image_id: detail.image_id,
                      save_run: saveRun,
                      limit: 50,
                    },
                    `Similar to ${filename(detail)}`,
                  )
                }
              >
                <Icon name="similar" size={17} />
                Find similar images
              </button>
              <h3>Source paths</h3>
              {detail.paths.map((p) => (
                <code className="path" key={p}>
                  {p}
                </code>
              ))}
              <h3>Extracted text preview</h3>
              <pre dir="auto">{detail.text || "No text extracted."}</pre>
              {detail.issues.map((issue, i) => (
                <p className="issue" key={i}>
                  {issue.stage}: {issue.error}
                </p>
              ))}
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
