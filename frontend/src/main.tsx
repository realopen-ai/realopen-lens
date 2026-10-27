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
  type QueryInterpretation,
} from "./api";
import "./style.css";
import { ImageViewer } from "./ImageViewer";
import { TimelineHeading, TimelineDate } from "./Timeline";
import { ThemeSwitch } from "./ThemeSwitch";
import { Duplicates } from "./Duplicates";
import { Workspace } from "./Workspace";
import { FolderPicker } from "./FolderPicker";
import { FolderActions } from "./FolderActions";
import { Setup } from "./Setup";
import { OcrLanguages, DEFAULT_OCR_OPTIONS } from "./OcrLanguages";
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
  const [allFolders, setAllFolders] = useState(false);
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
  const [browseOrder, setBrowseOrder] = useState<"newest" | "oldest">("newest");
  const [browseSnapshot, setBrowseSnapshot] = useState<string | null>(null);
  const [nextOffset, setNextOffset] = useState<number | null>(null);
  const [searchCursor, setSearchCursor] = useState<string | null>(null);
  const [searchCapped, setSearchCapped] = useState(false);
  const searchPageBusy = useRef(false);
  const pageSentinel = useRef<HTMLDivElement>(null);
  const pageBusy = useRef(false);
  const browsePreference = useRef({ library: false, order: browseOrder });
  browsePreference.current = { library: allFolders, order: browseOrder };
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
  const [interpretation, setInterpretation] =
    useState<QueryInterpretation | null>(null);
  const [detail, setDetail] = useState<ImageResult | null>(null);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [importing, setImporting] = useState(false);
  const [roots, setRoots] = useState<string[]>([]);
  const [folder, setFolder] = useState("");
  const [choosingFolder, setChoosingFolder] = useState(false);
  useEffect(() => {
    if (!importing) setChoosingFolder(false);
  }, [importing]);
  const [name, setName] = useState("personal");
  const [nameEdited, setNameEdited] = useState(false);
  const [importError, setImportError] = useState("");
  const [indexSubmitting, setIndexSubmitting] = useState(false);
  const [precision, setPrecision] = useState("int8");
  const [defaultPrecision, setDefaultPrecision] = useState("int8");
  const [precisionEdited, setPrecisionEdited] = useState(false);
  const [languagesEdited, setLanguagesEdited] = useState(false);
  const [languageOptions, setLanguageOptions] =
    useState<Record<string, string>>(DEFAULT_OCR_OPTIONS);
  const [defaultLanguages, setDefaultLanguages] = useState("");
  const [ocrFolder, setOcrFolder] = useState("");
  const [ocrSaving, setOcrSaving] = useState(false);
  const [ocrNotice, setOcrNotice] = useState("");
  const [ocrError, setOcrError] = useState("");
  useEffect(() => {
    if (!importing) return;
    let active = true;
    void api<{
      default_siglip_precision?: string;
      default_ocr_languages?: string;
      ocr_language_options?: Record<string, string>;
    }>("/session")
      .then((session) => {
        if (active) {
          if (session.ocr_language_options)
            setLanguageOptions(session.ocr_language_options);
          setDefaultPrecision(
            session.default_siglip_precision === "fp32" ? "fp32" : "int8",
          );
          setDefaultLanguages(session.default_ocr_languages || "");
        }
      })
      .catch((error) => {
        if (active) setImportError(String(error));
      });
    return () => {
      active = false;
    };
  }, [importing]);
  useEffect(() => {
    if (!importing || precisionEdited) return;
    const existing = collections.find((value) => value.root === folder);
    setPrecision(existing?.siglip_precision || defaultPrecision);
  }, [folder, collections, importing, defaultPrecision, precisionEdited]);
  useEffect(() => {
    if (!importing || languagesEdited) return;
    const existing = collections.find((value) => value.root === folder);
    if (existing?.languages || defaultLanguages)
      setLanguages(existing?.languages || defaultLanguages);
  }, [folder, collections, importing, defaultLanguages, languagesEdited]);
  useEffect(() => {
    if (!importing || nameEdited || !folder.trim()) return;
    const existing = collections.find((value) => value.root === folder);
    if (existing) {
      setName(existing.name);
      return;
    }
    const base =
      folder
        .replace(/\\/g, "/")
        .replace(/\/+$/, "")
        .split("/")
        .pop()
        ?.trim()
        .slice(0, 120) || "images";
    let candidate = base;
    let suffix = 2;
    while (collections.some((value) => value.name === candidate))
      candidate = `${base}-${suffix++}`;
    setName(candidate);
  }, [folder, collections, importing, nameEdited]);
  useEffect(() => {
    if (!importing) {
      setNameEdited(false);
      setPrecisionEdited(false);
      setLanguagesEdited(false);
      setImportError("");
    }
  }, [importing]);
  const [languages, setLanguages] = useState("eng+fra+ara");
  const streams = useRef(new Map<string, EventSource>());
  const seenTasks = useRef(new Map<string, number>());
  const automaticBrowse = useRef(false);
  useEffect(() => {
    automaticBrowse.current =
      view === "library" &&
      isBrowse &&
      !detail &&
      !importing &&
      !creatingVirtual &&
      selected.size === 0;
  }, [view, isBrowse, detail, importing, creatingVirtual, selected.size]);
  const foreground = useRef<string | null>(null);
  const requestVersion = useRef(0);
  const fileInput = useRef<HTMLInputElement>(null);

  async function loadCollections() {
    const values = await api<Collection[]>("/collections");
    setCollections(values);
    return values;
  }
  async function folderChanged(changed: Collection, name: string | null) {
    sessionStorage.removeItem("lens-task");
    const values = await loadCollections();
    if (browsePreference.current.library) {
      const source =
        values.find((value) => value.name === selectedCollection.current)
          ?.name ||
        values[0]?.name ||
        "";
      await browse(source, 0, virtualScope.current, true);
      return;
    }
    if (selectedCollection.current !== changed.name) return;
    if (name !== null) {
      await browse(name, 0, virtualScope.current);
    } else if (virtualScope.current?.collection_id === "library-v1") {
      await browse(values[0]?.name || "", 0, virtualScope.current);
    } else if (values.length) {
      await browse(values[0].name);
    } else {
      ++requestVersion.current;
      foreground.current = null;
      selectedCollection.current = "";
      virtualScope.current = null;
      setCollection("");
      setVirtualId(null);
      setVirtuals([]);
      setTargetVirtual("");
      setInterpretation(null);
      setImages([]);
      setTotal(0);
      setSelected(new Set());
      setDetail(null);
      setRunId(null);
      setLabels({});
      setRepeatSearch(null);
      setUndo(null);
      setMemoryEnabled(false);
      setGeneration(null);
      setLoading(false);
      setTitle("All images");
      setIsBrowse(true);
      setError("");
      setView("library");
      setAllFolders(true);
      await loadVirtuals("");
    }
  }
  async function loadVirtuals(source: string) {
    const [local, library] = await Promise.all([
      source
        ? api<VirtualCollection[]>(
            `/virtual-collections?collection=${encodeURIComponent(source)}`,
          )
        : Promise.resolve([]),
      api<VirtualCollection[]>("/virtual-collections?all_folders=true"),
    ]);
    const values = [
      ...new Map([...local, ...library].map((g) => [g.id, g])).values(),
    ];
    if (selectedCollection.current === source) {
      setVirtuals(values);
      setTargetVirtual((current) =>
        values.some((group) => group.id === current) ? current : "",
      );
    }
    return values;
  }
  function openLibrary() {
    void browse(selectedCollection.current, 0, null, true);
  }
  async function browse(
    value: string,
    offset = 0,
    group: VirtualCollection | null = null,
    library = false,
    order = browsePreference.current.order,
  ) {
    if (offset && pageBusy.current) return;
    pageBusy.current = true;
    setView("library");
    setSearchCursor(null);
    const libraryGroup = library || group?.collection_id === "library-v1";
    setAllFolders(libraryGroup);
    if (libraryGroup)
      setTargetVirtual((current) =>
        virtuals.some(
          (g) => g.id === current && g.collection_id === "library-v1",
        )
          ? current
          : "",
      );
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
    setRunId(null);
    setLabels({});
    if (!offset) {
      setIsBrowse(true);
      setInterpretation(null);
      setImages([]);
      setNextOffset(null);
      setBrowseSnapshot(null);
      setDetail(null);
      setSelected(new Set());
      setOrganizationNotice("");
    }
    try {
      const params = `&order=${order}${offset && browseSnapshot ? `&snapshot=${browseSnapshot}` : ""}${offset && generation ? `&generation=${generation}` : ""}`;
      const url =
        (group
          ? `/virtual-collections/${group.id}/images?all_folders=${libraryGroup}&collection=${encodeURIComponent(value)}&revision=${group.revision}&offset=${offset}&limit=60`
          : libraryGroup
            ? `/library/images?offset=${offset}&limit=60`
            : `/collections/${encodeURIComponent(value)}/images?offset=${offset}&limit=60`) +
        params;
      const data = await api<{
        results: ImageResult[];
        total: number;
        generation: string;
        unavailable_count?: number;
        snapshot?: string;
        next_offset?: number | null;
      }>(url);
      if (version !== requestVersion.current) return;
      setImages((old) => (offset ? [...old, ...data.results] : data.results));
      setTotal(data.total);
      setBrowseSnapshot(data.snapshot || null);
      setNextOffset(
        data.next_offset !== undefined
          ? data.next_offset
          : offset + data.results.length < data.total
            ? offset + data.results.length
            : null,
      );
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
      if (version === requestVersion.current) {
        pageBusy.current = false;
        setLoading(false);
      }
    }
  }
  useEffect(() => {
    const target = pageSentinel.current;
    if (
      !target ||
      !isBrowse ||
      loading ||
      error ||
      view !== "library" ||
      detail ||
      importing ||
      creatingVirtual ||
      nextOffset === null
    )
      return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting))
          void browse(collection, nextOffset, virtualScope.current, allFolders);
      },
      { rootMargin: "300px" },
    );
    observer.observe(target);
    return () => observer.disconnect();
  }, [
    isBrowse,
    loading,
    error,
    view,
    detail,
    importing,
    creatingVirtual,
    nextOffset,
    collection,
    allFolders,
    browseOrder,
    browseSnapshot,
    generation,
  ]);

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
    if (!group || (allFolders && group.collection_id !== "library-v1")) {
      setError("Choose an available virtual collection for this scope.");
      return;
    }
    setSelectionBusy(true);
    setError("");
    try {
      await api(`/virtual-collections/${group.id}/members`, {
        method: remove ? "DELETE" : "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          collection: source || "personal",
          all_folders: group.collection_id === "library-v1",
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
    if ((seenTasks.current.get(task.id) ?? -1) > task.revision) return;
    seenTasks.current.set(task.id, task.revision);
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
          if (task.origin === "automatic" && !automaticBrowse.current) {
            if (
              browsePreference.current.library ||
              selectedCollection.current === task.collection
            )
              setOrganizationNotice(
                "Folder updated. Select it again to load the latest images.",
              );
            return;
          }
          if (
            !foreground.current &&
            (browsePreference.current.library ||
              selectedCollection.current === task.collection)
          ) {
            const source = selectedCollection.current;
            void loadVirtuals(source)
              .then((values) => {
                if (task.origin === "automatic" && !automaticBrowse.current)
                  return;
                return browse(
                  source,
                  0,
                  values.find(
                    (group) => group.id === virtualScope.current?.id,
                  ) || null,
                  browsePreference.current.library,
                );
              })
              .catch((e) => setError(String(e)));
          }
        })
        .catch((e) => setError(String(e)));
      if (task.origin !== "automatic") setImporting(false);
    }
    if (foreground.current !== task.id) return;
    setLoading(false);
    if (task.status === "completed" && !task.result?.results) {
      setInterpretation(null);
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
      setAllFolders(Boolean(task.all_folders));
      if (
        task.result.unavailable?.length ||
        task.result.pending_folders?.length
      )
        setOrganizationNotice(
          [
            ...(task.result.unavailable || []).map(
              (f) => `${f.collection}: ${f.images} unavailable images`,
            ),
            ...(task.result.pending_folders || []).map(
              (f) => `${f}: not indexed yet`,
            ),
          ].join(" · "),
        );

      setInterpretation(task.result.interpretation || null);
      setResultRanking(
        task.result.ranking
          ? `${task.result.ranking.kind}${task.result.ranking.kind === "logistic" && task.result.pagination ? " · later pages use hybrid" : ""}${task.result.ranking.sha256 ? ` · ${task.result.ranking.sha256.slice(0, 8)}` : ""}${task.result.ranking.mixed_precision_experimental ? " · mixed precision (experimental)" : ""}`
          : "",
      );
      setMemoryEnabled(Boolean(task.result.feedback?.enabled));
      setImages(task.result.results);
      setTotal(task.result.total ?? task.result.results.length);
      setSearchCursor(task.result.next_cursor || null);
      setSearchCapped(Boolean(task.result.truncated));
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
      const session = await api<{
        allowed_roots: string[];
        default_siglip_precision?: string;
        default_ocr_languages?: string;
        ocr_language_options?: Record<string, string>;
      }>("/session");
      if (disposed) return;
      setRoots(session.allowed_roots);
      if (session.ocr_language_options)
        setLanguageOptions(session.ocr_language_options);
      if (session.default_siglip_precision) {
        setDefaultPrecision(session.default_siglip_precision);
        setPrecision(session.default_siglip_precision);
      }
      if (session.default_ocr_languages) {
        setDefaultLanguages(session.default_ocr_languages);
        setLanguages(session.default_ocr_languages);
      }
      setFolder(session.allowed_roots[0] || "");
      const values = await loadCollections();
      if (disposed) return;
      const remembered = sessionStorage.getItem("lens-task");
      const pending = await api<Task[]>("/tasks");
      if (disposed) return;
      const previous = pending.find(
        (t) =>
          t.id === remembered &&
          t.kind !== "index" &&
          (t.all_folders ||
            values.some((value) => value.name === t.collection)),
      );
      let restore = true;
      if (values.length) {
        const source = previous?.all_folders
          ? values[0].name
          : previous?.collection || values[0].name;
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
      if (!values.length) {
        await loadVirtuals("");
        setAllFolders(true);
      }
      setTasks(pending);
      pending.forEach((task) => seenTasks.current.set(task.id, task.revision));
      if (previous && restore) {
        setAllFolders(Boolean(previous.all_folders));
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
    let disposed = false;
    let pending = false;
    const timer = window.setInterval(() => {
      if (pending) return;
      pending = true;
      void (async () => {
        const values = await api<Task[]>("/tasks?summary=true");
        if (disposed) return;
        const folders = await api<Collection[]>("/collections");
        if (disposed) return;
        setCollections(folders);
        const changed = values.filter(
          (task) =>
            !seenTasks.current.has(task.id) ||
            task.revision > (seenTasks.current.get(task.id) ?? -1),
        );
        for (const task of changed) {
          // A job may finish between discovery polls; retrieve its result once.
          const value = terminal(task)
            ? await api<Task>(`/tasks/${task.id}`)
            : task;
          if (disposed) return;
          watch(value);
        }
        for (const id of seenTasks.current.keys())
          if (
            !values.some((task) => task.id === id) &&
            !streams.current.has(id)
          )
            seenTasks.current.delete(id);
      })()
        .catch(() => {
          /* Existing SSE connections retain live task progress. */
        })
        .finally(() => {
          pending = false;
        });
    }, 5000);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, []);
  useEffect(() => {
    if (!importing) return;
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
  }, [importing]);

  async function start(
    kind: "search" | "similar",
    body: Record<string, unknown>,
    heading: string,
  ) {
    setView("library");
    const library = Boolean(body.all_folders);
    const scope = virtualScope.current;
    if (kind === "similar" && !library) {
      const source = String(body.collection);
      const changed = selectedCollection.current !== source;
      setAllFolders(false);
      setCollection(source);
      selectedCollection.current = source;
      if (changed) {
        setVirtuals([]);
        setTargetVirtual("");
        setVirtualId(null);
        virtualScope.current = null;
        void loadVirtuals(source).catch((e) => setError(String(e)));
      }
    }
    body = {
      ...body,
      paginate: true,
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
    setInterpretation(null);
    setSearchCursor(null);
    setSearchCapped(false);
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
    setInterpretation(null);
    setSearchCursor(null);
    setSearchCapped(false);
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
        `/tasks/upload?paginate=true&collection=${encodeURIComponent(collection)}&save_run=${record}&all_folders=${allFolders}${scope ? `&virtual_collection=${scope.id}&virtual_revision=${scope.revision}` : ""}`,
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
  async function loadSearchPage() {
    if (!searchCursor || searchPageBusy.current) return;
    const version = requestVersion.current;
    searchPageBusy.current = true;
    setLoading(true);
    setError("");
    try {
      const page = await api<NonNullable<Task["result"]>>(
        `/search-pages?cursor=${encodeURIComponent(searchCursor)}`,
      );
      if (version !== requestVersion.current) return;
      setImages((old) => {
        const seen = new Set(old.map((row) => row.image_id));
        return [
          ...old,
          ...(page.results || []).filter((row) => !seen.has(row.image_id)),
        ];
      });
      setSearchCursor(page.next_cursor || null);
      setLabels((old) => ({
        ...Object.fromEntries(
          (page.results || [])
            .filter((row) => row.feedback?.label)
            .map((row) => [row.image_id, row.feedback!.label!]),
        ),
        ...old,
      }));
    } catch (error) {
      if (version === requestVersion.current) {
        setError(String(error));
        setSearchCursor(null);
      }
    } finally {
      searchPageBusy.current = false;
      if (version === requestVersion.current) setLoading(false);
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
    setError("");
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
          setDetail((current) =>
            current?.image_id === image.image_id ? null : current,
          );
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
    setImportError("");
    setIndexSubmitting(true);
    try {
      const task = await post<Task>("/tasks/index", {
        root: folder,
        collection: name.trim(),
        siglip_precision: precision,
        languages,
      });
      watch(task);
      setImporting(false);
    } catch (e) {
      setImportError(String(e));
    } finally {
      setIndexSubmitting(false);
    }
  }
  const active = tasks.filter((t) => !terminal(t));
  const latest =
    active.find((t) => t.status === "running") || active[0] || tasks[0];
  function imageCollection(image: ImageResult) {
    const source = image.sources?.[0];
    return (
      collections.find((c) => c.id === source?.collection_id)?.name ||
      source?.collection ||
      collection
    );
  }
  const current = collections.find((c) => c.name === collection);
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="/" aria-label="RealOpen Lens home">
          <span className="brand-mark">
            <img src="/assets/lens.svg" alt="" aria-hidden="true" />
          </span>
          <span>
            Lens<span className="brand-small">by RealOpen AI</span>
          </span>
        </a>
        <div className="workspace-label">YOUR WORKSPACE</div>
        <button
          className={`nav-item ${view === "library" && allFolders ? "selected" : ""}`}
          onClick={openLibrary}
          disabled={!collection}
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
            <div className="folder-row" key={c.id || c.name}>
              <button
                className={`collection-item ${!allFolders && collection === c.name ? "current" : ""}`}
                onClick={() => void browse(c.name)}
              >
                <Icon name="folder" size={17} />
                <span>{c.name}</span>
                <span className="collection-dot" />
              </button>
              <FolderActions
                folder={c}
                busy={tasks.some((task) => !terminal(task))}
                watch={watch}
                onChanged={folderChanged}
              />
            </div>
          ))}
          {!collections.length && (
            <p className="muted small">Your collections will appear here.</p>
          )}
        </div>
        {
          <VirtualCollections
            collection={collection || "personal"}
            allFolders={allFolders}
            groups={
              allFolders
                ? virtuals.filter((g) => g.collection_id === "library-v1")
                : virtuals
            }
            activeId={virtualId}
            creating={creatingVirtual}
            setCreating={setCreatingVirtual}
            onBrowse={(group) => void browse(collection, 0, group)}
            onChanged={organizationChanged}
            onCreated={setTargetVirtual}
          />
        }
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
            <OcrLanguages
              value={languages}
              options={languageOptions}
              onChange={(value) => {
                setLanguages(value);
                setLanguagesEdited(true);
              }}
            />
            <div className="workspace-ocr-actions">
              <p className="small muted">
                Save the default for new folders. Searches use the OCR already
                indexed; reprocess an existing folder to change its OCR
                languages. Ordinary folder refresh keeps that folder's saved
                settings.
              </p>
              <button
                className="secondary"
                disabled={
                  ocrSaving || !languages || languages === defaultLanguages
                }
                onClick={async () => {
                  setOcrSaving(true);
                  setOcrNotice("");
                  setOcrError("");
                  try {
                    const saved = await post<{ languages: string }>(
                      "/workspace/ocr",
                      { languages },
                    );
                    setDefaultLanguages(saved.languages);
                    setOcrNotice(
                      "OCR default saved. Existing folders are unchanged.",
                    );
                  } catch (error) {
                    setOcrError(String(error));
                  } finally {
                    setOcrSaving(false);
                  }
                }}
              >
                Save OCR default
              </button>
              {collections.length > 0 && (
                <div className="workspace-ocr-apply">
                  <label>
                    Reprocess a folder with these languages
                    <select
                      value={ocrFolder}
                      onChange={(event) => setOcrFolder(event.target.value)}
                    >
                      <option value="">Choose a folder</option>
                      {collections.map((folder) => (
                        <option key={folder.id} value={folder.id}>
                          {folder.name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <button
                    className="secondary"
                    disabled={ocrSaving || !ocrFolder || !languages}
                    onClick={async () => {
                      const folder = collections.find(
                        (item) => item.id === ocrFolder,
                      );
                      if (!folder) return;
                      setOcrSaving(true);
                      setOcrNotice("");
                      setOcrError("");
                      try {
                        const task = await post<Task>("/collections/refresh", {
                          collection: folder.name,
                          collection_id: folder.id,
                          languages,
                        });
                        watch(task);
                        setOcrNotice(
                          `Reprocessing ${folder.name}. Follow live progress in Activity. Search uses the previous index until the new one is ready.`,
                        );
                      } catch (error) {
                        setOcrError(String(error));
                      } finally {
                        setOcrSaving(false);
                      }
                    }}
                  >
                    Reprocess OCR
                  </button>
                </div>
              )}
              {ocrNotice && (
                <p role="status" className="small">
                  {ocrNotice}
                </p>
              )}
              {ocrError && (
                <p role="alert" className="error">
                  {ocrError}
                </p>
              )}
            </div>
            <Setup
              tasks={tasks}
              watch={watch}
              precision={precision}
              languages={languages}
              onLanguages={(value) => {
                setLanguages(value);
                setLanguagesEdited(true);
              }}
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
                    {task.origin === "automatic"
                      ? "automatic refresh"
                      : task.kind.replaceAll("_", " ")}{" "}
                    · {task.collection}
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
                      all_folders: allFolders,
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
                accept="image/jpeg,image/png,image/webp,image/tiff,image/bmp,image/heic,image/heif,.heic,.heif"
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
                  Search in{" "}
                  <select
                    aria-label="Search in"
                    disabled={!collection}
                    value={allFolders ? "" : collection}
                    onChange={(e) =>
                      e.target.value === ""
                        ? openLibrary()
                        : void browse(e.target.value)
                    }
                  >
                    <option value="">All folders</option>
                    {collections.map((c) => (
                      <option key={c.id} value={c.name}>
                        {c.name}
                      </option>
                    ))}
                  </select>
                </label>
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
                  interpretMetadata={
                    repeatSearch?.kind === "search"
                      ? repeatSearch.body.interpret_metadata !== false
                      : true
                  }
                  enabled={memoryEnabled}
                  allFolders={Boolean(repeatSearch.body.all_folders)}
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
                  onClick={() =>
                    allFolders ? openLibrary() : void browse(collection)
                  }
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
                    ? latest.origin === "automatic"
                      ? "Refreshing automatically"
                      : "Indexing"
                    : latest.kind === "setup"
                      ? "Checking device setup"
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
                    ? latest.progress.phase_cached_images !== undefined
                      ? ` · ${latest.progress.phase_processed_images || 0} processed · ${latest.progress.phase_cached_images} cached · ${latest.progress.unique_images} total`
                      : ` · ${latest.progress.phase_completed_images || 0}/${latest.progress.unique_images} images`
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
                  {latest.progress.phase_failed_images
                    ? ` · ${latest.progress.phase_failed_images} stage failures`
                    : ""}
                  {latest.progress.phase_skipped_images
                    ? ` · ${latest.progress.phase_skipped_images} skipped`
                    : ""}
                  {terminal(latest) &&
                  latest.kind === "index" &&
                  latest.result?.reused_stages
                    ? latest.result.processed_stages
                      ? ` · OCR: ${latest.result.processed_stages.ocr ?? 0} processed, ${latest.result.reused_stages.ocr ?? 0} cached`
                      : ` · OCR: ${latest.result.reused_stages.ocr ?? 0} cached`
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
                  {isBrowse
                    ? allFolders && !virtualId
                      ? "images in your library"
                      : "images in this collection"
                    : "results"}
                </span>
              </div>
              <div className="gallery-context">
                {allFolders ? (
                  <span className="pill">All folders</span>
                ) : (
                  current && <span className="pill">{current.name}</span>
                )}
                {!isBrowse && resultRanking && (
                  <span className="ranking-caption">{resultRanking}</span>
                )}
                {isBrowse && (
                  <label className="timeline-sort">
                    <span>Sort images</span>
                    <select
                      aria-label="Sort images"
                      value={browseOrder}
                      disabled={loading}
                      onChange={(event) => {
                        const order = event.target.value as "newest" | "oldest";
                        setBrowseOrder(order);
                        void browse(
                          collection,
                          0,
                          virtualScope.current,
                          allFolders,
                          order,
                        );
                      }}
                    >
                      <option value="newest">Newest first</option>
                      <option value="oldest">Oldest first</option>
                    </select>
                  </label>
                )}
                <Icon name="grid" size={18} />
              </div>
            </div>
            {isBrowse && error && (
              <button
                className="secondary timeline-reload"
                onClick={() =>
                  void browse(collection, 0, virtualScope.current, allFolders)
                }
              >
                Reload images
              </button>
            )}
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
                the indexed library. Membership is retained.
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
            {interpretation && !isBrowse && (
              <div
                className="query-interpretation"
                aria-label="Interpreted search"
              >
                <div className="metadata-chips">
                  {interpretation.constraints.map((rule, index) => (
                    <span className="pill" key={index}>
                      {rule.label}
                    </span>
                  ))}
                </div>
                <p>
                  {interpretation.visual_query
                    ? `Searching for “${interpretation.visual_query}” within these matches.`
                    : "Showing images matching these details."}
                </p>
                {interpretation.notices.map((notice, index) => (
                  <p className="muted" key={index}>
                    {notice}
                  </p>
                ))}
                {repeatSearch?.kind === "search" && (
                  <button
                    className="text-button"
                    onClick={() =>
                      void start(
                        "search",
                        { ...repeatSearch.body, interpret_metadata: false },
                        repeatSearch.heading,
                      )
                    }
                  >
                    Use entire query as search text
                  </button>
                )}
              </div>
            )}
            <SelectionToolbar
              selected={selected.size}
              groups={
                allFolders
                  ? virtuals.filter((g) => g.collection_id === "library-v1")
                  : virtuals
              }
              target={targetVirtual}
              setTarget={setTargetVirtual}
              activeId={virtualId}
              busy={selectionBusy}
              onAdd={() => void changeSelection()}
              onRemove={() => void changeSelection(true)}
              onClear={() => setSelected(new Set())}
              onCreate={() => setCreatingVirtual(true)}
            />
            {loading && (!isBrowse || !images.length) && (
              <div className="loading-label" role="status">
                <span className="spinner" />
                {isBrowse
                  ? "Loading images…"
                  : "Working locally. Progress appears above."}
              </div>
            )}
            {!images.length && !loading ? (
              <section className="empty">
                <div className="empty-icon">
                  <Icon name={collection ? "search" : "image"} size={35} />
                </div>
                <h2>
                  {allFolders &&
                  collections.length > 0 &&
                  !virtualId &&
                  !repeatSearch
                    ? "Search across your folders"
                    : collection
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
                  {allFolders &&
                  collections.length > 0 &&
                  !virtualId &&
                  !repeatSearch
                    ? "Describe a photo, screenshot or document. Lens searches every indexed folder."
                    : collection
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
                {!collection && (
                  <button
                    className="secondary"
                    onClick={() => setView("workspace")}
                  >
                    Check device setup
                  </button>
                )}
              </section>
            ) : (
              <div
                className={`gallery ${loading && !isBrowse ? "working" : ""}`}
              >
                {images.map((image, index) => (
                  <React.Fragment key={image.image_id}>
                    {isBrowse && (
                      <TimelineHeading
                        image={image}
                        previous={images[index - 1]}
                      />
                    )}
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
                          src={media(
                            collection,
                            image,
                            "thumbnail",
                            generation,
                          )}
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
                      {allFolders && image.sources && (
                        <p className="source-caption">
                          {[
                            ...new Set(image.sources.map((s) => s.collection)),
                          ].join(" · ")}
                        </p>
                      )}
                      <div className="card-actions">
                        <button
                          title="Find similar images"
                          onClick={() =>
                            void start(
                              "similar",
                              {
                                collection: imageCollection(image),
                                all_folders: allFolders,
                                image_id:
                                  image.sources?.[0]?.image_id ||
                                  image.image_id,
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
                      {isBrowse && <TimelineDate image={image} />}
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
                            aria-pressed={
                              labels[image.image_id] === "irrelevant"
                            }
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
                  </React.Fragment>
                ))}
              </div>
            )}
            {!isBrowse &&
              (searchCursor || (images.length > 0 && searchCapped)) && (
                <div className="load-more">
                  <p className="small muted">
                    Showing {images.length} of {total} retrieved results
                    {searchCapped
                      ? " · Search pool capped at 1,000 images; refine your query for more specific matches."
                      : "."}
                  </p>
                  {searchCursor && (
                    <button
                      className="secondary"
                      disabled={loading}
                      onClick={() => void loadSearchPage()}
                    >
                      {loading ? "Loading more results…" : "Load more results"}
                    </button>
                  )}
                </div>
              )}
            {isBrowse && nextOffset !== null && (
              <div className="load-more" ref={pageSentinel}>
                <button
                  className="secondary"
                  disabled={loading}
                  onClick={() =>
                    void browse(
                      collection,
                      nextOffset,
                      virtualScope.current,
                      allFolders,
                    )
                  }
                >
                  {loading ? "Loading more images…" : "Load more images"}
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
                  onChange={(e) => {
                    setNameEdited(true);
                    setName(e.target.value);
                  }}
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
              <button
                type="button"
                className="secondary"
                onClick={() => setChoosingFolder(!choosingFolder)}
              >
                Choose folder
              </button>
              {choosingFolder && (
                <FolderPicker
                  initial={folder}
                  onClose={() => setChoosingFolder(false)}
                  onSelect={(path) => {
                    setFolder(path);
                    setChoosingFolder(false);
                  }}
                />
              )}
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
                    onChange={(e) => {
                      setPrecisionEdited(true);
                      setPrecision(e.target.value);
                    }}
                  >
                    <option value="int8">INT8 · lower memory</option>
                    <option value="fp32">FP32 · higher memory</option>
                  </select>
                </label>
                <OcrLanguages
                  value={languages}
                  options={languageOptions}
                  disabled={indexSubmitting}
                  onChange={(value) => {
                    setLanguages(value);
                    setLanguagesEdited(true);
                  }}
                />
                <p className="field-help">
                  Models and language packs must already be installed. Existing
                  folders keep their current vision precision.
                </p>
              </details>
              <div className="import-setup">
                <Setup
                  tasks={tasks}
                  watch={watch}
                  precision={precision}
                  languages={languages}
                  source={folder}
                  onLanguages={(value) => {
                    setLanguages(value);
                    setLanguagesEdited(true);
                  }}
                />
              </div>
              {importError && <p role="alert">{importError}</p>}
              <button
                className="primary full-width"
                type="submit"
                disabled={indexSubmitting || !name.trim()}
              >
                {indexSubmitting ? "Starting indexing…" : "Start indexing"}
                <Icon name="arrow" size={18} />
              </button>
            </form>
          </section>
        </div>
      )}
      {detail && (
        <ImageViewer
          image={
            images.find((image) => image.image_id === detail.image_id) || detail
          }
          images={images}
          collection={collection}
          generation={generation}
          onClose={() => setDetail(null)}
          onNavigate={setDetail}
          onSimilar={(image) =>
            void start(
              "similar",
              {
                collection: imageCollection(image),
                all_folders: allFolders,
                image_id: image.sources?.[0]?.image_id || image.image_id,
                save_run: saveRun,
                limit: 50,
              },
              `Similar to ${filename(image)}`,
            )
          }
          onLabel={label}
          label={labels[detail.image_id]}
          feedbackEnabled={Boolean(runId) && !isBrowse}
          feedbackBusy={labelsLoading}
          feedbackError={error}
          onCollectionsChanged={organizationChanged}
        />
      )}
    </div>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
