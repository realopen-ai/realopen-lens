export interface MemoryDecision {
  image_id: string;
  label: "relevant" | "irrelevant" | null;
  event_id: number;
  run_id: string;
}
export interface ImageResult {
  rank: number;
  image_id: string;
  duplicate_ids?: string[];
  feedback?: MemoryDecision | null;
  paths: string[];
  thumbnail: string | null;
  width: number;
  height: number;
  text: string;
  visual_similarity: number | null;
  status: string;
  issues: { stage: string; error: string }[];
}
export interface Collection {
  id: string;
  name: string;
  root: string;
  generation: string | null;
  siglip_precision: string;
  languages?: string;
}
export interface SetupReport {
  ready: boolean;
  siglip_precision: string;
  languages: string;
  source: string | null;
  ocr: { installed_languages?: string[] };
  checks: {
    id: string;
    title: string;
    ok: boolean;
    detail: string;
    action?: string;
  }[];
  ocr_guide: string;
}
export interface Task {
  id: string;
  kind: string;
  collection: string;
  virtual_collection?: string | null;
  virtual_revision?: number | null;
  status: string;
  phase: string;
  elapsed_seconds: number;
  revision: number;
  error: string | null;
  progress: {
    step?: number;
    max_steps?: number;
    query?: number;
    queries?: number;
    files?: number;
    phase_completed_images?: number;
    unique_images?: number;
    current_image?: string;
  };
  result: {
    ready?: boolean;
    checks?: SetupReport["checks"];
    languages?: string;
    siglip_precision?: string;
    source?: string | null;
    ocr?: SetupReport["ocr"];
    ocr_guide?: string;
    feedback?: {
      enabled: boolean;
      applied: boolean;
      positive_groups?: number;
      hidden_groups?: number;
    };
    ranking?: { kind: string; sha256: string | null };
    results?: ImageResult[];
    run_id?: string | null;
    images?: number;
    fingerprinted?: number;
    errors?: unknown[];
    unsupported?: unknown[];
    failed_files?: number;
    degraded_files?: number;
  } | null;
}
export const terminal = (task: Task) =>
  [
    "completed",
    "completed_with_errors",
    "failed",
    "cancelled",
    "interrupted",
  ].includes(task.status);
export async function api<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch("/api" + url, init);
  if (!response.ok) {
    let message = await response.text();
    try {
      const value = JSON.parse(message);
      message =
        typeof value.detail === "string"
          ? value.detail
          : JSON.stringify(value.detail);
    } catch {
      /* Plain HTTP error. */
    }
    throw new Error(message || `Request failed (${response.status})`);
  }
  return response.json();
}
export const post = <T>(url: string, body: unknown) =>
  api<T>(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
export function media(
  collection: string,
  image: ImageResult,
  kind = "thumbnail",
  generation: string | null = null,
) {
  return (
    `/api/collections/${encodeURIComponent(collection)}/images/${image.image_id}/${kind}` +
    (generation ? `?generation=${encodeURIComponent(generation)}` : "")
  );
}
export const filename = (image: ImageResult) =>
  image.paths[0]?.split(/[\\/]/).pop() || image.image_id;

export function uploadImage(
  url: string,
  file: File,
  progress: (loaded: number, total: number) => void,
): Promise<Task> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api" + url);
    xhr.upload.onprogress = (event) =>
      progress(event.loaded, event.lengthComputable ? event.total : file.size);
    xhr.onerror = () =>
      reject(
        new Error("Image upload failed; check the local server connection."),
      );
    xhr.onabort = () => reject(new Error("Image upload cancelled."));
    xhr.onload = () => {
      try {
        const body = JSON.parse(xhr.responseText);
        if (xhr.status >= 200 && xhr.status < 300) resolve(body);
        else
          reject(
            new Error(
              typeof body.detail === "string"
                ? body.detail
                : "Image upload failed.",
            ),
          );
      } catch {
        reject(new Error(xhr.responseText || "Image upload failed."));
      }
    };
    xhr.send(file);
  });
}
