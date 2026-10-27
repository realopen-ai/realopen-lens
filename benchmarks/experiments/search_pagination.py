"""Compare ordinary top-50 with bounded paginated production API searches."""

import argparse
import json
from pathlib import Path
import platform
from time import monotonic, sleep

import numpy as np
from fastapi.testclient import TestClient

from benchmarks.runtime.resources import MemoryMonitor, write_resources
from realopen_lens.api.app import create_app
from realopen_lens.storage.artifacts import atomic_bytes


def task(client, kind, request):
    response = client.post(f"/api/tasks/{kind}", json=request)
    response.raise_for_status()
    key = response.json()["id"]
    deadline = monotonic() + 300
    while monotonic() < deadline:
        state = client.get(f"/api/tasks/{key}").json()
        if state["status"] in {"completed", "failed", "cancelled", "interrupted"}:
            if state["status"] != "completed":
                raise ValueError(state.get("error") or state["status"])
            return state["result"]
        sleep(0.02)
    raise TimeoutError("Search exceeded 300 seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--collection", default="personal")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", nargs="+", default=["pizza", "sunset", "invoice"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        "index_dir": str(args.index_dir.resolve()),
        "collection": args.collection,
        "platform": platform.platform(),
        "queries": args.queries,
        "page_size": 50,
        "pool_limit": 1000,
        "ground_truth": None,
    }
    atomic_bytes(args.output / "config.json", json.dumps(config, indent=2).encode())
    cases = []
    with (
        MemoryMonitor() as memory,
        TestClient(
            create_app(index_dir=args.index_dir, models=args.models),
            base_url="http://127.0.0.1:8765",
        ) as client,
    ):
        client.get("/api/session")
        for library in (False, True):
            requests = [("search", {"query": query, "mode": "hybrid"}) for query in args.queries]
            image = client.get(f"/api/collections/{args.collection}/images?limit=1").json()[
                "results"
            ][0]
            requests.append(("similar", {"image_id": image["image_id"]}))
            for index, (kind, query) in enumerate(requests):
                label = f"{'library' if library else 'folder'}-{kind}-{index}"
                memory.phase = label
                request = {
                    **query,
                    "collection": args.collection,
                    "all_folders": library,
                    "limit": 50,
                    "save_run": False,
                }
                started = monotonic()
                base = task(client, kind, request)
                baseline_seconds = monotonic() - started
                started = monotonic()
                first = task(client, kind, {**request, "paginate": True})
                initial_seconds = monotonic() - started
                rows = list(first["results"])
                cursor, latencies = first["next_cursor"], []
                while cursor:
                    before = monotonic()
                    response = client.get("/api/search-pages", params={"cursor": cursor})
                    response.raise_for_status()
                    page = response.json()
                    latencies.append(monotonic() - before)
                    rows.extend(page["results"])
                    cursor = page["next_cursor"]
                keys = [row["image_id"] for row in rows]
                prefix_equal = base["results"] == first["results"]
                aliases = [
                    key for row in rows for key in [row["image_id"], *row.get("duplicate_ids", [])]
                ]
                record = {
                    "case": label,
                    "request": request,
                    "baseline_seconds": baseline_seconds,
                    "initial_seconds": initial_seconds,
                    "continuation_seconds": latencies,
                    "continuation_p95_seconds": (
                        float(np.percentile(latencies, 95)) if latencies else None
                    ),
                    "prefix_equal": prefix_equal,
                    "unique": len(keys) == len(set(keys)),
                    "duplicate_groups_disjoint": len(aliases) == len(set(aliases)),
                    "count": len(rows),
                    "truncated": first["truncated"],
                    "results": rows,
                }
                if (
                    not prefix_equal
                    or not record["unique"]
                    or not record["duplicate_groups_disjoint"]
                ):
                    raise AssertionError(f"Pagination invariant failed: {label}")
                atomic_bytes(
                    args.output / f"{label}.json", json.dumps(record, ensure_ascii=False).encode()
                )
                cases.append({key: value for key, value in record.items() if key != "results"})
                print(f"{label}: {len(rows)} results, preserved head, unique pages", flush=True)
    write_resources(args.output, memory)
    atomic_bytes(args.output / "summary.json", json.dumps(cases, indent=2).encode())


if __name__ == "__main__":
    main()
