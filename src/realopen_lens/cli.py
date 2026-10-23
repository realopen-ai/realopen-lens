"""Application command line entry points for explicit setup and diagnostics."""

import json
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Annotated, Any
from collections.abc import Callable

import typer

from realopen_lens.workspace.cli import app as workspace_app
from realopen_lens.organization.cli import app as groups_app
from realopen_lens.organization.duplicates_cli import app as duplicates_app

app = typer.Typer(
    no_args_is_help=True, help="Private, offline collection indexing and image search."
)
models_app = typer.Typer(
    no_args_is_help=True, help="Explicit model downloads and integrity checks."
)
app.add_typer(models_app, name="models")
app.add_typer(workspace_app, name="workspace")
feedback_app = typer.Typer(no_args_is_help=True, help="Explicit local search relevance judgments.")
app.add_typer(feedback_app, name="feedback")


@models_app.command("fetch")
def fetch_models(
    directory: Annotated[Path, typer.Option("--directory")] = Path("models"),
    precision: Annotated[str | None, typer.Option()] = None,
    siglip_precision: Annotated[
        str | None, typer.Option(help="Override SigLIP2 precision only: int8 or fp32")
    ] = None,
) -> None:
    """Explicitly download pinned ONNX model artifacts."""
    from realopen_lens.models.manager import fetch

    try:
        from realopen_lens.runtime.precision import default_precision

        selected = default_precision(
            siglip_precision if siglip_precision is not None else precision
        )
        fetch(directory, precision or "int8", typer.echo, siglip_precision=selected)
    except (OSError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error


@models_app.command("verify")
def verify_models(
    directory: Annotated[Path, typer.Option("--directory")] = Path("models"),
    precision: Annotated[str | None, typer.Option()] = None,
    siglip_precision: Annotated[
        str | None, typer.Option(help="Override SigLIP2 precision only: int8 or fp32")
    ] = None,
) -> None:
    """Verify local model checksums without making network requests."""
    from realopen_lens.models.manager import verify

    try:
        from realopen_lens.runtime.precision import default_precision

        selected = default_precision(
            siglip_precision if siglip_precision is not None else precision
        )
        verify(directory, precision or "int8", selected)
    except (OSError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo("All selected model files verified.")


@models_app.command("fetch-text-detector")
def fetch_detector(
    directory: Annotated[Path, typer.Option("--directory")] = Path("models"),
) -> None:
    """Explicitly download the pinned experimental text-localization model."""
    from realopen_lens.models.manager import fetch_text_detector

    try:
        typer.echo(str(fetch_text_detector(directory)))
    except (OSError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error


@app.command()
def doctor() -> None:
    """Inspect local prerequisites without downloading or loading model weights."""
    import onnxruntime

    onnxruntime.disable_telemetry_events()
    from realopen_lens.ocr import runtime as ocr_runtime

    executable = ocr_runtime.resolve(shutil.which("tesseract"))
    database = sqlite3.connect(":memory:")
    try:
        database.execute("CREATE VIRTUAL TABLE check_fts USING fts5(text)")
    finally:
        database.close()
    result: dict = {
        "cpu_inference": "CPUExecutionProvider" in onnxruntime.get_available_providers(),
        "sqlite_fts5": True,
        "tesseract": executable,
    }
    if executable:
        version = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            check=True,
            env=ocr_runtime.environment(executable),
            timeout=10,
        )
        langs = subprocess.run(
            [executable, "--list-langs"],
            capture_output=True,
            text=True,
            check=True,
            env=ocr_runtime.environment(executable),
            timeout=10,
        )
        result["tesseract_version"] = version.stdout.splitlines()[0]
        result["ocr_languages"] = langs.stdout.splitlines()[1:]
    typer.echo(json.dumps(result, indent=2))
    if not executable:
        raise typer.Exit(1)


@app.command("ocr")
def recognize_image(
    image: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    engine: Annotated[str, typer.Option(help="tesseract (default) or glm-ocr")] = "tesseract",
    languages: Annotated[str, typer.Option()] = "eng+fra+ara",
    threads: Annotated[int, typer.Option(min=1, max=32)] = 2,
    model: Annotated[
        str, typer.Option(help="Explicitly installed local Ollama model")
    ] = "glm-ocr:q8_0",
    ollama_port: Annotated[int, typer.Option(min=1, max=65535)] = 11434,
    timeout: Annotated[float, typer.Option(min=1, max=300)] = 90,
    max_tokens: Annotated[int, typer.Option(min=1, max=8192)] = 4096,
    context: Annotated[int, typer.Option(min=1024, max=16384)] = 8192,
    detector_model: Annotated[
        Path | None,
        typer.Option(
            "--experimental-detector-model",
            help="Opt into experimental text-region OCR using this local model",
        ),
    ] = None,
    region_mode: Annotated[
        str, typer.Option(help="Experimental detector mode: gate or mask")
    ] = "gate",
) -> None:
    """Read an image locally; optional GLM-OCR uses only numeric loopback HTTP."""
    from realopen_lens.ocr.backends import create_backend

    backend = None
    try:
        backend = create_backend(
            engine,
            languages=languages,
            threads=threads,
            model=model,
            port=ollama_port,
            timeout=timeout,
            max_tokens=max_tokens,
            context=context,
            detector_model=detector_model,
            region_mode=region_mode,
        )
        if detector_model is not None:
            typer.echo(f"Experimental text-region OCR selected ({region_mode}).", err=True)
        typer.echo(json.dumps(backend.extract(image), ensure_ascii=False, indent=2))
    except (OSError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    finally:
        if backend is not None:
            try:
                backend.close()
            except (OSError, ValueError) as error:
                typer.echo(f"OCR cleanup warning: {error}", err=True)


@app.command("index")
def index_folder(
    folder: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    collection: Annotated[str, typer.Option()] = "personal",
    index_dir: Annotated[
        Path | None, typer.Option(help="Private catalog and derived-data directory")
    ] = None,
    models: Annotated[Path, typer.Option()] = Path("models"),
    languages: Annotated[str, typer.Option()] = "eng+fra+ara",
    threads: Annotated[int, typer.Option(min=1, max=8)] = 2,
    rehash: Annotated[
        bool, typer.Option(help="Rehash even files with unchanged filesystem metadata")
    ] = False,
    log: Annotated[bool, typer.Option("--log", help="Stream live indexing logs to stderr")] = False,
    siglip_precision: Annotated[
        str | None, typer.Option(help="Override device/index SigLIP2 precision; E5 stays int8")
    ] = None,
):
    """Index or refresh a folder; unchanged processing resumes from durable cache."""
    from dataclasses import asdict
    from realopen_lens.indexing.service import index_collection

    try:
        result = index_collection(
            folder,
            collection=collection,
            index_dir=index_dir,
            models=models,
            languages=languages,
            threads=threads,
            rehash=rehash,
            log=log,
            siglip_precision=siglip_precision,
        )
        typer.echo(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    if result.failed_files or result.degraded_files:
        raise typer.Exit(1)


@app.command("status")
def show_status(
    collection: Annotated[str, typer.Option()] = "personal",
    index_dir: Annotated[Path | None, typer.Option()] = None,
):
    """Inspect the latest indexing job, failures, and published generation."""
    from realopen_lens.indexing.service import collection_status

    try:
        typer.echo(
            json.dumps(collection_status(collection, index_dir), ensure_ascii=False, indent=2)
        )
    except (OSError, ValueError, sqlite3.Error) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error


@app.command("search")
def find_images(
    query: Annotated[str, typer.Argument()],
    all_folders: Annotated[bool, typer.Option(help="Search every indexed folder")] = False,
    collection: Annotated[str, typer.Option()] = "personal",
    virtual_collection: Annotated[
        str | None, typer.Option(help="Restrict to this virtual collection ID")
    ] = None,
    virtual_revision: Annotated[int | None, typer.Option(min=1)] = None,
    index_dir: Annotated[Path | None, typer.Option()] = None,
    models: Annotated[Path, typer.Option()] = Path("models"),
    limit: Annotated[int, typer.Option(min=1, max=100)] = 10,
    mode: Annotated[str, typer.Option(help="hybrid, visual, lexical, or semantic")] = "hybrid",
    threads: Annotated[int, typer.Option(min=1, max=8)] = 2,
    reranker: Annotated[
        Path | None, typer.Option(help="Experimental local logistic JSON model; hybrid mode only")
    ] = None,
    visual_candidates: Annotated[int | None, typer.Option(min=1, max=1000)] = None,
    semantic_candidates: Annotated[int | None, typer.Option(min=1, max=1000)] = None,
    lexical_candidates: Annotated[int | None, typer.Option(min=1, max=1000)] = None,
    siglip_precision: Annotated[
        str | None, typer.Option(help="Assert int8 or fp32; otherwise use collection precision")
    ] = None,
    save_run: Annotated[
        bool, typer.Option(help="Save this search for explicit relevance feedback")
    ] = False,
    feedback: Annotated[
        bool,
        typer.Option("--feedback/--no-feedback", help="Apply remembered exact-query relevance"),
    ] = True,
    interactive: Annotated[
        bool, typer.Option(help="Reuse encoders for subsequent queries; empty line or EOF exits")
    ] = False,
):
    """Search a saved collection after indexing, including across restarts."""
    from dataclasses import asdict
    from realopen_lens.retrieval.search import search_collection
    from realopen_lens.retrieval.candidates import CandidateDepths

    try:
        depths = None
        if any(v is not None for v in (visual_candidates, semantic_candidates, lexical_candidates)):
            depths = CandidateDepths(
                visual=50 if visual_candidates is None else visual_candidates,
                semantic=50 if semantic_candidates is None else semantic_candidates,
                lexical=50 if lexical_candidates is None else lexical_candidates,
            )
        if all_folders:
            from realopen_lens.retrieval.library import LibrarySession
            from realopen_lens.storage.catalog import default_directory

            if (
                virtual_collection
                or virtual_revision
                or reranker
                or depths
                or siglip_precision
                or collection != "personal"
            ):
                raise ValueError("Library search cannot use folder-specific overrides")
            session = LibrarySession(
                index_dir=index_dir or default_directory(), models=models, threads=threads
            )
            try:
                while query.strip():
                    result = session.search(
                        query, limit=limit, mode=mode, use_feedback=feedback, save_run=save_run
                    )
                    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
                    if not interactive:
                        break
                    try:
                        query = input("Search (empty line exits): ")
                    except EOFError:
                        break
            finally:
                session.close()
            return
        options: dict[str, Any] = dict(
            limit=limit,
            mode=mode,
            virtual_collection=virtual_collection,
            virtual_revision=virtual_revision,
            reranker=reranker,
            use_feedback=feedback,
            candidate_depths=depths,
            siglip_precision=siglip_precision,
        )
        if not interactive:
            if save_run:
                from realopen_lens.learning.feedback import search_and_record

                recorded = search_and_record(
                    query,
                    collection=collection,
                    index_dir=index_dir,
                    models=models,
                    threads=threads,
                    **options,
                )
                results = recorded.results
                typer.echo(f"Saved search: {recorded.run_id}", err=True)
            else:
                results = search_collection(
                    query,
                    collection=collection,
                    index_dir=index_dir,
                    models=models,
                    threads=threads,
                    **options,
                )
            typer.echo(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))
        else:
            import sys
            from realopen_lens.search import SearchSession

            failed = False
            with SearchSession(
                collection=collection, index_dir=index_dir, models=models, threads=threads
            ) as session:
                while query.strip():
                    try:
                        if save_run:
                            recorded = session.search_recorded(query, **options)
                            results = recorded.results
                            typer.echo(f"Saved search: {recorded.run_id}", err=True)
                        else:
                            results = session.search(query, **options)
                        typer.echo(
                            json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2)
                        )
                    except (
                        OSError,
                        ValueError,
                        RuntimeError,
                        TimeoutError,
                        sqlite3.Error,
                    ) as error:
                        failed = True
                        typer.echo(f"Error: {error}", err=True)
                    typer.echo("Query (empty line to exit): ", err=True, nl=False)
                    query = sys.stdin.readline().rstrip("\r\n")
            if failed:
                raise typer.Exit(1)
    except (OSError, ValueError, RuntimeError, TimeoutError, sqlite3.Error) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error


@app.command("similar")
def similar_images(
    image: Annotated[
        Path | None, typer.Argument(help="Local query image; omit with --image-id")
    ] = None,
    image_id: Annotated[str | None, typer.Option(help="Image ID in the active generation")] = None,
    collection: Annotated[str, typer.Option()] = "personal",
    all_folders: Annotated[bool, typer.Option(help="Search every indexed folder")] = False,
    virtual_collection: Annotated[
        str | None, typer.Option(help="Restrict to this virtual collection ID")
    ] = None,
    virtual_revision: Annotated[int | None, typer.Option(min=1)] = None,
    index_dir: Annotated[Path | None, typer.Option()] = None,
    models: Annotated[Path, typer.Option()] = Path("models"),
    limit: Annotated[int, typer.Option(min=1, max=100)] = 10,
    threads: Annotated[int, typer.Option(min=1, max=8)] = 2,
    include_self: Annotated[
        bool, typer.Option(help="Include exact query bytes in results")
    ] = False,
    save_run: Annotated[
        bool, typer.Option(help="Save this search for explicit relevance feedback")
    ] = False,
    siglip_precision: Annotated[
        str | None, typer.Option(help="Assert int8 or fp32; otherwise use collection precision")
    ] = None,
):
    """Find visually similar images from a local image or indexed image ID."""
    from dataclasses import asdict
    from realopen_lens.search import search_similar

    try:
        if all_folders:
            from realopen_lens.retrieval.library import LibrarySession
            from realopen_lens.storage.catalog import default_directory

            if virtual_collection or virtual_revision or siglip_precision:
                raise ValueError("Library similarity cannot use folder-specific overrides")
            session = LibrarySession(
                index_dir=index_dir or default_directory(), models=models, threads=threads
            )
            try:
                result = session.similar(
                    image,
                    image_id=image_id,
                    collection=collection,
                    limit=limit,
                    include_self=include_self,
                    save_run=save_run,
                )
                typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
            finally:
                session.close()
            return
        options: dict[str, Any] = dict(
            image_id=image_id,
            collection=collection,
            virtual_collection=virtual_collection,
            virtual_revision=virtual_revision,
            index_dir=index_dir,
            models=models,
            limit=limit,
            threads=threads,
            include_self=include_self,
            siglip_precision=siglip_precision,
        )
        if save_run:
            from realopen_lens.learning.feedback import similar_and_record

            recorded = similar_and_record(image, **options)
            results = recorded.results
            typer.echo(f"Saved search: {recorded.run_id}", err=True)
        else:
            results = search_similar(image, **options)
        typer.echo(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError, TimeoutError, sqlite3.Error) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error


def _feedback_echo(operation: Callable[..., Any], *args: Any, **options: Any) -> None:
    try:
        typer.echo(json.dumps(operation(*args, **options), ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error


@feedback_app.command("runs")
def feedback_runs(
    index_dir: Annotated[Path | None, typer.Option()] = None,
    limit: Annotated[int, typer.Option(min=1, max=100)] = 20,
):
    """List recent explicitly saved searches."""
    from realopen_lens.learning.feedback import saved_runs

    _feedback_echo(saved_runs, index_dir=index_dir, limit=limit)


@feedback_app.command("hidden")
def feedback_hidden(
    query: str,
    index_dir: Annotated[Path | None, typer.Option()] = None,
    collection: Annotated[str, typer.Option()] = "personal",
    virtual_collection: Annotated[str | None, typer.Option()] = None,
    offset: Annotated[int, typer.Option(min=0)] = 0,
    limit: Annotated[int, typer.Option(min=1, max=100)] = 50,
):
    """Browse hidden exact-query results without model inference."""
    from realopen_lens.retrieval.memory import hidden
    from realopen_lens.storage.catalog import default_directory

    _feedback_echo(
        hidden,
        index_dir or default_directory(),
        query,
        collection=collection,
        virtual_collection=virtual_collection,
        offset=offset,
        limit=limit,
    )


@feedback_app.command("show")
def feedback_show(run_id: str, index_dir: Annotated[Path | None, typer.Option()] = None):
    """Inspect the original search, returned results and current judgments."""
    from realopen_lens.learning.feedback import saved_search

    _feedback_echo(saved_search, run_id, index_dir=index_dir)


@feedback_app.command("set")
def feedback_set(
    run_id: str,
    image_id: str,
    label: str,
    index_dir: Annotated[Path | None, typer.Option()] = None,
):
    """Set or correct a result's explicit label: relevant or irrelevant."""
    from realopen_lens.learning.feedback import judge

    _feedback_echo(judge, run_id, image_id, label, index_dir=index_dir)


@feedback_app.command("remove")
def feedback_remove(
    run_id: str,
    image_id: str,
    index_dir: Annotated[Path | None, typer.Option()] = None,
):
    """Remove an active label; audit history is retained."""
    from realopen_lens.learning.feedback import remove_judgment

    _feedback_echo(remove_judgment, run_id, image_id, index_dir=index_dir)


@feedback_app.command("history")
def feedback_history(run_id: str, index_dir: Annotated[Path | None, typer.Option()] = None):
    """Show label corrections and removals in order."""
    from realopen_lens.learning.feedback import judgment_history

    _feedback_echo(judgment_history, run_id, index_dir=index_dir)


@feedback_app.command("export")
def feedback_export(
    output: Path,
    index_dir: Annotated[Path | None, typer.Option()] = None,
    overwrite: Annotated[bool, typer.Option(help="Replace an existing export")] = False,
):
    """Export current explicit labels as UTF-8 JSONL; prints the record count."""
    from realopen_lens.learning.feedback import export_feedback

    _feedback_echo(export_feedback, output, index_dir=index_dir, overwrite=overwrite)


app.add_typer(groups_app, name="groups")
app.add_typer(duplicates_app, name="duplicates")


training_app = typer.Typer(
    no_args_is_help=True, help="Reviewed feedback datasets and local reranker versions."
)
app.add_typer(training_app, name="training")


@training_app.command("audit")
def training_audit(
    index_dir: Annotated[Path | None, typer.Option()] = None,
    collection: Annotated[str, typer.Option()] = "personal",
):
    from realopen_lens.learning.review import audit
    from realopen_lens.storage.catalog import default_directory

    _feedback_echo(audit, index_dir or default_directory(), collection)


@training_app.command("prepare")
def training_prepare(
    review: Path, output: Path, index_dir: Annotated[Path | None, typer.Option()] = None
):
    """Freeze reviewed labels and group-separated train/validation/test data."""
    from realopen_lens.learning.dataset import prepare
    from realopen_lens.storage.catalog import default_directory

    _feedback_echo(prepare, index_dir or default_directory(), review, output)


@training_app.command("train")
def training_train(
    dataset: Path,
    index_dir: Annotated[Path | None, typer.Option()] = None,
    models: Annotated[Path, typer.Option()] = Path("models"),
):
    """Train, select on validation, then evaluate production queries; progress goes to stderr."""
    from realopen_lens.learning.workflow import train
    from realopen_lens.storage.catalog import default_directory

    try:
        result = train(
            dataset,
            index_dir or default_directory(),
            models,
            on_progress=lambda p: typer.echo(json.dumps(p), err=True),
        )
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        if result["status"] != "completed":
            raise typer.Exit(1)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error


@training_app.command("list")
def training_list(index_dir: Annotated[Path | None, typer.Option()] = None):
    from realopen_lens.learning.registry import versions
    from realopen_lens.storage.catalog import default_directory

    _feedback_echo(versions, index_dir or default_directory())


@training_app.command("activate")
def training_activate(
    collection: str,
    version: str,
    index_dir: Annotated[Path | None, typer.Option()] = None,
    acknowledge_experimental: Annotated[bool, typer.Option()] = False,
    rollback: Annotated[bool, typer.Option()] = False,
):
    from realopen_lens.learning.registry import change
    from realopen_lens.storage.catalog import default_directory

    _feedback_echo(
        change,
        index_dir or default_directory(),
        collection,
        version,
        acknowledge_experimental=acknowledge_experimental,
        rollback=rollback,
    )


@training_app.command("disable")
def training_disable(collection: str, index_dir: Annotated[Path | None, typer.Option()] = None):
    """Restore hybrid ranking for this collection; retain model and audit history."""
    from realopen_lens.learning.registry import change
    from realopen_lens.storage.catalog import default_directory

    _feedback_echo(change, index_dir or default_directory(), collection, None)


@app.command("serve")
def serve_gallery(
    index_dir: Annotated[Path | None, typer.Option()] = None,
    models: Annotated[Path | None, typer.Option()] = None,
    allow_root: Annotated[
        list[Path] | None, typer.Option(help="Allowed image import root; repeatable")
    ] = None,
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8765,
    threads: Annotated[int, typer.Option(min=1, max=8)] = 2,
    dev_origin: Annotated[str | None, typer.Option(help="Explicit loopback Vite origin")] = None,
):
    """Serve the private local gallery on 127.0.0.1 (one API worker)."""
    try:
        import uvicorn
        from realopen_lens.api.app import create_app
        from realopen_lens.storage.catalog import default_directory

        from realopen_lens.workspace.identity import models_directory

        workspace_root = index_dir or default_directory()
        roots = tuple(allow_root) if allow_root is not None else None
        api = create_app(
            index_dir=workspace_root,
            models=models or models_directory(workspace_root),
            allowed_roots=roots,
            threads=threads,
            port=port,
            dev_origin=dev_origin,
        )
        typer.echo(f"RealOpen Lens: http://127.0.0.1:{port} (local access only)")
        uvicorn.run(api, host="127.0.0.1", port=port, workers=1, access_log=False)
    except ImportError as error:
        typer.echo("Install the web adapter with 'uv sync --extra web'.", err=True)
        raise typer.Exit(1) from error
    except (OSError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error


if __name__ == "__main__":
    app()
