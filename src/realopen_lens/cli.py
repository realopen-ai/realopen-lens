"""Application command line entry points for explicit setup and diagnostics."""

import json
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Annotated, Any

import typer

app = typer.Typer(
    no_args_is_help=True, help="Private, offline collection indexing and image search."
)
models_app = typer.Typer(
    no_args_is_help=True, help="Explicit model downloads and integrity checks."
)
app.add_typer(models_app, name="models")


@models_app.command("fetch")
def fetch_models(
    directory: Annotated[Path, typer.Option("--directory")] = Path("models"),
    precision: Annotated[str, typer.Option()] = "int8",
    siglip_precision: Annotated[
        str | None, typer.Option(help="Override SigLIP2 precision only: int8 or fp32")
    ] = None,
) -> None:
    """Explicitly download pinned ONNX model artifacts."""
    from realopen_lens.models.manager import fetch

    try:
        fetch(directory, precision, typer.echo, siglip_precision=siglip_precision)
    except (OSError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error


@models_app.command("verify")
def verify_models(
    directory: Annotated[Path, typer.Option("--directory")] = Path("models"),
    precision: Annotated[str, typer.Option()] = "int8",
    siglip_precision: Annotated[
        str | None, typer.Option(help="Override SigLIP2 precision only: int8 or fp32")
    ] = None,
) -> None:
    """Verify local model checksums without making network requests."""
    from realopen_lens.models.manager import verify

    try:
        verify(directory, precision, siglip_precision)
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
    executable = shutil.which("tesseract")
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
            [executable, "--version"], capture_output=True, text=True, check=True
        )
        langs = subprocess.run(
            [executable, "--list-langs"], capture_output=True, text=True, check=True
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
        str, typer.Option(help="SigLIP2 precision: int8 or fp32; E5 stays int8")
    ] = "int8",
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
    collection: Annotated[str, typer.Option()] = "personal",
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
        options: dict[str, Any] = dict(
            limit=limit,
            mode=mode,
            reranker=reranker,
            candidate_depths=depths,
            siglip_precision=siglip_precision,
        )
        if not interactive:
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
    index_dir: Annotated[Path | None, typer.Option()] = None,
    models: Annotated[Path, typer.Option()] = Path("models"),
    limit: Annotated[int, typer.Option(min=1, max=100)] = 10,
    threads: Annotated[int, typer.Option(min=1, max=8)] = 2,
    include_self: Annotated[
        bool, typer.Option(help="Include exact query bytes in results")
    ] = False,
    siglip_precision: Annotated[
        str | None, typer.Option(help="Assert int8 or fp32; otherwise use collection precision")
    ] = None,
):
    """Find visually similar images from a local image or indexed image ID."""
    from dataclasses import asdict
    from realopen_lens.search import search_similar

    try:
        results = search_similar(
            image,
            image_id=image_id,
            collection=collection,
            index_dir=index_dir,
            models=models,
            limit=limit,
            threads=threads,
            include_self=include_self,
            siglip_precision=siglip_precision,
        )
        typer.echo(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError, TimeoutError, sqlite3.Error) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error


if __name__ == "__main__":
    app()
