"""Application command line entry points for explicit setup and diagnostics."""

import json
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    no_args_is_help=True, help="Private, offline image search: setup and diagnostics."
)
models_app = typer.Typer(
    no_args_is_help=True, help="Explicit model downloads and integrity checks."
)
app.add_typer(models_app, name="models")


@models_app.command("fetch")
def fetch_models(
    directory: Annotated[Path, typer.Option("--directory")] = Path("models"),
    precision: Annotated[str, typer.Option()] = "int8",
) -> None:
    """Explicitly download pinned ONNX model artifacts."""
    from realopen_lens.models import fetch

    try:
        fetch(directory, precision, typer.echo)
    except (OSError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error


@models_app.command("verify")
def verify_models(
    directory: Annotated[Path, typer.Option("--directory")] = Path("models"),
    precision: Annotated[str, typer.Option()] = "int8",
) -> None:
    """Verify local model checksums without making network requests."""
    from realopen_lens.models import verify

    try:
        verify(directory, precision)
    except (OSError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo("All selected model files verified.")


@models_app.command("fetch-text-detector")
def fetch_detector(
    directory: Annotated[Path, typer.Option("--directory")] = Path("models"),
) -> None:
    """Explicitly download the pinned optional text-localization model."""
    from realopen_lens.models import fetch_text_detector

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
        Path | None, typer.Option(help="Enable text-region detection with this local model")
    ] = None,
    region_mode: Annotated[
        str, typer.Option(help="gate whole-image OCR or mask detected regions")
    ] = "gate",
) -> None:
    """Read an image locally; optional GLM-OCR uses only numeric loopback HTTP."""
    from realopen_lens.ocr_backends import create_backend

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


if __name__ == "__main__":
    app()
