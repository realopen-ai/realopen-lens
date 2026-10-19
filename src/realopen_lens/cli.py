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
    """Download pinned model artifacts. This is the only command that uses the network."""
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


if __name__ == "__main__":
    app()
