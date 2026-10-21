"""Explicit workspace inspection, import preview and initialization commands."""

import json
from pathlib import Path
import sqlite3
from typing import Annotated

import typer

from realopen_lens.storage.catalog import default_directory
from realopen_lens.workspace.identity import describe
from realopen_lens.workspace.service import preview, initialize

app = typer.Typer(no_args_is_help=True, help="Inspect and set up a persistent everyday workspace.")


@app.command("info")
def info(index_dir: Annotated[Path | None, typer.Option()] = None):
    try:
        typer.echo(
            json.dumps(describe(index_dir or default_directory()), ensure_ascii=False, indent=2)
        )
    except (ValueError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error


@app.command("import")
def import_indexes(
    source: Annotated[
        list[Path],
        typer.Option("--source", help="Source index; repeat to combine compatible histories"),
    ],
    index_dir: Annotated[
        Path | None, typer.Option(help="New destination; defaults to application data")
    ] = None,
    apply: Annotated[
        str | None, typer.Option(help="Apply the exact SHA256 from a reviewed preview")
    ] = None,
    models: Annotated[Path, typer.Option()] = Path("models"),
    log: Annotated[bool, typer.Option(help="Stream live import progress to stderr")] = False,
):
    """Preview by default. Apply copies into a new directory and preserves originals."""

    def progress(value):
        if log:
            typer.echo(json.dumps(value, ensure_ascii=False), err=True)

    try:
        root = index_dir or default_directory()
        result = (
            initialize(source, root, apply, models=models, on_progress=progress)
            if apply
            else preview(source, root, progress)
        )
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, sqlite3.Error) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
