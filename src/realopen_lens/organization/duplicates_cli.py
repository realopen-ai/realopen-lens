"""Command-line adapter for duplicate scans and review."""

import json
from pathlib import Path
from typing import Annotated

import typer

from realopen_lens.organization import duplicates
from realopen_lens.storage.catalog import default_directory

app = typer.Typer(
    no_args_is_help=True, help="Scan and review image duplicates without deleting files."
)


def call(method: str, index_dir: Path | None, collection: str, **kwargs):
    try:
        result = getattr(duplicates, method)(index_dir or default_directory(), collection, **kwargs)
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get("status") == "completed_with_errors":
            raise typer.Exit(1)
    except (OSError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error


@app.command("scan")
def scan(
    index_dir: Annotated[Path | None, typer.Option()] = None,
    collection: Annotated[str, typer.Option()] = "personal",
    log: Annotated[bool, typer.Option(help="Show scan progress on stderr")] = False,
):
    def progress(value):
        typer.echo(
            f"{value['phase']}: {value.get('step', 0)}/{value.get('max_steps', 0)}", err=True
        )

    call("scan", index_dir, collection, on_progress=progress if log else None)


@app.command("list")
def list_groups(
    index_dir: Annotated[Path | None, typer.Option()] = None,
    collection: Annotated[str, typer.Option()] = "personal",
    offset: Annotated[int, typer.Option(min=0)] = 0,
    limit: Annotated[int, typer.Option(min=1, max=100)] = 20,
):
    call("browse", index_dir, collection, offset=offset, limit=limit)


@app.command("review")
def review(
    pair_id: str,
    state: str,
    index_dir: Annotated[Path | None, typer.Option()] = None,
    collection: Annotated[str, typer.Option()] = "personal",
):
    call("review", index_dir, collection, pair_id=pair_id, state=state)
