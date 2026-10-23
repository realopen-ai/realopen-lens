"""CLI adapter for explicit virtual collection operations."""

import json
from pathlib import Path
import sqlite3
from typing import Annotated, Any

import typer

from realopen_lens.organization import collections
from realopen_lens.storage.catalog import default_directory

app = typer.Typer(no_args_is_help=True, help="Virtual image collections without moving files.")


def call(method: str, index_dir: Path | None, collection: str, *args: Any, **options: Any):
    try:
        result = getattr(collections, method)(
            index_dir or default_directory(), collection, *args, **options
        )
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, sqlite3.Error) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error


@app.command("list")
def list_groups(
    index_dir: Annotated[Path | None, typer.Option()] = None,
    collection: Annotated[str, typer.Option()] = "personal",
    all_folders: Annotated[
        bool, typer.Option(help="Use a collection spanning all folders")
    ] = False,
):
    call("list_collections", index_dir, collection, all_folders=all_folders)


@app.command("create")
def create_group(
    name: str,
    index_dir: Annotated[Path | None, typer.Option()] = None,
    collection: Annotated[str, typer.Option()] = "personal",
    all_folders: Annotated[
        bool, typer.Option(help="Use a collection spanning all folders")
    ] = False,
):
    call("create", index_dir, collection, name, all_folders=all_folders)


@app.command("rename")
def rename_group(
    group_id: str,
    name: str,
    revision: Annotated[int, typer.Option(min=1)],
    index_dir: Annotated[Path | None, typer.Option()] = None,
    collection: Annotated[str, typer.Option()] = "personal",
    all_folders: Annotated[
        bool, typer.Option(help="Use a collection spanning all folders")
    ] = False,
):
    call("rename", index_dir, collection, group_id, name, revision, all_folders=all_folders)


@app.command("delete")
def delete_group(
    group_id: str,
    revision: Annotated[int, typer.Option(min=1)],
    index_dir: Annotated[Path | None, typer.Option()] = None,
    collection: Annotated[str, typer.Option()] = "personal",
    all_folders: Annotated[
        bool, typer.Option(help="Use a collection spanning all folders")
    ] = False,
):
    """Delete the group and memberships; source images remain in place."""
    call("remove", index_dir, collection, group_id, revision, all_folders=all_folders)


@app.command("add")
def add_images(
    group_id: str,
    image_ids: list[str],
    revision: Annotated[int, typer.Option(min=1)],
    index_dir: Annotated[Path | None, typer.Option()] = None,
    collection: Annotated[str, typer.Option()] = "personal",
    all_folders: Annotated[
        bool, typer.Option(help="Use a collection spanning all folders")
    ] = False,
):
    call(
        "change_members",
        index_dir,
        collection,
        group_id,
        image_ids,
        revision,
        all_folders=all_folders,
    )


@app.command("remove")
def remove_images(
    group_id: str,
    image_ids: list[str],
    revision: Annotated[int, typer.Option(min=1)],
    index_dir: Annotated[Path | None, typer.Option()] = None,
    collection: Annotated[str, typer.Option()] = "personal",
    all_folders: Annotated[
        bool, typer.Option(help="Use a collection spanning all folders")
    ] = False,
):
    call(
        "change_members",
        index_dir,
        collection,
        group_id,
        image_ids,
        revision,
        remove=True,
        all_folders=all_folders,
    )


@app.command("images")
def browse_group(
    group_id: str,
    index_dir: Annotated[Path | None, typer.Option()] = None,
    collection: Annotated[str, typer.Option()] = "personal",
    all_folders: Annotated[
        bool, typer.Option(help="Use a collection spanning all folders")
    ] = False,
    offset: Annotated[int, typer.Option(min=0)] = 0,
    limit: Annotated[int, typer.Option(min=1, max=100)] = 60,
):
    call(
        "browse",
        index_dir,
        collection,
        group_id,
        offset=offset,
        limit=limit,
        all_folders=all_folders,
    )
