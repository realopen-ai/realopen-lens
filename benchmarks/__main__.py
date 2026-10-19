"""Run development benchmarks and generate smoke fixtures from the checkout."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, help="Local inference and retrieval benchmarks.")


@app.command()
def fixture(
    output: Annotated[Path, typer.Option()] = Path(".cache/smoke-images"),
    font: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Generate multilingual text images for smoke testing, not quality evaluation."""
    from benchmarks.datasets.fixtures import create

    try:
        create(output, font)
    except (OSError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Generated smoke fixtures: {output}; queries: {output / 'queries.json'}")


@app.command()
def run(
    data: Annotated[Path, typer.Option()] = Path("data"),
    models: Annotated[Path, typer.Option()] = Path("models"),
    output: Annotated[Path | None, typer.Option()] = None,
    queries: Annotated[Path | None, typer.Option()] = None,
    ocr_truth: Annotated[Path | None, typer.Option()] = None,
    precision: Annotated[str, typer.Option()] = "int8",
    languages: Annotated[
        str, typer.Option(help="Tesseract language codes joined by +")
    ] = "eng+fra+ara",
    limit: Annotated[int, typer.Option(min=1, max=5000)] = 100,
    batch_size: Annotated[int, typer.Option(min=1, max=32)] = 1,
    threads: Annotated[int, typer.Option(min=1, max=32)] = 2,
    repeats: Annotated[int, typer.Option(min=1, max=20)] = 3,
    corpus_kind: Annotated[
        str, typer.Option(help="Use generated-smoke for generated fixtures")
    ] = "user",
) -> None:
    """Run sequential local inference stages and write private JSON/Markdown reports."""
    from benchmarks.experiments.baseline import run as run_benchmark
    from realopen_lens.runtime.offline import enforce

    enforce()
    if output is None:
        output = Path("runs") / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        run_benchmark(
            data,
            models,
            output,
            queries,
            precision,
            languages,
            limit,
            batch_size,
            threads,
            repeats,
            corpus_kind,
            typer.echo,
            ocr_truth,
        )
    except (OSError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error


if __name__ == "__main__":
    app()
