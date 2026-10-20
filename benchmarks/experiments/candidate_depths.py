"""Run resumable production collection evaluations at predeclared equal channel depths."""

import argparse
from pathlib import Path
from types import SimpleNamespace

from benchmarks.experiments.collections import run
from benchmarks.reporting.candidates import compare
from realopen_lens.runtime.offline import enforce


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depths", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if len(set(args.depths)) != len(args.depths) or any(not 1 <= v <= 1000 for v in args.depths):
        parser.error("Depths must be distinct integers from 1 to 1000")
    enforce()
    directories, errors = [], []
    for depth in args.depths:
        output = args.output / f"depth-{depth}"
        config = SimpleNamespace(
            **{key: value for key, value in vars(args).items() if key not in ("depths", "output")},
            output=output,
            visual_candidates=depth,
            semantic_candidates=depth,
            lexical_candidates=depth,
            defer_test_metrics=False,
        )
        try:
            run(config)
            directories.append(output)
        except Exception as error:
            errors.append(f"depth {depth}: {type(error).__name__}: {error}")
            print(errors[-1], flush=True)
    if errors:
        raise ValueError(
            "Incomplete candidate study; retained runs can resume: " + "; ".join(errors)
        )
    compare(directories, args.output)


if __name__ == "__main__":
    main()
