"""Build the pinned offline city index from locally downloaded GeoNames files."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import zipfile


def build(cities: Path, countries: Path, output: Path):
    names = {}
    for line in countries.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            row = line.split("\t")
            names[row[0]] = row[4]
    with zipfile.ZipFile(cities) as archive:
        data = archive.read("cities15000.txt").decode("utf-8")
    rows = []
    for line in data.splitlines():
        row = line.split("\t")
        rows.append(
            [
                row[0],
                row[1],
                row[8],
                row[10],
                float(row[4]),
                float(row[5]),
                int(row[14]),
                sorted(set([row[1], row[2], *row[3].split(",")]) - {""}),
            ]
        )
    rows.sort(key=lambda row: int(row[0]))
    payload = {
        "schema": 1,
        "source": "GeoNames cities15000",
        "license": "CC-BY-4.0",
        "cities_sha256": hashlib.sha256(cities.read_bytes()).hexdigest(),
        "countries_sha256": hashlib.sha256(countries.read_bytes()).hexdigest(),
        "countries": names,
        "cities": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(
        gzip.compress(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(), mtime=0
        )
    )
    print(f"{len(rows)} cities; {output.stat().st_size:,} compressed bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cities", type=Path)
    parser.add_argument("countries", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.cities, args.countries, args.output)
