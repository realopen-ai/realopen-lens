"""Bundled GeoNames city lookup. No runtime network or geocoding service."""

from functools import lru_cache
import gzip
import json
import math
from pathlib import Path
import unicodedata


def normalize(value: str) -> str:
    return " ".join(
        "".join(
            c
            for c in unicodedata.normalize("NFKD", value.casefold())
            if not unicodedata.combining(c)
        ).split()
    )


def distance(lat: float, lon: float, other_lat: float, other_lon: float) -> float:
    a, b = math.radians(lat), math.radians(other_lat)
    value = (
        math.sin((b - a) / 2) ** 2
        + math.cos(a) * math.cos(b) * math.sin(math.radians(other_lon - lon) / 2) ** 2
    )
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(max(0.0, value))))


@lru_cache(maxsize=1)
def gazetteer() -> tuple[dict, dict]:
    value = json.loads(
        gzip.decompress((Path(__file__).parent / "data/cities.json.gz").read_bytes())
    )
    countries = {code: name for code, name in value["countries"].items()}
    names: dict[str, list[dict]] = {}
    for key, name, country, region, lat, lon, population, aliases in value["cities"]:
        city = dict(
            id=key,
            name=name,
            country=country,
            region=region,
            latitude=lat,
            longitude=lon,
            population=population,
        )
        for alias in {normalize(alias) for alias in aliases}:
            # Short airport/admin abbreviations collide with ordinary prose
            # (API, UNE, etc.). Keep actual city names, not these aliases.
            if len(alias) < 4 and alias != normalize(name):
                continue
            if len(alias) < 3:
                continue
            names.setdefault(alias, []).append(city)
    return names, countries


def candidates(value: str) -> list[dict]:
    names, countries = gazetteer()
    value = normalize(value.strip().strip('"'))
    country = None
    region = None
    for code, name in countries.items():
        for suffix in (", " + normalize(name), " " + normalize(name), ", " + code.lower()):
            if value.endswith(suffix):
                value, country = value[: -len(suffix)], code
                break
        if country:
            break
    if "," in value:
        value, region = (part.strip() for part in value.split(",", 1))
    return [
        row
        for row in names.get(value, [])
        if (country is None or row["country"] == country)
        and (region is None or normalize(row["region"]) == region)
    ]


def resolve(value: str, gps: list[dict], radius: float) -> dict | None:
    found = {row["id"]: row for row in candidates(value)}
    if not found:
        return None
    if len(found) > 64:
        raise ValueError(f"Ambiguous place '{value}'; specify a country or region")
    if len(found) > 1:
        nearby = {
            key: row
            for key, row in found.items()
            if any(
                distance(point["latitude"], point["longitude"], row["latitude"], row["longitude"])
                <= radius
                for point in gps
            )
        }
        if len(nearby) == 1:
            found = nearby
        else:
            _, countries = gazetteer()
            labels = sorted(
                {
                    f"{r['name']}, {r['region']}, {countries.get(r['country'], r['country'])}"
                    for r in found.values()
                }
            )
            raise ValueError(
                f"Ambiguous place '{value}'; specify city and country/region: "
                + "; ".join(labels[:8])
            )
    result = next(iter(found.values()))
    _, countries = gazetteer()
    return {
        **result,
        "label": f"{result['name']}, {countries.get(result['country'], result['country'])}",
    }
