"""Deterministic natural-language metadata constraints applied before candidate selection."""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import re

from realopen_lens.metadata.places import normalize, distance, resolve

MONTHS = [
    ("january", "janvier", "يناير"),
    ("february", "février", "فبراير"),
    ("march", "mars", "مارس"),
    ("april", "avril", "أبريل"),
    ("may", "mai", "مايو"),
    ("june", "juin", "يونيو"),
    ("july", "juillet", "يوليو"),
    ("august", "août", "أغسطس"),
    ("september", "septembre", "سبتمبر"),
    ("october", "octobre", "أكتوبر"),
    ("november", "novembre", "نوفمبر"),
    ("december", "décembre", "ديسمبر"),
]


@dataclass
class QueryPlan:
    original: str
    text: str
    constraints: list[dict] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)

    def report(self) -> dict:
        return {
            "version": 1,
            "original": self.original,
            "visual_query": self.text,
            "constraints": self.constraints,
            "notices": self.notices,
        }

    def matches(self, metadata: dict, modified: list[datetime], folder: str = "") -> bool:
        exif = metadata.get("exif", {})
        for rule in self.constraints:
            kind = rule["kind"]
            if kind == "screenshot":
                if (
                    rule["policy"] == "marked_only"
                    and exif.get("screenshot", {}).get("state") != "marked"
                ):
                    return False
            elif kind == "camera":
                model = normalize(exif.get("camera_model") or "")
                make = normalize(exif.get("camera_make") or "")
                target = normalize(rule["value"])
                if not any(
                    value == target or value.startswith(target + " ")
                    for value in (model, (make + " " + model).strip(), make)
                ):
                    return False
            elif kind == "format":
                if metadata.get("format", "").upper() not in rule["values"]:
                    return False
            elif kind == "folder":
                if normalize(folder) != normalize(rule["value"]):
                    return False
            elif kind == "date":
                captured = exif.get("captured_at")
                values = []
                if rule["basis"] != "modified" and captured:
                    try:
                        values = [date.fromisoformat(captured[:10])]
                    except ValueError:
                        pass
                if rule["basis"] == "modified" or (
                    rule["basis"] == "capture_or_modified" and not values
                ):
                    values = [value.date() for value in modified]
                if not any(rule["start"] <= value.isoformat() < rule["end"] for value in values):
                    return False
            elif kind == "location":
                gps = exif.get("gps")
                if (
                    not gps
                    or distance(
                        gps["latitude"], gps["longitude"], rule["latitude"], rule["longitude"]
                    )
                    > rule["radius_km"]
                ):
                    return False
            elif kind == "dimension":
                if metadata.get(rule["field"], 0) < rule["minimum"]:
                    return False
            elif kind == "size":
                if metadata.get("bytes", 0) < rule["minimum"]:
                    return False
        return True


def marked_screenshot(row: dict) -> bool:
    return row.get("metadata", {}).get("exif", {}).get("screenshot", {}).get("state") == "marked"


def screenshot_ranking(plan: QueryPlan, ranked: list[str], lookup: dict, scores: dict) -> list[str]:
    """Only broad screenshot requests prioritize markers; scene-specific ranking stays intact."""
    rule = next((rule for rule in plan.constraints if rule["kind"] == "screenshot"), None)
    if not rule or not rule.get("promote"):
        return ranked
    marked = {key for key, row in lookup.items() if marked_screenshot(row)}
    preferred = sorted(marked, key=lambda key: (-scores.get(key, float("-inf")), key))
    return preferred + [key for key in ranked if key not in marked]


def modified_dates(row: dict, paths: tuple[str, ...] | list[str]) -> list[datetime]:
    # File modification dates belong to each source path, not the content hash.
    from pathlib import Path

    values = []
    for path in paths:
        try:
            values.append(datetime.fromtimestamp(Path(path).stat().st_mtime).astimezone())
        except OSError:
            continue
    return values


def compile_query(query: str, rows: list[dict], *, today: date | None = None) -> QueryPlan:
    plan = QueryPlan(query, query)
    today = today or date.today()
    content = query

    def remove(match, rule):
        nonlocal content
        plan.constraints.append(rule)
        content = content[: match.start()] + " " + content[match.end() :]

    # Quoted folder names are explicit. Never expand the caller's search scope.
    found = re.search(r'\b(?:from|in)\s+(?:the\s+)?folder\s+"([^"\n]{1,128})"', content, re.I)
    if found:
        remove(found, {"kind": "folder", "value": found[1], "label": f"Folder: {found[1]}"})

    camera = re.search(
        r"(?:\b(?:taken|shot|photographed)\s+(?:with|by|on)|"
        r"\b(?:prises?|capturées?)\s+avec|بكاميرا|بواسطة)\s+"
        r"(?:my\s+|an?\s+|mon\s+)?(.+?)"
        r"(?=\s+(?:in|near|around|on|from|between|en|à|في)\b|$)",
        content,
        re.I,
    )
    if not camera:
        implicit = re.search(r"\b(?:with|on|avec)\s+(?:my\s+|mon\s+)?(.+)", content, re.I)
        if implicit:
            aliases = {r["metadata"].get("exif", {}).get("camera_model") for r in rows}
            aliases.discard(None)
            for alias in sorted(aliases, key=len, reverse=True):
                if re.match(re.escape(alias) + r"\b", implicit[1], re.I):
                    camera = implicit
                    break
    if camera:
        value = camera[1].strip(' .,"')
        aliases = {r["metadata"].get("exif", {}).get("camera_model") for r in rows}
        aliases.discard(None)
        matched = next(
            (
                re.match(re.escape(alias) + r"\b", camera[1], re.I)
                for alias in sorted(aliases, key=len, reverse=True)
                if re.match(re.escape(alias) + r"\b", camera[1], re.I)
            ),
            None,
        )
        if matched:
            value = matched[0]
            plan.constraints.append({"kind": "camera", "value": value, "label": f"Camera: {value}"})
            content = content[: camera.start()] + " " + content[camera.start(1) + matched.end() :]
        else:
            remove(camera, {"kind": "camera", "value": value, "label": f"Camera: {value}"})

    # A place printed inside a document is content, not proof of where the
    # image was captured. Explicit capture wording can opt into GPS constraints.
    document_content = re.search(
        r"\b(?:document|invoice|receipt|form|diagram|advertisement|poster|terminal|"
        r"publicité|affiche|facture|reçu|diagramme)\b|إعلان|فاتورة|مخطط|نموذج",
        query,
        re.I,
    )
    capture_location = re.search(
        r"(?:taken|shot|photographed|captured|prises?|photographiées?|التقطت).*?"
        r"(?:\bin\b|\bnear\b|\bà\b|في)",
        query,
        re.I,
    )
    allow_location = not document_content or bool(capture_location)

    # Explicit dates use local calendar dates. Unknown capture timezone stays unknown.
    prefix = r"(?:\b(?:(?:taken|shot|photographed|modified)\s+)?(?:in|on|from)|\ben|في)\s+"
    months = {normalize(name): index + 1 for index, names in enumerate(MONTHS) for name in names}
    month_names = "|".join(re.escape(name) for names in MONTHS for name in names)
    found = re.search(prefix + rf"({month_names})\s+(\d{{4}})\b", content, re.I)
    start = end = None
    if found:
        start = date(int(found[2]), months[normalize(found[1])], 1)
        end = date(start.year + (start.month == 12), start.month % 12 + 1, 1)
    else:
        found = re.search(
            r"\b(?:taken\s+|modified\s+)?between\s+(\d{4}-\d{2}-\d{2})"
            r"\s+and\s+(\d{4}-\d{2}-\d{2})\b",
            content,
            re.I,
        )
        if found:
            start, end = date.fromisoformat(found[1]), date.fromisoformat(found[2]) + timedelta(
                days=1
            )
        else:
            found = re.search(prefix + r"(\d{4}-\d{2}-\d{2}|(?:19|20)\d{2})\b", content, re.I)
            if found:
                start = (
                    date(int(found[1]), 1, 1)
                    if len(found[1]) == 4
                    else date.fromisoformat(found[1])
                )
                end = (
                    date(start.year + 1, 1, 1) if len(found[1]) == 4 else start + timedelta(days=1)
                )
    if not found:
        found = re.search(
            r"\b(?:(?:taken|shot|modified)\s+|from\s+|in\s+)?"
            r"(last week|last month|last year|this week|today|yesterday)\b",
            content,
            re.I,
        )
        if found:
            period = found[1].lower()
            if period == "last week":
                end = today - timedelta(days=today.weekday())
                start = end - timedelta(days=7)
            elif period == "this week":
                start = today - timedelta(days=today.weekday())
                end = start + timedelta(days=7)
            elif period == "last month":
                end = today.replace(day=1)
                start = (end - timedelta(days=1)).replace(day=1)
            elif period == "last year":
                start, end = date(today.year - 1, 1, 1), date(today.year, 1, 1)
            else:
                start = today - timedelta(days=int(period == "yesterday"))
                end = start + timedelta(days=1)
    if found and start is not None and end is not None:
        if end <= start:
            raise ValueError("Date range ends before it starts")
        phrase = found[0].lower()
        basis = (
            "modified"
            if "modified" in phrase
            else (
                "capture"
                if (
                    any(word in phrase for word in ("taken", "shot", "photographed"))
                    or any(rule["kind"] == "camera" for rule in plan.constraints)
                )
                else "capture_or_modified"
            )
        )
        remove(
            found,
            {
                "kind": "date",
                "basis": basis,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "label": f"Date: {start} to {end - timedelta(days=1)} ({basis.replace('_', ' ')})",
            },
        )
        if basis == "capture_or_modified":
            plan.notices.append(
                "Dates use capture date when present, otherwise the source file's "
                "modification date. "
                "No timezone is inferred for EXIF capture dates."
            )

    formats = list(re.finditer(r"\b(jpe?g|png|webp|heic|heif|tiff?|bmp)\b", content, re.I))
    if formats:
        format_aliases = {"JPG": "JPEG", "TIF": "TIFF", "HEIC": "HEIF"}
        values = sorted(
            {format_aliases.get(match[1].upper(), match[1].upper()) for match in formats}
        )
        plan.constraints.append(
            {"kind": "format", "values": values, "label": "Format: " + ", ".join(values)}
        )
        for match in reversed(formats):
            content = content[: match.start()] + " " + content[match.end() :]

    found = re.search(
        r"\b(?:at least|minimum)\s+(\d{1,6})\s+(?:pixels?\s+)?(wide|tall)\b", content, re.I
    )
    if found:
        remove(
            found,
            {
                "kind": "dimension",
                "field": "width" if found[2].lower() == "wide" else "height",
                "minimum": int(found[1]),
                "label": found[0],
            },
        )
    found = re.search(
        r"\b(?:larger than|at least)\s+(\d{1,12}(?:\.\d{1,6})?)\s*(KB|MB|GB)\b", content, re.I
    )
    if found:
        remove(
            found,
            {
                "kind": "size",
                "minimum": int(
                    float(found[1]) * 1024 ** ({"KB": 1, "MB": 2, "GB": 3}[found[2].upper()])
                ),
                "label": found[0],
            },
        )

    found = re.search(
        r"(?:\bwithin\s+(\d+(?:\.\d+)?)\s*km\s+of|\bnear)"
        r"\s+(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)",
        content,
        re.I,
    )
    if found:
        radius, lat, lon = float(found[1] or 25), float(found[2]), float(found[3])
        if not (0 < radius <= 1000 and -90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Use valid latitude/longitude and a radius from 0 to 1000 km")
        remove(
            found,
            {
                "kind": "location",
                "latitude": lat,
                "longitude": lon,
                "radius_km": radius,
                "label": f"Within {radius:g} km of {lat:g}, {lon:g}",
            },
        )
    elif allow_location:
        gps = [row["metadata"].get("exif", {}).get("gps") for row in rows]
        for place in list(
            re.finditer(
                r"(?:\bwithin\s+(\d+(?:\.\d+)?)\s*km\s+of|"
                r"\b(?:in|near|around)|près de|à|في|قرب)\s+([^\n]+)",
                content,
                re.I,
            )
        ):
            radius = float(place[1] or 25)
            if not 0 < radius <= 1000:
                raise ValueError("Use a radius from 0 to 1000 km")
            spans = list(re.finditer(r"\S+", place[2]))
            words = [item[0] for item in spans]
            for count in range(min(len(words), 8), 0, -1):
                name = " ".join(words[:count]).strip(' ."')
                city = resolve(name, [point for point in gps if point], radius)
                if city is not None:
                    end_pos = place.start(2) + spans[count - 1].end()
                    plan.constraints.append(
                        {
                            "kind": "location",
                            "latitude": city["latitude"],
                            "longitude": city["longitude"],
                            "radius_km": radius,
                            "geonames_id": city["id"],
                            "label": city["label"] + f" · within {radius:g} km",
                        }
                    )
                    content = content[: place.start()] + " " + content[end_pos:]
                    break
            if any(rule["kind"] == "location" for rule in plan.constraints):
                break
    screenshot_pattern = r"\bscreenshots?\b|\bscreen captures?\b|captures? d[’']écran|لقطات? شاشة"
    screenshot = re.search(screenshot_pattern, content, re.I)
    negative = re.search(
        r"(?:not|without|excluding|sans|pas de|بدون)\s+(?:" + screenshot_pattern + ")",
        content,
        re.I,
    )
    if screenshot and not negative:
        strict = re.search(
            r"\b(?:marked|metadata-marked)\s+(?:" + screenshot_pattern + ")", content, re.I
        )
        broad_text = re.sub(screenshot_pattern, "", content, flags=re.I)
        broad = (
            re.sub(
                r"\b(?:show me|find|all|photos?|images?|pictures?)\b", "", broad_text, flags=re.I
            ).strip(" ,.")
            == ""
        )
        if strict or any(marked_screenshot(row) for row in rows):
            plan.constraints.append(
                {
                    "kind": "screenshot",
                    "policy": "marked_only" if strict else "markers_and_visual",
                    "promote": bool(broad and not strict),
                    "label": (
                        "Screenshot markers only"
                        if strict
                        else "Screenshots: markers + visual fallback"
                    ),
                }
            )
            if strict:
                content = content[: strict.start()] + " " + content[strict.end() :]
                content = re.sub(r"^\s*(?:of|showing)\s+", "", content, flags=re.I)
        plan.notices.append(
            "Screenshot markers are evidence, not proof. Unmarked files remain unknown"
            + (
                " and are excluded by this explicit marker-only request."
                if strict
                else " and stay eligible for visual/OCR retrieval."
            )
        )
        if any("screenshot" not in row["metadata"].get("exif", {}) for row in rows):
            plan.notices.append("Refresh older folders to extract embedded screenshot markers.")
    if plan.constraints:
        content = re.sub(r"\s+", " ", content).strip(" ,.")
        content = re.sub(
            r"^(?:show me\s+|find\s+|all\s+)?(?:photos?|images?|pictures?|صور)(?:\s+of)?\b",
            "",
            content,
            flags=re.I,
        ).strip()
        content = re.sub(r"\b(?:photos?|images?|pictures?)$", "", content, flags=re.I).strip()
        content = content.strip(" ,.")
        plan.text = content
        missing = sum(row["metadata"].get("exif", {}).get("schema") != 1 for row in rows)
        if missing:
            plan.notices.append(
                f"{missing} indexed images lack extracted EXIF fields; "
                "refresh their folders to backfill metadata."
            )
    return plan


def filter_paths(plan: QueryPlan, manifest: dict, paths: dict, folder: str) -> dict:
    if not plan.constraints:
        return paths
    lookup = {row["image_id"]: row for row in manifest["images"]}
    return {
        key: values
        for key, values in paths.items()
        if plan.matches(lookup[key]["metadata"], modified_dates(lookup[key], values), folder)
    }
