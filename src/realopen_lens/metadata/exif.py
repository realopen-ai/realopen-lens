"""Bounded, JSON-safe EXIF extraction; absent/invalid values remain unknown."""

from datetime import datetime
import math
import re
from typing import Any
from xml.etree import ElementTree

SCHEMA = 1


def text(value: Any) -> str | None:
    if isinstance(value, bytes):
        value = value[:1024].decode("utf-8", errors="replace")
    if not isinstance(value, str):
        return None
    return " ".join("".join(c for c in value[:1024] if c.isprintable()).split())[:256] or None


def number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None


def timestamp(value: Any, offset: Any = None, subsecond: Any = None) -> str | None:
    value = text(value)
    if not value or not re.fullmatch(r"\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}", value):
        return None
    try:
        result = datetime.strptime(value, "%Y:%m:%d %H:%M:%S").isoformat()
    except ValueError:
        return None
    fraction = text(subsecond)
    if fraction and fraction.isdigit():
        result += "." + fraction[:6]
    zone = text(offset)
    if zone and re.fullmatch(r"[+-](?:0\d|1[0-4]):[0-5]\d", zone):
        if zone[1:3] != "14" or zone[4:] == "00":
            result += zone
    return result


def coordinate(value: Any, reference: Any, limit: int) -> float | None:
    reference = text(reference)
    if reference not in ({"N", "S"} if limit == 90 else {"E", "W"}):
        return None
    try:
        parts = [number(item) for item in value]
        if len(parts) != 3 or any(item is None for item in parts):
            return None
        degrees, minutes, seconds = [item for item in parts if item is not None]
        if not (0 <= degrees <= limit and 0 <= minutes < 60 and 0 <= seconds < 60):
            return None
        result = degrees + minutes / 60 + seconds / 3600
        if result > limit:
            return None
        return -result if reference in {"S", "W"} else result
    except (TypeError, ValueError):
        return None


def user_comment(value: Any) -> str | None:
    """EXIF UserComment has an eight-byte charset header, unlike ordinary text."""
    if not isinstance(value, bytes):
        return text(value)
    value = value[:2048]
    header, payload = value[:8], value[8:]
    try:
        if header == b"ASCII\0\0\0":
            return text(payload.decode("ascii", errors="replace"))
        if header == b"UNICODE\0":
            if payload.startswith((b"\xff\xfe", b"\xfe\xff")):
                return text(payload.decode("utf-16"))
            codec = "utf-16-be" if payload[0::2].count(0) >= payload[1::2].count(0) else "utf-16-le"
            return text(payload.decode(codec))
        if header == b"JIS\0\0\0\0\0":
            return text(payload.decode("shift_jis", errors="replace"))
    except UnicodeError:
        return None
    return text(value)


def screenshot_metadata(image, basic: dict, exif: dict) -> dict:
    """Explicit embedded markers only; absence is unknown, never a negative label."""
    evidence = []

    def consider(source, value):
        value = text(value)
        if value and value.casefold() in {"screenshot", "screen capture"}:
            evidence.append({"source": source, "value": value})

    consider("EXIF.UserComment", user_comment(exif.get(37510)))
    consider("EXIF.ImageDescription", basic.get(270))
    if image.format == "PNG":
        for key in ("Description", "Comment"):
            consider("PNG." + key, image.info.get(key))
    # Parse only bounded UTF-8 XMP packets and the specific Dublin Core field.
    # Reject declarations before XML parsing, including entity expansion payloads.
    packet = image.info.get("xmp", image.info.get("XML:com.adobe.xmp"))
    if isinstance(packet, (bytes, str)) and len(packet) <= 65536:
        try:
            xml = packet.decode("utf-8") if isinstance(packet, bytes) else packet
            if "<!doctype" not in xml.casefold() and "<!entity" not in xml.casefold():
                root = ElementTree.fromstring(xml)
                for node in root.iter():
                    consider(
                        "XMP.dc.description",
                        node.get("{http://purl.org/dc/elements/1.1/}description"),
                    )
                for node in root.iter("{http://purl.org/dc/elements/1.1/}description"):
                    if node.text and node.text.strip():
                        consider("XMP.dc.description", node.text)
                    for item in list(node.iter("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li"))[
                        :8
                    ]:
                        consider("XMP.dc.description", item.text)
        except (UnicodeError, ElementTree.ParseError, ValueError):
            pass  # Malformed/unsupported XMP must not break image ingestion.
    evidence = [dict(item) for item in {tuple(sorted(row.items())) for row in evidence}]
    evidence.sort(key=lambda item: (item["source"], item["value"]))
    return {"state": "marked" if evidence else "unknown", "evidence": evidence}


def extract(image) -> dict:
    warnings = []
    try:
        basic = image.getexif()
        exif = basic.get_ifd(34665) if basic.get(34665) else {}
        gps = basic.get_ifd(34853) if basic.get(34853) else {}
    except (OSError, ValueError, TypeError, SyntaxError, OverflowError) as error:
        basic, exif, gps = {}, {}, {}
        warnings.append(f"EXIF unavailable: {type(error).__name__}")
    captured = timestamp(exif.get(36867), exif.get(36881), exif.get(37521))
    digitized = timestamp(exif.get(36868), exif.get(36882), exif.get(37522))
    latitude = coordinate(gps.get(2), gps.get(1), 90)
    longitude = coordinate(gps.get(4), gps.get(3), 180)
    position = None
    if latitude is not None and longitude is not None:
        altitude = number(gps.get(6))
        if altitude is not None and gps.get(5) in (1, b"\x01"):
            altitude = -altitude
        position = {"latitude": latitude, "longitude": longitude, "altitude_m": altitude}
    elif gps:
        warnings.append("GPS coordinates invalid or incomplete")
    for field, value in (("capture date", exif.get(36867)), ("digitized date", exif.get(36868))):
        if value and timestamp(value) is None:
            warnings.append(f"Invalid {field}")
    return {
        "schema": SCHEMA,
        "user_comment": user_comment(exif.get(37510)),
        "image_description": text(basic.get(270)),
        "software": text(basic.get(305)),
        "screenshot": screenshot_metadata(image, basic, exif),
        "camera_make": text(basic.get(271)),
        "camera_model": text(basic.get(272)),
        "lens_make": text(exif.get(42035)),
        "lens_model": text(exif.get(42036)),
        "captured_at": captured,
        "digitized_at": digitized,
        "exif_modified_at": timestamp(basic.get(306), exif.get(36880)),
        "capture_timezone_known": bool(captured and re.search(r"[+-]\d\d:\d\d$", captured)),
        "orientation": basic.get(274) if basic.get(274) in range(1, 9) else None,
        "iso": number(exif.get(34855)),
        "exposure_seconds": number(exif.get(33434)),
        "aperture": number(exif.get(33437)),
        "focal_length_mm": number(exif.get(37386)),
        "focal_length_35mm": number(exif.get(41989)),
        "gps": position,
        "warnings": warnings,
    }
