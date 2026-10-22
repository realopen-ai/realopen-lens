"""Render the existing Lens vector mark into desktop icon containers."""

from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw


def build_icons(destination: Path):
    source = ET.parse(
        Path(__file__).resolve().parents[1] / "frontend/public/assets/lens.svg"
    ).getroot()
    scale = 2  # Supersampling keeps the small Dock sizes crisp.
    image = Image.new("RGB", (1024 * scale, 1024 * scale))
    draw = ImageDraw.Draw(image)
    for element in source:
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "rect":
            draw.rectangle((0, 0, image.width, image.height), fill=element.attrib["fill"])
        elif tag == "circle":
            x, y, radius = (float(element.attrib[key]) * scale for key in ("cx", "cy", "r"))
            width = float(element.attrib["stroke-width"]) * scale
            # SVG strokes straddle the path; Pillow's ellipse stroke is internal.
            radius += width / 2
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=element.attrib["stroke"],
                width=round(width),
            )
        elif tag == "path":
            points = [float(value) * scale for value in element.attrib["d"][1:].split()]
            draw.line(
                points,
                fill=element.attrib["stroke"],
                width=round(float(element.attrib["stroke-width"]) * scale),
            )
    destination.mkdir(parents=True, exist_ok=True)
    # Dock artwork needs breathing room inside the 1024px icon canvas. Filling the
    # canvas makes the tile appear larger than neighboring macOS application icons.
    tile_size, inset, corner_radius = 824 * scale, 100 * scale, 184 * scale
    tile = image.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
    mask = Image.new("L", tile.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, tile_size - 1, tile_size - 1), radius=corner_radius, fill=255
    )
    mac_icon = Image.new("RGBA", image.size, (0, 0, 0, 0))
    mac_icon.paste(tile, (inset, inset), mask)
    mac_icon = mac_icon.resize((1024, 1024), Image.Resampling.LANCZOS)
    mac_icon.save(destination / "Lens.png")
    mac_icon.save(destination / "Lens.icns", format="ICNS")
    image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
    image.save(
        destination / "Lens.ico", format="ICO", sizes=[(n, n) for n in (16, 32, 48, 64, 128, 256)]
    )


if __name__ == "__main__":
    build_icons(Path(__file__).resolve().parents[1] / "build/desktop-icons")
