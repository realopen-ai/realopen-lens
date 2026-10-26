"""Copy, relocate and sign a private macOS OCR runtime; never modify its source."""

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import urllib.request


def run(*command, env=None):
    return subprocess.run(
        command, check=True, capture_output=True, text=True, encoding="utf-8", env=env, timeout=60
    ).stdout


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dependencies(path, executable):
    """Resolve non-system load commands, including loader-relative rpaths."""

    def expand(value):
        return value.replace("@loader_path", str(path.parent)).replace(
            "@executable_path", str(executable.parent)
        )

    rpaths = re.findall(
        r"cmd LC_RPATH\s+cmdsize \d+\s+path (.*?) \(offset", run("/usr/bin/otool", "-l", str(path))
    )
    result = []
    for line in run("/usr/bin/otool", "-L", str(path)).splitlines()[1:]:
        name = line.strip().split(" (", 1)[0]
        if name.startswith(("/usr/lib/", "/System/Library/")):
            continue
        candidates = [Path(expand(name))]
        if name.startswith("@rpath/"):
            candidates = [Path(expand(prefix)) / name[7:] for prefix in rpaths]
        found = next((item.resolve() for item in candidates if item.is_file()), None)
        if found is None:
            raise ValueError(f"Cannot resolve OCR dependency {name} in {path.name}")
        if found != path:
            result.append((name, found))
    return result


def language_data(destination, source, registry):
    """Use matching local files or fetch the pinned build-time assets, verifying both."""
    destination.mkdir(parents=True)
    for language, spec in registry.items():
        target = destination / f"{language}.traineddata"
        local = source / target.name
        if (
            local.is_file()
            and local.stat().st_size == spec["size"]
            and checksum(local) == spec["sha256"]
        ):
            shutil.copyfile(local, target)
        else:
            print(f"Fetching pinned OCR data: {language}", flush=True)
            with (
                urllib.request.urlopen(spec["url"], timeout=60) as response,
                target.open("wb") as out,
            ):
                remaining = spec["size"] + 1
                while remaining and (block := response.read(min(1024 * 1024, remaining))):
                    out.write(block)
                    remaining -= len(block)
        if target.stat().st_size != spec["size"] or checksum(target) != spec["sha256"]:
            target.unlink()
            raise ValueError(f"Integrity check failed for OCR language {language}")
    configs = destination / "configs"
    configs.mkdir()
    (configs / "tsv").write_text("tessedit_create_tsv 1\n", encoding="utf-8")


def bundle_ocr(bundle, executable, source_data=None, identity=None):
    executable = executable.resolve()
    version = run(str(executable), "--version").splitlines()[0]
    if not re.search(r"tesseract 5\.", version):
        raise ValueError("Build with a Tesseract 5 executable")
    if source_data is None:
        listed = run(str(executable), "--list-langs")
        match = re.search(r'in "([^"]+)"', listed)
        if not match:
            raise ValueError("Cannot locate source tessdata; provide --tessdata-dir")
        source_data = Path(match[1])
    code = bundle / "Contents/Frameworks/lens-ocr"
    resources = bundle / "Contents/Resources/lens-ocr"
    (code / "bin").mkdir(parents=True)
    (code / "lib").mkdir()
    resources.mkdir(parents=True)
    pending, graph = [executable], {}
    while pending:
        path = pending.pop()
        if path in graph:
            continue
        graph[path] = dependencies(path, executable)
        pending.extend(target for _, target in graph[path])
    mapping = {path: code / ("bin" if path == executable else "lib") / path.name for path in graph}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("OCR libraries have conflicting filenames")
    records = []
    for path, target in mapping.items():
        architectures = run("/usr/bin/lipo", "-archs", str(path)).split()
        if platform.machine() not in architectures:
            raise ValueError(f"Wrong OCR architecture: {path.name}: {architectures}")
        shutil.copyfile(path, target)
        target.chmod(0o755)
        for original, dependency in graph[path]:
            relative = os.path.relpath(mapping[dependency], target.parent)
            run(
                "/usr/bin/install_name_tool",
                "-change",
                original,
                f"@loader_path/{relative}",
                str(target),
            )
        if path != executable:
            run("/usr/bin/install_name_tool", "-id", f"@loader_path/{target.name}", str(target))
        for prefix in re.findall(
            r"cmd LC_RPATH\s+cmdsize \d+\s+path (.*?) \(offset",
            run("/usr/bin/otool", "-l", str(target)),
        ):
            run("/usr/bin/install_name_tool", "-delete_rpath", prefix, str(target))
        command = ["/usr/bin/codesign", "--force", "--sign", identity or "-"]
        if identity:
            command += ["--options", "runtime", "--timestamp"]
        run(*command, str(target))
        # Reject any remaining dependency on a developer's installation.
        loads = run("/usr/bin/otool", "-L", str(target)).splitlines()[1:]
        if any(
            not line.strip().startswith(("@loader_path/", "/usr/lib/", "/System/Library/"))
            for line in loads
        ):
            raise ValueError(f"Unrelocated OCR dependency in {target.name}")
        records.append(
            {
                "file": target.relative_to(bundle).as_posix(),
                "source_sha256": checksum(path),
                "sha256": checksum(target),
                "architectures": architectures,
            }
        )
    registry = json.loads(Path(__file__).with_name("tessdata.json").read_text(encoding="utf-8"))
    language_data(resources / "tessdata", source_data, registry)
    manifest = {
        "version": 1,
        "engine": version,
        "files": sorted(records, key=lambda r: r["file"]),
        "tessdata": registry,
        "tsv_sha256": checksum(resources / "tessdata/configs/tsv"),
    }
    (resources / "runtime.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("DYLD_")}
    env.update(PATH="/usr/bin:/bin", TESSDATA_PREFIX=str(resources / "tessdata"))
    copied = code / "bin" / executable.name
    listed = run(str(copied), "--list-langs", env=env)
    if not set(registry).issubset(set(listed.splitlines()[1:])):
        raise ValueError("Private OCR runtime cannot load its language/orientation packs")
    command = ["/usr/bin/codesign", "--force", "--sign", identity or "-"]
    if identity:
        command += ["--options", "runtime", "--timestamp"]
    run(*command, str(bundle))
    print(
        f"Bundled {version}: {len(records)} binaries, {len(registry) - 1} languages + orientation",
        flush=True,
    )
