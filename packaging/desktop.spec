"""Build on each target OS; never cross-compile a release claim."""

from pathlib import Path
import os
import sys
import tomllib
from PyInstaller.utils.hooks import collect_all, copy_metadata

root = Path(SPECPATH).parent
version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
identity = os.environ.get("LENS_CODESIGN_IDENTITY")
datas, binaries, hiddenimports = collect_all("realopen_lens", include_py_files=True)
datas += [(str(root / "LICENSE"), "licenses/realopen-lens")]
datas += [(str(root / "THIRD_PARTY_NOTICES.md"), "licenses/realopen-lens")]
icons = root / "build/desktop-icons"
if not (icons / "Lens.icns").is_file():
    raise RuntimeError("Run packaging/icons.py before freezing the desktop app")
datas += [(str(icons / "Lens.png"), "realopen_lens/desktop/assets")]
for distribution in ("realopen-lens", "Pillow", "numpy", "onnxruntime", "tokenizers",
                     "psutil", "fastapi", "uvicorn"):
    datas += copy_metadata(distribution)
# Follow ONNX Runtime's inference imports normally. PyInstaller's onnxruntime
# hook collects native provider libraries; collect_all would also import optional
# backend/conversion tools that require `onnx`, which inference does not need.
data, binary, hidden = collect_all("tokenizers")
datas += data
binaries += binary
hiddenimports += hidden
if not (root / "src/realopen_lens/api/static/index.html").is_file():
    raise RuntimeError("Build the gallery before freezing the desktop app")
a = Analysis(
    [str(root / "src/realopen_lens/desktop/entry.py")],
    pathex=[str(root / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + ["uvicorn.logging", "uvicorn.loops.asyncio",
                                 "uvicorn.protocols.http.h11_impl", "uvicorn.lifespan.on"],
    excludes=["pytest", "benchmarks", "rapidocr", "cv2", "onnxruntime.quantization"],
)
pyz = PYZ(a.pure)
# Keep standard streams: isolated engine children use stdin/stdout JSON protocols.
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="RealOpenLens", console=True,
          hide_console="hide-early" if sys.platform == "win32" else None,
          codesign_identity=identity,
          entitlements_file=str(root / "packaging/entitlements.plist") if identity else None,
          icon=str(icons / "Lens.ico") if sys.platform == "win32" else None)
coll = COLLECT(exe, a.binaries, a.datas, name="RealOpenLens")
if sys.platform == "darwin":
    app = BUNDLE(
        coll, name="RealOpen Lens.app", bundle_identifier="org.realopen.lens",
        icon=str(icons / "Lens.icns"),
        info_plist={"CFBundleShortVersionString": version, "CFBundleVersion": version,
                    "NSHighResolutionCapable": True},
    )
