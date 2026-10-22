"""Build on each target OS; never cross-compile a release claim."""

from pathlib import Path
import sys
from PyInstaller.utils.hooks import collect_all, copy_metadata

root = Path(SPECPATH).parent
datas, binaries, hiddenimports = collect_all("realopen_lens", include_py_files=True)
icons = root / "build/desktop-icons"
if not (icons / "Lens.icns").is_file():
    raise RuntimeError("Run packaging/icons.py before freezing the desktop app")
datas += [(str(icons / "Lens.png"), "realopen_lens/desktop/assets")]
for distribution in ("realopen-lens", "Pillow", "numpy", "onnxruntime", "tokenizers",
                     "psutil", "fastapi", "uvicorn"):
    datas += copy_metadata(distribution)
for name in ("onnxruntime", "tokenizers"):
    data, binary, hidden = collect_all(name)
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
    excludes=["pytest", "benchmarks", "rapidocr", "cv2"],
)
pyz = PYZ(a.pure)
# Keep standard streams: isolated engine children use stdin/stdout JSON protocols.
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="RealOpenLens", console=True,
          icon=str(icons / "Lens.ico") if sys.platform == "win32" else None)
coll = COLLECT(exe, a.binaries, a.datas, name="RealOpenLens")
if sys.platform == "darwin":
    app = BUNDLE(
        coll, name="RealOpen Lens.app", bundle_identifier="org.realopen.lens",
        icon=str(icons / "Lens.icns"),
        info_plist={"CFBundleShortVersionString": "0.1.0", "NSHighResolutionCapable": True},
    )
