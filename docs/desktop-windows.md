# Windows desktop package (experimental)

The native Windows x64 build produces `RealOpenLens-windows-x64.zip` and a SHA256
sidecar. It includes Python, Tcl/Tk, the browser gallery, ONNX Runtime and a private
Tesseract 5.5.3 engine with [19 OCR language packs](multilingual-ocr.md) and orientation data. Node.js,
Python, Homebrew and a separate OCR installation are not required on the target PC.
Model weights are downloaded separately with explicit consent during device setup.

## Starting Lens

1. Extract the **entire** ZIP into a local folder. Keep `_internal` beside
   `RealOpenLens.exe`; do not run the executable from inside the ZIP.
2. Open `RealOpenLens.exe`. This unsigned experimental application can trigger
   Windows security prompts. Only run a trusted project download; do not disable
   antivirus protections.
3. In **Device setup**, choose an existing model folder or confirm the model
   download. INT8 SigLIP2 is suggested below 16 GB RAM and FP32 at 16 GB or above;
   E5 stays INT8. A saved manual choice takes precedence.
4. Open the image library and use **Add folder**. Indexing reports live progress
   and leaves original files untouched. Subsequent indexing/search works offline.
5. Use **Quit Lens** to stop its server and workers. Closing the browser alone
   leaves Lens running.

Workspace, model downloads and logs are stored under
`%LOCALAPPDATA%\RealOpenLens`, independently of the extracted application folder.
The default loopback port is 8765. To use another workspace or port from PowerShell:

```powershell
.\RealOpenLens.exe --workspace "$env:LOCALAPPDATA\Lens-test" --port 8766
.\RealOpenLens.exe --cli workspace info
```

The executable keeps standard streams for isolated JSON workers and CLI commands.
PyInstaller hides its own console when opened outside an existing terminal; a
brief console flash can occur. CLI/worker streams use UTF-8, including paths and
queries in non-Latin scripts.

## Native CI and candidate downloads

Pull requests and pushes to `main` build Windows packages when desktop inputs
change, using the same component gate as macOS. Documentation-only changes skip
the builds. **Desktop release candidate** also builds both platforms manually.
The Windows steps live in the reusable `desktop-windows.yml` workflow. Both platforms
use the same dependency setup action and core smoke checks; manual candidates use
the same checksum-bound acceptance wrapper and archive-mutation check.

After a successful run, download the `desktop-windows-x64-<commit>` artifact and
extract the contained application ZIP. Its accompanying evidence artifact has
build logs, the exact archive SHA256, per-check outcomes/timings, native dependency
versions and failure diagnostics. Checks exercise:

- frozen imports, gallery assets, license notices and Tcl/Tk resources;
- the actual launcher view in both themes and its collapsible setup card;
- private OCR pack initialization and orientation with host tools hidden;
- allowed/rejected worker dispatch and resident JSON protocol handling;
- workspace CLI and authenticated loopback API, including non-Latin paths;
- setup reporting missing model weights while bundled OCR is ready;
- graceful server shutdown, failing the check if forced termination is needed.
  Windows CRT signal exit code 3 is accepted only with completed Uvicorn shutdown,
  a finished-server log and no logged server errors.

Manual candidate artifacts use `desktop-candidate-windows-x64-<commit>` and retain
`candidate-acceptance/acceptance.json` plus individual package results/logs. Ordinary
PR/main runs retain their smoke report; both modes test the extracted ZIP.

These package checks do **not** download embedding models or establish retrieval
accuracy. Real-model indexing/search can additionally be checked with existing
weights using `packaging/smoke.py`. Native Windows execution is performed by the
Windows runner; macOS unit tests do not substitute for it. Publish a Windows
release asset only after the native package job succeeds for that exact commit.

## Building locally on Windows x64

Install Git, Visual Studio 2022 Build Tools with the C++ desktop workload, CMake,
Node.js 22 and uv. From a checkout in PowerShell:

```powershell
uv python install 3.12
uv sync --locked --python 3.12 --extra web --group packaging --no-editable
npm --prefix frontend ci
$baseline = (Get-Content packaging/windows-ocr/vcpkg.json -Raw | ConvertFrom-Json).'builtin-baseline'
git clone https://github.com/microsoft/vcpkg.git build/vcpkg
git -C build/vcpkg checkout $baseline
& .\build\vcpkg\bootstrap-vcpkg.bat -disableMetrics
uv run --no-sync python packaging/build_windows.py --vcpkg-root build/vcpkg
uv run --no-sync python packaging/ci_smoke.py dist/RealOpenLens-windows-x64.zip
```

The first build downloads and compiles OCR dependencies. CI caches vcpkg binary
packages with compiler/ABI tracking; changing the manifest/triplet or hosted runner
image changes the binary cache key. Source/tool downloads have a separate cache
keyed by the pinned vcpkg revision. OCR language data is cached and hash-verified
on every run, then reused by the build. Completed OCR packages are saved even if later packaging checks fail, so
fixing those checks does not require another full compilation. Cache upload
failures do not waive package validation. Compilation uses two concurrent jobs
in CI. The source revision is
pinned and verified before building. The custom triplet uses static libraries and
CRT, disables Tesseract OpenMP and CPU-native tuning, and builds only Release.
The builder rejects the wrong PE architecture and dependencies on non-system DLLs
or separately installed VC/OpenMP runtimes.

Language assets use the same size/SHA256 pins as macOS. The resulting OCR manifest
hashes its executable, data, configuration and native dependency notices; device
setup checks integrity instead of falling back to host OCR. vcpkg's package status
and copyright files are preserved under `_internal/lens-ocr` and appended to the
bundled `THIRD_PARTY_NOTICES.md`. No target-machine installer, admin operation or
system PATH modification is used.

The native build follows the [vcpkg manifest workflow](https://learn.microsoft.com/en-us/vcpkg/consume/manifest-mode)
and [static triplet configuration](https://learn.microsoft.com/en-us/vcpkg/users/triplets).
Console behavior follows [PyInstaller's Windows console support](https://pyinstaller.org/en/stable/feature-notes.html).
