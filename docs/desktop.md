# Desktop build (experimental)

The Finder/Dock application icon uses the mint Lens mark. Its source is
`frontend/public/assets/lens.svg` (also the gallery mark and favicon);
`packaging/icons.py` generates PNG/ICNS/ICO assets
during the build, without downloading artwork or design libraries.
The macOS icon adds a transparent margin and rounded tile so its visual size fits
the Dock. The web logo/favicon retain the original square SVG.

The desktop launcher starts the existing FastAPI adapter on loopback and opens the
gallery in the default browser. A small native control window stays open to
select existing model weights, choose/download pinned INT8 or FP32 SigLIP2 models,
open logs and quit the server. Closing the browser does not stop Lens; use **Quit
Lens** or close the control window. Shutdown waits for the API task manager to
release owned workers. The normal port remains 8765; an occupied port produces a
startup message instead of attaching to an unknown process.

The control window uses the gallery's green/cream palette, serif heading, Lens
branding, styled actions and separate activity/workspace cards. Its light/dark
switch persists in `desktop.json`; this is a launcher preference, independent of
the browser gallery's theme. Native folder dialogs remain provided by the OS.

The launcher uses the normal persistent application-data workspace. Original images
are referenced in place. Desktop settings live in `desktop.json`, separately from
import metadata. Existing imported workspaces retain their remembered model path;
new workspaces default to a `models` subdirectory. Add image folders once, from the
gallery: the current user's home directory and its subfolders are allowed by
default. This is an application boundary, not an OS permission grant; protected
folders still require normal OS access. No folder is scanned automatically. The
launcher no longer maintains a separate folder allowlist; legacy `roots` settings
are ignored. Stop the gallery before changing model directories, then restart it.
The gallery's **Choose folder** control browses folders locally and fills the
indexing form; no path typing or image upload is needed.
Existing collection paths remain readable. CLI `--allow-root` options can narrow
the default boundary or explicitly permit an external drive.

Model downloads require explicit confirmation and show received/total megabytes.
Press the download button again to cancel. Partial artifacts are removed and only
verified weights are published. Downloads use the existing locked model registry;
inference remains offline. The SigLIP2 selector initially recommends INT8 below
16 GB of physical RAM and FP32 at 16 GB or more. Detection failure defaults to
INT8. A saved manual choice overrides that recommendation on later launches.
E5 always stays INT8; the download prompt states both choices explicitly, and other
installed precision files are preserved. The selected precision also becomes the
gallery's default when adding a new folder, without restarting the server. Refreshing
an existing collection retains its indexed precision, and the form still allows a
manual override. Switching the launcher selection does not convert existing indexes
or download weights automatically. Opening a pre-existing FP32 collection still
requires its FP32 weights in the selected model directory.

## Build on macOS

Use Python 3.12 with working Tk support (`python -c 'import tkinter'`), Node/npm,
and the project checkout:

```sh
uv sync --locked --extra web --group packaging --no-editable
npm --prefix frontend ci
uv run --no-sync python packaging/build_macos.py
```

Output: `dist/RealOpenLens-macos-ARCH.zip`. Extract the archive and open
`RealOpen Lens.app` in Finder. The helper builds in temporary storage, verifies
ad-hoc signing and archives without incidental file-provider attributes. The bundle contains Python,
the inference libraries, engine workers and gallery assets; it does not require
Python/uv/Node on the target machine. It does not contain model weights, source
images or Tesseract. Tesseract and the desired language data must be installed
separately; the launcher displays whether it is discoverable. macOS Homebrew paths
are included when Finder supplies a minimal PATH. Missing OCR is not treated as
successful OCR; ordinary pipeline error reporting still applies.

The current artifact is a development build. PyInstaller applies an ad-hoc macOS
signature; there is no Developer ID signature or notarization. Check build logs for
signing errors, especially extended attributes from file providers. Do not publish
the artifact as a signed installer. Build/sign/notarize releases on their target OS
and architecture, then validate on a clean machine.

## Validate the frozen engine

With existing INT8 models and English Tesseract data:

```sh
uv run --no-sync python packaging/smoke.py \
  '/path/to/RealOpen Lens.app/Contents/MacOS/RealOpenLens' --models models
```

This creates a temporary one-image collection, runs real frozen indexing and text
search (including child workers), starts the packaged HTTP gallery on an isolated
port, verifies HTML delivery, then shuts it down and removes the temporary data.
It is a packaging regression, not a retrieval-quality benchmark.

For manual isolated UI checks, invoke the bundle executable with
`--workspace NEW_DIRECTORY --port 8769 --no-browser`. These are testing overrides;
normal startup uses the default workspace and port. The executable also supports
`--cli` followed by normal Lens CLI arguments for diagnostics.

Rotating logs are in `WORKSPACE/logs/desktop.log` (2 MB, three backups). Logs may
contain local paths and task errors; review them before sharing. Workspace settings
in the gallery shows build identity, paths and installation information.

## Portability and remaining release work

### CI packaging gate

The `Desktop package (macOS arm64)` job runs on pushes and pull requests using
[`macos-15`](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
It installs the locked web/packaging environment, checks Tcl availability, builds
the gallery and application, and tests the extracted archive rather than source
imports. Locally, the same model-free checks run with:

```sh
uv run --no-sync python packaging/ci_smoke.py dist/RealOpenLens-macos-arm64.zip
```

The ten checks cover archive extraction, ad-hoc signing, icon metadata/padding,
frozen dependency imports/version metadata, Tcl/Tk scripts, pinned model registry,
stage/resident worker dispatch/protocol handling, CLI workspace info, loopback API
startup, gallery assets/favicon, authenticated empty-workspace reads and graceful
shutdown. Host Python path/environment overrides are removed for frozen subprocesses.
No model weights, Tesseract installation or GUI window are required. Uvicorn's
post-cleanup SIGTERM exit is accepted only with an application-shutdown completion
log; startup timeouts and forced kills fail the check.

Successful development ZIPs and always-uploaded diagnostics have seven-day artifact
retention. The JSON report records individual check outcomes/timing and is written
even on failure; build/server logs are retained alongside it. These artifacts are
not notarized releases. GUI usability, real inference, model downloading and
Windows/Linux native packaging remain separate validation work.

The macOS arm64 development build was exercised with real local INT8 models:
frozen indexing completed all four stages without failed/degraded files, OCR
returned the fixture text, text search returned the indexed image, the HTTP gallery
served HTML, and an isolated resident process encoded both SigLIP2 (768 dimensions)
and E5 (384). The native control window started and shut down an isolated server.
The Python suite passed 555 tests with one optional OCR skip; formatting, lint,
native/Windows type checks and the frontend build passed. These checks used a host
with existing models and Tesseract; they do not establish clean-machine setup or
retrieval quality. Download progress/checksum/cancellation were tested with controlled
responses; a fresh network model download was not executed.

The launcher uses portable Tk/Python APIs and the spec emits a directory executable
on Windows/Linux. These outputs require native builds and runtime validation; a
macOS build cannot establish support on those platforms. Native folder permissions,
Tesseract distribution/licensing, clean-machine inference, OS signing/installers
and physical 8 GB validation remain release gates. This is an experimental macOS
packaging milestone, not a three-platform release.
