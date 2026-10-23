# Linux desktop packaging (experimental)

The initial target is **Ubuntu 24.04 x64 with a graphical desktop**. The native
build produces `RealOpenLens-linux-x64.tar.gz` and its `.tar.gz.sha256` file.
Python, Node.js, Tesseract and an inference server are not required on the target.
A browser, fonts and an X11/XWayland desktop remain host requirements.
ARM, Alpine/musl and other distributions are not validated targets.

## Install and start

Download the application artifact, extract GitHub's outer artifact ZIP, then verify
the inner tarball before extracting it:

```sh
sha256sum -c RealOpenLens-linux-x64.tar.gz.sha256
tar -xzf RealOpenLens-linux-x64.tar.gz
./RealOpenLens/RealOpenLens
```

Keep the whole `RealOpenLens` directory together. The tarball preserves executable
permissions and the relative library symlinks used by PyInstaller. Do not run as
root. The launcher opens the gallery in the default browser at `127.0.0.1:8765`;
it reports an occupied port rather than attaching to an unrelated service.

Device setup checks bundled OCR and model availability. Confirm the model download
or select existing weights. SigLIP2 defaults to INT8 below 16 GiB RAM and FP32 at
or above 16 GiB; manual/saved choices are preserved and E5 remains INT8. Model
weights are not inside the archive. Indexing, search and feedback run locally;
source images stay in place. Use **Add folder** in the gallery to start indexing
with live progress. **Quit Lens** stops the local server.

## Native build and CI

`.github/workflows/desktop-linux.yml` runs on `ubuntu-24.04`, alongside the macOS
and Windows reusable workflows. PR checks use the same desktop change filter;
docs-only PRs do not build the application. Manual **Desktop release candidate**
runs all three platforms with `candidate: true` and 30-day artifact retention.
The workflow does not publish a release.

Linux freezes Ubuntu's Python 3.12 and dynamically linked Tk. The shared setup
accepts this explicit interpreter path; macOS and Windows keep their managed Python
selection. This avoids the standalone Python/Tk X11 conflict tracked in
[python-build-standalone #503](https://github.com/astral-sh/python-build-standalone/issues/503).
The frozen app still includes Python/Tk; users do not install Python.

The shared setup action installs locked Python/npm dependencies and verifies the
cached, pinned `eng`, `fra`, `ara` and `osd` data. Linux additionally caches Ubuntu
package downloads by runner image, architecture and package-list hash. Package
installation and dependency checks still run on cache hits. Final archives,
PyInstaller analysis and installed system directories are not cached.

```sh
# On Ubuntu 24.04 x64; apt needs administrator privileges only on the build host.
sudo apt-get update
sudo apt-get install --no-install-recommends -y $(sed '/^#/d; /^$/d' packaging/linux-packages.txt)
uv sync --locked --python /usr/bin/python3.12 --extra web --group packaging --no-editable
npm --prefix frontend ci
uv run --no-sync python packaging/prepare_tessdata.py --cache-dir build/tessdata
uv run --no-sync python packaging/build_linux.py --tessdata-dir build/tessdata
xvfb-run -a uv run --no-sync python packaging/acceptance.py \
  dist/RealOpenLens-linux-x64.tar.gz --output runs/linux-acceptance
```

On a graphical desktop, Xvfb is unnecessary. Optional `--models models` exercises
existing INT8 and FP32 weights through frozen indexing/OCR/search, refresh,
resident encoding and gallery startup/shutdown. `--precision int8` or `fp32`
selects one profile. Without weights, acceptance records inference as `not_run`.
CI does not download embedding models.

Eleven extracted-package checks cover ELF/icon/license metadata, frozen imports
and Tcl/Tk, launcher rendering in both themes under Xvfb, private multilingual OCR
and orientation with host tools hidden, worker protocols, Unicode workspace paths,
gallery assets, authenticated API, setup diagnostics and clean shutdown. Candidate
reports bind outcomes to the archive SHA256 and reject archive mutation. Build
logs and per-check timing/errors are retained even when validation fails.

## OCR isolation and compatibility

The build copies the installed Tesseract 5 executable and its transitive non-glibc
ELF library dependencies, checks x86-64 architecture and patches private origin
paths. The app uses this private engine with hash-verified English, French, Arabic
and orientation data. Frozen Python loader overrides and host tessdata settings
are removed for OCR subprocesses. Missing/corrupt bundled OCR reports an error;
it does not silently fall back to a system installation.

Native package versions, source/copied hashes and required system libraries are
recorded in `_internal/lens-ocr/runtime.json`. Debian package copyright notices
and common license texts accompany the bundle and are appended to its third-party
notices. Build-time apt package revisions follow the Ubuntu runner's repositories;
the exact native binary versions are recorded rather than claimed to be pinned.
Language assets and Python/npm dependencies remain pinned.

The archive relies on host glibc. Ubuntu 24.04 builds require glibc 2.39 or newer;
older distributions are not supported. This follows [PyInstaller's Linux build
compatibility guidance](https://pyinstaller.org/en/stable/usage.html#making-gnu-linux-apps-forward-compatible).
The portable tarball is not an AppImage, system installer or desktop-menu installer.
Linux CI validates the hosted Ubuntu environment with a virtual X display; it does
not establish browser launching or integration with every desktop environment.
