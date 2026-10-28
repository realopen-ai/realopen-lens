RealOpen Lens 0.1.0 — by RealOpen AI

Linux x64 portable application, built on Ubuntu 24.04 (glibc 2.39).
Ubuntu 24.04 x64 with a graphical desktop is the initial supported target.
Older glibc versions, musl/Alpine, ARM and other distributions are not validated.

Verify the .tar.gz using its sibling .sha256 file, then extract the tarball.
Keep the whole RealOpenLens directory together; run ./RealOpenLens/RealOpenLens.
Do not launch as root. No Python, Node.js, Tesseract or model server is required.
The launcher opens the gallery in your default browser on loopback port 8765.
A browser and a working X11/XWayland desktop with fonts are required.

Tesseract 5, 19 OCR language packs and orientation data are included.
On first launch, Device setup checks these resources and model availability.
Download models only after explicit confirmation, or choose existing weights.
Model inference and image indexing/search run locally and offline afterwards.
Source images stay in place. Removing a folder from Lens never deletes originals.
Quit Lens closes its local gallery server. Logs are available from the launcher.

Licenses/notices: _internal/licenses/realopen-lens/ and
_internal/lens-ocr/licenses/. Native OCR package versions and file checksums:
_internal/lens-ocr/runtime.json. glibc and the graphical system remain host-provided.
