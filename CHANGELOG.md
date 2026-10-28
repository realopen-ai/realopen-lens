# Changelog

## 0.1.0

First stable release. Full features, downloads, setup and scope are described in
the [release notes](docs/releases/v0.1.0.md).

- Local desktop packages for macOS arm64, Windows x64 and Ubuntu 24.04 x64.
- Bundled Tesseract with 19 OCR languages and orientation data.
- Resumable/incremental indexing with live progress and stage-cache reuse.
- Hybrid text and image search across folders and virtual collections, with pagination.
- EXIF/HEIF metadata, offline GPS lookup, timeline browsing and image inspection.
- Explicit relevance feedback, exact-query memory and reviewed local model controls.
- Folder management, automatic refresh, virtual collections and duplicate review.
- Native archive acceptance with both SigLIP2 precisions and INT8 E5.

Automatic script-group OCR, learned reranking, mixed-precision library ranking and
text-region OCR remain optional experiments. Hybrid/manual whole-image Tesseract
remain the defaults. Model weights are provisioned separately with explicit consent.

## 0.1.0-alpha.1

Initial macOS desktop prerelease. Its existing tag and artifacts remain historical;
they are not replaced by the stable release.
