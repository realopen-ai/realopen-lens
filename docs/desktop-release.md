# Desktop release readiness

The current supported build target for release tooling is **macOS arm64**.
The standard archive is ad-hoc signed for development. The manual **Desktop release
candidate** GitHub Actions workflow builds that archive and validates the extracted
bytes, then retains the ZIP, SHA256 checksum and evidence for 30 days. It requires
read-only repository permissions and does not create tags or publish releases.
Run it from Actions on the reviewed commit. Windows/Linux builds remain separate work.

## Reproduce archive acceptance

```sh
uv sync --locked --extra web --group packaging --no-editable
npm --prefix frontend ci
uv run --no-sync python packaging/build_macos.py
uv run --no-sync python packaging/acceptance.py \
  dist/RealOpenLens-macos-arm64.zip \
  --models models --output runs/release-candidate-acceptance
```

Use a new output directory for each run. Existing evidence is never overwritten.
The report binds all checks to the ZIP's SHA256 and records per-stage outcomes,
errors and elapsed time. The ten model-free package checks include startup without
external developer tools/OCR and missing-prerequisite reporting. Supplying `--models`
also exercises real INT8 and FP32 frozen indexing, lexical/hybrid retrieval with
OCR ground truth, cached refresh, resident encoder workers and HTTP startup/shutdown.
Use `--precision int8` or `--precision fp32` to test one installed profile.
All indexes and fixture images are temporary; existing models are read in place.

Without `--models`, the report explicitly records inference as **not_run**. A
passing packaging-only run does not imply inference acceptance. CI intentionally
does not download weights or install OCR. These checks run on an existing host;
they do not establish fresh-OS installation, Finder privacy prompts or Gatekeeper
behavior. No retrieval-quality metric is inferred from the synthetic invoice.

Each build writes a sibling `.zip.sha256` file. Check it from `dist/` using:

```sh
shasum -a 256 -c RealOpenLens-macos-arm64.zip.sha256
```

## OCR provisioning

For this milestone, **Tesseract 5 and language data remain external prerequisites**.
The app includes neither Homebrew nor an OCR installer. The Device setup card
reports installed languages, preserves explicit preferences and links to the
[official installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html).
On a Mac with Homebrew already installed, that guide documents installing
`tesseract` and, when additional languages are needed, `tesseract-lang`.
Installing Homebrew/system software remains an explicit user action.

After installation, reopen/check Device setup. Finder's minimal PATH is extended
with `/opt/homebrew/bin` and `/usr/local/bin`. Missing requested languages are
reported; Lens does not substitute English silently. Gallery browsing remains
available before OCR/models are installed. Model downloads require confirmation;
normal inference stays local and offline.

The launcher recommends INT8 SigLIP2 below 16 GiB RAM and FP32 at or above 16 GiB;
saved/manual choices override this recommendation. E5 remains INT8 in both cases.

## Optional Developer ID signing and notarization

The planned hobby-project prerelease uses the ad-hoc signed ZIP. Paid Apple
membership and notarization are optional, not publication gates for that
experimental build. Clearly label it unnotarized and link users to
[Apple's instructions for opening an app from an unidentified developer](https://support.apple.com/en-us/102445).

The release path requires an installed **Developer ID Application** certificate
and an existing `notarytool` Keychain profile. The build helper never imports
certificates, changes Keychains or stores Apple passwords. Prepare credentials
outside the repository following [Apple's notarization workflow](https://developer.apple.com/documentation/security/customizing-the-notarization-workflow).

```sh
uv run --no-sync python packaging/build_macos.py --release \
  --identity 'Developer ID Application: YOUR ORGANIZATION (TEAMID)' \
  --notary-profile lens-notary
```

PyInstaller receives the signing identity and the bundled entitlement for unsigned
executable memory; its real-identity signing enables hardened runtime. See
[PyInstaller's macOS signing notes](https://pyinstaller.org/en/stable/feature-notes.html#macos-binary-code-signing).
The helper verifies the bundle, submits it to Apple, requires `Accepted`, staples
and validates the ticket, and requires Gatekeeper assessment before emitting a
versioned release ZIP and checksum. The app version comes from `pyproject.toml`.
Submission to Apple's notarization service occurs only when `--release` is explicitly requested.
The request result is retained in `build/notarization.json`.

Signing options without `--release`, ad-hoc identities in release mode, missing
identities, rejection and failed stapling/assessment are errors. No development
fallback is used. Run archive acceptance against the final versioned ZIP too.
Validate the distributed signed archive with `packaging/acceptance.py` using
`--require-notarized` to require its extracted stapled ticket and Gatekeeper approval.
This code path requires actual credentials and signed-runtime acceptance before
being considered validated; development archive tests cannot establish that.

## Clean-machine acceptance before public publication

Use a separate Mac/user or clean macOS VM with no checkout, Python, uv, Node,
models or Tesseract. Retain a dated record with macOS version, architecture,
archive checksum, outcomes and errors for each item below. Do not include personal
photos, paths or credentials in a public report.

1. Download the exact ZIP through a browser, verify its checksum, extract and move
   the application to Applications. Open it from Finder with its normal quarantine
   metadata intact. A signed release must pass Gatekeeper normally.
2. Verify that the missing-model/OCR guidance appears and the empty gallery opens.
   Check dark/light layouts, collapse/expand behavior and folder selection.
3. Confirm the model download explicitly. Test cancellation, retry, progress and
   checksum failure reporting; confirm completed downloads pass setup verification.
4. Install Tesseract/languages using the documented external path. Verify detection
   from Finder, missing-language errors, and successful explicit language selection.
5. Index a small disposable folder containing a known text image and a photograph.
   Observe live progress; verify the reference text, expected results and thumbnails.
   Restart Lens and verify the folder/results persist without reimporting.
6. Add/remove an image, refresh and check counts. Rename the Lens folder; remove it
   from Lens after confirmation and verify original files remain untouched.
7. Disable network access and repeat search/browse. Quit Lens; confirm its server
   stops. Check the occupied-port message without attaching to another service.

This walkthrough is not yet recorded as completed. The host archive/inference
checks provide complementary automated evidence, not a replacement.

## Publication

Project source is licensed under [Apache-2.0](../LICENSE). Downloaded model weights,
OCR language data and bundled third-party components retain their own licenses.
The versioned inventory and upstream license texts are in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md), also included at
`Contents/Resources/licenses/realopen-lens/THIRD_PARTY_NOTICES.md` in the macOS app
and in Python distribution license metadata. Refresh it against the actual frozen
inventory when dependency/runtime versions change. The project license does not
replace these terms.

After reviewed-commit CI, final archive acceptance, clean-machine acceptance
and dependency-notice review, prepare a **draft prerelease** in
GitHub Releases. Attach the exact validated ZIP and checksum, supported OS/architecture,
OCR setup instructions and known limitations. Review diagnostic reports for local
paths before attaching them. A release tag must identify the source used for that
archive. Identify the current build as ad-hoc signed and unnotarized; only claim
Developer ID signing/notarization if the optional release path was completed and
validated. Publishing the draft is a separate explicit action. GitHub Packages is
not needed to distribute this desktop ZIP; Python package publication is separate.
