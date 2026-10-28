# Release versioning

The first stable release is **0.1.0**, published as **`v0.1.0`**. The earlier
**`v0.1.0-alpha.1`** tag is a separate historical prerelease and must not be moved
or replaced. The existing package/runtime/gallery metadata already uses `0.1.0`;
no version bump is needed to prepare this stable tag.

## Version sources

`pyproject.toml` is the canonical application/package version. Keep these values
in agreement for every release:

- `[project].version` in `pyproject.toml`;
- `__version__` in `src/realopen_lens/__init__.py`;
- the root project package version in `uv.lock`;
- `version` in `frontend/package.json`;
- the top-level and root-package versions in `frontend/package-lock.json`.

`packaging/desktop.spec` derives the macOS bundle version from `pyproject.toml`.
The frozen runtime and workspace build information report the runtime version.
The gallery's abbreviated `0.1` sidebar credit is a product-series label, not the
full installed build identifier; Workspace settings shows the full runtime/build
identity. Update the bundled Windows/Linux READMEs and release notes to match the
release version too.

A normal release tag is `v<version>` with no prerelease suffix. Patch releases
(`0.1.1`) contain compatible corrections; minor releases (`0.2.0`) introduce a
new feature set and may change APIs or derived index formats during the 0.x
series. Document any migration requirements. Existing source images remain under
the user's control. Optional experiments can remain clearly labelled inside a
stable-channel release; a release number does not certify their ranking quality.

## Preparing a later version

Update the canonical package version and runtime constant, regenerate `uv.lock`
with `uv lock`, and update the frontend version/lockfile together using npm's
version command with `--no-git-tag-version`. Review lockfile changes so unrelated
dependencies are not upgraded accidentally. Update release notes and bundled
READMEs. Check the synchronized values before building.

Do not retag or replace an already published release with different bytes. A fix
after publication receives a new patch version, candidate build and checksum set.
Model revisions and hashes remain separately pinned in `models.lock.json`; model
changes need their own compatibility/evaluation review, not just an app version
change.

## Tag and artifact provenance

Run the model-bearing candidate workflow from the **final release source commit**.
Require the per-platform acceptance reports to identify the exact archive SHA256,
show `status: passed` and `inference: passed`, and contain both precision results.
The release tag must point to that source SHA. Upload those exact archives and
sidecars; do not rebuild or alter them after acceptance. Asset filenames need not
contain a version because their GitHub Release and tag identify it.

The repository's candidate workflow does not tag or publish. See
[publication instructions](desktop-release.md#publication) and
[the 0.1.0 notes](releases/v0.1.0.md). Creating/publishing the release remains a
separate explicit action.
