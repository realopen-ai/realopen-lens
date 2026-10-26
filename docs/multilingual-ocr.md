# Multilingual OCR

Desktop builds include the Tesseract engine, 19 pinned `tessdata_fast` 4.1.0 language
packs and `osd` orientation data. The 16 additions use approximately 29.9 MiB before
archive compression. Build-time fetching is explicit and size/SHA-256 verified;
recognition and search remain offline. There is no language-pack download manager.

## Choose the document languages

Use **OCR languages** in desktop Device setup, gallery Workspace settings, or
Add folder → Indexing options. **Choose languages** is selected by default, with
English/French/Arabic enabled. Named checkboxes add/remove packs; the code field
also supports other explicitly installed source-language codes. Alternatively,
select **Automatic script groups**, or pass `--languages auto` to the CLI. Packs being installed does not enable all of them
in every recognition pass. Existing defaults (`eng+fra+ara`) and saved selections
are preserved. Source installs need the corresponding packs in their Tesseract
installation; the standalone desktop bundle supplies them itself.

| Language | Code | Language | Code |
| --- | --- | --- | --- |
| English | `eng` | French | `fra` |
| Arabic | `ara` | Spanish | `spa` |
| Portuguese | `por` | German | `deu` |
| Italian | `ita` | Indonesian | `ind` |
| Turkish | `tur` | Vietnamese | `vie` |
| Russian | `rus` | Hindi | `hin` |
| Bengali | `ben` | Urdu | `urd` |
| Persian | `fas` | Chinese, simplified | `chi_sim` |
| Chinese, traditional | `chi_tra` | Japanese | `jpn` |
| Korean | `kor` | Orientation only | `osd` |

For example:

```sh
lens index /path/to/images --index-dir /path/to/index --languages deu+eng --log
lens index /path/to/images --index-dir /path/to/index --languages jpn+eng --log
```

Set the field to `spa+eng` for Spanish/English documents or `hin+eng` for Hindi/English.
Language order can affect recognition. `osd` is used internally for orientation;
do not add it to the text-recognition language list.

Enable the languages actually present in the folder. Enabling every pack increases
OCR work and can change output quality. The [diagnostic comparison](benchmarks/2026-10-26-multilingual-ocr.md)
measures this tradeoff. Automatic mode uses the separate detector described below. The benchmark's
`matched` and `script` profiles still use reference labels as oracles; they must
not be confused with the measured `auto` profile.

Changing a folder's language selection reprocesses OCR and text embeddings while
reusing compatible metadata and visual embeddings. Adding unused bundled packs
does not change the engine's inference identity. The Korean layout fix changes OCR
implementation provenance, so upgrading from an older build conservatively
invalidates prior OCR stage caches once; compatible metadata/vision stages remain
reusable.

## Automatic script groups

Automatic mode uses local Tesseract orientation/script detection, with at most
three probes (whole image and two horizontal bands) within a three-second detector
budget and the existing total OCR deadline. It selects groups such as Latin,
Arabic, Cyrillic, Devanagari, Bengali, Han/Japanese and Korean, using only installed
supported packs. English is included when available. This is script detection,
not identification of French versus Spanish or other languages sharing a script.

A conservative score margin accepts a whole-image probe, or two agreeing band
probes. Strong conflicting groups are combined. Low confidence, unknown scripts,
missing corresponding packs, timeouts and short/blank text fall back to all
installed supported packs and retain an explicit warning. Group detection can miss
minority scripts; no mixed-script accuracy guarantee is made. `osd` is required
for automatic mode and setup reports it if missing.

The selected packs and detector/fallback information are retained per image and
shown under **Text recognition** in image details. Saved folder configuration
retains the mode for refresh and automatic folder watching. Device preferences
apply to new folders; change an existing folder's indexing options and re-index to
apply a different mode. Switching modes retains the current manual choices in the
picker. Manual selection remains the performance/accuracy default.

Detector policy and candidate-pack hashes are part of automatic OCR cache identity.
An unchanged automatic configuration reuses OCR; policy/pack changes invalidate
OCR and text embeddings while leaving compatible metadata and vision reusable.
The [automatic-mode study](benchmarks/2026-10-26-automatic-ocr.md) reports actual
routing and fallback rates, rather than assuming perfect script knowledge.

## Retrieval

Raw recognized text, including diacritics, feeds multilingual E5 semantic retrieval.
Lexical retrieval preserves Indic vowel marks and composed Hangul. Han/Kana
character terms support phrases inside longer unsegmented text; spaces commonly
inserted between OCR glyphs are normalized. Single-character Han/Kana lookup is
also supported. Latin accent folding and Arabic vowel folding retain their existing
behavior; accent-sensitive minimal pairs can remain ambiguous in the lexical channel.

Character terms are a bounded substring aid, not a linguistic word segmenter. Lexical
queries retain their existing OR semantics, so partial phrase matches can appear
below stronger matches. Searching identifiers or phrases is not equivalent to an
exact metadata constraint.

Saved text indexes have a tokenization version. Older versions are rebuilt in
memory from saved OCR without modifying immutable artifacts; refreshing the folder
publishes the new version. Re-indexing is required to extract text with newly
selected OCR languages. Existing learned rerankers are guarded by runtime identity
and require compatible retraining/evaluation after retrieval implementation changes.

## Coverage boundaries

Bundled OCR support, semantic query-language coverage, visual query-language coverage,
and metadata-command grammar are separate capabilities. Metadata phrase parsing
and numeric intent vocabulary retain their current English/French/Arabic coverage.
The interface is not translated by installing OCR packs. Script/font/layout quality
varies; the synthetic benchmark is not an independently reviewed multilingual
accuracy certification. No translation service or network inference is used.
