# Candidate retrieval: depth, precision, language and OCR

Date: 2026-10-20. **Diagnostic study; hybrid search and int8 remain the defaults.**

The largest measured retrieval weakness is the pinned int8 SigLIP2 artifacts.
Replacing both SigLIP towers with their pinned FP32 counterparts raises known-judgment
Recall@50 from 0.9223 to 0.9930 on the inspected query set. Increasing candidate
depth recovers more targets into the pool but barely improves final ranking.
These findings justify a fresh precision-policy evaluation before neural reranking;
they do not establish a new production default or general accuracy guarantee.

The [structured evidence](2026-10-20-candidate-retrieval.json) publishes individual
query metrics, positive ranks, parity probes, model fingerprints, OCR controls and
remaining misses. The [reproduction guide](../candidate-retrieval.md) documents
commands and retained local artifacts.

## Scope and evaluation conditions

The saved production collection contains 2,052 unique images: 1,657 photographs,
145 screenshots, 124 documents, 79 diagrams and 47 challenging cases. The same
357 English/French/Arabic queries and frozen judgments are used throughout. All
were inspected in earlier work, including earlier training queries; every cohort
is diagnostic for this study. Agent-reviewed judgments remain incomplete and lack
independent human/native-speaker review. Unknown results are not training negatives.

Depth tests execute full production search requests. Precision tests reuse the
same OCR and production fusion function with complete image/passage embeddings;
expansion reuses recorded production query requests. Their latencies are not
interchangeable. No GLM or RapidOCR evaluation was performed. Models and language
packs are local; the principal inference/evaluation runs used OS-level networking
denied after explicit model downloads.

## 1. Candidate depth

| Depth per channel | Full-pool recall | Recall@50 | MRR@10 | nDCG@10 | Median request s | P95 s | Peak tree MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.9428 | 0.9223 | 0.5585 | 0.6064 | 1.601 | 2.133 | 1171.7 |
| 100 | 0.9563 | 0.9223 | 0.5582 | 0.6061 | 1.583 | 1.822 | 1173.3 |
| 200 | 0.9818 | 0.9223 | 0.5593 | 0.6069 | 1.549 | 1.646 | 1175.9 |

All three runs completed 357 requests with zero query errors. The default run
exactly reproduces all 357 earlier candidate dictionaries and hybrid rankings.
Budgets are independently configurable in the API/CLI and form part of resume
identity. Existing learned artifacts reject non-50 budgets rather than accepting
a changed training distribution.

Larger pools recover the three previously investigated Arabic targets at visual
ranks 105, 195 and 165, but the int8-based fusion generally leaves them below the
final cutoff. Pool inclusion is not ranking success. The slightly lower later-run
times are not a speedup claim: runs were sequential, OS caches/background work were
not controlled, and request time includes verification and encoder startup.

## 2. Encoder parity and full-corpus precision

Pinned ONNX graphs were compared against official local PyTorch implementations.
The initial probes cover 33 fixed-seed/diagnostic images, 360 text inputs
(357 queries plus three controls), and 36 OCR passages. Production EXIF/alpha
decoding is shared for the small image probe; official resize/normalization differs
by at most 1.2e-7 per pixel. The full precision comparison encodes all 2,052 original
images and 2,037 OCR passages, retaining each embedding and input result.

| Probe | Mean cosine: int8 vs FP32 | FP32 vs reference |
|---|---:|---:|
| SigLIP2 vision | 0.7848 | approximately 1.0000 |
| SigLIP2 text | 0.8959 | approximately 1.0000 with identical lowercase inputs |
| E5 queries | 0.9965 | approximately 1.0000 |
| E5 passages | 0.9960 | approximately 1.0000 |

Native reference tokenization initially differs on 274/360 SigLIP inputs because
the fast Gemma tokenizer does not apply the adapter's explicit lowercasing. The
86 identical-token cases already agree numerically. A separate lowercase reference
run produces zero token mismatches and mean cosine 0.999999989 across all 360.
This distinguishes preprocessing from graph conversion; it does not support
calling the FP32 exporter broken. Native-case retrieval is a separate control below.

With unchanged OCR, 50 candidates per channel and unchanged fusion thresholds:

| Precision policy | Pool recall | Recall@50 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| Current int8 | 0.9428 | 0.9223 | 0.5585 | 0.6064 |
| FP32 SigLIP vision only | 0.9727 | 0.9698 | 0.7054 | 0.7466 |
| FP32 SigLIP text only | 0.9574 | 0.9452 | 0.6497 | 0.6979 |
| FP32 both SigLIP towers, int8 E5 | 0.9936 | 0.9930 | 0.8310 | 0.8602 |
| FP32 E5 only | 0.9400 | 0.9195 | 0.5590 | 0.6073 |
| FP32 all encoders | 0.9936 | 0.9930 | 0.8294 | 0.8591 |
| FP32 SigLIP with native-case reference queries | 0.9850 | 0.9762 | 0.7554 | 0.7984 |

Every policy retains 357 individual query results. Int8 replay must match all
production channel/hybrid rankings exactly before the command accepts the comparison.
The separate tower ablations are follow-up diagnostics after the initial precision
comparison; no test-based model/default selection is claimed.

Pure visual Recall@50 rises from 0.8519 to 0.9905 when both SigLIP towers use FP32,
so the effect is not merely the OCR fusion threshold. Arabic hybrid Recall@50 rises
from 0.8359 to 0.9832; MRR rises from 0.4098 to 0.7419. The Arabic cafeteria,
Inkscape welcome and Wikipedia targets move from absent pools to hybrid ranks
3, 2 and 2 respectively. This identifies a significant degradation in these
specific quantized artifacts; it does not prove every int8 export has this problem.
The observation is tied to the recorded macOS CPU/ONNX Runtime environment.
These tests do not isolate quantization/export choices from int8 kernel behavior;
alternate runtime/platform checks remain necessary.

There are still seven known-judgment nDCG regressions with FP32 SigLIP, including
three English photo targets moving from rank 1 to 2 and Arabic FileZilla moving
from 9 to 18. Unjudged competitors are not automatically incorrect. Broad Arabic
operations-report retrieval still misses all 20 positives; their FP32 visual ranks
are 67–344 and E5 ranks 1,065–1,678 despite available English OCR. The Arabic Audacity
metadata target has visual rank 83 and E5 rank 620. These inputs have 27 and 22
SigLIP tokens, below the 64-token limit: query truncation does not explain them.
Original images and OCR were inspected; cross-language representation/coverage
remains an issue even after the precision improvement.

Costs matter. The small int8 text probe peaks at 678.8 MiB versus 2,107.0 MiB for
FP32; measured mean text inference is about 24 ms versus 64 ms, with model loads
about 0.89 s versus 2.30 s. Full-corpus FP32 vision peaks at 791.7 MiB. These are
isolated probe measurements, not certified production request or 8 GB hardware
results. A production precision policy needs consistent index/query configuration,
fresh judgments and end-to-end latency/memory measurements before promotion.

## 3. Bounded reviewed-English expansion

At most one existing reviewed English translation is added to an original query.
Translations are selected by family/fold, never relevance labels. Candidate unions
are deduplicated; the RRF experiment takes one maximum vote per channel across
language variants. Its original-query RRF control separates fusion effects from
expansion. Max-hybrid fusion is experimental because scores across languages are
not calibrated.

| Max-hybrid experiment | Pool recall | Recall@50 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| Original, all languages | 0.9428 | 0.9223 | 0.5585 | 0.6064 |
| Original + reviewed English | 0.9748 | 0.9595 | 0.6484 | 0.6971 |
| Original Arabic | 0.8863 | 0.8359 | 0.4098 | 0.4483 |
| Arabic + reviewed English | 0.9790 | 0.9433 | 0.6438 | 0.6869 |

English controls use one request. Other languages can use twice the encodings and
up to twice the candidate channels. This is a translation-oracle diagnostic, not
a measured automatic local translator or expansion latency result. Given the
precision findings, adding a translation model is not the first production change.

## 4. Missing OCR, normalization and passage coverage

The first Tesseract pass could consume its 15-second cap and abort while the
30-second overall budget still had time available. A second full-page sparse pass
also timed out, increasing latency without recovering text. That failed trial is
retained. Monolingual/low-resolution probes did not establish a usable replacement;
low-confidence OSD even misclassified the French page as Arabic, so no script-based
language routing was adopted.

The implemented fallback uses up to three disjoint horizontal bands only after
the whole-image timeout. All failed/successful native attempts count toward four;
remaining time is shared between bands, and PNG serialization consumes its attempt
budget. Successful whole-image OCR keeps its existing path. Word boxes are mapped
back into full OCR-input coordinates. Missing bands and original failures remain
explicit; this is partial recovery, not a complete transcript.

The frozen native control set contains four images per original benchmark category
plus the dense newspaper: 21 total. The 20 previously successful transcripts are
identical after the fix; six have existing quality references and unchanged quality
metrics. The newspaper changes from a 15.4-second failure to a 30.17-second partial
result with 1,494 characters/272 words from the top band. It recognizes AURORE and
ZOLA; the main title is still imperfect and the dense body remains unavailable.
No full-reference OCR accuracy is invented for it.

A separate production collection verifies that recovered text reaches indexing,
semantic processing and keyword search. `AURORE` returns the newspaper with a
structured `PartialOCR` issue; indexing reports two searchable images, zero failed
paths and one degraded image. An integration check caught and fixed warnings being
published as healthy status. An earlier temporary-source ctime change correctly
blocked publication; retry evidence is retained without weakening that guard.
Partial analyses remain reusable cache entries; repeated indexing does not imply
a fresh full-page recognition attempt.

An additional Arabic-normalized keyword view strips tatweel, folds selected letter
variants and normalizes digits while retaining the original view. It recovers zero
additional known positives across all 357 queries and leaves lexical Recall@50
unchanged. Production normalization is therefore unchanged. Passage inspection
finds a maximum of 6,249 OCR characters and four passages per image, below the
32,000-character/16-passage limits; these caps do not explain the observed misses.

## Validation, artifacts and remaining work

Regression coverage includes independent candidate budgets, preserved defaults,
reranker rejection, resume identity, paired-comparison guards, reference input/hash
integrity, bounded timeout recovery, word coordinates, partial status propagation
and usable partial cache entries. Existing training workflows also reject wider
pools rather than mislabeling their artifacts as budget-50 models. Native binary
and language-pack fingerprints guard OCR resume; worker JSON survives CP1252 pipes.

All 253 tests pass, together with Black, Flake8, Ruff and mypy (default, Windows
and Linux targets). The built/installed application wheel matches the production
source and passes real CLI/API budget and reranker-rejection checks. Native
Windows/Linux inference and 8 GB hardware remain unverified by these macOS runs.

Complete evidence remains under `runs/candidate-retrieval-v1/`: depth SQLite/JSONL
records, input/model fingerprints, parity arrays and per-input outputs, precision
candidate traces, expansion/normalization results, three OCR trials, and production
index/search checks. Historical runs are preserved. Images, downloaded models and
large raw run directories remain ignored; the public JSON retains compact individual
results and evidence hashes. Large-scale retrieval comparisons reuse unchanged OCR;
the recovery change is evaluated separately, not presented as a fully reindexed
2,052-image after-run.

An actual encoder interruption preserved 51 completed numerical checkpoints.
Resuming finished all 360 inputs with bit-identical vectors to the uninterrupted
probe; a subsequent completed-run resume did not rewrite its report. Isolated
encoder workers now have a stage deadline and atomic per-input vector checkpoints.

Keep current production model defaults while freezing a new independently reviewed
multilingual holdout. Next, validate a better SigLIP precision/quantization policy
through production indexing/search, including index/query compatibility and actual
low-memory hardware. Broaden Arabic document/screenshot coverage, investigate the
remaining broad-query misses and dense body OCR, then reconsider translation and
learned reranking against the corrected embedding baseline.
