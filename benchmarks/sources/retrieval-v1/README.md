# Retrieval corpus and judgments v1

These files freeze public metadata and reviewed relevance, not image bytes.
See [the evaluation guide](../../../docs/retrieval-evaluation.md) for restoration,
production indexing, evaluation, and export commands.

- `corpus.json`: 2,000 distinct encoded hashes and production-normalized RGB pixel identities.
  Includes 492 valid, deduplicated images from the earlier mixed corpus and 1,508
  additional licensed COCO photographs. Three expected decode guards and five
  pixel-identical encodings from the earlier corpus were excluded.
- `reviewed-intents.json`: 60 inspected intents covering 88 distinct images,
  English/French/Arabic query text, review provenance, and ambiguous pairs that
  remain unjudged. Twelve photo targets were sampled with seed 20261020 from new
  scale images, four per split, before observing retrieval output.
- `judgments.json` (revision 2): 180 queries and 5,952 explicit query-image judgments, bound to
  the exact corpus and intent-file checksums. Grades are 2 for direct relevance
  and 0 for distinct reviewed required content. The evaluator also supports grade 1.

| Split | Corpus images | Intent families | Queries | Positive source/duplicate components |
|---|---:|---:|---:|---:|
| Train | 1,226 | 36 | 108 | 16 |
| Validation | 346 | 10 | 30 | 7 |
| Test | 428 | 14 | 42 | 11 |

| Category | Images |
|---|---:|
| Photos | 1,657 |
| Screenshots | 118 |
| Documents | 99 |
| Diagrams | 79 |
| Challenging variants | 47 |

Source and likely duplicate components remain indivisible. Consequently validation
contains photos and diagrams but no screenshots or documents. Test includes photos,
screenshots, synthetic reports and synthetic workflow diagrams; its text coverage
is not representative of independently collected real documents. Do not solve this
by moving images from a source family into another split. Add independent source
families and review them in a new version.

The reviewer is **Codex**, using visual inspection; translations were authored and
checked by that same agent. There is no independent human or native-speaker review.
The judgments are incomplete and mostly specific known positives and inspected
counterexamples. An unjudged image may be relevant, especially a related or derived
image. Missing judgments are never negative training labels.

Legacy targets were used in previous pipeline development. These splits isolate
future supervised reranker training, but do not retroactively make those targets
an untouched test of the existing hybrid heuristics. Results separate `new-scale`
queries from `legacy-benchmark` queries. The new subset is small and photo-only;
source-family counts, not the number of translations, determine independent sample
size. Only 34 components contain reviewed positives. This supports an initial logistic
regression experiment; it is small for selecting or validating a neural reranker.
Further pooled review is necessary before strong quality or generalization claims.

Per-image provenance and license attribution are in `corpus.json`; repository
licenses remain in the parent `sources/` directory. Image files, saved collections,
raw candidates, and run evidence remain under ignored `data/` and `runs/` directories.

## Reference audit

Revision 2 corrects the report intent in all three languages. A full-resolution
inspection and the source generator show sections numbered 1–9; the original
query mistakenly specified A–H. The correction removes that unsupported constraint
without changing relevance grades, split membership, or ranking code. The entire
evaluation is repeated using the corrected frozen bundle.

`reviewed-intents.initial.json` and `judgments.initial.json` preserve the original
inputs for audit. They contain a known reference error and are not training inputs.
`corrections-v2.json` records original/corrected checksums, exact query changes,
evidence, and run locations. The corrected test intent is not an untouched holdout.
Use the canonical `judgments.json` for new evaluations and training exports.

## Training-only pooled review

`training-review-v1.json` adds explicit agent-reviewed decisions for 212 training
family/image pairs from the original hybrid top 10. Its four direct matches and
165 negatives add 507 judgments across translations; 43 ambiguous decisions remain
unjudged. The canonical bundle and all validation/test judgments remain unchanged.
Per-image observations and decision reasons preserve the review evidence. See the
[logistic guide](../../../docs/logistic-reranker.md) for application and fitting.
