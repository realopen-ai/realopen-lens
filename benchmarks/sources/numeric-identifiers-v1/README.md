# Numeric identifier regression protocol

The 30 literal cases in `cases.json` were frozen before ranking changes on
2026-10-20. SHA-256: `c646cf008506ec291b576ee3f0955a0b64f4393ec4b5bcdf2427872f088cb207`.
These are diagnostic specifications, not independent real-world relevance labels.

The new signal requires agreement between explicit form/invoice/error intent and
an identifying field in OCR. Do not infer an identifier from rarity alone. Bare
numbers, ordinary dates/amounts, type/value mismatches and multiple-number ambiguity
must not activate the new signal. Preserve leading zeros and token boundaries.
Test English, French, Arabic, Arabic-Indic digits and existing exact mixed codes.

Use the existing exact-code evidence ceiling (1.4) and existing OCR fusion weight
(0.16); do not tune them against the 357 already inspected multilingual queries.
Do not change encoders, OCR, index membership, model precision, candidate budgets
or thresholds during the before/after comparison. Query aliases may retrieve the
same typed numeric value across existing digit scripts without modifying the index.

Run both INT8 and FP32 full saved-collection evaluations, with resident query runtime
in both phases. Preserve initial failures and every query result. Compare overall,
language, kind and individual results; inspect every changed query and report all
regressions. This corpus is diagnostic, not a new validation/test holdout.

Supplement literal tests with deterministic synthetic documents rendered as images,
processed through production indexing/OCR/search. Record invented content as
synthetic provenance. Avoid treating general number queries as negative relevance
judgments: the constraint is absence of a new identifier boost, not irrelevance.
