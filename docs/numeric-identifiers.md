# Numeric identifiers in hybrid search

Typed form, invoice and error numbers can provide identifying OCR evidence even
when the query and document use different languages. The query must name exactly
one supported type and one standalone numeric value explicitly bound to that type
or an identifier qualifier. Phrases such as `invoice for 2026` are not explicit ID
requests. Forms also recognize the supported IRS agency modifiers. OCR must contain the same
value at the first explicit identifier of that type, optionally separated by a small number of
identifier qualifiers such as `No.`, `number`, `code`, `numéro`, or `رقم`.

Examples include `Form 4868`, `facture numéro 846231`, and `خطأ رقم 1045`.
The current marker vocabulary supports English, French and Arabic. This is an
explicit vocabulary, not general translation or broad semantic understanding.

## Conservative boundaries

- Preserve leading zeros, exact values and word boundaries: `004231` differs from
  `4231`, and `4868` differs from `48680`.
- Require type agreement. An order number alone does not establish an invoice ID.
- Preserve complete compound identifiers: `1099-MISC` and `1099-NEC` must not
  become the same number. A form's later cross-reference does not override its
  first explicit identifying number.
- Suppress the new signal for multiple-number/type ambiguity and explicit year,
  date, amount or quantity cues. Bare rare numbers receive no new identifier boost.
- Treat a year-shaped value as an identifier only with explicit typed agreement;
  `Form 2026` can be an identifier while `invoice for year 2026` is ambiguous.
- Ordinary numeric dates and decimal amounts do not qualify as strong exact codes
  merely because they contain punctuation. Explicitly quoted literals and valid
  IP addresses and valid IP:port endpoints retain their existing exact-match behavior.

Strong typed evidence uses the existing exact-code strength (1.4) and existing
bounded OCR fusion weight (0.16). It does not add another independent OCR vote or
change semantic thresholds, model defaults, candidate budgets, or embedding files.
Existing mixed identifiers such as `INV-2026-0042` and `ERR_42` retain their path.

Typed numeric queries also include ASCII, Arabic-Indic and Persian digit aliases
within the existing lexical term limit. The stored FTS tokenization is unchanged,
so existing published indexes can be searched without rebuilding them. The query
alias improves candidate retrieval; evidence still requires matching typed OCR.

## Limitations

This is not fuzzy OCR repair. Missing or incorrect type/value text may prevent
activation. The first explicit identifier is a conservative heuristic, not layout
understanding: a body reference can still qualify if the true heading is missing
and no earlier typed identifier is recognized. Queries with multiple
numbers intentionally fall back to existing retrieval. Strong quoted or ordinary
lexical evidence can still match numbers independently of this new signal.

Existing experimental logistic rerankers were not retrained against this changed
evidence distribution. This branch evaluates default hybrid search; those artifacts
remain experimental and INT8-only. The Arabic-to-English semantic candidate gap is
separate and is not repaired by a numeric rule.

## Reproduce diagnostics

```sh
uv run --no-sync python -m benchmarks.datasets.identifier_fixtures \
  --output data/numeric-identifiers-local

uv run --no-sync lens index data/numeric-identifiers-local \
  --index-dir runs/numeric-identifiers-local --languages eng --log

uv run --no-sync lens search "facture numéro 846231" \
  --index-dir runs/numeric-identifiers-local --interactive
```

The frozen literal cases and document references live under
`benchmarks/sources/numeric-identifiers-v1/`. Diagnostic content is synthetic and
explicitly identified; it is not an independent relevance holdout.
The renderer records Pillow/FreeType versions and actual output hashes. Platform
rendering differences are marked against the original byte references rather than
silently treated as identical inputs. Before/after indexing must use the same
unchanged generated images; existing different files are never overwritten.

The collection evaluator supports `--resident` for bounded repeated production
queries. Runtime selection is part of run identity; changing it requires a new
output directory. Sessions close on completion or error. Both baseline and final
phases must use the same runtime, snapshots, model configuration and budgets.

To exercise all 30 queries through the generated production index, save individual
outputs using the same session API:

```sh
uv run --no-sync python - <<'PY' > runs/numeric-identifier-cases.jsonl
import json
from pathlib import Path
from realopen_lens.search import SearchSession

cases = json.loads(Path("benchmarks/sources/numeric-identifiers-v1/cases.json")
                   .read_text(encoding="utf-8"))["cases"]
failed = False
with SearchSession(index_dir=Path("runs/numeric-identifiers-local")) as session:
    for case in cases:
        try:
            trace = {}
            results = session.search(case["query"], limit=50, trace=trace)
            row = {"id": case["id"], "status": "ok", "trace": trace,
                   "paths": [r.paths for r in results]}
        except Exception as error:
            failed = True
            row = {"id": case["id"], "status": "error", "error": str(error)}
        print(json.dumps(row, ensure_ascii=False))
raise SystemExit(int(failed))
PY
```

Compare saved outputs from before and after the ranking change using identical
images/indexes. For the full multilingual corpus, use the
[saved-collection evaluator](retrieval-evaluation.md) with `--resident` in both
phases. Retain separate output directories and their source/configuration snapshots;
a change to scoring code must start a new run rather than resume an old identity.
