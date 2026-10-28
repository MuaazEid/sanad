# Sanad — evidence-grounded Arabic/English RAG

Sanad (سند, "support") is a local retrieval-augmented generation system for
Arabic and English documents. It answers only from a pinned corpus, cites the
exact chunk behind every claim, verifies those claims before showing them, and
ships with a gold-labeled evaluation harness so quality is measured rather than
asserted.

No cloud APIs. Python standard library + SQLite FTS5, two Ollama models
(`bge-m3` for embeddings, `qwen2.5:7b` for generation), and a laptop GPU.

## The problem

Arabic RAG demos usually fail in quiet ways:

- Normalization is skipped, so `محطة` does not match `محطه`, `أ` does not match
  `ا`, and diacritics split tokens. Retrieval looks fine in English and silently
  degrades in Arabic.
- The model is trusted to quote sources. A 7B model will happily turn "six
  lines" into "seven lines" while citing the correct chunk.
- There is no evaluation, so "it works" means three cherry-picked examples.

Sanad treats both retrieval and generation as measurable components: Arabic-aware
indexing on one side, sentence-level claim verification on the other, and a
50-question labeled test set over the whole pipeline.

## Results (held-out test split)

Corpus: 10 pinned Wikipedia documents (6 Arabic, 4 English), 585 chunks.
Generation: `qwen2.5:7b`, top-k 5. Full report:
[`reports/e2e-test-final.md`](reports/e2e-test-final.md).

### Retrieval

| mode | recall@5 | MRR | nDCG@5 | recall@10 |
|---|---|---|---|---|
| hybrid (BM25 + bge-m3, RRF) | **0.950** | **0.792** | **0.833** | **1.000** |
| lexical (BM25 only) | 0.900 | 0.664 | 0.722 | 0.900 |

### End-to-end answers (20 answerable + 5 unanswerable)

| metric | result |
|---|---|
| answer accuracy (gold fact present) | **0.950** (19/20) |
| false abstention on answerable questions | 0.050 (1/20) |
| answers fully grounded in their citations | 0.950 |
| abstention on unanswerable questions | **1.000** (5/5) |
| citation validity among answered questions | **1.000** |
| latency p50 / p95 | 8.14 s / 9.10 s |

The single failure is a retrieval miss, not a hallucination: the question asks
for Khartoum's area (`مساحة`), while the source chunk says `مساحتها` ("its
area"). Arabic morphology defeats the lexical index at k=5; the model
correctly abstains instead of guessing, and hybrid recall@10 is 1.000.

An earlier run, before the citation parser understood grouped markers such as
`[S1, S2]`, scored 0.500 accuracy while every answer was actually supported.
That report is kept as
[`reports/e2e-test-before-citation-fix.md`](reports/e2e-test-before-citation-fix.md);
fixing the parser and re-running the same test split produced 0.950.

## Architecture

```
question
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│ Hybrid retrieval                                             │
│  BM25 (SQLite FTS5, Arabic-folded) ─┐                        │
│  dense (bge-m3, 1024-d cosine) ─────┴─► RRF (k=60) ─► top-k │
└──────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│ Grounded generation (qwen2.5:7b, temperature 0)              │
│  strict prompt: numbered sources, [S#] after every claim,    │
│  exact numbers, INSUFFICIENT_EVIDENCE when unsupported       │
└──────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│ Verification — the model is an untrusted writer              │
│  citation IDs exist?                                         │
│  sentence content overlaps the chunk it cites?               │
│  every number in the claim appears in that chunk?            │
│  trailing citation-only sentences attached to their claim?   │
└──────────────────────────────────────────────────────────────┘
   │                  │                         │
   ▼                  ▼                         ▼
ANSWERED          one repair round         UNGROUNDED / INSUFFICIENT
(citations)       (same checks)            (answer withheld)
```

Key decisions:

- **Arabic normalization before indexing**: NFKC, harakat and tatweel removal,
  alef/yaa/taa-marbuta folding, Arabic-Indic digits, Arabic punctuation.
  The original text is stored untouched for display and citation.
- **Heading-aware chunking** with sentence overlap, stable IDs
  (`doc_id:NNNN`), dedupe, and size limits.
- **RRF instead of score blending**: BM25 and cosine scores live on different
  scales; rank fusion needs no per-corpus calibration and degrades cleanly when
  one signal is missing.
- **Numeric consistency check**: lexical overlap alone accepts "seven lines"
  when the source says "six"; digits and Arabic/English number words are
  compared against the cited chunk.
- **Abstention is a first-class status** (`INSUFFICIENT`, `UNGROUNDED`), not a
  sentence appended to a confident answer.
- **One repair round**: if verification fails, the model is shown the rejection
  reason and asked once to rewrite. The repaired answer must pass the same
  checks; otherwise the original rejection stands.
- **Deterministic offline embedder** (`HashingEmbedder`) for unit tests: every
  test runs without Ollama. Live-model behavior is tested separately.

## Corpus and evaluation design

`corpus/` holds 10 Wikipedia articles fetched with pinned revision IDs by
[`scripts/fetch_corpus.py`](scripts/fetch_corpus.py); each document's URL,
revision, timestamp and CC BY-SA 4.0 license are recorded in
[`corpus/manifest.json`](corpus/manifest.json). The article text is not
committed (it is CC BY-SA data, not code); the fetch script materializes it and
re-running it fetches the same revisions. The manifest is committed.

[`eval/questions.jsonl`](eval/questions.jsonl) contains 50 hand-written
questions: 40 answerable across Arabic and English, 10 unanswerable (the
corpus genuinely lacks the fact). Every answerable question carries one or more
verbatim evidence quotes. The dataset is validated against the corpus before
any evaluation runs:

```bash
sanad validate
# dataset OK: 50 questions (dev 25, test 25), 10 unanswerable
```

Labels are split 25 dev / 25 test; all numbers above are from the untouched
test split. Metrics: recall@k, MRR, nDCG@k, evidence coverage, answer accuracy
(normalized gold containment, date-order aware), false abstention, full
grounding, unanswerable abstention, citation validity, and latency.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) with:
  - `ollama pull bge-m3` (multilingual embeddings)
  - `ollama pull qwen2.5:7b` (generation)
- ~6 GB VRAM recommended; it also runs on CPU, slower.

No Python runtime dependencies.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"

ollama pull bge-m3
ollama pull qwen2.5:7b

python scripts/fetch_corpus.py   # pinned Wikipedia revisions
sanad ingest                     # build data/sanad.db (585 chunks, ~8s warm)
sanad ask "كم عدد محطات مترو الرياض؟" -k 5 --show-sources
sanad eval --split test -k 5 --generate \
    --json-out reports/e2e-test-final.json \
    --md-out  reports/e2e-test-final.md
```

Tests:

```bash
python scripts/fetch_corpus.py # once; tests validate labels against the corpus
pytest -m "not live"           # 105 tests, no Ollama needed
SANAD_LIVE=1 pytest -m live    # 1 end-to-end test against the real models
```

## Usage

```bash
sanad ingest [--force] [--reset] [--max-chars 900]
sanad search "Riyadh Metro opening" -k 5 [--lang en] [--mode hybrid|lexical] [--json]
sanad ask "متى افتتح مترو الرياض؟" -k 5 [--show-sources] [--json]
sanad validate
sanad eval --split test|dev|all -k 5 [--modes hybrid,lexical] [--generate] \
    [--support-threshold 0.6] [--json-out FILE] [--md-out FILE]
sanad stats
```

Exit codes: `0` success, `3` no hits / abstained, `1` invalid dataset.

Example — the verifier refusing a wrong number instead of presenting it:

```
$ sanad ask "كم عدد محطات مترو الرياض؟" -k 5
[UNGROUNDED] يضم مترو الرياض 85 محطة، منها 8 محطات رئيسة تربط الخطوط السبعة بعضها [S1].
grounding: 0.00 (0/1 sentences)
exit code 3
```

The model copied 85 and 8 but invented "seven lines" where the source says
"six"; the numeric check rejects the whole sentence. This is the intended
behavior, and it is covered by a live test that asserts the pipeline invariant:
an answer is either verified as grounded or withheld.

A grounded run:

```
$ sanad ask "ما إجمالي طول شبكة مترو الرياض؟" -k 5
[ANSWERED] إجمالي طول شبكة مترو الرياض يبلغ 176 كيلومتراً [S1, S2, S3, S4].
grounding: 1.00 (1/1 sentences)
```

## Layout

| Path | Role |
|---|---|
| `sanad/text.py` | Arabic/English normalization, tokenization, chunking |
| `sanad/store.py` | SQLite schema, FTS5 index, vectors, ingest/search |
| `sanad/embed.py` | Ollama and deterministic hashing embedders |
| `sanad/retrieve.py` | BM25 + dense + RRF hybrid retrieval |
| `sanad/generate.py` | prompt construction, citation parsing, claim verification |
| `sanad/pipeline.py` | retrieve → generate → verify → repair/abstain |
| `sanad/evaluate.py` | dataset validation, metrics, report rendering |
| `sanad/cli.py` | `sanad` command line |
| `scripts/fetch_corpus.py` | pinned Wikipedia corpus fetch |
| `eval/questions.jsonl` | 50 gold-labeled questions |
| `reports/` | measured evaluation reports |
| `tests/` | 106 tests (105 offline + 1 live) |

## Known limitations

- **Corpus is small and encyclopedic.** Ten documents are enough to measure the
  mechanics honestly, not to claim open-domain coverage. Numbers will change on
  a different corpus; the harness is the reusable part.
- **One retrieval miss at k=5** (`مساحة` vs `مساحتها`). No Arabic stemmer or
  reranker is applied yet; recall@10 is 1.000, and the pipeline abstains rather
  than hallucinating when the evidence is missing.
- **Lexical verification, not entailment.** The support check is content-word
  overlap plus numeric consistency. It catches wrong numbers and unrelated
  citations, but it can reject a valid paraphrase or accept a claim that
  reuses the source's words incorrectly. A cross-encoder entailment check is
  the natural next step.
- **Answer accuracy is strict containment** of a gold fact, with date-order
  normalization. "22.5 billion" and "$22.5B" match; a correct answer phrased
  around the fact may not.
- **Latency is local-model latency**: p50 ~8 s per question on a 6 GB RTX 4050.
  Ingestion took 8.2 s warm and 188 s on a cold model load for 585 chunks.
- **Live test is a contract test.** It asserts "verified or withheld", not that
  a 7B model is always right; accuracy is measured by `sanad eval`.
- **No streaming UI.** This is a CLI and library. The web layer was deliberately
  left out to keep the claim-verification path the only thing under test.

## License

Code: MIT. Corpus files: CC BY-SA 4.0, attribution in `corpus/manifest.json`.
