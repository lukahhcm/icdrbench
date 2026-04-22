# CDR-Bench Domain Design

This document records the current domain design for **CDR-Bench: Benchmarking LLMs for Compositional Data Refinement**.

CDR-Bench domains are organized by realistic data-refinement scenarios. They are not meant to be one-to-one mappings from data sources to operators.

## 1. Unified Task Protocol

All benchmark tasks use the same output contract:

```json
{"status": "KEEP", "clean_text": "..."}
```

or:

```json
{"status": "DROP", "clean_text": ""}
```

This protocol lets the benchmark compare domains, workflow types, and model families under a single evaluation interface.

The model-facing task should be expressed as a natural-language data-refinement request. Operator names, Data-Juicer configs, and internal thresholds are implementation metadata unless the prompt generator intentionally verbalizes them in a user-like way.

## 2. Active Domains

### 2.1 Web

Scenario: web crawl cleanup and filtering for pretraining corpora, search indexing, and web-scale ingestion.

Typical raw data:

- Common Crawl pages
- raw or semi-raw HTML
- pages with navigation, footers, boilerplate, links, contact information, and table residue

Representative operations:

- HTML cleanup
- link/contact removal
- copyright/template cleanup
- Unicode and whitespace normalization
- table extraction into deterministic text
- length, repetition, alphanumeric, and line-quality filtering

Example user-facing request:

> Remove HTML, links, contact information, and template residue from the page, normalize whitespace, then decide whether the cleaned page is worth keeping.

### 2.2 Arxiv

Scenario: scientific TeX/source cleanup and canonicalization for scientific corpus construction.

Typical raw data:

- arXiv source text
- TeX fragments
- comments, headers, bibliography blocks, macros, and figure context

Representative operations:

- comment removal
- bibliography removal
- header removal
- macro expansion
- figure context extraction into deterministic text
- Unicode and punctuation normalization
- source-length and repetition filtering

Example user-facing request:

> Remove LaTeX comments, headers, and bibliography residue, expand macros, normalize the source, then decide whether the scientific source should be kept.

### 2.3 Knowledge Base

Scenario: support corpus and knowledge-base preparation for RAG, help-center search, and assistant grounding.

Typical raw data:

- help-center pages
- documentation text
- manuals, FAQ, README-like pages
- support snippets with repeated boilerplate or malformed tokens

Representative operations:

- repeated sentence removal
- long malformed word removal
- Unicode normalization
- whitespace/segmentation cleanup
- length, stopword, repetition, and line-quality filtering

Example user-facing request:

> Clean repeated support-template sentences and malformed tokens from the document, normalize the text, then decide whether it should enter the knowledge base.

### 2.4 PII

Scenario: sanitization and redaction before storing or releasing user-facing text.

Typical raw data:

- text containing emails, IPs, paths, phone numbers, secrets, IDs, JWTs, PEM blocks, MAC addresses, and channel identifiers
- mixed web/support/report snippets with privacy-sensitive spans

Representative operations:

- email redaction
- IP redaction
- path redaction
- phone and ID-card redaction
- secret/channel/JWT/PEM/MAC redaction
- link cleanup
- malformed substring removal

Example user-facing request:

> Redact private identifiers and credentials from the text, remove unsafe links or malformed fragments, then return the sanitized version if the result is still usable.

## 3. Compatibility Domain

### Image Safety

`image_safety` remains in `configs/domains.yaml` as a compatibility placeholder because earlier scans included image filters.

It is skipped in the current text-first CDR-Bench pipeline and should not be counted as a v1 benchmark domain.

## 4. Shared Filters

Current shared text filters:

- `alphanumeric_filter`
- `average_line_length_filter`
- `character_repetition_filter`
- `flagged_words_filter`
- `maximum_line_length_filter`
- `stopwords_filter`
- `text_length_filter`
- `word_repetition_filter`
- `words_num_filter`

These filters are not treated as fixed-default operators. The benchmark scans their statistics and recalibrates thresholds when materializing benchmark instances.

## 5. Workflow Construction Policy

CDR-Bench uses a two-part construction strategy.

Bottom-up mining:

- run candidate operators on real domain data
- record active operators per sample
- mine frequent clean/operator combinations
- keep concrete workflow candidates only when they have enough supporting samples
- exclude fallback coverage candidates from benchmark materialization

Top-down completion:

- impose a deterministic order on mined clean/operator sets
- attach filters at meaningful positions
- keep main-track variants as `clean-only`, `filter-then-clean`, and `clean-then-filter`
- reserve middle insertion for the order-sensitivity track

This design keeps workflows grounded in real data while still covering important application scenarios that pure frequency mining might miss.

## 6. Main Track vs Order Track

Main track:

- evaluates realistic compositional data refinement
- contains `clean-only`, `filter-then-clean`, and `clean-then-filter`
- reports performance by domain, workflow type, workflow length, and difficulty tier

Order-sensitivity track:

- evaluates whether the model respects operation order
- groups `front / middle / end` variants together
- requires all slots in a group to be correct
- is reported as a separate diagnostic table

## 7. Atomic Calibration

Atomic single-operator tasks are used for difficulty calibration, not as the headline benchmark.

Atomic tasks are sampled by operator globally rather than by domain/operator pair. Each instance keeps `source_domain` for analysis.

The intended use is:

- estimate which operators are intrinsically hard
- compare workflow failures against atomic failure rates
- measure compositional gap beyond isolated operator difficulty

## 8. Domain Value Summary

| Domain | Scenario | Representative Challenge |
| --- | --- | --- |
| `web` | Web-scale ingestion and pretraining data cleanup | HTML/noise removal plus quality filtering |
| `arxiv` | Scientific source corpus construction | TeX cleanup, macro/source normalization, source-specific filtering |
| `knowledge_base` | RAG/support corpus preparation | Repetition, malformed text, and knowledge-base suitability |
| `pii` | Privacy-preserving data release or storage | Multi-type redaction plus usability preservation |

Together these domains cover common real-world data-refinement settings while keeping the deterministic `status + clean_text` evaluation protocol stable.
