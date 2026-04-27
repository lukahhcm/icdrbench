# CDR-Bench Research Plan

## 1. Title and Scope

**CDR-Bench: Benchmarking LLMs on Compositional Data Refinement Recipes**

CDR-Bench studies an execution fidelity gap in natural-language data refinement: models often understand what a user wants, but fail to faithfully execute the requested refinement policy on individual samples when cleaning, filtering, and ordering constraints interact.

CDR-Bench therefore evaluates whether LLMs can directly execute realistic, multi-step data-refinement requests over raw text and document-source data.

The core task is not workflow generation, tool calling, code generation, or user-simulator interaction. A model receives a raw sample and a complete user-facing refinement request, then outputs the final deterministic result:

```json
{"status": "KEEP", "clean_text": "..."}
```

or:

```json
{"status": "DROP", "clean_text": "..."}
```

For `DROP`, `clean_text` is the current text at the point where the deterministic workflow rejects the sample. For `filter-then-clean`, this is usually the raw input text; for `clean-then-filter`, it is the cleaned text that failed the filter.

The hidden reference is produced by deterministic Data-Juicer-backed operators. Final prompts should describe the user need in natural language rather than exposing operator names.

This benchmark intentionally isolates direct refinement execution from agentic tool use. Full code-generation or workflow-synthesis settings entangle planning, tool invocation, environment interaction, and debugging. CDR-Bench instead measures a lower-level capability: whether a model can operationalize a compositional refinement policy directly and faithfully on the sample itself.

## 2. Core Research Question

Given a raw unstructured data sample and a natural-language compositional refinement request, can an LLM produce the same final keep/drop decision and refined text as a deterministic reference pipeline?

This question matters because realistic data curation is rarely a single rewrite. It often combines cleaning, normalization, redaction, extraction, and filtering, and the final answer depends on both the operations and their order.

More specifically, CDR-Bench asks whether LLMs can faithfully operationalize refinement policies rather than merely:

- understand the request at a high level
- solve isolated single-step editing problems
- generate code or workflow plans that might, in principle, implement the request

The benchmark is designed to expose three concrete forms of the execution fidelity gap:

- intent understanding does not guarantee instance-level faithful execution
- atomic operator competence does not guarantee correct compositional execution
- recognizing an operation order does not guarantee following that order faithfully in the output

In this sense, CDR-Bench is not a generic instruction-following benchmark. It measures whether a model can execute a compositional data-refinement policy on individual instances with faithful keep/drop decisions, faithful text transformation, and faithful order adherence.

## 3. Motivation

Data refinement is a routine step before pretraining, RAG ingestion, search indexing, policy/report analysis, and scientific corpus construction.

In many realistic settings, users express these refinement needs in natural language first, not as finalized code. A scientist, analyst, or data engineer may know the desired policy, but still want the system to directly return cleaned data rather than a Python script, workflow graph, or partial plan. This is especially common in lightweight, iterative, or human-in-the-loop data preparation.

Representative user-facing requests:

- "Remove HTML, links, and contact information from the page, normalize whitespace, then decide whether the cleaned page is worth keeping."
- "Clean links, copyright headers, template residue, and repeated sentences from help documents before adding them to the support corpus."
- "Remove disclaimers, table residue, and abnormal long lines from reports, then decide whether the document is suitable for retrieval."
- "Remove LaTeX comments and bibliography, expand macros, normalize the source, then decide whether the source should be kept."

Existing work covers adjacent but different capabilities:

- workflow generation and code synthesis benchmark whether a model can propose or implement a pipeline
- data selection work studies which samples should be kept at corpus scale
- text editing work studies whether a model can perform precise modifications

What remains under-measured is whether a model can directly and faithfully execute a user-stated refinement policy on a specific sample.

The benchmark should therefore measure compositional execution rather than isolated text editing, isolated binary filtering, or end-to-end code generation. This framing also gives CDR-Bench practical relevance: direct execution is useful in its own right, and it is a prerequisite capability for stronger agentic data-processing systems.

## 4. Domains

CDR-Bench v1 is text-first and organized by application scenario.

Current domains:

- `web`: web crawl cleanup and filtering
- `arxiv`: scientific TeX/source cleanup and canonicalization
- `knowledge_base`: support and knowledge-base corpus preparation
- `pii`: PII sanitization and redaction

The `image_safety` config remains as a compatibility placeholder for future extensions and is skipped in the text-first benchmark.

Each active domain defines:

- a realistic data pool
- domain-relevant deterministic clean/filter operators
- mined clean-operation combinations
- materialized main-track variants
- optional order-sensitivity families

## 5. Benchmark Tracks

### 5.1 Main Track

The main track evaluates direct execution of compositional data refinement.

Workflow types:

- `clean-only`: clean, normalize, redact, or extract without filtering
- `filter-then-clean`: decide on the raw input first, then clean kept samples
- `clean-then-filter`: clean first, then decide whether to keep the final text

Main-track sampling rules:

- clean-only instances require the reference text to differ from the input
- filter variants recalibrate thresholds to target balanced `KEEP` / `DROP`
- default target drop rate is `0.5`
- fallback workflows are excluded from benchmark materialization

### 5.2 Order-Sensitivity Track

The order-sensitivity track is a grouped diagnostic track, not part of the main score.

Each order family shares:

- the same raw input
- the same clean skeleton
- the same filter
- the same calibrated threshold

Each family contains three slots:

- `front`: `filter-then-clean`
- `middle`: `clean-filter-clean`
- `end`: `clean-then-filter`

A group is kept only if at least two slots produce different references. Group-level success requires all three slots to be correct, which tests whether a model can follow the requested order rather than collapse to a generic cleaning strategy.

### 5.3 Atomic Operator Calibration

The atomic set estimates single-operator difficulty.

Atomic tasks are keyed by global operator, not by domain/operator pair. Instances keep `source_domain` only for diagnostics.

Use cases:

- estimate `atomic_failure_rate(operator)`
- calibrate workflow difficulty
- compute atomic-to-compositional gaps
- verify whether main-track failures exceed isolated operator failures

## 6. Construction Pipeline

### 6.1 Data Download

Download curated raw JSONL files into `data/raw/` from the Hugging Face raw-data repository.

### 6.2 Data-Juicer Tagging

Run each candidate operator through the repo-local Data-Juicer checkout and record whether each operator is active on each sample.

Outputs:

- `data/processed/domain_tags/*.jsonl`
- `data/processed/domain_filtered/*.jsonl`
- `data/processed/domain_filtered/all.jsonl`
- `data/processed/domain_operator_catalog.csv`

### 6.3 Bottom-Up Workflow Mining

Mine frequent active mapper/operator combinations per domain.

Rules:

- a concrete workflow candidate needs at least 5 supporting samples by default
- selected workflows are data-supported candidates
- fallback coverage candidates are recorded for inspection but excluded from `selected_workflows.csv`

Outputs:

- `data/processed/workflow_mining/<domain>/selected_workflows.csv`
- `data/processed/workflow_mining/<domain>/workflow_families.csv`
- `data/processed/workflow_mining/<domain>/workflow_candidates.yaml`

### 6.4 Workflow Library Materialization

Convert mined mapper sets into ordered clean sequences and attach candidate filters.

The materializer scans filter statistics at every checkpoint:

- `S0`: raw input
- `S1...Sfinal`: after each clean step

For the main track, only `clean-only`, `filter-then-clean`, and `clean-then-filter` variants are kept. Middle insertion is used only for the order-sensitivity track.

Outputs:

- `data/processed/workflow_library/<domain>/workflow_library.yaml`
- `data/processed/workflow_library/<domain>/workflow_variants.csv`
- `data/processed/workflow_library/<domain>/filter_attachments.csv`
- `data/processed/workflow_library/<domain>/checkpoint_filter_stats.csv`
- `data/processed/workflow_library/<domain>/order_sensitivity_families.csv`

### 6.5 Benchmark Instance Materialization

Generate final benchmark instances and deterministic references.

Outputs:

- `data/benchmark/main.jsonl`
- `data/benchmark/order_sensitivity.jsonl`
- `data/benchmark/atomic_ops.jsonl`
- summary CSVs for all three sets

This step intentionally does not generate prompts yet. Prompt generation is separated so we can later compare robust natural-language phrasings without changing the benchmark references.

## 7. Threshold and Sampling Policy

Filter thresholds should not blindly use Data-Juicer defaults. The pipeline first scans filter status/statistics, then recalibrates thresholds during benchmark materialization.

Main-track filter variants:

- target balanced `KEEP` / `DROP`
- choose thresholds from eligible candidate statistic distributions
- skip variants that cannot satisfy minimum keep/drop counts
- prefer source-record diversity when selecting final rows, so different workflows do not repeatedly use the same few examples
- cap or disable raw input length before materialization to match the intended LLM prompt budget

Order-sensitivity variants:

- use a shared threshold for `front / middle / end`
- keep only input groups with genuine order-dependent references
- require a minimum number of order-sensitive groups

Human-facing threshold values are rounded:

- length/count thresholds use coarse readable values such as 5, 10, 50, 100, 1000
- ordinary ratios use a 0.01 grid
- very small ratios may keep finer grids such as 0.001 or 0.0001
- calibrated ratio thresholds equal to 0 are treated as degenerate; the default materialization policy first tries a small positive threshold such as 0.001, and the variant is skipped if it cannot maintain enough keep/drop examples

## 8. Prompting Plan

Final prompts should be user-facing data-refinement requests, not operator lists.

Good prompt style:

- describes the desired data-cleaning outcome
- mentions ordering when order matters
- states the output contract
- avoids Data-Juicer operator names
- avoids exposing hidden thresholds in unnatural forms unless the task naturally requires a numeric constraint

Internal metadata such as `operator_sequence`, `filter_params_by_name`, and `reference_trace` is kept for deterministic execution, debugging, and later prompt generation.

## 9. Difficulty Calibration

Difficulty should not be assigned only by workflow length.

Recommended components:

- full workflow length
- number of clean steps
- number of filter steps
- threshold grounding
- input length bucket
- domain-specific formats such as HTML or LaTeX
- extraction/serialization operators
- atomic operator difficulty
- order-sensitivity membership

Atomic-to-compositional analysis is central:

```text
atomic difficulty = failure rate on single-op tasks
compositional gap = workflow failure beyond what atomic difficulty predicts
```

This supports the claim that CDR-Bench measures compositional execution, not only isolated operator imitation.

## 10. Metrics

Main metrics:

- `Workflow Success`
- `Status Accuracy`
- `CleanText Exact Match`
- `CleanText Canonical Match`

Main slices:

- domain
- workflow type
- workflow length
- input length bucket
- difficulty tier
- `KEEP` vs `DROP`

Order-sensitivity metrics:

- `Order-Variant Success`
- `Order-Family Success`
- `Order-Consistent Success`
- `Wrong-Order Collapse Rate`

Atomic metrics:

- per-operator success rate
- per-operator failure rate
- atomic-to-compositional gap

## 11. Target Scale

Planned formal scale:

- main track: around 5,000 instances
- order-sensitivity track: around 1,000 order groups, or about 3,000 slot instances
- atomic calibration: around 1,000 instances, roughly up to 10 per operator

These are targets rather than hard guarantees. Actual size depends on how many variants pass quality gates.

## 12. Research Value

CDR-Bench is valuable if it can support the following claims:

1. LLMs struggle with direct execution of compositional data-refinement requests.
2. Errors are not reducible to single-operator failure; composition introduces additional difficulty.
3. Filtering decisions and final text quality must be evaluated jointly.
4. Order-sensitive families reveal whether models follow the requested operation order.
5. Realistic domain-grounded construction is stronger than synthetic operator lists alone.

Key risks and mitigations:

- Prompts may become too operator-like. Mitigation: keep prompt generation separate and user-facing.
- References may overfit Data-Juicer quirks. Mitigation: report deterministic-reference scope clearly and use canonical matching where appropriate.
- Thresholds may become unnatural. Mitigation: recalibrate and round to human-readable values.
- Order-sensitive examples may be rare. Mitigation: keep them as a diagnostic track rather than forcing them into the main score.

## 13. Immediate Next Steps

1. Regenerate workflow mining outputs after fallback exclusion.
2. Regenerate workflow library with current threshold rounding.
3. Regenerate benchmark instances for main, order sensitivity, and atomic operators.
4. Inspect summary CSVs for skipped variants, keep/drop balance, and order-family counts.
5. Implement workflow-to-prompt generation from metadata.
6. Run pilot models on atomic, main, and order tracks.
7. Calibrate difficulty tiers using atomic failure rates and pilot workflow results.
