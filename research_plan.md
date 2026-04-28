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

The domain split is important for construction as well as evaluation. We do not start from abstract operator lists and then synthesize fake tasks. Instead, each domain is grounded in raw corpora that naturally exhibit the corresponding refinement needs:

- `web`: HTML cleanup, link removal, whitespace normalization, and content-quality filtering
- `arxiv`: LaTeX/source cleanup, macro expansion, bibliography/header/comment removal, and source-text filtering
- `knowledge_base`: support-style corpus cleanup for retrieval and indexing
- `pii`: privacy-sensitive text sanitization, identifier removal, and release-oriented filtering

This domain grounding serves two purposes:

1. it makes the benchmark requests realistic rather than operator-game prompts
2. it lets us mine workflows from actual operator activity on raw data rather than manually inventing all compositions

## 5. Retained Operator Set

CDR-Bench v1 currently keeps a restricted text-first operator set that can be replayed deterministically and mapped cleanly into a single final text output.

Retained mappers:

- `clean_channel_id_mapper`
- `clean_copyright_mapper`
- `clean_email_mapper`
- `clean_html_mapper`
- `clean_ip_mapper`
- `clean_jwt_mapper`
- `clean_links_mapper`
- `clean_mac_mapper`
- `clean_path_mapper`
- `clean_phone_mapper`
- `clean_secret_mapper`
- `expand_macro_mapper`
- `extract_tables_from_html_mapper`
- `fix_unicode_mapper`
- `latex_figure_context_extractor_mapper`
- `punctuation_normalization_mapper`
- `remove_bibliography_mapper`
- `remove_comments_mapper`
- `remove_header_mapper`
- `remove_long_words_mapper`
- `remove_repeat_sentences_mapper`
- `remove_specific_chars_mapper`
- `remove_words_with_incorrect_substrings_mapper`
- `whitespace_normalization_mapper`

Retained filters:

- `alphanumeric_filter`
- `average_line_length_filter`
- `character_repetition_filter`
- `maximum_line_length_filter`
- `text_length_filter`
- `word_repetition_filter`
- `words_num_filter`

Selection constraints for v1:

- benchmark mappers must preserve one input row to one output row
- operators must admit deterministic replay in the local Data-Juicer checkout
- the final benchmark output for each sample must still be representable as a single `status + clean_text` pair
- operators with unstable semantics, external dependencies that break determinism, or poor promptability are excluded from the benchmark set

## 6. Benchmark Tracks

CDR-Bench is materialized as three related but distinct tracks.

### 6.1 Main Track

The main track evaluates direct execution of compositional data refinement.

Workflow types:

- `clean-only`: clean, normalize, redact, or extract without filtering
- `filter-then-clean`: decide on the raw input first, then clean kept samples
- `clean-then-filter`: clean first, then decide whether to keep the final text

Main-track sampling rules:

- clean-only instances require the reference text to differ from the input
- filter variants recalibrate thresholds to target balanced `KEEP` / `DROP`
- the default target drop rate is `0.5`
- fallback workflows are excluded from benchmark materialization

### 6.2 Order-Sensitivity Track

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

### 6.3 Atomic Operator Calibration

The atomic set estimates single-operator difficulty.

Atomic tasks are keyed by global operator, not by domain/operator pair. Instances keep `source_domain` only for diagnostics.

Use cases:

- estimate `atomic_failure_rate(operator)`
- calibrate workflow difficulty
- compute atomic-to-compositional gaps
- verify whether main-track failures exceed isolated operator failures

## 7. End-to-End Construction Pipeline

This section is the most important one for the benchmark-construction method. The current repository implements a seven-stage pipeline that starts from raw JSONL corpora and ends with eval-ready prompt files and model-scoring scripts.

### 7.1 Stage 1: Curated Raw Data Collection

We first collect domain-relevant raw corpora and store them as JSONL files under `data/raw/`. The current release manifest pulls curated files from a Hugging Face raw-data repository. The raw pool includes sources such as web pages, arXiv/LaTeX source text, support-style documents, and PII-heavy corpora.

The design principle is to start from naturally messy text rather than synthetic operator inputs. This increases ecological validity and allows the subsequent mining stages to discover realistic operator combinations.

Primary output:

- `data/raw/<domain>/*.jsonl`

### 7.2 Stage 2: Operator Tagging and Domain Assignment

Next, we run the repo-local Data-Juicer checkout to probe each raw sample with the retained operators. This stage has two goals:

1. assign each sample to a benchmark domain
2. record which operators are actually active on that sample

The tagging logic distinguishes between two cases:

- mappers are replayed to see whether they materially change the text
- filters are analyzed through their statistics and keep/drop behavior

For each sample, we save operator-activity metadata such as active mapper names, filter statistics, and domain labels. This stage is the bridge between raw corpora and downstream workflow mining.

Primary outputs:

- `data/processed/domain_tags/*.jsonl`
- `data/processed/domain_filtered/*.jsonl`
- `data/processed/domain_filtered/all.jsonl`
- `data/processed/domain_operator_catalog.csv`
- `data/processed/domain_labeling_summary.csv`
- `data/processed/dj_cli_tagging/`

Construction notes:

- most text-cleaning mappers are executed through `dj-process`
- meta/stat filters are executed through `dj-analyze`
- only one-row-to-one-row mapper behavior is retained for benchmark construction
- extraction operators such as `extract_tables_from_html_mapper` and `latex_figure_context_extractor_mapper` are serialized back into deterministic text in the `text` field

### 7.3 Stage 3: Bottom-Up Workflow Mining

After tagging, we mine candidate workflows separately inside each domain. The unit we mine first is not a fully ordered pipeline, but a supported set of active clean operators observed on real records.

The mining stage answers:

- which clean operators frequently co-occur on real samples in a domain
- which combinations have enough support to justify benchmark inclusion
- which domain/operator combinations are too rare and should remain fallback-only

By default, a concrete workflow candidate must have at least five supporting samples. Candidates that fail the support threshold may still be recorded as fallback coverage artifacts, but they are not allowed into the benchmark materialization path.

Primary outputs:

- `data/processed/recipe_mining/<domain>/selected_recipes.csv`
- `data/processed/recipe_mining/<domain>/recipe_families.csv`
- `data/processed/recipe_mining/<domain>/recipe_candidates.yaml`
- `data/processed/recipe_mining/domain_recipe_mining_summary.csv`

Methodologically, this stage is what keeps the benchmark from becoming a hand-written list of arbitrary operator recipes. The main workflow pool is data-supported.

### 7.4 Stage 4: Workflow Library Materialization

The mining output still does not define fully replayable benchmark workflows. The next step converts mined operator sets into ordered deterministic clean sequences and attaches candidate filters.

This stage performs four concrete operations:

1. order mined clean sets into deterministic clean sequences
2. replay those clean sequences to construct intermediate checkpoints `S0, S1, ..., Sfinal`
3. scan filter statistics at each checkpoint
4. materialize benchmark-ready workflow variants and order-sensitivity families

Checkpoint semantics:

- `S0`: the raw input text
- `S1 ... Sfinal`: the text after each successive clean step

The checkpoint scan is what lets us attach filters at different positions instead of always filtering only raw text or only final text.

The workflow library stage produces:

- main-track workflow variants: `clean-only`, `filter-then-clean`, `clean-then-filter`
- order-sensitivity families: `front`, `middle`, `end`
- filter-attachment metadata
- checkpoint-level filter statistics used later for threshold calibration

Primary outputs:

- `data/processed/recipe_library/<domain>/recipe_library.yaml`
- `data/processed/recipe_library/<domain>/recipe_variants.csv`
- `data/processed/recipe_library/<domain>/filter_attachments.csv`
- `data/processed/recipe_library/<domain>/checkpoint_filter_stats.csv`
- `data/processed/recipe_library/<domain>/order_sensitivity_families.csv`
- `data/processed/recipe_library/recipe_library_summary.csv`

### 7.5 Stage 5: Benchmark Instance Materialization and Deterministic References

The workflow library defines workflow templates; it does not yet define the final benchmark instances. In this stage, we select concrete samples, calibrate thresholds, replay deterministic references, and write the final benchmark JSONL files.

This stage is implemented in `materialize_benchmark_instances.py` and currently handles all three tracks.

#### Main-track instance materialization

For each workflow variant:

1. gather candidate records that support the required clean operators
2. exclude samples above the configured raw-input length cap
3. if the workflow includes a filter, compute the relevant filter statistic at the filter insertion point
4. recalibrate the threshold on the candidate pool toward a target drop rate
5. replay the full deterministic workflow on the selected records
6. keep only instances that satisfy balance and quality constraints

For `clean-only` workflows, we keep only instances whose final text differs from the raw input.

#### Order-sensitivity materialization

For each order family:

1. use the same raw record across `front`, `middle`, and `end`
2. calibrate one shared threshold across the family
3. replay all three variants
4. keep the group only if at least two slots have different deterministic references
5. require a minimum number of genuinely order-sensitive groups per family

#### Atomic materialization

Atomic construction is global rather than domain-local.

- for mapper operators, we search all records where the mapper is active and keep outputs that actually change the text
- for filter operators, we compute the relevant statistic on the raw text, recalibrate a balanced threshold, and then sample balanced `KEEP` / `DROP` cases

Atomic instances are keyed by operator and retain `source_domain` only as diagnostic metadata.

Deterministic reference execution records:

- `reference_status`
- `reference_text`
- `intermediate_text_at_drop`
- `reference_trace`

The `reference_trace` field stores step-level replay metadata, which is useful for debugging and future error analysis but remains hidden from the model during evaluation.

Primary outputs:

- `data/benchmark/main.jsonl`
- `data/benchmark/order_sensitivity.jsonl`
- `data/benchmark/atomic_ops.jsonl`
- `data/benchmark/main_summary.csv`
- `data/benchmark/order_sensitivity_summary.csv`
- `data/benchmark/atomic_ops_summary.csv`

### 7.6 Stage 6: Prompt Library Generation

Prompt generation is intentionally separated from deterministic benchmark construction. This separation matters because we want to vary user-facing wording without regenerating the hidden references.

The prompt pipeline operates at the workflow level rather than the instance level. Given a workflow definition, operator evidence, and filter thresholds, it produces multiple stylistically distinct but functionally equivalent natural-language requests.

Current prompt generation principles:

- preserve the exact internal workflow order
- express any active filter threshold in human-facing language
- hide Data-Juicer operator names and implementation details
- produce prompts that sound like plausible user requests rather than operator checklists

Each generated candidate is then judged for:

- functional equivalence
- order correctness
- threshold grounding
- absence of code leakage
- compatibility with a fixed benchmark output wrapper

Primary outputs:

- `data/benchmark_prompts/<track>/recipe_prompt_library.jsonl`
- `data/benchmark_prompts/<track>/prompt_generation_summary.jsonl`

### 7.7 Stage 7: Eval-Ready Prompt Track Construction

After prompt generation, we combine accepted workflow-level prompt candidates with benchmark rows to create eval-ready files. Each benchmark instance is assigned a small set of prompt variants sampled deterministically from distinct accepted styles.

This stage keeps:

- the raw input text
- the deterministic reference
- the workflow-level prompt key
- multiple user-facing prompt variants for the same underlying workflow

Primary outputs:

- `data/benchmark_prompts/<track>/eval/<track>.jsonl`
- `data/benchmark_prompts/<track>/eval/prompt_eval_build_summary.jsonl`

At this point the benchmark is ready for model inference.

## 8. Threshold Calibration and Sampling Policy

Filter thresholds are not copied blindly from Data-Juicer defaults. Instead, CDR-Bench recalibrates them on the candidate pool used for benchmark construction.

The main reason is methodological: default thresholds are often too loose, too strict, or too domain-misaligned for benchmark evaluation. We need task instances that are meaningful, balanced, and promptable.

### 8.1 Main-Track Filter Calibration

For a main-track workflow with one filter:

1. compute the relevant filter statistic at the actual insertion point of the filter
2. gather the statistic over all eligible candidate records
3. choose a threshold by quantile so the target drop rate is approximately met
4. round the threshold into a human-readable value
5. replay the deterministic workflow and keep the variant only if minimum `KEEP` and `DROP` counts are satisfied

The default target drop rate is `0.5`.

### 8.2 Order-Family Filter Calibration

Order families use one shared filter threshold across `front`, `middle`, and `end`. The statistic pool is collected across the relevant insertion points for all family members, then one calibrated threshold is applied to all three slots.

This is necessary so the comparison isolates order effects rather than threshold mismatches.

### 8.3 Atomic Filter Calibration

Atomic filter tasks are calibrated globally per operator. We compute the operator statistic on raw text, estimate a balanced threshold, and then select balanced `KEEP` / `DROP` rows subject to minimum class counts.

### 8.4 Degenerate Ratio Thresholds

Very small ratio filters create a special problem: a calibrated threshold can round to exactly `0`, which produces unnatural user-facing tasks. The current policy therefore treats zero-ratio thresholds as degenerate.

Default handling:

- if a ratio threshold rounds to `0`, first try a small positive threshold such as `0.001`
- if the variant still cannot sustain enough `KEEP` and `DROP` examples, skip it

Alternative handling:

- skip the variant immediately

### 8.5 Diversity-Aware Sampling

Final row selection is best-effort diversity-aware. When the same source record is eligible for many workflows, earlier selections increase that record's usage count, and later selections deprioritize it. This helps reduce over-reuse of a few easy samples across the benchmark.

### 8.6 Input-Length Control

Raw inputs can be capped before materialization using `--max-input-chars`. This is a practical rather than conceptual constraint: the benchmark should stay within a feasible prompt budget for current strong models while still covering non-trivial long-context cases.

## 9. Prompting and Evaluation Pipeline

The deterministic benchmark and the model-evaluation pipeline are separate layers.

### 9.1 Model Input Format

At evaluation time, the model sees:

- a user-facing natural-language refinement request
- the raw input text
- a fixed JSON-only output contract

The required output is:

```json
{"status": "KEEP", "clean_text": "..."}
```

or:

```json
{"status": "DROP", "clean_text": "..."}
```

If the deterministic workflow drops the sample, `clean_text` is defined as the text state at the drop point.

### 9.2 API-Compatible Inference

The current evaluation code supports OpenAI-compatible APIs for both external hosted models and local models served through `vllm`. This keeps the model interface uniform across evaluation settings.

The current atomic-first evaluation stack supports:

- direct online inference on eval-ready JSONL files
- score-only evaluation for prediction files already generated elsewhere
- local `vllm` serving for open-weight models

### 9.3 Why Prompt Generation Is Separate

Keeping prompt generation separate from reference construction has three benefits:

1. prompt variants can be revised without re-running deterministic references
2. benchmark validity does not depend on one brittle prompt template
3. prompt robustness can be studied as an independent axis

### 9.4 What Is Hidden from the Model

The model never sees:

- operator names
- operator code or documentation
- filter parameter keys
- threshold-calibration metadata
- the deterministic `reference_trace`

These remain internal benchmark metadata only.

## 10. Difficulty Calibration

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

## 11. Metrics

We report two primary instance-level metrics and one order-sensitive group metric.

Primary metrics:

- `Recipe Success (RS)`
- `Refinement Gain (RG)`
- `Order-Consistent Success (OCS)` for the order-sensitivity track

### 11.1 Recipe Success

`RS` measures exact recipe execution for one prompt variant:

```text
status_match = predicted_status == reference_status
text_exact_match = predicted_clean_text == reference_text
recipe_success = status_match AND text_exact_match
```

This is the main exact-match metric. It does not canonicalize whitespace, Unicode, or punctuation away before comparison.

When multiple prompt variants are used for the same benchmark instance, the natural multi-prompt extension is:

```text
RS@K = 1 if any of the K prompt variants succeeds exactly, else 0
```

and an average per-style success rate can also be reported across the same set of prompt variants.

### 11.2 Refinement Gain

`RG` measures progress toward the deterministic reference using raw-string edit distance.

```text
d_input = edit_distance(input_text, reference_text)
d_pred  = edit_distance(predicted_clean_text, reference_text)

if d_input == 0:
    refinement_gain = 1.0 if d_pred == 0 else 0.0
else:
    refinement_gain = 1 - d_pred / d_input
```

Notes:

- `RG = 1` means the prediction exactly matches the deterministic reference text
- `RG = 0` means the prediction makes no progress relative to the raw input
- `RG < 0` means the prediction moves farther away from the reference than the raw input was
- the `d_input = 0` case is necessary for filter-style tasks whose correct reference text is identical to the raw input at the drop point

### 11.3 Order-Consistent Success

For the order-sensitivity track, benchmark rows are grouped by `order_group_instance_id`. A group succeeds only if every slot in the group succeeds under exact recipe execution:

```text
OCS(group) = 1 if RS(slot) = 1 for all slots in the group, else 0
```

The reported score is the mean of `OCS(group)` over all order groups.

Auxiliary metrics that may still be useful for analysis:

- `Status Accuracy`
- `Exact Text Match Rate`
- `Canonical Text Match Rate` as a relaxed secondary analysis only

Main slices:

- domain
- workflow type
- workflow length
- input length bucket
- difficulty tier
- `KEEP` vs `DROP`

Order-sensitivity metrics:

- `Order-Consistent Success`
- optional wrong-order collapse diagnostics

Atomic metrics:

- per-operator recipe success rate
- per-operator failure rate
- per-operator average refinement gain
- atomic-to-compositional gap

## 12. Target Scale

Planned formal scale:

- main track: around 5,000 instances
- order-sensitivity track: around 1,000 order groups, or about 3,000 slot instances
- atomic calibration: around 1,000 instances, roughly up to 10 per operator

These are targets rather than hard guarantees. Actual size depends on how many variants pass quality gates.

## 13. Research Value

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

## 14. Current Status and Remaining Paper Tasks

Implemented in the current repository:

- raw-data download and manifest handling
- domain assignment and operator tagging
- bottom-up workflow mining with fallback exclusion
- workflow-library materialization with checkpoint-level filter stats
- benchmark-instance materialization for main, order-sensitivity, and atomic tracks
- workflow-level prompt generation and judging
- eval-ready prompt-track construction
- API-compatible model evaluation
- atomic-first scoring with `workflow_success` and `refinement_gain`
- local `vllm` serving support for open-weight model evaluation

What remains for the paper:

1. freeze the reported benchmark snapshot and final dataset counts
2. write the benchmark-construction method section from Sections 4--9 of this plan
3. summarize the retained operator set in an appendix table
4. run pilot and final model suites on atomic, main, and order tracks
5. define the reported difficulty tiers using atomic and compositional results
6. choose the final paper tables for benchmark composition, operator coverage, and evaluation slices
