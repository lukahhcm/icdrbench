# CDR-Bench

**CDR-Bench: Benchmarking LLMs for Compositional Data Refinement**

This repository builds CDR-Bench data from raw JSONL corpora using a repo-local Data-Juicer checkout. The current pipeline downloads raw data, tags operator activity, mines domain workflows, materializes workflow variants, and generates deterministic references for the main, order-sensitivity, and atomic calibration sets.

The repository and package are now named `cdrbench`.

## Pipeline

Run the full construction flow in this order:

1. Download raw JSONL files into `data/raw/`.
2. Run Data-Juicer CLI tagging with `tag_and_assign_domains.py`.
3. Mine per-domain workflow candidates with `mine_domain_workflows.py`.
4. Materialize workflow libraries with `materialize_domain_workflows.py`.
5. Generate benchmark instances and deterministic references with `materialize_benchmark_instances.py`.

The final benchmark files are written under `data/benchmark/`.

## 1. Clone

```bash
git clone https://github.com/lukahhcm/cdrbench.git
cd cdrbench
```

## 2. Environment

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create a small project environment:

```bash
uv venv .venv-ops --python 3.11
uv pip install --python .venv-ops/bin/python -e .
uv pip install --python .venv-ops/bin/python -U huggingface_hub py-data-juicer
```

The repository already vendors `data-juicer/`, including the customized operators used by CDR-Bench. The preparation scripts prefer:

```bash
python data-juicer/tools/process_data.py
python data-juicer/tools/analyze_data.py
```

Only if `./data-juicer` is unavailable do they fall back to system `dj-process` / `dj-analyze`.

## 3. Download Raw Data

```bash
HF_TOKEN=<your_hf_token_if_needed> \
PYTHONPATH=src .venv-ops/bin/python -m cdrbench.release.download_hf_jsonl \
  --repo-id lukahh/cdrbench-raw \
  --repo-root .
```

The manifest downloads JSONL files into `data/raw/`, including arXiv, Common Crawl, Wikipedia/help-style text, government reports, and PII corpora.

## 4. Tag Operators and Assign Domains

Small smoke test:

```bash
PYTHONPATH=src .venv-ops/bin/python -m cdrbench.prepare_data.tag_and_assign_domains --max-records 200
```

Full resumable run:

```bash
PYTHONPATH=src .venv-ops/bin/python -m cdrbench.prepare_data.tag_and_assign_domains --resume
```

Useful overrides:

```bash
PYTHONPATH=src .venv-ops/bin/python -m cdrbench.prepare_data.tag_and_assign_domains \
  --dj-repo-root /path/to/data-juicer \
  --dj-python /usr/bin/python3 \
  --resume
```

Outputs:

- `data/processed/domain_tags/*.jsonl`
- `data/processed/domain_filtered/*.jsonl`
- `data/processed/domain_filtered/all.jsonl`
- `data/processed/domain_operator_catalog.csv`
- `data/processed/domain_labeling_summary.csv`
- `data/processed/dj_cli_tagging/`

Notes:

- Most text-cleaning mappers run through `dj-process`.
- Meta/stat filters run through `dj-analyze`.
- Mappers used in the final benchmark should preserve one input row to one output row.
- `extract_tables_from_html_mapper` writes deterministic TSV text back into `text`.
- `latex_figure_context_extractor_mapper` writes merged figure-context text back into `text`.

## 5. Mine Workflow Candidates

```bash
PYTHONPATH=src .venv-ops/bin/python -m cdrbench.prepare_data.mine_domain_workflows \
  --tagged-dir data/processed/domain_tags \
  --output-dir data/processed/workflow_mining
```

By default, a concrete workflow candidate needs at least `5` supporting samples. Adjust this with `--min-workflow-support`.

Key outputs:

- `data/processed/workflow_mining/<domain>/workflow_families.csv`
- `data/processed/workflow_mining/<domain>/selected_workflows.csv`
- `data/processed/workflow_mining/<domain>/workflow_candidates.yaml`
- `data/processed/workflow_mining/domain_workflow_mining_summary.csv`

Quick inspection:

```bash
column -s, -t < data/processed/workflow_mining/domain_workflow_mining_summary.csv | less -S
column -s, -t < data/processed/workflow_mining/web/selected_workflows.csv | less -S
sed -n '1,160p' data/processed/workflow_mining/web/workflow_candidates.yaml
```

Fallback workflow candidates are kept for inspection but excluded from benchmark materialization.

## 6. Materialize Workflow Libraries

```bash
PYTHONPATH=src .venv-ops/bin/python -m cdrbench.prepare_data.materialize_domain_workflows \
  --workflow-mining-dir data/processed/workflow_mining \
  --filtered-path data/processed/domain_filtered/all.jsonl \
  --output-dir data/processed/workflow_library \
  --resume
```

This step:

- orders mined clean/operator sets into deterministic clean sequences
- replays clean prefixes to build `S0, S1, ..., Sfinal` checkpoints
- scans filter statistics at each checkpoint
- produces main-track variants and order-sensitivity families

Main-track workflow types:

- `clean-only`
- `filter-then-clean`
- `clean-then-filter`

Order-sensitivity families:

- `front`: `filter-then-clean`
- `middle`: `clean-filter-clean`
- `end`: `clean-then-filter`

Key outputs:

- `data/processed/workflow_library/<domain>/workflow_library.yaml`
- `data/processed/workflow_library/<domain>/workflow_variants.csv`
- `data/processed/workflow_library/<domain>/filter_attachments.csv`
- `data/processed/workflow_library/<domain>/checkpoint_filter_stats.csv`
- `data/processed/workflow_library/<domain>/order_sensitivity_families.csv`
- `data/processed/workflow_library/workflow_library_summary.csv`

Useful checks:

```bash
column -s, -t < data/processed/workflow_library/workflow_library_summary.csv | less -S
column -s, -t < data/processed/workflow_library/web/workflow_variants.csv | less -S
column -s, -t < data/processed/workflow_library/web/checkpoint_filter_stats.csv | less -S
```

## 7. Generate Benchmark Instances and GT

```bash
PYTHONPATH=src .venv-ops/bin/python -m cdrbench.prepare_data.materialize_benchmark_instances \
  --workflow-library-dir data/processed/workflow_library \
  --filtered-path data/processed/domain_filtered/all.jsonl \
  --output-dir data/benchmark \
  --target-drop-rate 0.5 \
  --max-candidate-records 0 \
  --max-input-chars 50000 \
  --min-positive-ratio-threshold 0.001 \
  --zero-ratio-threshold-policy min-positive \
  --max-instances-per-variant 50 \
  --max-order-groups-per-family 30 \
  --max-atomic-candidate-records 1000 \
  --max-atomic-instances-per-op 10 \
  --min-keep 5 \
  --min-drop 5 \
  --min-order-sensitive-groups 5 \
  --min-atomic-keep 3 \
  --min-atomic-drop 3 \
  --resume
```

This step does not generate prompts yet. It selects samples and runs deterministic references.

Outputs:

- `data/benchmark/main.jsonl`
- `data/benchmark/order_sensitivity.jsonl`
- `data/benchmark/atomic_ops.jsonl`
- `data/benchmark/main_summary.csv`
- `data/benchmark/order_sensitivity_summary.csv`
- `data/benchmark/atomic_ops_summary.csv`

Target scale:

- main track: about `5k` instances
- order-sensitivity track: about `1k` order groups, or `3k` variant instances
- atomic calibration: about `1k` instances, up to 10 per operator

Important parameters:

- `--target-drop-rate 0.5`: calibrate filter tasks toward 50% `KEEP` and 50% `DROP`
- `--max-candidate-records 0`: scan all eligible candidate samples per workflow variant or order family; set a positive number such as `2000` for a faster capped run
- `--max-input-chars 50000`: skip raw inputs above 50k characters before GT materialization, keeping tasks within a practical prompt budget for recent strong models; use `0` to disable
- `--min-positive-ratio-threshold 0.001`: when a calibrated ratio threshold is exactly `0`, first try a small positive threshold instead of creating an unnatural zero-ratio task
- `--max-instances-per-variant 50`: cap main-track instances per workflow variant
- `--max-order-groups-per-family 30`: cap order groups per order family
- `--max-atomic-candidate-records 1000`: scan at most 1000 candidates per atomic operator
- `--max-atomic-instances-per-op 10`: cap atomic instances per operator
- `--min-keep / --min-drop`: skip main filter variants without enough keep/drop candidates
- `--min-order-sensitive-groups`: skip order families without enough genuinely order-sensitive groups
- `--min-atomic-keep / --min-atomic-drop`: skip atomic filters without enough keep/drop candidates
- `--zero-ratio-threshold-policy min-positive`: use `0.001` for degenerate calibrated ratio thresholds and let normal keep/drop balance checks decide whether to keep the variant; use `skip` to drop such variants immediately
- `--resume`: reuse per-variant cache shards in `data/benchmark/_materialize_cache_v2/` after an interrupted run

Thresholds are recalibrated during materialization. Length/count thresholds are rounded to human-readable values such as 5, 10, 50, 100, and 1000. Ratios usually use a 0.01 grid, while very small ratios may keep finer 0.001 or 0.0001 grids. Final row selection is best-effort diversity-aware: rows from source records that have already been selected by earlier workflows are deprioritized.

## 8. How to Read the Outputs

Main benchmark:

- `main.jsonl` contains one task per row.
- `workflow_type` tells whether the row is `clean-only`, `filter-then-clean`, or `clean-then-filter`.
- `reference_status` and `reference_text` are the deterministic GT.
- Workflow length can be recovered from the operator sequence fields in each row.
- `input_length_chars` and `input_length_bucket` support difficulty analysis without splitting the main benchmark by length.

Order-sensitivity benchmark:

- `order_sensitivity.jsonl` contains `front / middle / end` variant rows.
- Rows with the same `order_group_instance_id` belong together.
- Group success requires all three slots to be correct.

Atomic calibration:

- `atomic_ops.jsonl` is keyed by global operator.
- `source_domain` is diagnostic only.
- Use it to estimate atomic operator difficulty and compositional gaps.

Benchmark composition visualization:

If you want a quick paper-style overview of what the benchmark is made of, generate the composition plots:

```bash
PYTHONPATH=src .venv-ops/bin/python -m cdrbench.reporting.plot_benchmark_composition \
  --benchmark-dir data/benchmark \
  --workflow-library-dir data/processed/workflow_library \
  --output-dir data/paper_stats/plots
```

Outputs:

- `data/paper_stats/plots/benchmark_composition_overview.png`
- `data/paper_stats/plots/benchmark_composition_overview.pdf`
- `data/paper_stats/plots/benchmark_composition_summary.json`

The overview figure shows:

- workflow-library domain composition
- main-track domain composition
- main-track workflow-type composition
- main-variant `kept / skipped` status composition
- order-sensitive group composition by domain
- order-family `kept / skipped` status composition
- atomic-track source-domain composition
- atomic-operator `kept / skipped` status composition

## 9. Generate Model Prompts

Prompt generation is intentionally separate from GT construction, so prompt wording can be revised without rerunning Data-Juicer references.

The prompt pipeline now has two stages:

1. Build a workflow-level prompt library:
   Generate multiple prompt candidates for each style, judge them immediately, and keep only the accepted prompts in the workflow pool.
2. Build eval-ready prompt tracks:
   For each benchmark sample, deterministically sample `3` distinct styles from the accepted workflow pool.

Atomic preview first, using the merged generate+judge stage:

```bash
PYTHONPATH=src .venv-ops/bin/python -m cdrbench.prompting.generate_workflow_prompt_library \
  --benchmark-dir data/benchmark \
  --output-dir data/benchmark_prompts_atomic_preview \
  --prompt-config configs/workflow_prompting.yaml \
  --prompt-source llm \
  --tracks atomic_ops \
  --variants-per-workflow 11 \
  --candidates-per-style 3 \
  --cache-path data/benchmark_prompts_atomic_preview/workflow_prompt_library_cache.jsonl \
  --resume
```

Then build the eval-ready atomic prompt track from the accepted pool:

```bash
PYTHONPATH=src .venv-ops/bin/python -m cdrbench.prompting.build_eval_prompt_tracks \
  --benchmark-dir data/benchmark \
  --prompt-library data/benchmark_prompts_atomic_preview/workflow_prompt_library.jsonl \
  --output-dir data/benchmark_prompts_atomic_preview/eval \
  --tracks atomic_ops \
  --prompt-variants-per-sample 3 \
  --prompt-sampling-seed 0 \
  --min-prompt-variants-per-sample 3
```

If the atomic prompts look good, continue with the main and order-sensitivity tracks:

```bash
PYTHONPATH=src .venv-ops/bin/python -m cdrbench.prompting.generate_workflow_prompt_library \
  --benchmark-dir data/benchmark \
  --output-dir data/benchmark_prompts \
  --prompt-config configs/workflow_prompting.yaml \
  --prompt-source llm \
  --tracks main order_sensitivity \
  --variants-per-workflow 11 \
  --candidates-per-style 3 \
  --cache-path data/benchmark_prompts/workflow_prompt_library_cache.jsonl \
  --resume
```

```bash
PYTHONPATH=src .venv-ops/bin/python -m cdrbench.prompting.build_eval_prompt_tracks \
  --benchmark-dir data/benchmark \
  --prompt-library data/benchmark_prompts/workflow_prompt_library.jsonl \
  --output-dir data/benchmark_prompts/eval \
  --tracks main order_sensitivity \
  --prompt-variants-per-sample 3 \
  --prompt-sampling-seed 0 \
  --min-prompt-variants-per-sample 3
```

Outputs:

- `data/benchmark_prompts/workflow_prompt_library.jsonl`
- `data/benchmark_prompts/prompt_generation_summary.jsonl`
- `data/benchmark_prompts/eval/main.jsonl`
- `data/benchmark_prompts/eval/order_sensitivity.jsonl`
- `data/benchmark_prompts/eval/atomic_ops.jsonl`
- `data/benchmark_prompts/eval/prompt_eval_build_summary.jsonl`

`workflow_prompt_library.jsonl` stores the accepted workflow-level prompt pool. Each row corresponds to one unique workflow signature and includes:

- `workflow_prompt_key`
- `benchmark_track`
- `domain`
- `workflow_type`
- `order_slot`
- `operator_sequence`
- `filter_params_by_name`
- `threshold_meta`
- `requested_style_count`
- `candidates_per_style`
- `generated_candidate_count`
- `accepted_candidate_count`
- `accepted_style_count`
- `candidates`

Track files are eval-ready sample files. They keep only the benchmark fields needed for evaluation plus:

- `workflow_prompt_key`
- `prompt_candidate_pool_count`
- `prompt_variant_count`
- `prompt_sampling_policy`
- `prompt_sampling_seed`
- `prompt_variants`

Each `prompt_variants` entry is lightweight and contains only:

- `style_id`
- `style_label`
- `user_requirement`

This means the workflow library keeps only judge-passed prompts, while each benchmark sample keeps a fixed deterministic subset of `3` styles for direct evaluation.

The actual model prompt should be assembled by evaluation code:

```text
{{LLM-generated user requirement}}

{{fixed CDR-Bench output contract}}

Raw input text:
<<<CDR_INPUT
{{input_text}}
CDR_INPUT>>>
```

This means natural-language diversity comes from the requirement body, while the output protocol remains fixed across all styles.

The default generation flow hides operator names and parameter names from the user-facing requirement, while still preserving:

- `data/processed/workflow_library/<domain>/workflow_library.yaml`
- `configs/workflow_prompting.yaml`
- `data/benchmark/*.jsonl`

The generated requirement candidates should preserve:

- the user-facing refinement goal
- the required order when order matters
- natural threshold semantics when needed
- stylistic diversity across different users

Current style pool includes imperative checklist, goal-oriented description, application-context task, quality-control request, analyst handoff, concise brief, policy-like requirement, workflow narrative, end-weighted instruction, negative-constraint driven, and conversational cooperative styles.

By default, prompt generation skips workflows containing `flagged_words_filter` and `stopwords_filter`.

Notes:

- Prompt generation is grouped by workflow signature rather than by individual sample, so all samples sharing the same workflow reuse the same accepted prompt pool.
- `--variants-per-workflow` controls how many style presets to request for each workflow.
- `--candidates-per-style` controls how many prompt candidates to generate for each style before judging.
- `generate_workflow_prompt_library.py` runs generation and judging in one pass, and stores only the accepted prompts in `workflow_prompt_library.jsonl`.
- `build_eval_prompt_tracks.py` deterministically samples distinct styles from the accepted workflow pool for each benchmark sample.
- `--prompt-sampling-seed` fixes the deterministic sample of styles used for each benchmark sample, so repeated evaluations stay reproducible.
- `--min-prompt-variants-per-sample 3` ensures the final eval tracks only keep samples whose workflow prompt pool can supply at least `3` distinct accepted styles.
- `--resume` reuses the workflow-level cache at `--cache-path`, so an interrupted workflow-library run can continue without re-calling the LLM for finished workflows.
- clarity and format consistency

## 10. Troubleshooting Data-Juicer Imports

If tagging fails with errors such as:

- `No module named 'data_juicer.core.data'`
- the same Python command worked before but now fails

Run:

```bash
PYTHONPATH=src python -m cdrbench.debug_tools.debug_data_juicer_env
```

This checks:

- current Python executable and version
- whether the system `data_juicer` package is shadowing the repo-local checkout
- whether vendored `data-juicer/` imports correctly
- whether `process_data.py` and `analyze_data.py` import cleanly

Paste the full output when debugging the server environment.
