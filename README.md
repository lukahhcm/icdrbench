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
.venv-ops/bin/python scripts/release/download_hf_jsonl.py \
  --repo-id lukahh/cdrbench-raw \
  --repo-root .
```

The manifest downloads JSONL files into `data/raw/`, including arXiv, Common Crawl, Wikipedia/help-style text, government reports, and PII corpora.

## 4. Tag Operators and Assign Domains

Small smoke test:

```bash
.venv-ops/bin/python scripts/prepare_data/tag_and_assign_domains.py --max-records 200
```

Full resumable run:

```bash
.venv-ops/bin/python scripts/prepare_data/tag_and_assign_domains.py --resume
```

Useful overrides:

```bash
.venv-ops/bin/python scripts/prepare_data/tag_and_assign_domains.py \
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
.venv-ops/bin/python scripts/prepare_data/mine_domain_workflows.py \
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
.venv-ops/bin/python scripts/prepare_data/materialize_domain_workflows.py \
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
.venv-ops/bin/python scripts/prepare_data/materialize_benchmark_instances.py \
  --workflow-library-dir data/processed/workflow_library \
  --filtered-path data/processed/domain_filtered/all.jsonl \
  --output-dir data/benchmark \
  --target-drop-rate 0.5 \
  --max-candidate-records 2000 \
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
- `--max-candidate-records 2000`: scan at most 2000 candidate samples per workflow variant or order family
- `--max-instances-per-variant 50`: cap main-track instances per workflow variant
- `--max-order-groups-per-family 30`: cap order groups per order family
- `--max-atomic-candidate-records 1000`: scan at most 1000 candidates per atomic operator
- `--max-atomic-instances-per-op 10`: cap atomic instances per operator
- `--min-keep / --min-drop`: skip main filter variants without enough keep/drop candidates
- `--min-order-sensitive-groups`: skip order families without enough genuinely order-sensitive groups
- `--min-atomic-keep / --min-atomic-drop`: skip atomic filters without enough keep/drop candidates
- `--resume`: reuse per-variant cache shards in `data/benchmark/_materialize_cache/` after an interrupted run

Thresholds are recalibrated during materialization. Length/count thresholds are rounded to human-readable values such as 5, 10, 50, 100, and 1000. Ratios usually use a 0.01 grid, while very small ratios may keep finer 0.001 or 0.0001 grids.

## 8. How to Read the Outputs

Main benchmark:

- `main.jsonl` contains one task per row.
- `workflow_type` tells whether the row is `clean-only`, `filter-then-clean`, or `clean-then-filter`.
- `reference_status` and `reference_clean_text` are the deterministic GT.
- Workflow length can be recovered from the operator sequence fields in each row.

Order-sensitivity benchmark:

- `order_sensitivity.jsonl` contains `front / middle / end` variant rows.
- Rows with the same `order_group_id` belong together.
- Group success requires all three slots to be correct.

Atomic calibration:

- `atomic_ops.jsonl` is keyed by global operator.
- `source_domain` is diagnostic only.
- Use it to estimate atomic operator difficulty and compositional gaps.

## 9. Prompt Generation

Prompt generation is intentionally separate from GT construction.

The final model prompt should be a natural-language user request, not an operator list. Use these files as metadata sources:

- `data/processed/workflow_library/<domain>/workflow_library.yaml`
- `configs/workflow_prompting.yaml`
- `data/benchmark/*.jsonl`

The prompt should state:

- the user-facing refinement goal
- the required order when order matters
- the output contract: `status` and `clean_text`

## 10. Troubleshooting Data-Juicer Imports

If tagging fails with errors such as:

- `No module named 'data_juicer.core.data'`
- the same Python command worked before but now fails

Run:

```bash
python scripts/debug/debug_data_juicer_env.py
```

This checks:

- current Python executable and version
- whether the system `data_juicer` package is shadowing the repo-local checkout
- whether vendored `data-juicer/` imports correctly
- whether `process_data.py` and `analyze_data.py` import cleanly

Paste the full output when debugging the server environment.
