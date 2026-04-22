# External Benchmark Data Status

This note tracks which external benchmark assets have already been brought into the CDR-Bench workspace, and which ones were inspected but not yet integrated.

## Integrated Now

### External PII benchmarks

Integrated source directories:
- `data/raw/external_benchmarks/pii/docpii-redaction-benchmark`
- `data/raw/external_benchmarks/pii/synthetic-text-anonymizer-dataset-v1`

Converted with:

```bash
PYTHONPATH=src python3 scripts/prepare_data/convert_raw_corpus.py pii_docpii
PYTHONPATH=src python3 scripts/prepare_data/convert_raw_corpus.py pii_synthetic
```

Output raw corpora:
- `data/raw/pii/docpii-contextual-1k.jsonl`
- `data/raw/pii/synthetic-anonymizer-8k.jsonl`

Added to `configs/corpora.yaml` as:
- `pii_docpii`
- `pii_synthetic`

Why they fit:
- Both are text-first PII redaction/NER-style corpora
- Both can be normalized into the same raw JSONL schema used by the current tagging pipeline

## Investigated But Not Integrated Yet

### DataGovBench / GovBench-150

Confirmed from the paper (`arXiv:2512.04416`):
- The benchmark is constructed from 30 curated tables sourced from `Statista (2025)`
- The benchmark contains 150 tasks
- Each task includes a natural language description, raw datasets, and an executable scoring script
- Tasks are split into operator-level and DAG-level settings
- The split is `100` operator-level tasks and `50` DAG-level tasks
- The operator-level tasks are organized around six structured data governance scenarios:
  - Filtering
  - Refinement
  - Imputation
  - Deduplication & Consistency
  - Data Integration
  - Classification & Labeling

Why this matters for CDR-Bench:
- This is a valuable related source of task design ideas
- But the underlying data shape is still table-centric / structured-data-centric rather than text-first
- So it does not drop directly into the current Data-Juicer text-operator pipeline in its current form
- In method comparison terms, it is closer to AutoDCWorkflow-style structured-data governance than to CDR-Bench's text-first compositional data-refinement setting

Current blocker:
- I did not find a public benchmark repo or dataset release containing the 150 tasks themselves
- The paper HTML does not expose a public download link
- The `GovBench-AI` GitHub organization appears public, but I only found a `Hackathon` repo rather than the benchmark data release

Conclusion:
- DataGovBench is conceptually very relevant
- But there is no confirmed public data drop in this workspace yet, so nothing has been integrated into `configs/corpora.yaml`

### AutoDCWorkflow

Why not integrated:
- Its public task framing is structured-table workflow generation for OpenRefine
- That does not match the current CDR-Bench text-first Data-Juicer operator pipeline

Conclusion:
- Useful as a related benchmark in the paper
- Not currently a compatible raw corpus source for CDR-Bench

## Explicitly Dropped

### DCA-Bench

Reason dropped:
- It is more about issue discovery / auditing on dataset platforms than text curation execution
- The task/data shape does not match the current benchmark direction closely enough

## Next Best External Data Direction

If we want more external assets that fit the current pipeline, the most promising next steps are:

1. More text-first privacy/sanitization corpora with document-level context
2. Public text-centric curation or sanitization datasets with before/after pairs
3. Any future public release of GovBench-150 task files
