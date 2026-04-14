# DJBench Bootstrap Workspace

This directory is the new development workspace for the benchmark.

## Current workflow
- raw corpus conversion scripts for the local datasets
- domain operator tagging using official Data-Juicer operator source files
- per-record domain assignment and filtering based on active mapper evidence

## Quick start
```bash
cd /Users/tarak30/Downloads/djbench/djbench
PYTHONPATH=src python3 scripts/prepare_data/convert_raw_corpus.py arxiv
PYTHONPATH=src python3 scripts/prepare_data/tag_and_assign_domains.py
```

For a quick pilot before a full run, add `--max-records 200` or another sample size:
```bash
PYTHONPATH=src python3 scripts/prepare_data/tag_and_assign_domains.py --max-records 200
```

This writes:
- `data/processed/domain_tags/*.jsonl`: per-record operator tags and candidate domains
- `data/processed/domain_filtered/*.jsonl`: records kept after domain assignment and mapper-based filtering
- `data/processed/domain_filtered/all.jsonl`: merged kept records across corpora
- `outputs/domain_operator_catalog.csv`: which operators belong to which benchmark domains
- `outputs/domain_labeling_summary.csv`: corpus-level keep/drop summary
- `outputs/domain_assignment_counts.csv`: assigned domain counts after tagging

## Configs
- `configs/corpora.yaml`: the local raw corpora
- `configs/domains.yaml`: the four benchmark domains and their operator sets

## Notes
- Only four benchmark domains are used for assignment: `web`, `kb_support`, `reports_policy`, `scientific`.
- A record is kept only if at least two mapper operators actually change the text.
- Domain assignment is rule-based: domain-unique operator hits are prioritized over shared cleanup operators.
