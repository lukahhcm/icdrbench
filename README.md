# ICDR-Bench

这份仓库用于下载 ICDR-Bench 的已整理 JSONL 数据，并用 Data-Juicer CLI 做打标与 workflow 挖掘。

## 1. 拉代码

```bash
git clone https://github.com/lukahhcm/icdrbench.git
cd icdrbench
```

## 2. 准备 Data-Juicer CLI 环境

如果服务器还没装 `uv`，先安装：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

如果你服务器上已经有能直接运行的 `dj-process` / `dj-analyze` 环境，这一步可以跳过。否则可以单独拉一份 `data-juicer`：

```bash
git clone https://github.com/datajuicer/data-juicer.git /path/to/data-juicer
export ICDRBENCH_DATA_JUICER_ROOT=/path/to/data-juicer
```

推荐再用 `uv` 建一个单独环境，后面 HF 下载和聚合脚本都用这个环境跑：

```bash
uv venv .venv-ops --python 3.11
uv pip install --python .venv-ops/bin/python -e .
uv pip install --python .venv-ops/bin/python -U huggingface_hub py-data-juicer
```

如果 `data-juicer` 就放在仓库根目录下的 `./data-juicer`，可以不设 `ICDRBENCH_DATA_JUICER_ROOT`。主打标脚本实际调用的是 `dj-process` 和 `dj-analyze`。

这些脚本现在会自动把仓库里的 `src/` 加到 `sys.path`，所以不需要额外设置 `PYTHONPATH=src`。

## 3. 下载数据

从 Hugging Face 下载当前 manifest 里的 JSONL：

```bash
HF_TOKEN=<your_hf_token_if_needed> \
.venv-ops/bin/python scripts/release/download_hf_jsonl.py \
  --repo-id lukahh/icdrbench-raw \
  --repo-root .
```

默认会下载这些文件到 `data/raw/`：

- `arxiv/arxiv-4k.jsonl`
- `commoncrawl/cc-10k.jsonl`
- `enwiki/enwiki-pages-110k.jsonl`
- `govreport/govreport-20k.jsonl`
- `pii/pii-43k.jsonl`
- `pii/docpii-contextual-1k.jsonl`
- `pii/synthetic-anonymizer-8k.jsonl`

## 4. 用 Data-Juicer CLI 打标

先小规模试跑：

```bash
.venv-ops/bin/python scripts/prepare_data/tag_and_assign_domains.py --max-records 200
```

正式继续跑：

```bash
.venv-ops/bin/python scripts/prepare_data/tag_and_assign_domains.py --resume
```

如果你的环境里 `dj-process` / `dj-analyze` 不在默认 PATH，可以显式传：

```bash
.venv-ops/bin/python scripts/prepare_data/tag_and_assign_domains.py \
  --dj-process-bin /path/to/dj-process \
  --dj-analyze-bin /path/to/dj-analyze \
  --resume
```

输出文件：

- `data/processed/domain_tags/*.jsonl`
- `data/processed/domain_filtered/*.jsonl`
- `data/processed/domain_filtered/all.jsonl`
- `outputs/domain_operator_catalog.csv`
- `outputs/domain_labeling_summary.csv`
- `outputs/domain_assignment_counts.csv`
- `outputs/dj_cli_tagging/`
  - Data-Juicer YAML、per-op CLI 输出、logs

## 5. 挖掘每个 domain 的 workflow

```bash
.venv-ops/bin/python scripts/prepare_data/mine_domain_workflows.py \
  --tagged-dir data/processed/domain_tags \
  --output-dir outputs/workflow_mining
```

输出文件：

- `outputs/workflow_mining/<domain>/workflow_families.csv`
- `outputs/workflow_mining/<domain>/selected_workflows.csv`
- `outputs/workflow_mining/<domain>/workflow_candidates.yaml`

## 常用补充

只跑部分语料：

```bash
.venv-ops/bin/python scripts/prepare_data/tag_and_assign_domains.py --corpora arxiv pii --resume
```

如果要单独跑 per-op CLI probe：

```bash
.venv-ops/bin/python scripts/prepare_data/run_dj_per_op_probe.py --execute --resume
```
