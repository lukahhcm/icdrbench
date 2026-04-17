# ICDR-Bench

这份仓库用于下载 ICDR-Bench 的已整理 JSONL 数据，并用 Data-Juicer CLI 做打标、domain 数据池构建和 workflow 挖掘。

## 一条主线

如果你只想知道“接下来该做什么”，就按这四步：

1. 下载 raw JSONL 到 `data/raw/`
2. 用 `tag_and_assign_domains.py` 跑 Data-Juicer CLI 打标
3. 用 `mine_domain_workflows.py` 从 `domain_tags` 里挖 workflow families 和 concrete workflow candidates
4. 查看 `outputs/workflow_mining/<domain>/workflow_candidates.yaml`，人工挑每个 domain 的 workflow

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

这一步跑完以后，你已经有了两类中间结果：

- `domain_tags/*.jsonl`
  每条样本的 per-op tagging 结果，后面 workflow 挖掘吃这个
- `domain_filtered/*.jsonl`
  已经按 `assigned_domain` 留下来的样本，后面可作为每个 domain 的候选数据池

## 5. 挖掘每个 domain 的 workflow

```bash
.venv-ops/bin/python scripts/prepare_data/mine_domain_workflows.py \
  --tagged-dir data/processed/domain_tags \
  --output-dir outputs/workflow_mining
```

默认规则里，某个具体 workflow 至少要有 `5` 条样本支持，才会被保留为有效 workflow candidate。这个阈值可以用 `--min-workflow-support` 调整。

输出文件：

- `outputs/workflow_mining/<domain>/workflow_families.csv`
- `outputs/workflow_mining/<domain>/selected_workflows.csv`
- `outputs/workflow_mining/<domain>/workflow_candidates.yaml`
- `outputs/workflow_mining/domain_workflow_mining_summary.csv`

## 6. 怎么看 workflow 结果

建议按这个顺序看：

1. 先看总览

```bash
column -s, -t < outputs/workflow_mining/domain_workflow_mining_summary.csv | less -S
```

这个文件告诉你每个 domain 挖出了多少 family、多少 candidate workflow。

2. 再看某个 domain 的 family

```bash
column -s, -t < outputs/workflow_mining/web/workflow_families.csv | less -S
```

这里看的是：

- 这个 domain 有几个 workflow family
- 每个 family 的 anchor operator set 是什么
- family support 大概有多高

3. 最后看某个 domain 的具体 workflow 候选

```bash
column -s, -t < outputs/workflow_mining/web/selected_workflows.csv | less -S
sed -n '1,160p' outputs/workflow_mining/web/workflow_candidates.yaml
```

这里最重要的是：

- `selected_workflows.csv`
  方便快速扫 operator 组合、长度、support
- `workflow_candidates.yaml`
  更适合人工整理成最终 workflow library

注意：这里看到的还是 `operator-set based workflow candidates`，不是最终已经定好顺序的 benchmark workflow。下一步通常是：

1. 从每个 domain 的 `workflow_candidates.yaml` 里挑 family 和 concrete workflow candidates
2. 补 operator 顺序
3. 补 activation spec
4. 再做 workflow-level executor validation

## 7. 最终你会拿到什么

如果整条流程跑完，当前 repo 里最重要的产物是：

- 每个 domain 的候选数据：
  - `data/processed/domain_filtered/*.jsonl`
  - `data/processed/domain_filtered/all.jsonl`
- 每个 domain 的 workflow 候选：
  - `outputs/workflow_mining/<domain>/workflow_candidates.yaml`
  - `outputs/workflow_mining/<domain>/selected_workflows.csv`
- 底层追踪信息：
  - `data/processed/domain_tags/*.jsonl`
  - `outputs/dj_cli_tagging/`

## 常用补充

只跑部分语料：

```bash
.venv-ops/bin/python scripts/prepare_data/tag_and_assign_domains.py --corpora arxiv pii --resume
```

如果要单独跑 per-op CLI probe：

```bash
.venv-ops/bin/python scripts/prepare_data/run_dj_per_op_probe.py --execute --resume
```
