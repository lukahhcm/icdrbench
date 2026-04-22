# ICDR-Bench

这份仓库用于下载 ICDR-Bench 的已整理 JSONL 数据，并用 Data-Juicer CLI 做打标、domain 数据池构建和 workflow 挖掘。

## 一条主线

如果你只想知道“接下来该做什么”，就按这四步：

1. 下载 raw JSONL 到 `data/raw/`
2. 用 `tag_and_assign_domains.py` 跑 Data-Juicer CLI 打标
3. 用 `mine_domain_workflows.py` 从 `domain_tags` 里挖 workflow families 和 concrete workflow candidates
4. 用 `materialize_domain_workflows.py` 把 mapper skeleton 落成主榜和顺序敏感拓展两版 workflow library
5. 查看 `data/processed/workflow_library/<domain>/workflow_library.yaml`，再做最终人工筛选

## 1. 拉代码

```bash
git clone https://github.com/lukahhcm/icdrbench.git
cd icdrbench
```

## 2. 准备环境

如果服务器还没装 `uv`，先安装：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

这个仓库已经自带 `data-juicer/`，里面包含 ICDR-Bench 当前用到的定制算子，所以默认不需要你再单独 clone 一份 DJ repo。

推荐再用 `uv` 建一个单独环境，后面 HF 下载和聚合脚本都用这个环境跑：

```bash
uv venv .venv-ops --python 3.11
uv pip install --python .venv-ops/bin/python -e .
uv pip install --python .venv-ops/bin/python -U huggingface_hub py-data-juicer
```

主打标脚本默认会优先走仓库里的：

```bash
python data-juicer/tools/process_data.py
python data-juicer/tools/analyze_data.py
```

并自动把 `./data-juicer` 注入到 `PYTHONPATH`。只有在仓库里没有这份目录时，才会退回系统里的 `dj-process` / `dj-analyze`。

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

因为仓库已经自带 `./data-juicer`，最短就是直接这样跑。

先小规模试跑：

```bash
.venv-ops/bin/python scripts/prepare_data/tag_and_assign_domains.py --max-records 200
```

正式继续跑：

```bash
.venv-ops/bin/python scripts/prepare_data/tag_and_assign_domains.py --resume
```

如果你想强制改用系统里的 `dj-process` / `dj-analyze`，可以这样传：

```bash
.venv-ops/bin/python scripts/prepare_data/tag_and_assign_domains.py \
  --dj-process-bin /path/to/dj-process \
  --dj-analyze-bin /path/to/dj-analyze \
  --resume
```

如果你以后把 `data-juicer` 挪到别处，也可以显式指定 repo 路径和 Python：

```bash
.venv-ops/bin/python scripts/prepare_data/tag_and_assign_domains.py \
  --dj-repo-root /path/to/data-juicer \
  --dj-python /usr/bin/python3 \
  --resume
```

输出文件：

- `data/processed/domain_tags/*.jsonl`
- `data/processed/domain_filtered/*.jsonl`
- `data/processed/domain_filtered/all.jsonl`
- `data/processed/domain_operator_catalog.csv`
- `data/processed/domain_labeling_summary.csv`
- `data/processed/domain_assignment_counts.csv`
- `data/processed/dj_cli_tagging/`
  - Data-Juicer YAML、per-op CLI 输出、logs

这一步跑完以后，你已经有了两类中间结果：

- `domain_tags/*.jsonl`
  每条样本的 per-op tagging 结果，后面 workflow 挖掘吃这个
- `domain_filtered/*.jsonl`
  已经按 `assigned_domain` 留下来的样本，后面可作为每个 domain 的候选数据池

补充说明：
- 大多数 text-cleaning mapper 会走 `dj-process`
- 如果某个 mapper 在 benchmark 里最终要按 `text` 对齐，就优先配置成文本输出模式
- 例如 `extract_tables_from_html_mapper` 在当前 ICDR-Bench 配置里会把表格抽成 TSV 文本写回 `text`，而不是只写嵌套 `meta.html_tables`
- `latex_figure_context_extractor_mapper` 在当前 ICDR-Bench 配置里也走一对一 `text` 模式：会把一篇论文里的 figure / subfigure 信息合并成规范化文本块写回 `text`
- 只有明确走 meta tagging 的 mapper，才会走 `dj-analyze`

## 5. 挖掘每个 domain 的 workflow

```bash
.venv-ops/bin/python scripts/prepare_data/mine_domain_workflows.py \
  --tagged-dir data/processed/domain_tags \
  --output-dir data/processed/workflow_mining
```

默认规则里，某个具体 workflow 至少要有 `5` 条样本支持，才会被保留为有效 workflow candidate。这个阈值可以用 `--min-workflow-support` 调整。

输出文件：

- `data/processed/workflow_mining/<domain>/workflow_families.csv`
- `data/processed/workflow_mining/<domain>/selected_workflows.csv`
- `data/processed/workflow_mining/<domain>/workflow_candidates.yaml`
- `data/processed/workflow_mining/domain_workflow_mining_summary.csv`

## 6. 怎么看 workflow 结果

建议按这个顺序看：

1. 先看总览

```bash
column -s, -t < data/processed/workflow_mining/domain_workflow_mining_summary.csv | less -S
```

这个文件告诉你每个 domain 挖出了多少 family、多少 candidate workflow。

2. 再看某个 domain 的 family

```bash
column -s, -t < data/processed/workflow_mining/web/workflow_families.csv | less -S
```

这里看的是：

- 这个 domain 有几个 workflow family
- 每个 family 的 anchor operator set 是什么
- family support 大概有多高

3. 最后看某个 domain 的具体 workflow 候选

```bash
column -s, -t < data/processed/workflow_mining/web/selected_workflows.csv | less -S
sed -n '1,160p' data/processed/workflow_mining/web/workflow_candidates.yaml
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

## 7. 直接产出 workflow library

如果你不想停在 mapper operator set，而是想直接拿到一版可用 workflow 草案，继续跑：

```bash
.venv-ops/bin/python scripts/prepare_data/materialize_domain_workflows.py \
  --workflow-mining-dir data/processed/workflow_mining \
  --filtered-path data/processed/domain_filtered/all.jsonl \
  --output-dir data/processed/workflow_library \
  --resume
```

`--resume` 会按 domain 跳过已经完整生成的结果；如果某个 domain 的输出缺文件或 yaml 读不出来，会自动重新跑这个 domain。运行时脚本会打印当前 domain、workflow 进度，以及每个 workflow 产出的 checkpoint stats / 主榜 variants / order families 数量。

这一步会做三件事：

1. 按 domain 配置里的 operator 顺序，把 mapper set 排成一条确定的 mapper sequence
2. 在支持这些 mapper 的真实样本上重放 mapper prefixes，形成 `S0, S1, ..., Sfinal` 中间状态
3. 对每个 checkpoint 扫描 filter status 分布，并产出主榜和顺序敏感拓展实验的 workflow variants

这里的 filter 不直接相信 DJ 默认阈值。脚本会先记录每个 checkpoint 的统计量分布，再用一个简单的分位数规则生成 provisional calibrated params：

- `min` 型 filter 默认用 `p20` 作为下界
- `max` 型 filter 默认用 `p80` 作为上界
- 完整分布会保存在 `checkpoint_filter_stats.csv`，后续可以人工或自动重新调阈值

输出文件：

- `data/processed/workflow_library/<domain>/workflow_library.yaml`
- `data/processed/workflow_library/<domain>/workflow_variants.csv`
- `data/processed/workflow_library/<domain>/filter_attachments.csv`
- `data/processed/workflow_library/<domain>/checkpoint_filter_stats.csv`
- `data/processed/workflow_library/<domain>/order_sensitivity_families.csv`
- `data/processed/workflow_library/<domain>/order_sensitivity_candidates.csv`
- `data/processed/workflow_library/workflow_library_summary.csv`

其中最重要的是：

- `workflow_library.yaml`
  里面会同时给出：
  - `ordered_clean_sequence`
  - `main_workflow_variants`
  - `order_sensitivity_families`
  - `order_sensitivity_variants`
  - `selected_filter_attachments`

主榜目前固定分成三类：

- `clean-only`
- `filter-then-clean`
- `clean-then-filter`

顺序敏感拓展实验单独放在：

- `order_sensitivity_families.csv`
- `order_sensitivity_candidates.csv`

这里不是把中间插入的 workflow 混进主榜，而是单独构造成组的次榜：

- 同一个 `workflow_id + filter_name` 形成一个 `order_family`
- 每个 `order_family` 必须同时包含 `front / middle / end` 三个 slot
- `front` 对应 `filter-then-clean`
- `middle` 对应 `clean-filter-clean`
- `end` 对应 `clean-then-filter`
- 组级 metric 要求三个 slot 都做对，才算这个 `order_family` 成功

可以把这一步理解成：

- `workflow_mining` 产出 mapper skeleton
- `workflow_library` 产出可用于主榜和拓展实验的完整 workflow 草案

如果想快速看每个 domain 产出了多少 workflow：

```bash
column -s, -t < data/processed/workflow_library/workflow_library_summary.csv | less -S
```

如果想看某个 workflow 的 filter 为什么被选中，优先看：

- `checkpoint_filter_stats.csv`

这里会显示每个 filter 在 `S0/S1/.../Sfinal` 的 `mean/p20/p50/p80` 以及相邻 checkpoint 的变化量。

## 8. 生成 benchmark 样本和 GT

`workflow_library` 只是 workflow 草案。要生成真正可评测的数据和 Data-Juicer reference，继续跑：

```bash
.venv-ops/bin/python scripts/prepare_data/materialize_benchmark_instances.py \
  --workflow-library-dir data/processed/workflow_library \
  --filtered-path data/processed/domain_filtered/all.jsonl \
  --output-dir data/benchmark \
  --target-drop-rate 0.5 \
  --max-atomic-instances-per-op 20
```

这一步暂时不生成 prompt，只做样本选择和 GT：

- 主榜写到 `data/benchmark/main.jsonl`
- 顺序敏感次榜写到 `data/benchmark/order_sensitivity.jsonl`
- 单算子 atomic 集写到 `data/benchmark/atomic_ops.jsonl`
- 主榜 summary 写到 `data/benchmark/main_summary.csv`
- 次榜 summary 写到 `data/benchmark/order_sensitivity_summary.csv`
- 单算子 summary 写到 `data/benchmark/atomic_ops_summary.csv`

主榜带 filter 的 workflow 会重新按目标 drop rate 校准阈值，然后尽量均衡抽取 KEEP/DROP 样本。次榜按 `order_family` 校准共享阈值，同一个输入同时跑 `front / middle / end`，只保留至少两个 slot 的 reference 不同的样本。

单算子 atomic 集用于后续估计 `Operator Atomic Difficulty`：mapper 只保留输出确实变化的样本，filter 会按同样的 `target-drop-rate` 校准阈值并尽量均衡 KEEP/DROP。若暂时不想生成 atomic 集，可以加 `--skip-atomic`。

校准出来的阈值会被转成更像真实需求里的粗粒度数值：长度/数量阈值会落到 5、10、50、100、1000 等可读档位；ratio 通常落到 0.01 的网格，但极小比例会保留到 0.001 或 0.0001。summary 里仍会保留 `threshold_raw_value` 方便 debug。

如果你后面要把 workflow 自动转成自然语言指令，可以直接参考：

- `configs/workflow_prompting.yaml`

这个文件现在已经记录了：

- `extract_tables_from_html_mapper` 的自然语言意图
- 它在 benchmark 里的文本输出约定
- TSV 序列化规范
- 后续 `workflow -> prompt` 代码可以直接复用的 prompt template

## 9. 最终你会拿到什么

如果整条流程跑完，当前 repo 里最重要的产物是：

- 每个 domain 的候选数据：
  - `data/processed/domain_filtered/*.jsonl`
  - `data/processed/domain_filtered/all.jsonl`
- 每个 domain 的 workflow 候选：
  - `data/processed/workflow_mining/<domain>/workflow_candidates.yaml`
  - `data/processed/workflow_mining/<domain>/selected_workflows.csv`
- 底层追踪信息：
  - `data/processed/domain_tags/*.jsonl`
  - `data/processed/dj_cli_tagging/`
- benchmark 样本和 GT：
  - `data/benchmark/main.jsonl`
  - `data/benchmark/order_sensitivity.jsonl`
  - `data/benchmark/atomic_ops.jsonl`
  - `data/benchmark/main_summary.csv`
  - `data/benchmark/order_sensitivity_summary.csv`
  - `data/benchmark/atomic_ops_summary.csv`

## 常用补充

只跑部分语料：

```bash
.venv-ops/bin/python scripts/prepare_data/tag_and_assign_domains.py --corpora arxiv pii --resume
```

如果要单独跑 per-op CLI probe：

```bash
.venv-ops/bin/python scripts/prepare_data/run_dj_per_op_probe.py --execute --resume
```

## DJ 排错

如果服务器上跑打标时出现这类错误：

- `No module named 'data_juicer.core.data'`
- 同样的 `python` 命令之前能跑、现在不能跑

先不要手动猜环境。直接在仓库根目录运行：

```bash
python scripts/debug/debug_data_juicer_env.py
```

这个脚本会检查：

- 当前 `python` 可执行文件和版本
- 是否先加载到了系统安装的 `data_juicer`
- 仓库内 vendored 的 `data-juicer/` 能不能正常 import
- `data-juicer/tools/process_data.py` 和 `analyze_data.py` 在 import 阶段会不会报错

把完整输出贴回来，就能更快定位是：

- 还在误用系统里的旧 DJ
- vendored DJ 没被正确加载
- 还是当前 Python 环境和 vendored DJ 本身不兼容
