# ICDR-Bench Research Plan v3

## 1. 核心问题

ICDR-Bench 希望回答两个互补问题：

1. **Solo**: 当用户用自然语言完整描述一个数据治理流程时，foundation models 能否在不给工具、不给显式算子名的前提下，直接把原始非结构化数据处理成正确的最终结果？
2. **Interactive**: 当用户只给出高层数据治理目标、保留标准和清洗约束尚未说全时，foundation models 能否通过多轮澄清把需求补全，并继续把原始非结构化数据处理成正确的最终结果？

这里的“最终结果”统一表示为：

- `status`: `KEEP` 或 `DROP`
- `clean_text`: 若 `KEEP`，则为处理后的最终文本；若 `DROP`，则为空

ICDR-Bench 的核心任务不是 workflow generation（可以考虑让其显示输出workflow作为验证），也不是 tool-calling / operator selection（并非调用工具），而是 **compositional data refinement through natural-language specification and execution**。

---

## 2. 为什么这个问题值得单独做 benchmark

数据治理是 foundation-model pipeline 的前置环节，典型场景包括：

- 公开网页语料清洗与过滤
- 企业知识库 / 帮助中心 / FAQ 语料准备
- 政策、财报、合规类长文档清洗
- 学术源码清理与规范化

真实使用中，用户越来越可能直接描述目标，而不是手写规则或 pipeline，例如：

- “把页面里的 HTML、链接和联系方式去掉，统一空白，再判断是否值得保留。”
- “把帮助文档里的链接、版权头、模板残留和重复句去掉，再判断是否值得入库。”
- “把报告里的免责声明、表格残留和异常长行清掉，再判断是否适合进入检索库。”
- “去掉 LaTeX 注释和 bibliography，展开宏并规范化，再判断这篇 source 是否值得保留。”

这些任务有三个关键特征：

1. **多步组合**：通常由多个 mapper/filter 组成；
2. **顺序依赖**：步骤顺序变化会改变最终结果；
3. **步骤激活要求**：workflow 中的关键步骤必须真的改变最终结果，而不是可有可无的 no-op。

ICDR-Bench 的目标是把这类能力从“demo”变成一个可以系统比较、可复现、可分析的 benchmark。

---

## 3. ICDR-Bench 的任务定义

### 3.1 输入

每个样本都包含：

- 一个原始非结构化样本 `x`
- 一个隐藏的 deterministic Data-Juicer recipe `r`

不同 track 的可见输入不同：

- `Solo`: 给模型一条完整自然语言 workflow specification `w`；
- `Interactive`: 只给模型一个高层 refinement request `q`，模型可在限定轮数内提出澄清问题，最后恢复出足以执行的隐式 workflow。

### 3.2 输出

模型需要直接输出：

- `DROP`
- 或 `KEEP + clean_text`

### 3.3 隐藏 reference

benchmark 使用隐藏 recipe `r` 执行得到 reference `status + clean_text`。

### 3.4 主指标

主指标为 **Workflow Success**：

- 若 reference 为 `DROP`，模型必须正确输出 `DROP`；
- 若 reference 为 `KEEP`，模型必须输出与 reference 一致的最终 `clean_text`；
- filter 错误是一票否决。

---

## 4. ICDR-Bench Position

为了避免与现有工作混淆，ICDR-Bench 的定位需要明确为：

**a benchmark for solo and interactive compositional data refinement on unstructured corpora, with deterministic operator-backed clean-text verification.**

ICDR-Bench 的核心评测包含两条主线：

- `Solo`: 给定完整自然语言 workflow，直接执行，输出 `KEEP + clean_text` / `DROP`
- `Interactive`: 给定不完全 specification，先多轮澄清，再输出 `KEEP + clean_text` / `DROP`
- 两条主线都通过 deterministic reference 做 clean-text 级验证

ICDR-Bench **不**评：

- workflow generation
- transformation code generation 作为主任务
- issue discovery
- document parsing / OCR understanding
- SQL / BI / analytics agent

### 4.1 与最相关 benchmark 的区别

这张表的目的不是证明 “ICDR-Bench 什么都做”，而是让读者一眼看清：**现有 benchmark 各自覆盖了哪一块，而 ICDR-Bench 首次把“非结构化文本 + compositional refinement + 程序化验证 + 顺序敏感”放在了一起**。

`✓` = 明确覆盖；`△` = 部分覆盖 / 不是主线；`✗` = 不覆盖

| Benchmark | 非结构化文本 / 文档 | 不依赖外部工具 / 代码生成 | 程序化可验证 | 不依赖 LLM judge / 高成本人工审查 | 显式评测顺序敏感性 |
| --------- | ------------------- | --------------------------- | ------------ | --------------------------------- | -------------------- |
| **AutoDCWorkflow** | ✗ | ✗ | ✓ | ✓ | ✗ |
| **DCA-Bench** | ✗ | ✓ | ✗ | ✗ | ✗ |
| **DataGovBench** | ✗ | ✗ | △ | △ | ✗ |
| **IDA-Bench** | ✗ | ✗ | ✓ | ✓ | ✗ |
| **ICDR-Bench (ours)** | **✓** | **✓** | **✓** | **✓** | **✓** |

`OmniDocBench` 更适合作为旁系相关工作：它评的是 document parsing、layout、table/formula/OCR 等文档理解能力，不是 parsing 之后的 text curation workflow execution，因此不放在主对比表里。

`IDA-Bench`（arXiv 2505.18223）是一个很值得正面回应的近邻 benchmark：它强调 **multi-round interactive guided data analysis**，核心设定是 simulated user 逐轮给出指导，agent 在 Python sandbox 中做 CSV / tabular predictive modeling，并用最终数值结果是否达到 human baseline 来自动评测。它与 ICDR-Bench 的重叠点在于都不是纯主观 LLM-judge 任务；但二者的主任务仍然明显不同，前者强调交互式分析与 evolving guidance，后者强调围绕 **compositional data refinement** 的两类能力：fully specified workflow 下的直接执行，以及 under-specified refinement request 下的 requirement clarification + execution。

如果后续需要分析模型“理解成了什么 workflow”，可以把 workflow/code elicitation 作为**辅助诊断维度**，与隐藏的 Data-Juicer recipe 做 alignment 对比；但 ICDR-Bench 的核心定义仍然应是 end-to-end execution correctness。

### 4.2 具体边界

- **相比 AutoDCWorkflow**：两者都与 data cleaning 相关，但 AutoDCWorkflow 的核心问题是“能否为表格自动生成 OpenRefine 工作流”；ICDR-Bench 的核心问题是“给定或逐步澄清自然语言 refinement specification 后，模型能否对非结构化文本真正执行 clean / sanitize / normalize / filter”。
- **相比 DCA-Bench**：DCA-Bench 更像 dataset auditor benchmark，考察 agent 能否在真实平台上发现隐蔽问题；ICDR-Bench 则假设清洗目标最终可以被唯一确定，评测模型能否把 transformation / filtering 真的做对。两者关系是“discovery” 对 “refinement execution”。
- **相比 DataGovBench**：DataGovBench 的重点是把用户意图转成 transformation code，并在 agent 框架里调用工具、调试和编排 workflow；ICDR-Bench 的核心评测是另一件事，即模型能否**不借助外部工具**直接把非结构化文本处理对，而且其中一类核心难点就是**顺序敏感性**。即便加入 Interactive track，重点也仍是 requirement clarification 后的 direct execution，而不是 NL-to-code benchmark。
- **相比 IDA-Bench**：IDA-Bench 评测的是 simulated user 持续给指令时，agent 能否在 Python sandbox 里完成交互式数据分析，当前任务主体主要是 CSV / tabular predictive modeling，最终按 submission 的数值表现是否达到 human baseline 计分。ICDR-Bench 则围绕非结构化文本的数据治理，统一评测 `KEEP + clean_text` / `DROP` 的最终正确性；其中 `Solo` 聚焦 fully specified workflow execution，`Interactive` 聚焦多轮澄清数据需求、保留标准和清洗约束后的 execution。换句话说，两者分别更接近 “interactive guided analysis” 与 “interactive/solo compositional refinement”。
- **相比 OmniDocBench**：OmniDocBench 解决的是“看懂文档”与“解析版面”，ICDR-Bench 解决的是“拿到文本后如何做程序化可验证的治理与清洗”。两者可串联，但不应混为同一任务。

### 4.3 与通用 instruction-following benchmark 的关系

IFEval、FollowBench、SIFo 等工作评测的是通用约束遵循或序列指令跟随，它们是邻近相关工作，但不是 ICDR-Bench 的直接主对比对象。ICDR-Bench 的独特点在于：

- 真实数据治理流程
- deterministic curation operators
- clean-text reference execution
- clean / sanitize / normalize / filter 一体化 end-state 评测

---

## 5. 范围与非目标

### 5.1 v1 范围

ICDR-Bench v1 只做 **text-first** benchmark，覆盖：

1. Web Crawl Cleanup & Filtering
2. Knowledge Base / Support Corpus Preparation
3. Report / Policy / Compliance Document Cleanup
4. Scientific Source Cleanup & Canonicalization

### 5.2 不纳入 v1 核心评测的内容

- 多模态图像/图文治理
- tool-calling
- workflow generation
- code generation 作为主任务
- 纯 document parsing / OCR benchmark

这样做的原因不是这些方向不重要，而是 ICDR-Bench v1 的 novelty 更稳地来自“任务定义和评测协议”，而不是模态数量。

---

## 6. Benchmark 核心设计

### 6.1 Domain 组织方式

benchmark 按真实数据治理场景组织，而不是按单个算子组织。每个 domain 都包含：

- 一个相对统一的数据来源分布
- 一组相关 deterministic operators
- 若干 workflow families
- easy / medium / hard 难度层

ICDR-Bench v1 中，domain 的边界优先由 **应用场景 + 数据基底 + domain-native operators** 决定，而不是由单个共享 text cleanup 算子决定。更具体地说：

- `CleanHtmlMapper` 及其后续清洗链定义 web crawl domain；
- `CleanLinks`、`CleanEmail`、`CleanIp`、`CleanCopyright`、`RemoveTableText` 等共享 text cleanup ops 可以作为 KB/support 与 report/policy/compliance 两个 domain 的 backbone，但两者仍由数据来源和真实 pipeline 明确区分；
- `LatexMergeTexMapper`、`ExpandMacroMapper`、`RemoveCommentsMapper`、`RemoveBibliographyMapper`、`RemoveHeaderMapper` 定义 scientific source domain。

也就是说，ICDR-Bench v1 不追求“每个 domain 都有完全独占的算子”，而是追求：

- 数据分布不同；
- 用户工作流语义不同；
- 最终服务的真实应用 pipeline 不同。

### 6.2 Workflow families 组织原则

family 不按 operator catalog 命名，而按真实目标组织，例如：

- web cleanup
- support-corpus sanitization
- report residue cleanup
- cleanup-then-filter
- source cleanup
- canonicalize-then-filter

### 6.3 难度来源

难度不只由步数决定，而由以下机制决定：

1. workflow 长度
2. 顺序敏感性
3. 参数 grounding
4. 不同清洗子目标的组合
5. near-threshold filtering

### 6.4 如果目标是 oral，这些难度机制必须成为主线

如果目标不只是“做出一个能投的 benchmark”，而是瞄准更强的投稿版本，那么下面两点不能只是附加分析，而应成为 benchmark 的主卖点：

1. **顺序敏感性**

   - 明确构造 `A -> B` 成功但 `B -> A` 失败的 workflow family；
   - 它强调的是“交换两个 active steps 的顺序，最终结果会变”；
   - paper 里要单独汇报 `Order-Sensitive Success`。
2. **组合复杂度**

   - 必须有专门的组合保持子集或组合复杂度切片；
   - 评测时要区分简单组合与长链、新顺序组合，而不是只做 paraphrase 或常规子集。

`中间状态依赖` 仍然重要，但更适合作为**任务构造与质量控制要求**，用来保证 workflow 中的每个 step 都是强 active，而不再单独作为 headline 难度机制。

如果没有前面这两类机制，ICDR-Bench 仍然可能是一个扎实的 benchmark，但更像“清晰的 dataset contribution”；只有把它们做成核心实验，工作才更接近 oral 级别的 evaluative claim。

### 6.5 输出协议

Solo 与 Interactive 两条主线统一输出为：

- `DROP`
- 或 `KEEP + clean_text`

核心协议中 `clean_text` 是唯一 artifact 类型。

### 6.6 核心协议的 artifact 设计约束

ICDR-Bench v1 优先采用 **closed-form、边界唯一、易 canonicalize** 的 artifact。

因此，核心协议不仅排除 `sentence_split_mapper` 和 `text_chunk_mapper` 这类边界敏感的 one-to-many 输出，也暂不把 `ExtractTablesFromHtmlMapper`、`LatexFigureContextExtractorMapper` 这类更自然输出结构化结果的任务纳入主协议。

结论是：

- `SentenceSplit` / `TextChunk` 不进入 v1 核心 family；
- `table extraction` / `figure extraction` 更适合作为附录实验或 extension，而不是 v1 核心协议。

### 6.7 指标

主指标：

- `Workflow Success`

次指标：

- `Status Accuracy`
- `CleanText Exact Match`
- `CleanText Canonical Match`
- `Order-Sensitive Success`
- `Composition Complexity Gap`

可选诊断：

- step probes / sentinel checks
- per-family breakdown
- per-difficulty breakdown

### 6.8 评测切片

建议至少包含三类 evaluation slices：

1. `Core slice`：同 family 分布下的常规样本
2. `Composition slice`：更长链或新顺序的 workflow 组合
3. `Paraphrase slice`：workflow 用不同自然语言表达

如果时间允许，可增加：
4. `Source-shift slice`：来自不同数据源或不同站点/学科子域的样本

### 6.9 如果目标是 oral，主实验应该长什么样

oral 级别的 benchmark 不能只给一个总分表，还需要明确展示 benchmark 揭示了什么新的能力缺口。最建议的主实验结构是：

1. **Main leaderboard**

   - 在 `Workflow Success` 上比较主流 frontier models；
   - 同时报告 `Status Accuracy` 与 `CleanText Exact Match`，避免总分难以解释。
2. **Mechanism breakdown**

   - 单独汇报 `Core / Order-sensitive / Composition` 几类切片；
   - 让读者直接看到模型不是“普遍都差”，而是在哪种机制下系统性失效。
3. **Solo vs Interactive gap**

   - 用同一 domain / family 设计下的 paired subset 比较 `Solo` 和 `Interactive`；
   - 若 `Interactive` 明显更强，就能支持“部分失败来自 specification ambiguity，而不只是 execution failure”。
4. **Solo vs Code gap**

   - 用同一批 fully specified 任务比较 `Solo` 和 `Code`；
   - 若 `Code` 明显更强，就能支持“模型理解了任务，但直接执行仍不稳定”这一更有价值的结论。
5. **Trivial baseline check**

   - 报告 regex / heuristic / scripted baseline；
   - 证明 benchmark 的难度不是来自格式噪声，而是真正来自 workflow execution。
6. **Per-domain application value**

   - 分别展示 web、KB/support、report/policy/compliance、scientific source 四个 domain 的难点不同；
   - 避免 reviewer 觉得这只是同一类清洗任务的重复换皮。

---

## 7. 三种评测模式的定位

### 7.1 Solo 模式（主线之一）

这是 ICDR-Bench 的第一条核心主线。

- 只给模型原始样本和自然语言 workflow
- 不允许用户额外澄清
- 不允许外部数据处理工具
- 不允许先显式生成 workflow 作为中间结果

Solo 模式评测的是模型在无外部帮助时，是否能直接把流程执行正确。

### 7.2 Interactive 模式（主线之一）

这不是附录，而是 ICDR-Bench 的第二条核心主线。

Interactive 模式的目标是回答：

- 当初始需求不完整时，模型能否通过多轮澄清恢复出关键 refinement constraints？
- 澄清完成后，模型能否继续把同一条 compositional refinement 流程执行正确？

Interactive 模式建议采用严格边界：

- 与 Solo 共享同一 domain、operator family、输出协议和最终 reference
- 初始指令只给高层目标，不直接给完整 workflow
- 模型可在限定轮数内追问保留标准、敏感信息策略、是否删除版权/联系方式/模板残留、去重粒度等关键约束
- 澄清结束后，模型再输出最终 `KEEP + clean_text` / `DROP`
- 不把任务主体改成开放式分析助手，也不演化成一般性 agent planning benchmark

Interactive 模式除最终 `Workflow Success` 外，还可增加：

- `Clarification Efficiency`
- `Requirement Coverage`
- `Interactive-to-Solo Gain`

这样定义的好处是：

- 与 Solo 形成清晰互补：`fully specified execution` 与 `underspecified requirement elicitation + execution`
- 仍然服务于数据治理任务，而不是漂移到一般性分析 agent benchmark
- 与 IDA-Bench 形成可解释区分：我们交互的是 **specification**，而不是围绕中间分析结果持续改变任务目标

### 7.3 Code 模式（辅助分析）

Code 模式应当保留，但**不是 headline 主线**。

它的作用是回答：

- 模型失败是因为没理解 refinement specification？
- 还是理解了，但不会精确执行？

Code 模式的约束应尽量严格：

- 与 Solo 使用同一批 fully specified 任务、同一批 reference、同一套指标
- 允许模型写 Python
- 不引入 retrieval / agent planning / debugging benchmark 逻辑
- 不把 code generation 单独包装成新的 benchmark 主任务

换言之，Code 模式是 **upper-bound / diagnostic track**，而不是 ICDR-Bench 的 headline。
---

## 8. 数据构造原则：不能只靠 pattern noise

这是 ICDR-Bench 能否站住的关键。

**结论：pattern-based noise 只能做 bootstrapping，不能做 final benchmark 的主体。**

如果 benchmark 主要由简单规则注入生成，容易出现两类问题：

- 样本过于模板化，模型和规则系统都容易做对；
- benchmark 更像“识别注入模式”，而不是执行真实数据治理流程。

### 8.1 最可行的任务构造路线：template first, sample second

如果你希望**保证每个 step 都是 active**，最可行的方案不是先让 LLM 自由脑暴 workflow，再去海量原始数据里“碰运气”搜索；更稳的做法是：

1. **先人工冻结一小批 workflow templates**

   - 以 family 为单位，先定 `2-4` 步的高价值 workflow；
   - 每个 workflow 都写清楚 step list、参数、目标 domain、预期输出。
2. **为每个 step 写前置条件与 activation 条件**

   - 例如 `CleanEmail` 要求样本中至少有一个真实 email；
   - `RemoveTableText` 要求样本中存在表格残留；
   - `TextLengthFilter` 不只要求可执行，还要要求前序清洗会让长度跨过阈值或接近阈值。
3. **先做 corpus capability index，再做匹配**

   - 不直接找“完美样本”，而是先为语料池中的每个样本打 capability tags；
   - 例如：`has_html`、`has_links`、`has_email`、`has_ip`、`has_repeat_sentences`、`has_table_residue`、`has_long_lines`、`has_comments`、`has_bibliography`、`has_macros`。
4. **按 workflow 前置条件筛样本**

   - workflow 需要什么，就只从满足这些条件的样本子池里取；
   - 不满足就不实例化该 workflow，而不是强行套。
5. **若天然样本不够，再做同源真实污染补足**

   - 从同站点、同文档集合、同论文工程里拼接真实噪声片段；
   - 优先补足缺失的 active steps，而不是手写模板化假噪声。
6. **最后做 executor-based activation 验证**

   - 对每个候选任务跑完整 recipe；
   - 再做 `leave-one-step-out` ablation；
   - 若删掉任一步之后最终 `status` 或 `clean_text` 不变，则该 step 不是强 active，不应进入核心评测。

这个流程的核心是：**workflow 先小规模人工设计，数据按能力索引匹配，最后由 executor 反证每一步都真的重要。** 这里的 `step-active` 是数据质量约束，不再单独作为 headline 难度轴。

### 8.2 推荐的四路数据构造策略

#### 路线 A：Natural-noise route

直接从原始公开语料中取天然带噪样本：

- raw HTML
- docs/help/support text
- long-form reports / filings / policy documents
- LaTeX source
- 文档 boilerplate
- 重复段落 / 标题 / 导航 / footer / bibliography / comments

这类数据最真实，也最不容易被简单 pattern baseline 秒杀。

#### 路线 B：Operator-aware inverse synthesis

从较干净的 reference artifact 出发，按目标 workflow 的“逆方向”构造前像：

- 给 clean text 包回 HTML / links / footer / contact blocks
- 给 clean document 注入重复句、异常空白、异常标点
- 给 clean TeX 加回 comments、header、bibliography、macro aliases

这里的关键不是随便加噪，而是**围绕目标 operator 的可逆前像去设计 corruption**。

#### 路线 C：Contamination / splice route

把同站点、同论文、同文档集合中的真实干扰片段拼接进来，例如：

- 导航栏、目录、页脚、免责声明
- unrelated but same-site snippets
- duplicated boilerplate from neighboring sections
- figure captions or bibliography fragments leaking into body text

这种污染比单纯 regex noise 更接近真实脏数据来源。

#### 路线 D：Hard-threshold calibration

专门构造 near-threshold 样本：

- 长度刚好在阈值附近
- 特殊字符比例接近边界
- 重复率接近边界
- 平均行长 / 最大行长刚好越过边界

这些样本最能拉开模型执行细粒度规则的能力差异。

### 8.3 数据保留标准

最终进入 benchmark 的样本应满足：

1. workflow 的 reference 可被 deterministic executor 唯一确定；
2. artifact 比较可 canonicalize；
3. 不应被单条 regex 或单步启发式轻易解决；
4. 不应主要依赖显式算子名关键词触发；
5. 在不同组合复杂度切片中仍然具有区分度。
6. 对核心评测样本，移除任一步后，最终 `status` 或 `clean_text` 必须改变。

### 8.4 质量控制建议

建议维护三类 sanity baseline：

- regex / heuristic baseline
- simple scripted pipeline baseline
- frontier LLM zero-shot baseline

若某类任务被 trivial baseline 大规模秒杀，则应当：

- 提升组合复杂度
- 提升边界样本比例
- 换成更真实的自然噪声或污染源

---

## 9. Benchmark 规模建议（一个月可落地版本）

建议先做一个可投 MVP，而不是大而全版本。

### 9.1 推荐规模

- `4` 个 domain
- 每个 domain `3-4` 个 family
- 每个 family `3` 档难度
- 每档 `60-120` 个样本

这样得到的规模约为：

- `4 domains × 3 families × 3 difficulties × 80 ≈ 2880` 样本
- 若做到更完整版本，可扩到 `4 × 4 × 3 × 100 ≈ 4800` 样本

### 9.2 推荐优先级

第一优先级：

- Solo 主线
- Interactive 主线
- 4 个应用场景明确的 text-first domains：`web crawl`、`KB/support`、`reports/policy/compliance`、`scientific source`
- Workflow Success + clean-text metrics
- core slice + composition slice

第二优先级：

- Code upper-bound track
- paired ambiguity subset for Solo/Interactive comparison
- paraphrase split
- sentinel probes

第三优先级：

- 更大规模 domain 扩展

---

## 10. 一个月落地路线

### 第 1 周

- 固化任务定义、指标、输出协议
- 固化 4 个 domains 和 12-14 个 workflow families
- 固化 operator 子集

### 第 2 周

- 实现数据构造与 deterministic executor
- 先做 200-300 个 pilot 样本
- 跑 trivial baseline 和 1-2 个 LLM baseline 做难度校准

### 第 3 周

- 扩展到正式 benchmark 规模
- 完成正式评测集、composition slice、workflow paraphrase slice
- 完成 Interactive ambiguity subset 与 simulated-user protocol
- 跑主模型实验

### 第 4 周

- 跑 Solo / Interactive / Code 三条轨道
- 做 failure analysis
- 写论文图表与主要结论

---

## 11. 最终主张

ICDR-Bench 最稳的贡献不在于“算子更多”或“模态更多”，而在于以下组合：

- 非结构化文本/文档数据治理
- `Solo + Interactive` 双轨 compositional refinement
- direct execution rather than planning/generation
- deterministic operator-backed reference
- clean-text-level end-to-end evaluation

如果这五点同时成立，ICDR-Bench 就不会只是“又一个数据处理 benchmark”，而会是一个任务定义清楚、实验上可落地、审稿时也能说清楚边界的新 benchmark。
