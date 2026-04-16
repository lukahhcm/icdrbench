# ICDR-Bench Domain Design v6

## 0. Solo / Interactive 共用执行协议
ICDR-Bench v1 的 `Solo` 与 `Interactive` 两条主线在执行阶段统一采用：
- `status ∈ {KEEP, DROP}`
- `clean_text`

也就是说，两条主线最终都评测：
- 模型是否做出了正确的保留/丢弃决策；
- 若保留，模型是否输出了与 deterministic executor 一致的最终清洗文本。

两条主线的区别不在最终 artifact，而在输入设定：
- `Solo`: 用户一次性给出完整 workflow specification；
- `Interactive`: 用户先给出高层 refinement goal，模型可在限定轮数内澄清需求后再执行。

这样做的好处是：
- 评测协议简单；
- 不同 domain、不同交互设定之间可以直接横向比较；
- 避免结构化输出格式带来的额外评测争议。

因此，像 `ExtractTablesFromHtmlMapper`、`LatexFigureContextExtractorMapper` 这类更自然输出结构化 artifact 的算子，适合作为 appendix / extension，而不是 v1 核心协议。

---

## 1. domain 应该按应用场景来分
如果目标是一个更强、更有应用价值的 benchmark，domain 不应该只按“数据长什么样”来分，而要按“这个 benchmark 服务哪个真实 pipeline”来分。

ICDR-Bench v1 建议使用 4 个应用场景明确的 domain：
1. `Web Crawl Cleanup & Filtering`
2. `Knowledge Base / Support Corpus Preparation`
3. `Report / Policy / Compliance Document Cleanup`
4. `Scientific Source Cleanup & Canonicalization`

这 4 个 domain 覆盖 4 类明显不同的真实使用场景：
- 大规模 web corpus ingestion
- 企业知识库 / FAQ / 帮助中心 / RAG 准备
- 长文档、政策、合规、财报类文档清洗
- scientific source 级语料构建

---

## 2. 全局设计原则
### 2.1 仍然只做 text-first
ICDR-Bench v1 不做多模态，不做 tool-calling，不做 workflow generation。

### 2.2 仍然优先使用 deterministic Data-Juicer operators
两条主线在执行阶段都必须能由 deterministic executor 给出唯一 `status + clean_text`。

### 2.3 难度必须来自机制，不只是脏数据更复杂
oral 导向下，每个 domain 至少要覆盖三类机制：
1. `Order-sensitive workflows`
2. `Composition-complexity slices`
3. `Step-active quality control`

这里刻意不使用“compositional generalization”这一说法，因为 ICDR-Bench 不发布 train split；我们评的是**组合复杂度**，不是训练集意义上的泛化。

这里建议把“中间状态”降级为质检约束，而不是 headline 机制：
- `Order-sensitive`: 交换两个 active steps 的顺序，最终 `status` 或 `clean_text` 改变。
- `Composition-complexity`: workflow 由更多 active steps 组成，长度更长、组合更复杂。
- `Step-active quality control`: 删掉任一步，最终 `status` 或 `clean_text` 应改变，否则该任务不进入核心评测。

### 2.4 数据构造不能只靠 pattern noise
必须优先使用：
- natural noise
- operator-aware inverse synthesis
- contamination / splice
- hard-threshold calibration

### 2.5 数据保留规则
进入 benchmark 的样本至少满足：
1. deterministic executor 可唯一给出 `status + clean_text`；
2. `clean_text` 可 exact / canonical match；
3. 不能被单条 regex 或单步 heuristic 高准确率解决；
4. 不主要依赖显式算子名关键词；
5. 能体现 workflow 组合，而不只是单步 pattern matching。

---

## 3. D1: Web Crawl Cleanup & Filtering
### 3.1 应用场景
面向：
- pretraining web corpora 清洗
- 通用网页语料过滤
- search / retrieval 前的 web text ingestion

这是最典型的“原始网页变干净文本”的场景。

### 3.2 主数据池
优先使用：
- Common Crawl / CC-MAIN raw HTML
- benchmark-internal crawl 的公开网页，如 docs、tutorial、blog、wiki-like pages

### 3.3 候选公开来源
- Common Crawl
- 公开网站的原始 HTML 页面

### 3.4 这些数据如何变成非平凡脏数据
- `Natural noise`: 直接保留 raw HTML、导航、页脚、cookie/banner、站点模板、联系方式等真实噪声。
- `Contamination / splice`: 从同站点相邻页面拼接导航栏、相关推荐、版权区块、重复 footer，而不是手写模板字符串。
- `Inverse synthesis`: 从清洗后的正文反向包回链接块、联系信息、异常空白和符号，专门制造顺序敏感与 near-threshold filtering 样本。

### 3.5 Core operators
#### Domain-native
- `CleanHtmlMapper`

#### Shared support
- `CleanLinksMapper`
- `CleanEmailMapper`
- `CleanIpMapper`
- `FixUnicodeMapper`
- `PunctuationNormalizationMapper`
- `WhitespaceNormalizationMapper`
- `RemoveSpecificCharsMapper`
- `RemoveRepeatSentencesMapper`
- `RemoveWordsWithIncorrectSubstringsMapper`
- `TextLengthFilter`
- `WordsNumFilter`
- `SpecialCharactersFilter`
- `WordRepetitionFilter`
- `CharacterRepetitionFilter`
- `AverageLineLengthFilter`
- `MaximumLineLengthFilter`
- `AlphanumericFilter`

### 3.6 推荐 families
#### D1-F1. HTML body cleanup
- 目标：把 raw HTML 清洗成正文文本
- 输出：`status + clean_text`
- 典型 workflow：
  - `CleanHtml`
  - `CleanHtml -> CleanLinks`
  - `CleanHtml -> FixUnicode -> WhitespaceNormalization`

#### D1-F2. Web sanitization
- 目标：进一步清除链接、联系方式和模板残留
- 输出：`status + clean_text`
- 典型 workflow：
  - `CleanHtml -> CleanEmail -> CleanIp`
  - `CleanHtml -> CleanLinks -> RemoveSpecificChars -> WhitespaceNormalization`

#### D1-F3. Cleanup-then-filter
- 目标：先清洗，再判断网页是否值得保留
- 输出：`status + clean_text`
- 典型 workflow：
  - `CleanHtml -> TextLengthFilter`
  - `CleanHtml -> CleanLinks -> TextLengthFilter`
  - `CleanHtml -> RemoveRepeatSentences -> AverageLineLengthFilter`
  - `CleanHtml -> FixUnicode -> SpecialCharactersFilter`

### 3.7 oral 导向的难度机制
- `Order-sensitive`: 需要先把 HTML 解析成正文，再做后续基于纯文本的清洗；若把这些步骤交换到 HTML 解析之前，最终 `clean_text` 会变
- `Composition-complexity`: 在高难切片中加入三步、四步 workflow
- `Step-active QC`: 进入核心评测的 workflow 需要通过 leave-one-step-out 检查，确保关键清洗步不是 no-op

### 3.8 为什么有应用价值
它直接对应 web-scale data pipeline 的第一步，也是 foundation model 预训练和通用检索最常见的语料入口之一。

---

## 4. D2: Knowledge Base / Support Corpus Preparation
### 4.1 应用场景
面向：
- 企业知识库构建
- help-center / FAQ / product docs 准备
- RAG / support assistant 前的语料卫生处理

这个场景和 D1 的关键区别是：
- D1 处理的是 raw web pages；
- D2 处理的是已经进入知识库候选池的 docs / help / support text。

### 4.2 主数据池
优先使用：
- 产品文档站点的文本或固定抽取结果
- 帮助中心、FAQ、manual、README、技术文档的纯文本版本
- benchmark-internal crawl / export 的 docs corpora

### 4.3 候选公开来源
适合用作 benchmark-internal 语料池的公开来源包括：
- Kubernetes documentation
- Python documentation
- scikit-learn documentation
- MDN Web Docs
- Hugging Face documentation

### 4.4 这些数据如何变成非平凡脏数据
- `Natural noise`: 保留文档站自带的版本导航、面包屑、相关链接、版权尾注、复制按钮残留和 FAQ 模板重复段。
- `Contamination / splice`: 从同一文档站别的页面拼接“相关资源”“升级提示”“常见问题”片段，制造真实 support-corpus 污染。
- `Inverse synthesis`: 从较干净的文档正文反向加入链接、邮箱、IP、表格残留、异常长 token 和重复模板，并校准 filter 边界。

### 4.5 Core operators
#### Shared cleanup ops as the domain backbone
- `CleanLinksMapper`
- `CleanEmailMapper`
- `CleanIpMapper`
- `FixUnicodeMapper`
- `PunctuationNormalizationMapper`
- `WhitespaceNormalizationMapper`
- `RemoveSpecificCharsMapper`
- `RemoveRepeatSentencesMapper`
- `RemoveWordsWithIncorrectSubstringsMapper`
- `CleanCopyrightMapper`
- `RemoveTableTextMapper`
- `RemoveLongWordsMapper`

#### Filters
- `TextLengthFilter`
- `TokenNumFilter`
- `WordsNumFilter`
- `SpecialCharactersFilter`
- `WordRepetitionFilter`
- `CharacterRepetitionFilter`
- `AverageLineLengthFilter`
- `MaximumLineLengthFilter`
- `AlphanumericFilter`

### 4.6 推荐 families
#### D2-F1. Support-text sanitization
- 目标：清除链接、联系方式、异常符号和模板尾巴
- 输出：`status + clean_text`
- 典型 workflow：
  - `CleanLinks -> CleanEmail -> CleanIp`
  - `CleanLinks -> RemoveSpecificChars -> WhitespaceNormalization`

#### D2-F2. Support-corpus residue cleanup
- 目标：去掉版权头、表格残留、异常长 token、重复模板
- 输出：`status + clean_text`
- 典型 workflow：
  - `CleanCopyright -> RemoveTableText -> WhitespaceNormalization`
  - `RemoveTableText -> RemoveLongWords -> WhitespaceNormalization`
  - `CleanCopyright -> RemoveRepeatSentences -> WhitespaceNormalization`

#### D2-F3. Prepare-then-filter
- 目标：在知识库文本卫生处理后决定是否保留
- 输出：`status + clean_text`
- 典型 workflow：
  - `CleanCopyright -> RemoveTableText -> TextLengthFilter`
  - `FixUnicode -> WhitespaceNormalization -> SpecialCharactersFilter`
  - `RemoveRepeatSentences -> AverageLineLengthFilter`

### 4.7 oral 导向的难度机制
- `Order-sensitive`: 某些 support 文本需要先做残留清理，再做规范化或过滤；交换顺序后最终 `clean_text` 或 `status` 会改变
- `Composition-complexity`: 高难切片里引入三步混合 workflow，而核心切片只含二步组合
- `Step-active QC`: 只有删去任一步就会改变最终结果的样本，才进入核心评测

### 4.8 为什么有应用价值
这是企业知识库、support bot、FAQ 检索、RAG ingest 中非常真实的一步，而且和“原始网页清洗”不同，它更接近组织内部真正会维护的 corpus。

---

## 5. D3: Report / Policy / Compliance Document Cleanup
### 5.1 应用场景
面向：
- 政策文档、政府报告、财报、合规文档清洗
- 长文档检索与分析前的预处理
- compliance / finance / policy intelligence pipeline

这个 domain 和 D2 的区别在于：
- D2 偏 docs/help/support；
- D3 偏长文档、正式报告、政策/法规/合规材料。

### 5.2 主数据池
优先使用：
- PDF 或 HTML 形式的政府报告、政策文档、白皮书
- SEC EDGAR filing 文本
- GovInfo / GAO / EUR-Lex 等公开长文档来源的固定抽取文本

### 5.3 候选公开来源
- U.S. GAO reports
- GovInfo / U.S. Government Publishing Office
- SEC EDGAR filings
- EUR-Lex public legal documents

### 5.4 这些数据如何变成非平凡脏数据
- `Natural noise`: 直接使用 PDF/HTML 固定抽取链后的文本，保留目录、页眉页脚、免责声明、表格残留、长行和格式破碎。
- `Contamination / splice`: 从同一报告集合拼接附录、法律声明、参考页、表格块和相邻章节 boilerplate。
- `Hard-threshold calibration`: 专门保留清洗前后会卡在 `AverageLineLengthFilter`、`MaximumLineLengthFilter`、`WordsNumFilter` 边界附近的样本。

### 5.5 Core operators
- `CleanCopyrightMapper`
- `RemoveTableTextMapper`
- `RemoveLongWordsMapper`
- `FixUnicodeMapper`
- `PunctuationNormalizationMapper`
- `WhitespaceNormalizationMapper`
- `RemoveRepeatSentencesMapper`
- `RemoveSpecificCharsMapper`
- `TextLengthFilter`
- `TokenNumFilter`
- `WordsNumFilter`
- `SpecialCharactersFilter`
- `AverageLineLengthFilter`
- `MaximumLineLengthFilter`
- `AlphanumericFilter`

### 5.6 推荐 families
#### D3-F1. Front-matter / boilerplate cleanup
- 目标：去掉版权头、免责声明、长文档 boilerplate
- 输出：`status + clean_text`
- 典型 workflow：
  - `CleanCopyright -> WhitespaceNormalization`
  - `CleanCopyright -> RemoveSpecificChars -> WhitespaceNormalization`

#### D3-F2. Extraction-residue cleanup for long documents
- 目标：清除表格残留、异常长行、异常 token、重复模板
- 输出：`status + clean_text`
- 典型 workflow：
  - `RemoveTableText -> RemoveLongWords -> WhitespaceNormalization`
  - `RemoveTableText -> AverageLineLengthFilter`
  - `FixUnicode -> RemoveTableText -> MaximumLineLengthFilter`

#### D3-F3. Cleanup-then-filter for reports
- 目标：在长文档清理后判断是否保留
- 输出：`status + clean_text`
- 典型 workflow：
  - `CleanCopyright -> RemoveTableText -> TextLengthFilter`
  - `RemoveRepeatSentences -> AverageLineLengthFilter`
  - `FixUnicode -> SpecialCharactersFilter -> WordsNumFilter`

### 5.7 oral 导向的难度机制
- `Order-sensitive`: 某些报告样本必须先去掉 boilerplate / 表格残留，再做规范化或过滤；交换顺序后最终结果会变
- `Composition-complexity`: 高难切片使用三步甚至四步清理链
- `Step-active QC`: 通过 executor ablation 保证每一步都会影响最终 `status` 或 `clean_text`

### 5.8 为什么有应用价值
这一类文档是 policy retrieval、finance/copilot、compliance search、长文档分析的真实输入，应用价值比单纯“通用文本清洗”更强、更容易说服 reviewer。

---

## 6. D4: Scientific Source Cleanup & Canonicalization
### 6.1 应用场景
面向：
- scientific corpus 构建
- source-level 论文分析
- 学术语料检索和 ingestion

输入是压缩的 LaTeX project archive 或解包后的多文件 TeX 工程。
输出统一为 `status + clean_text`。

### 6.2 主数据池
- arXiv source tarballs
- 解包后的多文件 TeX 工程

### 6.3 候选公开来源
- arXiv bulk source access

### 6.4 这些数据如何变成非平凡脏数据
- `Natural noise`: 直接使用多文件 TeX 工程，保留 comments、header、bibliography、macro alias、跨文件引用等真实 source-level 噪声。
- `Contamination / splice`: 从同一论文工程中拼接 appendix、supplement、重复导言或 bibliography 片段，制造 source residue。
- `Inverse synthesis`: 从规范化后的 TeX 反向加回 comments、header、bibliography 和宏别名，专门构造展开顺序与过滤边界敏感样本。

### 6.5 Core operators
#### Domain-native
- `LatexMergeTexMapper`
- `ExpandMacroMapper`
- `RemoveCommentsMapper`
- `RemoveBibliographyMapper`
- `RemoveHeaderMapper`

#### Shared support
- `WhitespaceNormalizationMapper`
- `PunctuationNormalizationMapper`
- `FixUnicodeMapper`
- `TextLengthFilter`
- `TokenNumFilter`
- `WordsNumFilter`
- `SpecialCharactersFilter`
- `WordRepetitionFilter`

### 6.6 推荐 families
#### D4-F1. Source cleanup
- 目标：清理 comments、bibliography、header
- 输出：`status + clean_text`
- 典型 workflow：
  - `LatexMergeTex -> RemoveComments`
  - `LatexMergeTex -> RemoveComments -> RemoveBibliography`
  - `LatexMergeTex -> RemoveHeader -> RemoveComments`

#### D4-F2. Canonicalization
- 目标：合并多文件、展开宏并规范化
- 输出：`status + clean_text`
- 典型 workflow：
  - `LatexMergeTex -> ExpandMacro`
  - `LatexMergeTex -> RemoveComments -> ExpandMacro`
  - `LatexMergeTex -> RemoveComments -> ExpandMacro -> WhitespaceNormalization`

#### D4-F3. Canonicalize-then-filter
- 目标：在规范化 scientific text 上做最终保留判断
- 输出：`status + clean_text`
- 典型 workflow：
  - `LatexMergeTex -> ExpandMacro -> TextLengthFilter`
  - `LatexMergeTex -> RemoveComments -> ExpandMacro -> TokenNumFilter`
  - `LatexMergeTex -> RemoveComments -> ExpandMacro -> WordsNumFilter`

### 6.7 oral 导向的难度机制
- `Order-sensitive`: merge / cleanup / expand 之间的顺序交换会改变最终 `clean_text`
- `Composition-complexity`: 高难切片引入 `merge + cleanup + expand + filter` 这类更长链组合
- `Step-active QC`: merge、cleanup、expand、filter 中的任一步若删掉后结果不变，则该样本不纳入核心评测

### 6.8 为什么有应用价值
它对应 scientific data pipeline 的真实难点，而且和前三个 domain 在数据基底与应用目标上都明显不同。

---

## 7. 最终建议的核心配置
### 7.1 推荐 domain 数量
- `4` 个 domain

### 7.2 推荐 family 数量
- 每个 domain `3` 个核心 family
- 总计 `12` 个 family

### 7.3 推荐统一输出协议
Solo 与 Interactive 两条主线在最终执行阶段统一为：
```json
{"status": "KEEP", "clean_text": "..."}
```
或
```json
{"status": "DROP", "clean_text": ""}
```

### 7.4 推荐优先级
#### 必进核心集
- D1-F1 HTML body cleanup
- D1-F3 Cleanup-then-filter
- D2-F2 Support-corpus residue cleanup
- D2-F3 Prepare-then-filter
- D3-F2 Extraction-residue cleanup for long documents
- D3-F3 Cleanup-then-filter for reports
- D4-F1 Source cleanup
- D4-F3 Canonicalize-then-filter

#### 很适合核心集
- D1-F2 Web sanitization
- D2-F1 Support-text sanitization
- D3-F1 Front-matter / boilerplate cleanup
- D4-F2 Canonicalization

---

## 8. 一张总表：应用场景、数据池、价值
| Domain | 应用场景 | 主数据池 | 代表算子 | 输出 |
| --- | --- | --- | --- | --- |
| Web Crawl Cleanup & Filtering | web-scale ingest / pretraining corpora | raw HTML pages | `CleanHtmlMapper` | `status + clean_text` |
| Knowledge Base / Support Corpus Preparation | enterprise KB / help-center / RAG prep | docs/help/manual/FAQ text | shared text cleanup ops | `status + clean_text` |
| Report / Policy / Compliance Document Cleanup | policy / finance / compliance long docs | reports / filings / legal docs text | `CleanCopyright`, `RemoveTableText`, line-based filters | `status + clean_text` |
| Scientific Source Cleanup & Canonicalization | scientific corpus construction | arXiv LaTeX source | `LatexMergeTex`, `RemoveComments`, `RemoveBibliography`, `ExpandMacro` | `status + clean_text` |

---

## 9. 一句话总结
如果目标是一个更像 oral 候选的 benchmark，那么更合理的 domain 结构不是“网页 + 泛文本 + 两个很像的 scientific”，而是：
- `web-scale ingest`
- `KB / RAG ingest`
- `reports / policy / compliance`
- `scientific source`

再配上：
- 统一 `status + clean_text` 输出协议
- 顺序敏感 / 组合复杂度两类主难度机制
- 以及 `step-active` 质量控制

这样应用价值和评测价值都会更强。
