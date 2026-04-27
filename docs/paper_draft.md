# CDR-Bench Paper Draft

Working title: **CDR-Bench: Benchmarking LLMs on Compositional Data Refinement Recipes**

This draft follows the current research plan in `research_plan.md`. It intentionally marks result-dependent claims as placeholders because the current workspace contains construction and workflow-mining artifacts but not completed model evaluation results.

## Storyline and NeurIPS Positioning

CDR-Bench is about execution fidelity in natural-language data refinement: a model receives a raw sample and a user-facing refinement request, then must produce the exact final `KEEP` or `DROP` decision and the final cleaned text that a deterministic reference pipeline would produce.

The central gap is that current benchmarks mostly ask whether LLMs can audit datasets, generate data-cleaning workflows, write transformation code, or perform isolated edits. These are important, but they do not test whether a model can directly operationalize a compositional data-refinement policy on the instance itself. In real data work, the final artifact matters: a cleaned page, a sanitized document, a retained or rejected record, and a reproducible reason for that final state.

The key novelty is the benchmark granularity. CDR-Bench isolates direct refinement execution from tool use, code generation, and agentic debugging. This makes it possible to ask whether models fail because they cannot perform an atomic operation, because they cannot compose multiple operations, because they make the wrong keep/drop decision, or because they collapse different operation orders into the same generic cleaning behavior.

The practical value is that this capability sits under many high-impact workflows: pretraining corpus construction, RAG ingestion, search indexing, scientific source processing, support-corpus preparation, and privacy-preserving release. A model that can describe a policy but cannot faithfully execute it may still corrupt data pipelines at scale.

The NeurIPS-worthy claim should be framed as follows: CDR-Bench introduces a new evaluation target for data-centric AI systems: **compositional data-refinement execution**. It contributes a deterministic, domain-grounded benchmark with main, order-sensitivity, and atomic-calibration tracks; evaluates joint text transformation and filtering decisions; and enables an atomic-to-compositional gap analysis that existing data-agent, data-cleaning, and text-editing benchmarks do not provide.

## Compact Section Outline

- **Motivation:** Natural-language data refinement is increasingly realistic, but correctness is defined by the final data artifact, not by a plausible plan.
- **Gap:** Existing benchmarks cover dataset audit, workflow/code generation, interactive analysis, corpus curation, and precise editing, but not direct execution of compositional refinement policies.
- **Task:** Given a raw sample and a natural-language refinement request, output a deterministic final JSON object with `status` and `clean_text`.
- **Benchmark design:** Ground workflows in real domains, mine operator combinations from data, materialize deterministic Data-Juicer references, and separate prompt wording from reference generation.
- **Diagnostics:** Main tasks test direct execution, order-sensitive groups test operation-order adherence, and atomic tasks calibrate single-operator difficulty.
- **Evaluation plan:** Report workflow success, status accuracy, clean-text match, order-family success, wrong-order collapse, and atomic-to-compositional gaps.

## Introduction

[Opening / motivation] Data refinement is a routine but consequential part of modern AI pipelines. Before text is used for pretraining, retrieval-augmented generation, search indexing, scientific corpus construction, support-document grounding, or privacy-preserving release, raw records are often cleaned, normalized, redacted, extracted, filtered, and sometimes reordered through several dependent decisions. Although these operations are traditionally implemented as scripts or workflow graphs, many practical refinement needs are first expressed in natural language: a user knows that a web page should have boilerplate removed before quality filtering, or that a LaTeX source should be canonicalized before retention, but may not want to write or inspect a full processing pipeline for each sample.

[Challenge / gap] The growing ability of large language models to follow instructions, write code, and act as data agents does not by itself guarantee faithful data-refinement execution. A model may understand the user's intent at a high level, identify the right family of cleaning operations, or generate a plausible workflow, yet still produce the wrong final artifact when operations interact. For example, deciding whether to keep a document before cleaning can yield a different reference than cleaning it first and filtering afterward; similarly, a model can perform a redaction or normalization step in isolation but fail when the final answer depends on a specific sequence of redaction, cleanup, and filtering.

[Task definition] This paper introduces CDR-Bench, a benchmark for compositional data-refinement execution. In each task, a model receives a raw unstructured text sample and a natural-language refinement request, then returns a final JSON object, either `{"status": "KEEP", "clean_text": "..."}` or `{"status": "DROP", "clean_text": "..."}`. The hidden reference is produced by a deterministic Data-Juicer-backed workflow, while the model-facing prompt describes the user need rather than exposing operator names. Thus, CDR-Bench evaluates whether an LLM can operationalize a refinement policy directly on the sample itself, rather than merely propose code, invoke tools, or describe a plan.

[Technical novelty] CDR-Bench is designed around three forms of execution fidelity that are under-measured by existing evaluations. First, it evaluates the joint correctness of a binary retention decision and a text transformation, because real refinement workflows often require both. Second, it separates atomic operator competence from compositional execution by including single-operator calibration tasks alongside multi-step workflows. Third, it includes an order-sensitivity track in which related tasks share the same raw input, clean-operation skeleton, and filter, but differ in where the filter is applied; group-level success requires the model to respect the requested order rather than collapse the task into a generic cleanup strategy.

[Benchmark construction] The benchmark is grounded in realistic text-first data-refinement scenarios. The current design covers web crawl cleanup, scientific TeX/source cleanup, knowledge-base preparation, and PII sanitization. For each domain, CDR-Bench runs candidate Data-Juicer operators on real raw data, records operator activity, mines frequent data-supported clean-operation combinations, materializes ordered workflow variants, recalibrates filter thresholds to produce balanced keep/drop decisions, and finally generates deterministic references. The benchmark contains a main track with clean-only, filter-then-clean, and clean-then-filter workflows; an order-sensitivity diagnostic track with front, middle, and end filter placements; and an atomic calibration track keyed by global operator.

[Evaluation / evidence placeholder] CDR-Bench evaluates models with workflow success, status accuracy, exact and canonical clean-text match, order-family success, wrong-order collapse rate, and atomic-to-compositional gap. In the final paper, this paragraph should summarize the main empirical findings, for example: `[RESULT: best closed and open models on main workflow success]`, `[RESULT: gap between atomic and compositional success]`, `[RESULT: error concentration by workflow type or domain]`, and `[RESULT: order-sensitive family failure rate]`. These results should support the paper's core claim that direct refinement execution is a distinct bottleneck from isolated editing, workflow generation, and data-agent planning.

[Contributions] This work makes four contributions. First, we formalize compositional data-refinement execution as a benchmark task with a deterministic final-state output protocol. Second, we construct CDR-Bench from domain-grounded, data-supported workflows over web, arXiv, knowledge-base, and PII refinement scenarios. Third, we introduce diagnostics for order adherence and atomic-to-compositional gaps, allowing failures to be localized beyond aggregate instruction-following accuracy. Fourth, we provide a reproducible benchmark construction pipeline that separates reference generation from natural-language prompt generation, enabling prompt-style studies without changing the underlying ground truth.

## Related Work

[Topic: data agents and data-governance benchmarks] Recent benchmarks evaluate LLM agents across increasingly realistic data-centric workflows. DCA-Bench studies dataset curation agents that must discover hidden issues in real-world datasets, such as documentation problems, label errors, ethical concerns, and outdated information, across 221 test cases from dataset platforms [huang2024dcabench]. DataGovBench evaluates data-governance agents that translate user intent into executable transformation code over 150 real-world workflow tasks and reports that complex multi-step governance remains difficult for current models [liu2025datagovbench]. DAComp broadens the setting to the full data intelligence lifecycle, combining data engineering over industrial schemas with open-ended data analysis and showing low agent success on repository-level pipeline orchestration [lei2025dacomp]. IDA-Bench focuses on interactive guided data analysis, where agents respond to sequential natural-language instructions derived from Kaggle notebooks and are judged by final numerical outputs [li2025idabench]. These benchmarks are closest in motivation because they evaluate data work rather than generic language tasks, but their primary object is agentic discovery, code generation, SQL or table pipeline construction, or iterative analysis. CDR-Bench instead isolates a lower-level capability: given a raw sample and a complete refinement request, can the model directly produce the deterministic final refined artifact?

[Topic: automated data cleaning and workflow generation] Automated data-cleaning systems and benchmarks study how to generate or execute cleaning workflows from higher-level goals. AutoDCWorkflow generates OpenRefine operations from a raw table and an analysis purpose, then evaluates answer, data, and workflow quality over table-cleaning tasks [li2024autodcworkflow]. Data-Juicer and Data-Juicer 2.0 provide scalable operator abstractions and system support for constructing data recipes for foundation models, including composable operators, multimodal processing, and large-scale execution [chen2023datajuicer, chen2025datajuicer2]. CDR-Bench builds on this style of deterministic operator-backed data processing, but asks a different evaluation question. Rather than testing whether an LLM can synthesize a workflow or whether a data-processing system can scale, CDR-Bench tests whether an LLM itself can follow a user-facing refinement policy to the same final state as a hidden deterministic reference.

[Topic: data curation for pretraining, extraction, and alignment] A separate line of work shows that data refinement choices strongly affect downstream model quality. Dolma documents a three-trillion-token open pretraining corpus and reports analyses of intermediate curation stages and practices [soldaini2024dolma]. DataComp-LM provides a controlled testbed for dataset experiments, including deduplication, filtering, and data mixing over a 240T-token Common Crawl corpus and 53 downstream evaluations [li2024datacomp]. Recent work on HTML-to-text extraction shows that fixed extraction choices can substantially change the pages that survive filtering and can affect structured-content tasks such as table question answering and code performance [li2026htmltotext]. PrefCleanBench benchmarks preference-data cleaning strategies for LLM alignment across datasets, architectures, and optimization algorithms [yeh2025prefcleanbench]. These studies establish the importance of data quality and cleaning policies, but they generally evaluate corpus construction, downstream training outcomes, or the effectiveness of cleaning methods. CDR-Bench complements them by evaluating whether a model can execute a specified refinement policy on individual samples before those samples enter such large-scale pipelines.

[Topic: instruction-driven text editing] Precise text-editing benchmarks test whether models can apply targeted modifications to existing text. FineEdit introduces InstrEditBench, an automated benchmark with more than 30,000 structured editing tasks over Wikipedia articles, LaTeX documents, source code, and database languages, and trains a model specialized for context-aware edits [zeng2025fineedit]. This line is highly relevant because CDR-Bench also requires exact text transformations, including domain-specific formats such as LaTeX. However, CDR-Bench differs in both task structure and evaluation target: the model must jointly decide retention status and final text, handle cleaning and filtering as an ordered policy, and match deterministic references across multi-step workflows rather than complete isolated edit instructions alone.

## Reverse Outline

- Paragraph 1: Data refinement is common and consequential, and users often express refinement needs in natural language.
- Paragraph 2: High-level intent understanding or workflow generation does not guarantee faithful final artifacts under composition and ordering.
- Paragraph 3: CDR-Bench defines the direct refinement execution task and deterministic output protocol.
- Paragraph 4: The benchmark contributes joint decision/text evaluation, atomic-to-compositional calibration, and order-sensitivity diagnostics.
- Paragraph 5: The construction pipeline grounds workflows in realistic domains and deterministic Data-Juicer references.
- Paragraph 6: Empirical claims must be filled after model evaluation.
- Paragraph 7: Contributions are benchmark formalization, construction, diagnostics, and reproducible pipeline design.

## Claim-Evidence Map

- Claim: CDR-Bench evaluates direct compositional data-refinement execution rather than workflow generation or agentic tool use. | Evidence: Research plan task protocol, deterministic Data-Juicer references, model output contract. | Status: supported by design.
- Claim: Existing data-centric benchmarks focus on dataset issue discovery, data governance code/workflows, enterprise data intelligence, or interactive analysis. | Evidence: DCA-Bench, DataGovBench, DAComp, and IDA-Bench abstracts and task descriptions. | Status: supported by cited related work.
- Claim: Data-cleaning workflow generation is adjacent but different from final-state execution. | Evidence: AutoDCWorkflow generates OpenRefine workflows from raw tables and analysis purposes; CDR-Bench asks for final status/text on raw text samples. | Status: supported by cited related work and benchmark design.
- Claim: Data quality and extraction choices matter for downstream LLM performance. | Evidence: Dolma, DataComp-LM, HTML-to-text extraction, and PrefCleanBench. | Status: supported by cited related work.
- Claim: CDR-Bench can quantify atomic-to-compositional gaps. | Evidence: Atomic track and main-track design in research plan. | Status: supported by design; needs model evaluation.
- Claim: Current LLMs struggle with CDR-Bench. | Evidence: None in current workspace. | Status: needs pilot and full evaluation results.
- Claim: Order-sensitive examples reveal wrong-order collapse. | Evidence: Order-sensitivity track design with front/middle/end variants. | Status: supported by design; needs model evaluation.

## Self-Review Checklist

- Clarity: The task is defined before the contribution list, and CDR-Bench is repeatedly distinguished from workflow generation, agentic analysis, and isolated editing.
- Flow: The introduction moves from practical motivation to capability gap, then task definition, design, evaluation plan, and contributions.
- Terminology consistency: Use "compositional data-refinement execution", "refinement policy", "deterministic reference", "main track", "order-sensitivity track", and "atomic calibration track" consistently.
- Unsupported claims: The empirical struggle of LLMs, best-model scores, atomic-to-compositional gaps, and wrong-order collapse rates are placeholders until experiments are run.
- Missing evidence: Add final benchmark sizes, model list, score table, qualitative error taxonomy, and examples once benchmark materialization and pilot evaluation are complete.

## Citation Notes

- [huang2024dcabench] DCA-Bench: A Benchmark for Dataset Curation Agents. arXiv:2406.07275. https://arxiv.org/abs/2406.07275
- [liu2025datagovbench] DataGovBench: Benchmarking LLM Agents for Real-World Data Governance Workflows. arXiv:2512.04416. https://arxiv.org/abs/2512.04416
- [yeh2025prefcleanbench] Clean First, Align Later: Benchmarking Preference Data Cleaning for Reliable LLM Alignment. arXiv:2509.23564. https://arxiv.org/abs/2509.23564
- [lei2025dacomp] DAComp: Benchmarking Data Agents across the Full Data Intelligence Lifecycle. arXiv:2512.04324. https://arxiv.org/abs/2512.04324
- [li2026htmltotext] Beyond a Single Extractor: Re-thinking HTML-to-Text Extraction for LLM Pretraining. arXiv:2602.19548. https://arxiv.org/abs/2602.19548v1
- [li2025idabench] IDA-Bench: Evaluating LLMs on Interactive Guided Data Analysis. arXiv:2505.18223. https://arxiv.org/abs/2505.18223
- [li2024autodcworkflow] AutoDCWorkflow: LLM-based Data Cleaning Workflow Auto-Generation and Benchmark. arXiv:2412.06724. https://arxiv.org/abs/2412.06724
- [zeng2025fineedit] Bridging the Editing Gap in LLMs: FineEdit for Precise and Targeted Text Modifications. arXiv:2502.13358. https://arxiv.org/abs/2502.13358
- [chen2023datajuicer] Data-Juicer: A One-Stop Data Processing System for Large Language Models. arXiv:2309.02033. https://arxiv.org/abs/2309.02033
- [chen2025datajuicer2] Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing for and with Foundation Models. arXiv:2501.14755. https://arxiv.org/abs/2501.14755
- [li2024datacomp] DataComp-LM: In search of the next generation of training sets for language models. arXiv:2406.11794. https://arxiv.org/abs/2406.11794
- [soldaini2024dolma] Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research. arXiv:2402.00159. https://arxiv.org/abs/2402.00159
