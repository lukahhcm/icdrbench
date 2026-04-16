# ICDR-Bench Domain Download Guide

## 1. Guiding principle
Data should be downloaded in the **raw representation that makes the target operators meaningful**.

This means:
- if a domain relies on `clean_html_mapper`, prefer raw HTML instead of already cleaned text;
- if a domain relies on `remove_table_text_mapper`, prefer fixed extraction outputs that preserve table residue instead of already normalized corpora;
- if a domain relies on `remove_comments_mapper` or `expand_macro_mapper`, prefer LaTeX source instead of PDF text.

The first-stage goal is not to maximize dataset size, but to build a corpus where the chosen operators can be activated and measured.

## 2. Read the support summary correctly
The current bootstrap summary is useful, but it should be interpreted with operator semantics in mind:
- `special_characters_filter` and some line-length filters can look overly harsh on raw HTML because they are intended to matter **after** upstream cleanup.
- `clean_links_mapper` can be under-estimated if the fetch pipeline already removes URL strings from the text view.
- `clean_copyright_mapper` is code/comment-style and is a weaker match for already normalized report text.

So the right question is not only "what dataset should I use?", but also "in what raw form should I store it?".

## 3. D1: Web Crawl Cleanup & Filtering
### Download this kind of data
- raw HTML pages
- pages with real site templates, navigation, footer, cookie/banner blocks, repeated boilerplate
- pages where links are still present in HTML

### Best raw form
- store the original HTML response body
- keep `url`, `domain`, `source_name`, and raw `text=html`

### Why this matches the operators
- `clean_html_mapper` needs raw HTML
- `clean_links_mapper` is naturally active on HTML containing URLs and anchor targets
- `remove_repeat_sentences_mapper` becomes useful when navigation/footer text is duplicated across pages
- `whitespace_normalization_mapper` is naturally active on messy HTML-to-text projections or raw HTML with irregular whitespace

### Good public sources
- Common Crawl WARC / WET slices
- RFC Editor HTML pages
- IANA pages
- public docs / tutorial / blog pages with stable HTML templates
- Python.org / MDN / public product-doc pages in raw HTML form

### Avoid for this domain
- already cleaned webtext datasets
- DOM-cleaned text corpora such as "final plain text only"
- corpora that removed site template boilerplate before you see them

### Special note
`clean_email_mapper` and `clean_ip_mapper` should be treated as optional in web. They can appear, but should not define the domain.

## 4. D2: Knowledge Base / Support Corpus Preparation
### Download this kind of data
- documentation pages, help-center articles, FAQ pages, product manuals
- troubleshooting pages with CLI output, code snippets, config examples
- networking / security / infra docs where IPs and literal URLs are more likely to appear
- support or contact pages if you want `clean_email_mapper` to matter

### Best raw form
- prefer raw docs HTML or docs Markdown/source
- if you keep an extracted text view, preserve:
  - line breaks
  - code blocks
  - literal URLs
  - tabular residue
- do **not** over-normalize whitespace during download

### Why this matches the operators
- `clean_links_mapper` needs literal URLs to survive extraction
- `clean_email_mapper` only makes sense on contact/support-like pages
- `clean_ip_mapper` is strongest on networking/security/config docs
- `remove_table_text_mapper` needs plain-text table residue after extraction
- `remove_repeat_sentences_mapper` needs duplicated FAQ/template text
- `average_line_length_filter` works best when line structure is still preserved

### Good public sources
- Python docs
- Kubernetes docs
- MDN docs
- Hugging Face docs
- product docs with networking or deployment topics
- vendor help-center or troubleshooting docs that expose URLs, IP examples, or contact blocks

### Avoid for this domain
- text extracted with all URLs removed
- aggressively cleaned markdown/plain-text mirrors
- very short FAQ snippets with no structural residue

### Recommended subpools inside D2
- general docs/help pages
- networking/security/config pages
- support/contact pages
- docs with tables / CLI output / repeated templates

This split is important because `clean_email` and `clean_ip` are not high-coverage everywhere.

## 5. D3: Report / Policy / Compliance Document Cleanup
### Download this kind of data
- official PDFs or HTML long reports
- policy documents, government reports, filings, compliance documents, whitepapers
- multi-page documents with:
  - page headers/footers
  - front matter
  - tables
  - appendix sections
  - long broken lines from extraction

### Best raw form
- download the original PDF or original HTML
- run a **fixed** extractor yourself and keep the resulting extracted text
- preserve line breaks and page-break residue

### Why this matches the operators
- `remove_table_text_mapper` only becomes meaningful if extraction leaves whitespace-separated table residue
- `average_line_length_filter` and `maximum_line_length_filter` need broken line structure
- `remove_specific_chars_mapper` is useful when extraction introduces bullets/symbol residue
- `text_length_filter` is naturally meaningful for long-form documents

### Good public sources
- U.S. GAO report PDFs
- GovInfo publications and official PDFs
- SEC EDGAR filings
- EUR-Lex legal and regulatory documents
- public annual reports / policy whitepapers

### Avoid for this domain
- already normalized benchmark text like cleaned summarization datasets
- corpora where page structure and line breaks were flattened into one giant paragraph

### Important warning
The current `govreport` bootstrap source is useful for wiring the pipeline, but it is **too clean** for this domain's main claim. For the real benchmark, D3 should be built from your own fixed extraction outputs, not from already processed text datasets.

## 6. D4: Scientific Source Cleanup & Canonicalization
### Download this kind of data
- arXiv source tarballs
- multi-file LaTeX projects, not only PDFs
- papers with:
  - `%` comments
  - bibliography sections
  - section headers
  - user-defined macros, especially no-argument macros

### Best raw form
- store the source tarball metadata
- extract all `.tex` files
- keep:
  - main `.tex`
  - optional merged text
  - file list metadata

### Why this matches the operators
- `remove_comments_mapper` is TeX-comment specific
- `remove_bibliography_mapper` operates on bibliography commands and sections
- `remove_header_mapper` needs section/chapter-like commands
- `expand_macro_mapper` currently supports no-argument `\\newcommand` and `\\def`

### Good public sources
- arXiv source submissions
- cs.CL / cs.LG / stat.ML or similar TeX-heavy categories

### Avoid for this domain
- PDF-only corpora
- already converted plain text from papers
- LaTeX sources that have already been stripped or normalized

## 7. What to download first
For a first serious support scan, a good target is:
- D1 web: 500-1000 raw HTML pages
- D2 kb_support: 500-1000 docs pages, with at least one networking/config subpool
- D3 reports_policy: 300-500 original PDFs/HTML docs, then your own fixed extraction
- D4 scientific: 300-500 arXiv source tarballs

At this stage, coverage matters more than benchmark size.

## 8. Practical conclusion
If we align the download format with operator semantics, the domains should use:
- D1: raw HTML
- D2: raw docs HTML / Markdown plus minimally processed extracted text
- D3: original PDF/HTML plus fixed extraction outputs
- D4: raw LaTeX source

This is the cleanest path toward meaningful operator support tags and workflow construction.
