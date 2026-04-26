#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cdrbench.llm_utils import build_client, chat_completion, parse_json_response, resolve_model


TRACK_FILES = {
    'main': 'main.jsonl',
    'order_sensitivity': 'order_sensitivity.jsonl',
    'atomic_ops': 'atomic_ops.jsonl',
}

SKIPPED_OPERATORS = {'flagged_words_filter', 'stopwords_filter'}

FILTER_THRESHOLD_EXPRESSION_GUIDE = {
    'alphanumeric_filter': 'Say this as a minimum share of alphanumeric content, e.g., "keep only text where at least 60% of the characters are letters or numbers."',
    'average_line_length_filter': 'Say this as an average line-length limit, e.g., "keep only documents whose average line length is no more than 80 characters."',
    'character_repetition_filter': 'Say this as a maximum repeated-character ratio, e.g., "reject text where repeated character patterns exceed 20%."',
    'flagged_words_filter': 'Say this as a maximum flagged-word share, e.g., "keep only text where flagged terms make up less than 0.1%."',
    'maximum_line_length_filter': 'Say this as a longest-line limit, e.g., "reject samples with any line longer than 200 characters."',
    'special_characters_filter': 'Say this as a maximum special-character share, e.g., "keep only text where special characters are below 15%."',
    'stopwords_filter': 'Say this as a minimum natural-language stopword share, e.g., "keep only text with at least 5% common function words."',
    'text_length_filter': 'Say this as a character-count constraint, e.g., "keep only text with at least 100 characters."',
    'word_repetition_filter': 'Say this as a maximum repeated-word ratio, e.g., "reject text where repeated words exceed 20%."',
    'words_num_filter': 'Say this as a word-count constraint, e.g., "keep only documents with at least 50 words."',
}

STYLE_PRESETS = [
    {
        'style_id': 'imperative_checklist',
        'label': 'Imperative Checklist',
        'definition': 'A direct command that lists the required operations explicitly and in order.',
        'template': 'Please perform the following operations on the text: {requirement}.',
        'example': 'Please clean this web page by removing the HTML, stripping links, normalizing the whitespace, and then keep it only if the cleaned text has at least 100 characters.',
        'guidance': 'Write a direct step-by-step request with explicit sequencing, like a user telling an assistant exactly what to do.',
    },
    {
        'style_id': 'goal_oriented',
        'label': 'Goal-Oriented Description',
        'definition': 'A prose description that emphasizes the intended final state rather than a numbered procedure.',
        'template': 'The goal is to make the data suitable for {use_case}; it should end up with {requirement}.',
        'example': 'The goal is to make these help documents clean enough for a support index, with links removed, repeated template sentences cleaned up, spacing made consistent, and only documents with no more than 20% repeated words retained.',
        'guidance': 'Describe the goal and desired cleanup outcome in fluent prose without sounding like code or a numbered recipe.',
    },
    {
        'style_id': 'application_context',
        'label': 'Application-Context Task',
        'definition': 'A task framed around a downstream use case such as retrieval, indexing, release, compliance, or corpus construction.',
        'template': 'For {use_case}, process the following data so that {requirement}.',
        'example': 'For downstream retrieval, process these reports so that disclaimers, table residue, and abnormal long lines are removed, then keep only reports whose longest remaining line is no more than 200 characters.',
        'guidance': 'Frame the request around a realistic downstream use case such as retrieval, indexing, release, or corpus preparation.',
    },
    {
        'style_id': 'qa_request',
        'label': 'Quality-Control Request',
        'definition': 'A request that sounds like data quality screening, emphasizing retention criteria and rejection conditions.',
        'template': 'Quality-check this sample: {requirement}. Keep it only if it still satisfies the quality condition.',
        'example': 'Quality-check this page after cleaning away links and noisy HTML. Keep it only if the remaining text has at least 50 words and looks usable as corpus content.',
        'guidance': 'Phrase it like a quality-control request that focuses on what should be retained or rejected and why.',
    },
    {
        'style_id': 'analyst_handoff',
        'label': 'Analyst Handoff',
        'definition': 'A natural teammate-to-teammate handoff written in practical workplace language.',
        'template': 'Could you take this batch and {requirement}?',
        'example': 'Could you take these LaTeX sources, remove the comments and bibliography, expand the simple macros, and then keep only sources whose final text length is at least 1,000 characters?',
        'guidance': 'Phrase it like one teammate handing a dataset-cleaning request to another teammate in normal workplace language.',
    },
    {
        'style_id': 'concise_brief',
        'label': 'Concise Brief',
        'definition': 'A short, compact instruction that keeps all necessary behavior while minimizing wording.',
        'template': '{requirement}. Return the final keep/drop decision and cleaned text.',
        'example': 'Remove private identifiers and unsafe links, normalize the remaining text, then keep only sanitized samples whose special-character share is below 15%.',
        'guidance': 'Write a compact but complete user request with minimal fluff, while still preserving all important behavior.',
    },
    {
        'style_id': 'policy_like',
        'label': 'Policy-Like Requirement',
        'definition': 'A formal processing requirement that reads like a data handling policy, not code.',
        'template': 'Before this data can be used, it must satisfy the following processing requirement: {requirement}.',
        'example': 'Before release, the text must not contain emails, IP addresses, file paths, credentials, or long secret-like tokens; after sanitization, retain it only if the result still has at least 100 words.',
        'guidance': 'Write it like a processing requirement or policy note, but still from a user-facing perspective rather than code.',
    },
    {
        'style_id': 'workflow_narrative',
        'label': 'Workflow Narrative',
        'definition': 'A scenario-style request that first explains the messy data situation and then asks for the needed processing.',
        'template': 'I have {data_situation}. Please {requirement}.',
        'example': 'I have raw crawl pages with markup, navigation links, and messy whitespace. Please turn them into clean readable text and keep only pages whose alphanumeric content makes up at least 60% of the final text.',
        'guidance': 'Describe the data situation first, then explain the requested cleanup and filtering behavior as a realistic need.',
    },
    {
        'style_id': 'end_weighted_instruction',
        'label': 'End-Weighted Instruction',
        'definition': 'The raw data appears before the concrete instruction, so the task request is weighted toward the end of the prompt.',
        'template': '[Data first] ... Above is the raw data. Please apply this processing request: {requirement}.',
        'example': 'Above is the raw document. Please remove links and contact information, normalize spacing, and then keep the document only if its final text has at least 500 characters.',
        'guidance': 'Write a requirement intended to be placed after the raw data. Make the instruction self-contained and attention-grabbing at the end.',
    },
    {
        'style_id': 'negative_constraint_driven',
        'label': 'Negative-Constraint Driven',
        'definition': 'A request centered on what must not remain in the output, useful for cleaning and filtering workflows.',
        'template': 'When processing this text, make sure the final result does not contain {noise_list}; also {requirement}.',
        'example': 'When processing this report, make sure the final result contains no disclaimers, table residue, or lines longer than 180 characters. After that, judge whether it is suitable for retrieval.',
        'guidance': 'Emphasize unwanted content that should be absent from the final output, while preserving the actual workflow order.',
    },
    {
        'style_id': 'conversational_cooperative',
        'label': 'Conversational Cooperative',
        'definition': 'A casual chat-style request that sounds like a real user asking for help, while still being complete enough to execute.',
        'template': 'Hey, could you help me with this text? I need you to {requirement}.',
        'example': 'Hey, could you help me clean up these source files? I need the LaTeX comments and references removed, the macros expanded, and then please keep only sources with at least 300 words left.',
        'guidance': 'Use natural conversational wording, but do not omit key steps, order, or filtering behavior.',
    },
]

GENERATION_SYSTEM_PROMPT = """You are an expert at understanding data processing code and operator documentation, then translating the behavior into realistic user-facing data refinement requests.

You will be given:
1. The benchmark track and domain.
2. The internal workflow sequence and filter parameters.
3. Source code and documentation snippets for the operators.
4. A list of desired prompt-style variants.

Your task:
- Generate multiple diverse natural-language user requests that are FUNCTIONALLY EQUIVALENT to the workflow.
- Pretend the user has never seen the code.
- Never mention operator names, parameter names, class names, file names, YAML, Python, or implementation details.
- Preserve the exact workflow order and all essential filter semantics.
- If a workflow contains any filter/keep-drop step, you MUST express every active threshold naturally and explicitly in the user requirement.
- Express thresholds as user-style constraints such as "at least 100 characters", "no more than 200 characters per line", "below 15%", or "at least 50 words".
- Do NOT use parameter names such as min_len, max_len, min_ratio, max_ratio, min_num, max_num, or internal statistic names.
- Make the prompts stylistically diverse and realistic across different users.

The generated text should be ONLY the user requirement body.
Do not include raw data placeholders, JSON schema instructions, answer format instructions, or system/developer wording. The benchmark wrapper will add those later.

Hard requirements for every candidate requirement:
- The requested operation order must be correct.
- Filtering behavior must be correct.
- Any numeric threshold or ratio required by a filter must be explicitly grounded in natural language.
- Do not ask clarifying questions.
- Do not refer to the benchmark, hidden reference, or code.

Return JSON only, with this exact schema:
{
  "candidates": [
    {
      "style_id": "...",
      "style_label": "...",
      "user_requirement": "...",
      "style_notes": "short note on why this wording is stylistically distinct"
    }
  ]
}
"""

SYSTEM_PROMPT = """You are a careful data refinement engine. Follow the user's data-refinement request exactly and in order. Return only the required JSON object. Do not explain your reasoning."""

OUTPUT_CONTRACT = """Output format:
Return exactly one valid JSON object and nothing else.
Use one of these two forms:
{"status":"KEEP","clean_text":"..."}
{"status":"DROP","clean_text":"..."}

Rules:
- If the requested filtering/quality check rejects the current text, return DROP and set clean_text to the current text at the point where it was rejected.
- If the sample is kept, return KEEP and put the final refined text in clean_text.
- Do not include Markdown fences, comments, explanations, or extra keys."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    count = 0
    with tmp_path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
            count += 1
    tmp_path.replace(path)
    return count


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        payload = yaml.safe_load(f)
    return payload if isinstance(payload, dict) else {}


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _stable_id(*parts: Any, length: int = 16) -> str:
    blob = '||'.join(_stable_json(part) if isinstance(part, (dict, list)) else str(part) for part in parts)
    return hashlib.sha1(blob.encode('utf-8')).hexdigest()[:length]


def _find_operator_file(op_name: str, kind: str) -> Path | None:
    path = ROOT / 'data-juicer' / 'data_juicer' / 'ops' / kind / f'{op_name}.py'
    return path if path.exists() else None


def _find_operator_doc(op_name: str, kind: str) -> Path | None:
    path = ROOT / 'data-juicer' / 'docs' / 'operators' / kind / f'{op_name}.md'
    return path if path.exists() else None


def _trim_doc_text(text: str) -> str:
    markers = [
        '\n## 📊 Effect demonstration',
        '\n## Effect demonstration',
        '\n## 🔗 related links',
        '\n## Related links',
    ]
    trimmed = text
    for marker in markers:
        idx = trimmed.find(marker)
        if idx != -1:
            trimmed = trimmed[:idx]
    return trimmed.strip()


def _load_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ''
    return path.read_text(encoding='utf-8')


def _workflow_key(row: dict[str, Any]) -> str:
    operator_sequence = list(row.get('operator_sequence') or ([row['operator']] if row.get('operator') else []))
    return _stable_id(
        row.get('benchmark_track'),
        row.get('domain'),
        row.get('workflow_type'),
        row.get('order_slot'),
        operator_sequence,
        row.get('filter_params_by_name') or {},
    )


def _should_skip_row(row: dict[str, Any], skipped_ops: set[str]) -> bool:
    operator_sequence = list(row.get('operator_sequence') or ([row['operator']] if row.get('operator') else []))
    return any(op in skipped_ops for op in operator_sequence)


def _operator_kind(op_name: str) -> str:
    return 'filter' if op_name.endswith('_filter') else 'mapper'


def _group_rows(rows: list[dict[str, Any]], skipped_ops: set[str]) -> tuple[dict[str, list[dict[str, Any]]], int]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    skipped = 0
    for row in rows:
        if _should_skip_row(row, skipped_ops):
            skipped += 1
            continue
        key = _workflow_key(row)
        grouped.setdefault(key, []).append(row)
    return grouped, skipped


def _style_subset(count: int) -> list[dict[str, str]]:
    return STYLE_PRESETS[:count]


def _format_style_for_llm(style: dict[str, str]) -> str:
    return (
        f"- style_id: {style['style_id']}\n"
        f"  label: {style['label']}\n"
        f"  definition: {style['definition']}\n"
        f"  template: {style['template']}\n"
        f"  example: {style['example']}\n"
        f"  generation guidance: {style['guidance']}"
    )


def _workflow_bundle(workflow_key: str, rows: list[dict[str, Any]], prompt_cfg: dict[str, Any], variants_per_workflow: int) -> dict[str, Any]:
    row = rows[0]
    operator_sequence = list(row.get('operator_sequence') or ([row['operator']] if row.get('operator') else []))
    filter_params_by_name = row.get('filter_params_by_name') if isinstance(row.get('filter_params_by_name'), dict) else {}
    operators = []
    for op_name in operator_sequence:
        kind = _operator_kind(op_name)
        op_cfg = dict((prompt_cfg.get('operators') or {}).get(op_name) or {})
        filter_cfg = dict((prompt_cfg.get('filters') or {}).get(op_name) or {})
        code_path = _find_operator_file(op_name, kind)
        doc_path = _find_operator_doc(op_name, kind)
        operators.append(
            {
                'name': op_name,
                'kind': kind,
                'params': filter_params_by_name.get(op_name) if isinstance(filter_params_by_name.get(op_name), dict) else {},
                'render_hint': op_cfg.get('natural_language_intent') or filter_cfg.get('natural_language_intent'),
                'threshold_expression_guide': FILTER_THRESHOLD_EXPRESSION_GUIDE.get(op_name) if kind == 'filter' else None,
                'code_path': str(code_path) if code_path else None,
                'code_excerpt': _load_text(code_path),
                'doc_path': str(doc_path) if doc_path else None,
                'doc_excerpt': _trim_doc_text(_load_text(doc_path)),
            }
        )
    return {
        'workflow_prompt_key': workflow_key,
        'benchmark_track': row.get('benchmark_track'),
        'domain': row.get('domain'),
        'workflow_type': row.get('workflow_type'),
        'order_slot': row.get('order_slot'),
        'operator_sequence': operator_sequence,
        'filter_params_by_name': filter_params_by_name,
        'threshold_meta': row.get('threshold_meta') or {},
        'num_instances': len(rows),
        'style_requests': _style_subset(variants_per_workflow),
        'operators': operators,
    }


def _generation_user_prompt(bundle: dict[str, Any]) -> str:
    style_lines = [_format_style_for_llm(style) for style in bundle['style_requests']]
    op_blocks = []
    for op in bundle['operators']:
        params_json = json.dumps(op['params'], ensure_ascii=False, sort_keys=True)
        op_blocks.append(
            '\n'.join(
                [
                    f"[Operator] {op['name']} ({op['kind']})",
                    f"Parameters: {params_json}",
                    f"Human hint: {op.get('render_hint') or ''}",
                    f"Threshold expression guide: {op.get('threshold_expression_guide') or ''}",
                    "[Documentation excerpt]",
                    op['doc_excerpt'] or '(no operator doc excerpt found)',
                    "[Source code]",
                    op['code_excerpt'] or '(no operator source found)',
                ]
            )
        )

    return (
        f"Benchmark track: {bundle['benchmark_track']}\n"
        f"Domain: {bundle['domain']}\n"
        f"Workflow type: {bundle.get('workflow_type')}\n"
        f"Order slot: {bundle.get('order_slot')}\n"
        f"Internal operator sequence: {' -> '.join(bundle['operator_sequence'])}\n"
        f"Filter params by name: {json.dumps(bundle['filter_params_by_name'], ensure_ascii=False, sort_keys=True)}\n"
        f"Threshold meta: {json.dumps(bundle['threshold_meta'], ensure_ascii=False, sort_keys=True)}\n\n"
        "Generate one candidate user requirement body for each requested style below.\n"
        "Use the style definition, template, and example only as guidance; do not copy the example literally unless the workflow matches it.\n\n"
        f"{chr(10).join(style_lines)}\n\n"
        "Important: act as if the user has never seen code. Do not mention operator names, parameter names, class names, YAML, file paths, JSON schema, or implementation-specific terms.\n"
        "Return only the user requirement body for each candidate. The final benchmark prompt wrapper and output format will be added by code.\n\n"
        "Operator evidence:\n\n"
        f"{chr(10).join(op_blocks)}"
    )


def _cache_key(bundle: dict[str, Any], model: str) -> str:
    return _stable_id(model, bundle['workflow_prompt_key'], bundle['style_requests'], bundle['operator_sequence'], bundle['filter_params_by_name'])


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache = {}
    for row in _read_jsonl(path):
        key = row.get('cache_key')
        if isinstance(key, str):
            cache[key] = row
    return cache


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def _call_llm_for_candidates(
    *,
    bundle: dict[str, Any],
    client,
    model: str,
    temperature: float,
) -> list[dict[str, Any]]:
    content = chat_completion(
        client=client,
        model=model,
        system_prompt=GENERATION_SYSTEM_PROMPT,
        user_prompt=_generation_user_prompt(bundle),
        temperature=temperature,
    )
    payload = parse_json_response(content)
    if not isinstance(payload, dict) or not isinstance(payload.get('candidates'), list):
        raise RuntimeError('LLM did not return the expected JSON object with a candidates list.')
    candidates = []
    seen = set()
    for item in payload['candidates']:
        if not isinstance(item, dict):
            continue
        style_id = str(item.get('style_id') or '')
        user_requirement = str(item.get('user_requirement') or item.get('user_request') or '').strip()
        if not style_id or not user_requirement or user_requirement in seen:
            continue
        seen.add(user_requirement)
        candidates.append(
            {
                'style_id': style_id,
                'style_label': str(item.get('style_label') or style_id),
                'style_notes': str(item.get('style_notes') or ''),
                'user_requirement': user_requirement,
            }
        )
    return candidates


def _build_user_prompt(candidate: dict[str, Any], input_text: str) -> str:
    return (
        f"{candidate['user_requirement']}\n\n"
        f"{OUTPUT_CONTRACT}\n\n"
        "Raw input text:\n"
        "<<<CDR_INPUT\n"
        f"{input_text}\n"
        "CDR_INPUT>>>"
    )


def _prompt_variant(candidate: dict[str, Any], input_text: str) -> dict[str, Any]:
    user_prompt = _build_user_prompt(candidate, input_text)
    return {
        **candidate,
        'user_prompt': user_prompt,
        'system_prompt': SYSTEM_PROMPT,
        'prompt': SYSTEM_PROMPT + '\n\n' + user_prompt,
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt},
        ],
        'expected_response_format': {
            'type': 'json_object',
            'schema_hint': '{"status":"KEEP","clean_text":"..."} or {"status":"DROP","clean_text":"..."}',
        },
    }


def _template_candidates(bundle: dict[str, Any], prompt_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    domain_contexts = prompt_cfg.get('domain_contexts') or {}
    scenario = str(domain_contexts.get(str(bundle.get('domain'))) or 'Perform data refinement.')
    operator_lines = []
    operators_cfg = prompt_cfg.get('operators') or {}
    filters_cfg = prompt_cfg.get('filters') or {}
    for op_name in bundle['operator_sequence']:
        if op_name.endswith('_filter'):
            hint = (filters_cfg.get(op_name) or {}).get('natural_language_intent') or f'Apply {op_name}.'
        else:
            hint = (operators_cfg.get(op_name) or {}).get('natural_language_intent') or f'Apply {op_name}.'
        operator_lines.append(hint)
    joined = ' Then '.join(operator_lines)
    candidates = []
    for style in bundle['style_requests']:
        if style['style_id'] == 'imperative_checklist':
            text = f'{scenario} Please execute these steps in order: {joined}.'
        elif style['style_id'] == 'goal_oriented':
            text = f'The goal is to refine this text for {bundle.get("domain")} use. {joined}.'
        elif style['style_id'] == 'application_context':
            text = f'For downstream processing, please clean this data carefully. {joined}.'
        else:
            text = f'{scenario} {joined}.'
        candidates.append(
            {
                'style_id': style['style_id'],
                'style_label': style['label'],
                'style_notes': 'template fallback',
                'user_requirement': text,
            }
        )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate diverse workflow-level prompt candidates for CDR-Bench benchmark instances.')
    parser.add_argument('--benchmark-dir', default='data/benchmark')
    parser.add_argument('--output-dir', default='data/benchmark_prompts')
    parser.add_argument('--prompt-config', default='configs/workflow_prompting.yaml')
    parser.add_argument('--tracks', nargs='*', default=list(TRACK_FILES), choices=sorted(TRACK_FILES))
    parser.add_argument('--prompt-source', choices=['llm', 'template'], default='llm')
    parser.add_argument('--model', default=None, help='OpenAI-compatible model name. Defaults to OPENAI_MODEL / LLM_MODEL env.')
    parser.add_argument('--base-url', default=None, help='OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL / LLM_BASE_URL env.')
    parser.add_argument('--api-key', default=None, help='API key. Defaults to OPENAI_API_KEY / DASHSCOPE_API_KEY env.')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--variants-per-workflow', type=int, default=6)
    parser.add_argument(
        '--skip-operators',
        nargs='*',
        default=sorted(SKIPPED_OPERATORS),
        help='Skip workflows containing these operators when generating prompt candidates.',
    )
    parser.add_argument(
        '--cache-path',
        default='data/benchmark_prompts/llm_prompt_cache.jsonl',
        help='Cache file for workflow-level LLM prompt generation.',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Reuse workflow-level prompt cache from --cache-path so interrupted runs can continue without regenerating finished workflows.',
    )
    args = parser.parse_args()

    benchmark_dir = (ROOT / args.benchmark_dir).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_cfg = _load_yaml((ROOT / args.prompt_config).resolve())

    model = resolve_model(args.model)
    client = build_client(api_key=args.api_key, base_url=args.base_url) if args.prompt_source == 'llm' else None
    cache_path = (ROOT / args.cache_path).resolve()
    if args.resume:
        cache = _load_cache(cache_path)
        print(f'resume enabled: loaded {len(cache)} cached workflow prompt entries from {cache_path}', flush=True)
    else:
        cache = {}
        if cache_path.exists():
            cache_path.unlink()
            print(f'resume disabled: cleared existing prompt cache at {cache_path}', flush=True)

    prompt_library_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    skipped_ops = set(args.skip_operators)

    for track in args.tracks:
        input_path = benchmark_dir / TRACK_FILES[track]
        if not input_path.exists():
            print(f'skip missing {track}: {input_path}', flush=True)
            continue
        rows = _read_jsonl(input_path)
        grouped, skipped_count = _group_rows(rows, skipped_ops)
        track_output_rows = []
        for workflow_key, workflow_rows in grouped.items():
            bundle = _workflow_bundle(workflow_key, workflow_rows, prompt_cfg, args.variants_per_workflow)
            cache_key = _cache_key(bundle, model)
            cached = cache.get(cache_key)
            if cached is not None:
                candidates = list(cached.get('candidates') or [])
                source = 'cache'
            else:
                candidates = (
                    _call_llm_for_candidates(bundle=bundle, client=client, model=model, temperature=args.temperature)
                    if args.prompt_source == 'llm'
                    else _template_candidates(bundle, prompt_cfg)
                )
                cache_row = {
                    'cache_key': cache_key,
                    'workflow_prompt_key': workflow_key,
                    'model': model,
                    'prompt_source': args.prompt_source,
                    'candidates': candidates,
                }
                _append_jsonl(cache_path, cache_row)
                cache[cache_key] = cache_row
                source = args.prompt_source

            library_row = {
                **bundle,
                'candidate_count': len(candidates),
                'prompt_source': source,
                'model': model,
                'candidates': candidates,
            }
            prompt_library_rows.append(library_row)

            for row in workflow_rows:
                prompt_variants = [_prompt_variant(candidate, str(row.get('input_text', ''))) for candidate in candidates]
                track_output_rows.append(
                    {
                        **row,
                        'workflow_prompt_key': workflow_key,
                        'prompt_candidate_count': len(candidates),
                        'prompt_variants': prompt_variants,
                    }
                )

        count = _write_jsonl(output_dir / TRACK_FILES[track], track_output_rows)
        benchmark_rows.extend(track_output_rows)
        summary_rows.append(
            {
                'track': track,
                'input_rows': len(rows),
                'kept_rows': count,
                'skipped_rows': skipped_count,
                'workflow_count': len(grouped),
            }
        )
        print(f'wrote track mapping {track}: {count} rows -> {output_dir / TRACK_FILES[track]}', flush=True)

    _write_jsonl(output_dir / 'workflow_prompt_library.jsonl', prompt_library_rows)
    _write_jsonl(output_dir / 'prompt_generation_summary.jsonl', summary_rows)
    print(f'wrote workflow prompt library -> {output_dir / "workflow_prompt_library.jsonl"}', flush=True)
    print(f'wrote summary -> {output_dir / "prompt_generation_summary.jsonl"}', flush=True)


if __name__ == '__main__':
    main()
