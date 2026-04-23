#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cdrbench.llm_utils import build_client, chat_completion, parse_json_response, resolve_model


JUDGE_SYSTEM_PROMPT = """You are a strict benchmark prompt judge.

You will be given:
- The internal workflow definition and filter parameters.
- The operator code/doc evidence.
- One candidate user-facing prompt.

Judge whether the prompt is faithful to what the workflow actually does.

Mandatory keep conditions:
1. Functional equivalence: the prompt requests the same transformation and filtering behavior.
2. Order correctness: the requested order matches the internal workflow order.
3. Output contract correctness: the prompt clearly asks for JSON with status and clean_text.
4. No code leakage: the prompt does not mention operator names, parameter names, YAML, Python, hidden code, or implementation internals.

Also score:
- user_naturalness: does it sound like a plausible user request?
- threshold_grounding: are thresholds/conditions expressed naturally and correctly?
- clarity: is the request clear and executable?
- format_consistency: is the output contract explicit and unambiguous?

Return JSON only:
{
  "verdict": "keep" or "reject",
  "must_pass": {
    "functional_equivalence": true,
    "order_correct": true,
    "output_contract_correct": true,
    "no_code_leakage": true
  },
  "scores": {
    "user_naturalness": 1-5,
    "threshold_grounding": 1-5,
    "clarity": 1-5,
    "format_consistency": 1-5
  },
  "issues": ["..."],
  "summary": "one short sentence"
}
"""


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


def _judge_user_prompt(entry: dict[str, Any], candidate: dict[str, Any], *, client, model: str, temperature: float) -> dict[str, Any]:
    operator_blocks = []
    for op in entry.get('operators') or []:
        operator_blocks.append(
            '\n'.join(
                [
                    f"[Operator] {op.get('name')} ({op.get('kind')})",
                    f"Parameters: {json.dumps(op.get('params') or {}, ensure_ascii=False, sort_keys=True)}",
                    "[Documentation excerpt]",
                    str(op.get('doc_excerpt') or ''),
                    "[Source code]",
                    str(op.get('code_excerpt') or ''),
                ]
            )
        )
    user_prompt = (
        f"Benchmark track: {entry.get('benchmark_track')}\n"
        f"Domain: {entry.get('domain')}\n"
        f"Workflow type: {entry.get('workflow_type')}\n"
        f"Order slot: {entry.get('order_slot')}\n"
        f"Internal operator sequence: {' -> '.join(entry.get('operator_sequence') or [])}\n"
        f"Filter params by name: {json.dumps(entry.get('filter_params_by_name') or {}, ensure_ascii=False, sort_keys=True)}\n"
        f"Threshold meta: {json.dumps(entry.get('threshold_meta') or {}, ensure_ascii=False, sort_keys=True)}\n\n"
        f"Candidate style: {candidate.get('style_id')} / {candidate.get('style_label')}\n"
        f"Candidate prompt:\n{candidate.get('user_request')}\n\n"
        "Operator evidence:\n"
        f"{chr(10).join(operator_blocks)}"
    )
    content = chat_completion(
        client=client,
        model=model,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=temperature,
    )
    payload = parse_json_response(content)
    if not isinstance(payload, dict):
        raise RuntimeError('Judge did not return a JSON object.')
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description='LLM-judge workflow-level CDR-Bench prompt candidates.')
    parser.add_argument('--prompt-library', default='data/benchmark_prompts/workflow_prompt_library.jsonl')
    parser.add_argument('--output-dir', default='data/benchmark_prompts/judged')
    parser.add_argument('--model', default=None, help='OpenAI-compatible judge model. Defaults to OPENAI_MODEL / LLM_MODEL env.')
    parser.add_argument('--base-url', default=None)
    parser.add_argument('--api-key', default=None)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--min-average-score', type=float, default=3.5)
    args = parser.parse_args()

    model = resolve_model(args.model)
    client = build_client(api_key=args.api_key, base_url=args.base_url)
    library = _read_jsonl((ROOT / args.prompt_library).resolve())
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    judged_rows = []
    accepted_rows = []
    for entry in library:
        candidates = list(entry.get('candidates') or [])
        judged_candidates = []
        accepted_candidates = []
        for candidate in candidates:
            verdict = _judge_user_prompt(entry, candidate, client=client, model=model, temperature=args.temperature)
            must_pass = verdict.get('must_pass') if isinstance(verdict.get('must_pass'), dict) else {}
            scores = verdict.get('scores') if isinstance(verdict.get('scores'), dict) else {}
            average_score = sum(float(scores.get(k, 0)) for k in ('user_naturalness', 'threshold_grounding', 'clarity', 'format_consistency')) / 4.0
            keep = (
                all(bool(must_pass.get(key)) for key in ('functional_equivalence', 'order_correct', 'output_contract_correct', 'no_code_leakage'))
                and average_score >= args.min_average_score
                and str(verdict.get('verdict')).lower() == 'keep'
            )
            judged_candidate = {
                **candidate,
                'judge': verdict,
                'judge_average_score': round(average_score, 4),
                'accepted': keep,
            }
            judged_candidates.append(judged_candidate)
            if keep:
                accepted_candidates.append(judged_candidate)

        judged_entry = {**entry, 'judge_model': model, 'judged_candidates': judged_candidates}
        accepted_entry = {**entry, 'judge_model': model, 'accepted_candidates': accepted_candidates}
        judged_rows.append(judged_entry)
        accepted_rows.append(accepted_entry)

    _write_jsonl(output_dir / 'workflow_prompt_library.judged.jsonl', judged_rows)
    _write_jsonl(output_dir / 'workflow_prompt_library.accepted.jsonl', accepted_rows)
    print(f'wrote judged prompts -> {output_dir / "workflow_prompt_library.judged.jsonl"}', flush=True)
    print(f'wrote accepted prompts -> {output_dir / "workflow_prompt_library.accepted.jsonl"}', flush=True)


if __name__ == '__main__':
    main()
