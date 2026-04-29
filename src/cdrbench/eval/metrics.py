from __future__ import annotations

import re
import unicodedata
from typing import Any


_WHITESPACE_RE = re.compile(r'\s+')
_ZERO_WIDTH_RE = re.compile(r'[\u200b\u200c\u200d\ufeff]')


def canonicalize_text(value: Any) -> str:
    text = '' if value is None else str(value)
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = _ZERO_WIDTH_RE.sub('', text)
    text = text.strip()
    text = _WHITESPACE_RE.sub(' ', text)
    return text


def normalize_status(value: Any) -> str:
    text = '' if value is None else str(value)
    return text.strip().upper()


def edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    if len(left) < len(right):
        left, right = right, left

    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insert_cost = current[right_index - 1] + 1
            delete_cost = previous[right_index] + 1
            replace_cost = previous[right_index - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def compute_recipe_metrics(
    *,
    input_text: Any,
    reference_status: Any,
    reference_text: Any,
    predicted_status: Any,
    predicted_clean_text: Any,
) -> dict[str, Any]:
    raw_input = '' if input_text is None else str(input_text)
    raw_reference = '' if reference_text is None else str(reference_text)
    raw_prediction = '' if predicted_clean_text is None else str(predicted_clean_text)

    canonical_input = canonicalize_text(raw_input)
    canonical_reference = canonicalize_text(raw_reference)
    canonical_prediction = canonicalize_text(raw_prediction)
    normalized_reference_status = normalize_status(reference_status)
    normalized_prediction_status = normalize_status(predicted_status)

    status_match = normalized_prediction_status == normalized_reference_status
    text_exact_match = raw_prediction == raw_reference
    text_canonical_match = canonical_prediction == canonical_reference
    recipe_success = status_match and text_exact_match

    d_input = edit_distance(raw_input, raw_reference)
    d_pred = edit_distance(raw_prediction, raw_reference)
    if d_input == 0:
        refinement_gain = 1.0 if d_pred == 0 else 0.0
    else:
        refinement_gain = 1.0 - (d_pred / d_input)

    return {
        'raw_input_text': raw_input,
        'raw_reference_text': raw_reference,
        'raw_predicted_clean_text': raw_prediction,
        'canonical_input_text': canonical_input,
        'canonical_reference_text': canonical_reference,
        'canonical_predicted_clean_text': canonical_prediction,
        'normalized_reference_status': normalized_reference_status,
        'normalized_predicted_status': normalized_prediction_status,
        'status_match': status_match,
        'text_exact_match': text_exact_match,
        'text_canonical_match': text_canonical_match,
        'recipe_success': recipe_success,
        'text_match': text_exact_match,
        'edit_distance_input_to_reference': d_input,
        'edit_distance_prediction_to_reference': d_pred,
        'refinement_gain': refinement_gain,
    }
