from __future__ import annotations

from typing import Any

import editdistance


def normalize_status(value: Any) -> str:
    text = '' if value is None else str(value)
    return text.strip().upper()


def edit_distance(left: str, right: str) -> int:
    return int(editdistance.eval(left, right))


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

    normalized_reference_status = normalize_status(reference_status)
    normalized_prediction_status = normalize_status(predicted_status)

    status_match = normalized_prediction_status == normalized_reference_status
    text_exact_match = raw_prediction == raw_reference
    recipe_success = status_match and text_exact_match

    d_input = edit_distance(raw_input, raw_reference)
    d_pred = edit_distance(raw_prediction, raw_reference)
    distance_total = d_input + d_pred
    if distance_total == 0:
        refinement_gain = 1.0
    else:
        refinement_gain = (d_input - d_pred) / distance_total

    return {
        'normalized_reference_status': normalized_reference_status,
        'normalized_predicted_status': normalized_prediction_status,
        'status_match': status_match,
        'text_exact_match': text_exact_match,
        'recipe_success': recipe_success,
        'text_match': text_exact_match,
        'edit_distance_input_to_reference': d_input,
        'edit_distance_prediction_to_reference': d_pred,
        'refinement_gain': refinement_gain,
    }
