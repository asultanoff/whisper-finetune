from __future__ import annotations


def _edit_distance(left: list[str], right: list[str]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            substitution_cost = 0 if left_item == right_item else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(predictions: list[str], references: list[str]) -> float:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have identical lengths")

    total_words = 0
    total_errors = 0
    for prediction, reference in zip(predictions, references):
        prediction_words = prediction.split()
        reference_words = reference.split()
        total_words += len(reference_words)
        total_errors += _edit_distance(prediction_words, reference_words)

    if total_words == 0:
        return 0.0
    return total_errors / total_words
