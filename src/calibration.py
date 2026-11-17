"""Threshold selection utilities with false-internal and false-web rates."""
from __future__ import annotations
from typing import Iterable


def summarize_threshold(records: list[dict], threshold: float) -> dict:
    """High false-internal cost: wrong internal answers are riskier than extra web queries."""
    fp_internal = 0
    fn_web = 0
    correct = 0
    for record in records:
        predicted = "INTERNAL_RAG" if record["score"] >= threshold else "WEB_FALLBACK"
        expected = record["expected"]
        correct += int(predicted == expected)
        if expected == "WEB_FALLBACK" and predicted == "INTERNAL_RAG":
            fp_internal += 1
        if expected == "INTERNAL_RAG" and predicted == "WEB_FALLBACK":
            fn_web += 1
    external_count = sum(r["expected"] == "WEB_FALLBACK" for r in records)
    internal_count = sum(r["expected"] == "INTERNAL_RAG" for r in records)
    return {
        "threshold": round(threshold, 3),
        "accuracy": round(correct / max(len(records), 1), 4),
        "false_internal_count": fp_internal,
        "false_internal_rate": round(fp_internal / max(external_count, 1), 4),
        "false_web_count": fn_web,
        "false_web_rate": round(fn_web / max(internal_count, 1), 4),
        "weighted_error": fp_internal * 3 + fn_web,
    }


def choose_threshold(records: list[dict], thresholds: Iterable[float]) -> dict:
    if not records:
        raise ValueError("Threshold calibration requires labeled data")
    options = [summarize_threshold(records, t) for t in thresholds]
    if not options:
        raise ValueError("No candidate thresholds supplied")
    best = min(options, key=lambda item: (item["weighted_error"], -item["accuracy"], item["threshold"]))
    return {"selected": best, "grid": options}
