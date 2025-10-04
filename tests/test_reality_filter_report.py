from __future__ import annotations

import pytest

import scripts.reality_filter_report as report


def test_validate_eval_parity_accepts_matching_confusion() -> None:
    summary = {
        "metrics": {"total": 2},
        "confusion_matrix": {
            "SUPPORTED": {"SUPPORTED": 1, "UNVERIFIABLE": 0},
            "REFUTED": {"REFUTED": 1, "SUPPORTED": 0},
        },
    }
    detail = [
        {"expected": "SUPPORTED", "predicted": "SUPPORTED"},
        {"expected": "REFUTED", "predicted": "REFUTED"},
    ]

    report.validate_eval_parity(summary, detail)


def test_validate_eval_parity_raises_on_mismatch() -> None:
    summary = {
        "metrics": {"total": 2},
        "confusion_matrix": {"SUPPORTED": {"SUPPORTED": 2}},
    }
    detail = [
        {"expected": "SUPPORTED", "predicted": "SUPPORTED"},
        {"expected": "SUPPORTED", "predicted": "UNVERIFIABLE"},
    ]

    with pytest.raises(ValueError):
        report.validate_eval_parity(summary, detail)
