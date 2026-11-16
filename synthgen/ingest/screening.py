"""Personal data screening for uploaded example values.

A first-pass, pattern-based check for values that look like real personal data. It is
deliberately cautious: anything flagged must be confirmed as fictional or removed.
Name detection (for example Microsoft Presidio) is out of scope; see docs/decisions.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "uk_phone": re.compile(r"(?:\+44\s?7\d{3}|\b07\d{3})\s?\d{3}\s?\d{3}\b"),
    # Deliberately broad (any two letters) so example-style numbers are caught too
    "national_insurance": re.compile(
        r"\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b", re.IGNORECASE
    ),
    "uk_postcode": re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b", re.IGNORECASE),
}


@dataclass
class ScreeningFinding:
    column: str
    row: int
    kind: str


def screen_values(sample: pd.DataFrame) -> list[ScreeningFinding]:
    """Return one finding per suspicious value. An empty list means the check passed."""
    findings: list[ScreeningFinding] = []
    for column in sample.columns:
        for row, value in sample[column].items():
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            text = str(value)
            for kind, pattern in PATTERNS.items():
                if pattern.search(text):
                    findings.append(ScreeningFinding(column=column, row=int(row), kind=kind))
    return findings
