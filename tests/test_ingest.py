from pathlib import Path

import pandas as pd
import pytest

from synthgen.ingest import load_csv, screen_values

ROOT = Path(__file__).resolve().parents[1]


def test_load_example():
    schema = load_csv((ROOT / "examples" / "civil_claims_schema.csv").read_bytes())
    assert schema.headers[0] == "case_number"
    assert len(schema.sample) == 2


def test_headers_only():
    schema = load_csv("a,b,c\n")
    assert schema.headers == ["a", "b", "c"]
    assert not schema.has_rows


def test_duplicate_headers_rejected():
    with pytest.raises(ValueError):
        load_csv("a,a\n1,2\n")


def test_example_passes_screening():
    schema = load_csv((ROOT / "examples" / "civil_claims_schema.csv").read_bytes())
    assert screen_values(schema.sample) == []


@pytest.mark.parametrize(
    "value,kind",
    [
        ("jo.bloggs@example.com", "email"),
        ("07700 900123", "uk_phone"),
        ("QQ 12 34 56 C", "national_insurance"),
        ("SW1A 1AA", "uk_postcode"),
    ],
)
def test_screening_flags(value, kind):
    findings = screen_values(pd.DataFrame({"x": [value]}))
    assert kind in {f.kind for f in findings}
