import pandas as pd
import pytest

from synthgen.core.models import SchemaProfile
from synthgen.core.profile_manager import (
    build_column,
    coerce_sample,
    next_version,
    profile_id_from_filename,
    save_profile,
)


@pytest.fixture
def sample():
    return pd.DataFrame(
        {
            "claim_value": ["1250.00", "8900.5"],
            "hearing_date": ["2025-07-02", None],
            "status": ["Listed", "Defended"],
            "court": ["Cardiff", "Manchester"],
        }
    )


def test_build_column_uses_dictionary_and_sample_shape(dictionary, sample):
    col = build_column("claim_value", "claim_amount", dictionary, sample, {"rules": 1.0})
    assert col.type == "decimal"
    assert col.decimals == 2
    assert not col.nullable
    assert col.inferred_by == {"rules": 1.0}
    assert build_column("hearing_date", "hearing_date", dictionary, sample).nullable


def test_build_column_never_copies_example_values(dictionary, sample):
    # court_name has no dictionary list, so the example courts must not enter the profile
    col = build_column("court", "court_name", dictionary, sample)
    assert col.allowed_values is None
    status = build_column("status", "case_status", dictionary, sample)
    assert status.allowed_values == dictionary.get("case_status").allowed_values


def test_unknown_column_uses_detected_type(dictionary, sample):
    col = build_column("claim_value", None, dictionary, sample)
    assert col.concept is None
    assert col.type == "decimal"


def test_coerce_sample_converts_types(dictionary, sample):
    cols = [
        build_column(c, k, dictionary, sample)
        for c, k in [("claim_value", "claim_amount"), ("hearing_date", "hearing_date")]
    ]
    typed = coerce_sample(sample, cols)
    assert typed["claim_value"].dtype.kind == "f"
    assert typed["hearing_date"].dtype.kind == "M"
    assert sample["claim_value"].iloc[0] == "1250.00"  # original untouched


def test_profile_versions_and_change_log(tmp_path, profile):
    assert profile_id_from_filename("Civil Claims_schema.csv") == "civil-claims-schema"
    assert next_version(tmp_path, profile.profile_id) == 1
    path = save_profile(profile, [{"item": "rule:R1", "action": "approved"}], tmp_path)
    assert SchemaProfile.load(path).profile_id == profile.profile_id
    assert (path.parent / "v1.changes.yaml").exists()
    assert next_version(tmp_path, profile.profile_id) == 2
    with pytest.raises(FileExistsError):
        save_profile(profile, [], tmp_path)


def test_coerce_sample_reads_iso_and_uk_dates(dictionary):
    sample = pd.DataFrame({"d": ["2025-07-02", "03/04/2025", None]})
    col = build_column("d", "hearing_date", dictionary, sample)
    typed = coerce_sample(sample, [col])["d"]
    assert typed[0] == pd.Timestamp("2025-07-02")
    assert typed[1] == pd.Timestamp("2025-04-03")
    assert pd.isna(typed[2])
