from pathlib import Path

import pytest

from synthgen.core.models import SchemaProfile
from synthgen.schema.dictionary import ConceptDictionary

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def profile() -> SchemaProfile:
    return SchemaProfile.load(ROOT / "examples" / "civil_claims_profile.yaml")


@pytest.fixture
def data_dir() -> Path:
    """The built reference data (data/ is never committed): skip when it isn't there."""
    data = ROOT / "data"
    needed = [
        data / "reference" / "reference.csv",
        data / "calibration" / "calibration_stats.yaml",
        data / "evaluation" / "held_back_stats.yaml",
    ]
    if not all(p.exists() for p in needed):
        pytest.skip("Reference data not built (run make reference)")
    return data


@pytest.fixture
def dictionary() -> ConceptDictionary:
    return ConceptDictionary.load(ROOT / "config" / "concept_dictionary_extended.yaml")
