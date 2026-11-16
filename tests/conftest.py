from pathlib import Path

import pytest

from synthgen.core.models import SchemaProfile
from synthgen.schema.dictionary import ConceptDictionary

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def profile() -> SchemaProfile:
    return SchemaProfile.load(ROOT / "examples" / "civil_claims_profile.yaml")


@pytest.fixture
def dictionary() -> ConceptDictionary:
    return ConceptDictionary.load(ROOT / "config" / "concept_dictionary_extended.yaml")
