import pandas as pd
import pytest

from synthgen.schema import RulesMatcher, normalise_header
from synthgen.schema.combine import combine
from synthgen.schema.rules_matcher import ColumnInference, detect_type


@pytest.mark.parametrize(
    "raw,expected",
    [("hrg_dt", "hearing date"), ("HearingDate", "hearing date"), ("CASE-NO", "case number")],
)
def test_normalise(raw, expected):
    assert normalise_header(raw) == expected


def test_rules_matcher_synonym(dictionary):
    inf = RulesMatcher(dictionary).infer("hrg_dt", pd.Series(["2025-07-02"]))
    assert inf.concept == "hearing_date"
    assert inf.type == "date"
    assert inf.confidence == 1.0


def test_rules_matcher_unknown(dictionary):
    inf = RulesMatcher(dictionary).infer("xref_zq")
    assert inf.concept is None


@pytest.mark.parametrize(
    "values,expected",
    [(["1", "2"], "integer"), (["1.50", "2"], "decimal"), (["2025-01-01"], "date"), (["x"], "string")],
)
def test_detect_type(values, expected):
    assert detect_type(pd.Series(values)) == expected


def test_combine_flags_disagreement():
    infs = [
        ColumnInference("status", "case_status", "category", 0.9, "rules"),
        ColumnInference("status", "case_status", "category", 0.8, "llm"),
        ColumnInference("status", "payment_status", "category", 0.6, "embedding"),
    ]
    result = combine(infs, weights={"rules": 1, "llm": 1, "embedding": 1})
    assert result.concept == "case_status"
    assert result.methods_agree == 2
    assert result.needs_review


def _word_encoder(texts):
    """Stand-in for the sentence model: normalised bag-of-words vectors (hashed words).

    crc32, not hash(), so the buckets are the same in every test run.
    """
    import zlib

    import numpy as np

    out = np.zeros((len(texts), 4096))
    for i, t in enumerate(texts):
        for w in t.replace("|", " ").replace(";", " ").split():
            out[i, zlib.crc32(w.encode()) % 4096] += 1
        norm = np.linalg.norm(out[i])
        out[i] = out[i] / norm if norm else out[i]
    return out


def test_column_text_uses_normalised_header_and_five_values():
    from synthgen.schema.embedding_classifier import column_text

    values = pd.Series(["2025-07-02", None, "", "2025-08-11", "a", "b", "c", "d"])
    assert column_text("hrg_dt", values) == "hearing date | 2025-07-02; 2025-08-11; a; b; c"
    assert column_text("hrg_dt") == "hearing date"


def test_embedding_neighbours_match_synonyms_and_flag_unknowns(dictionary):
    from synthgen.schema.embedding_classifier import EmbeddingClassifier

    clf = EmbeddingClassifier(dictionary, encoder=_word_encoder)
    hearing, unknown = clf.infer_all(["hearing_dt", "xq_zz"])
    assert hearing.concept == "hearing_date" and hearing.method == "embedding"
    assert hearing.confidence > 0.9
    assert unknown.concept is None and unknown.confidence == 0.0


def test_embedding_learned_model_can_add_a_concept_neighbours_miss(dictionary):
    from synthgen.schema.embedding_classifier import NO_CONCEPT, EmbeddingClassifier

    labelled = pd.DataFrame(
        {
            "text": ["zz code | A", "zz code | B", "qq band | 1", "qq band | 2"],
            "concept": ["claim_type", "claim_type", NO_CONCEPT, NO_CONCEPT],
        }
    )
    clf = EmbeddingClassifier(dictionary, encoder=_word_encoder, learned_weight=1.0).fit(labelled)
    [result] = clf.infer_all(["zz_code"], pd.DataFrame({"zz_code": ["A"]}))
    assert result.concept == "claim_type"
    clf.learned_weight = 0.0  # neighbours only: "zz code" is nothing like the dictionary
    assert clf.infer_all(["zz_code"])[0].concept is None


def test_labelled_columns_split_by_schema():
    from pathlib import Path

    from synthgen.schema.embedding_classifier import NO_CONCEPT, load_labelled_columns

    root = Path(__file__).resolve().parents[1] / "examples" / "test_schemas"
    train = load_labelled_columns(root, {"train"})
    validation = load_labelled_columns(root, {"validation"})
    assert set(train["schema_id"]).isdisjoint(validation["schema_id"])
    assert len(train) > 100 and len(validation) > 30
    assert (train["concept"] == NO_CONCEPT).any()  # unlabelled columns become "no concept"
    assert train["text"].str.contains(r" \| ").any()  # sample values are included
