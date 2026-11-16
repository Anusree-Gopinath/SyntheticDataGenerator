"""Method 2 of 3: embedding classifier for column understanding.

Input text: the normalised header plus up to five sample values, for example
``hearing date | 2025-07-02; 2025-08-11``. Encoded locally with sentence-transformers,
then classified two ways:

- k-nearest neighbours: the normalised header against every dictionary concept name and
  synonym. Works with no training data.
- Logistic regression on the header-and-values text, trained on labelled columns from the
  test schemas (``fit``). Training and testing are split by schema, never by column, so no
  schema is seen in both. Columns labelled with no concept are learned as "no concept".

Without ``fit`` the classifier uses neighbours only; after ``fit`` it takes a weighted
average of the two (``learned_weight``).

Requires the optional extra: ``pip install -e ".[embeddings]"``. The model runs on this
machine; its weights are downloaded once from Hugging Face and cached.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from synthgen.schema.dictionary import ConceptDictionary
from synthgen.schema.normalise import normalise_header
from synthgen.schema.rules_matcher import ColumnInference

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NO_CONCEPT = "__none__"
K = 5
# Below this cosine similarity to every dictionary entry, neighbours vote "no concept"
MIN_SIMILARITY = 0.55
MAX_VALUES = 5
# Both chosen on the validation schemas (see docs/decisions.md), never on the test schemas.
# Each neighbour votes with similarity ** SHARPNESS, so one close match outweighs several
# loose ones (otherwise "case type" loses to the many "case ..." synonyms of case_reference).
SHARPNESS = 8
# Share of the vote given to the learned model once fitted; the rest goes to neighbours.
LEARNED_WEIGHT = 0.2

Encoder = Callable[[list[str]], np.ndarray]


def column_text(header: str, values: pd.Series | list | None = None) -> str:
    """``hearing date | 2025-07-02; 2025-08-11``: what the classifier reads for one column."""
    text = normalise_header(header)
    if values is not None:
        present = [str(v).strip() for v in values if pd.notna(v) and str(v).strip()]
        if present:
            text += " | " + "; ".join(present[:MAX_VALUES])
    return text


def load_encoder(model_name: str = DEFAULT_MODEL) -> Encoder:
    """Load the sentence model on the CPU, using the local cache when it has the weights."""
    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(model_name, device="cpu", local_files_only=True)
    except OSError:  # not cached yet: download the weights once
        model = SentenceTransformer(model_name, device="cpu")
    return lambda texts: model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def load_labelled_columns(schemas_dir: str | Path, splits: set[str]) -> pd.DataFrame:
    """Labelled columns from the test schemas in the given splits (train/validation/test).

    Returns one row per column: ``schema_id``, ``header``, ``text`` and ``concept``
    (``NO_CONCEPT`` for columns the labels say match no dictionary concept).
    """
    root = Path(schemas_dir)
    index = pd.read_csv(root / "index.csv", dtype=str)
    labels = pd.read_csv(root / "labels" / "column_labels.csv", dtype=str)
    rows = []
    for entry in index[index["split"].isin(splits)].itertuples():
        path = root / entry.file
        if not path.exists():  # schemas may sit directly in the folder
            path = root / Path(entry.file).name
        sample = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
        for label in labels[labels["schema_id"] == entry.schema_id].itertuples():
            values = sample[label.column] if label.column in sample.columns else None
            rows.append(
                {
                    "schema_id": entry.schema_id,
                    "header": label.column,
                    "text": column_text(label.column, values),
                    "concept": label.concept if isinstance(label.concept, str) else NO_CONCEPT,
                }
            )
    return pd.DataFrame(rows)


class EmbeddingClassifier:
    method = "embedding"

    def __init__(
        self,
        dictionary: ConceptDictionary,
        model_name: str = DEFAULT_MODEL,
        encoder: Encoder | None = None,
        learned_weight: float = LEARNED_WEIGHT,
    ):
        self.dictionary = dictionary
        self.learned_weight = learned_weight
        self.model_name = model_name
        self._encoder = encoder
        self.model = None  # logistic regression, once fitted
        self._index_concepts: list[str] = []
        self._index_vectors: np.ndarray | None = None

    @property
    def encode(self) -> Encoder:
        if self._encoder is None:
            self._encoder = load_encoder(self.model_name)
        return self._encoder

    def _index(self) -> tuple[list[str], np.ndarray]:
        """Embeddings of every dictionary concept name and synonym (built once)."""
        if self._index_vectors is None:
            texts, concepts = [], []
            for concept in self.dictionary:
                for name in [concept.concept, *concept.synonyms]:
                    texts.append(normalise_header(name))
                    concepts.append(concept.concept)
            self._index_concepts = concepts
            self._index_vectors = np.asarray(self.encode(texts))
        return self._index_concepts, self._index_vectors

    def fit(self, labelled_columns: pd.DataFrame) -> EmbeddingClassifier:
        """Train on labelled columns (columns: ``text``, ``concept``)."""
        from sklearn.linear_model import LogisticRegression

        vectors = np.asarray(self.encode(labelled_columns["text"].tolist()))
        self.model = LogisticRegression(max_iter=2000).fit(vectors, labelled_columns["concept"])
        return self

    def _neighbour_scores(self, headers: list[str]) -> list[tuple[dict[str, float], float]]:
        """For each header: each concept's share of the top-k similarity, and the best match."""
        concepts, index = self._index()
        queries = np.asarray(self.encode([normalise_header(h) for h in headers]))
        out = []
        for sims in queries @ index.T:
            top = np.argsort(sims)[::-1][:K]
            best = float(sims[top[0]])
            if best < MIN_SIMILARITY:
                out.append(({NO_CONCEPT: 1.0}, best))
                continue
            scores: dict[str, float] = {}
            for i in top:
                vote = max(float(sims[i]), 0.0) ** SHARPNESS
                scores[concepts[i]] = scores.get(concepts[i], 0.0) + vote
            total = sum(scores.values())
            out.append(({c: s / total for c, s in scores.items()}, best))
        return out

    def infer_all(
        self, headers: list[str], sample: pd.DataFrame | None = None
    ) -> list[ColumnInference]:
        neighbours = self._neighbour_scores(headers)
        probabilities = None
        if self.model is not None:
            texts = [
                column_text(h, sample[h] if sample is not None and h in sample.columns else None)
                for h in headers
            ]
            probabilities = self.model.predict_proba(np.asarray(self.encode(texts)))

        results = []
        for i, header in enumerate(headers):
            shares, best = neighbours[i]
            if probabilities is None:
                concept = max(shares, key=shares.get)
                # Neighbours alone: confidence is how close the header is to that concept
                confidence = best if concept != NO_CONCEPT else 0.0
            else:
                learned = dict(zip(self.model.classes_, probabilities[i], strict=True))
                w = self.learned_weight
                blended = {
                    c: (1 - w) * shares.get(c, 0.0) + w * learned.get(c, 0.0)
                    for c in set(shares) | set(learned)
                }
                concept = max(blended, key=blended.get)
                confidence = blended[concept]
            results.append(
                ColumnInference(
                    column=header,
                    concept=None if concept == NO_CONCEPT else concept,
                    type=None,
                    confidence=round(float(confidence), 3),
                    method=self.method,
                )
            )
        return results

    def infer(self, header: str, values: pd.Series | None = None) -> ColumnInference:
        sample = None if values is None else pd.DataFrame({header: values})
        return self.infer_all([header], sample)[0]
