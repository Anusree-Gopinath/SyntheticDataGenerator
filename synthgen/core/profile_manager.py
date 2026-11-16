"""Profile manager: combine inferences into a draft profile and version approved profiles.

Example rows are used to work out structure (type, whether a column is optional, how many
decimal places) but their values are never copied into a profile, so nothing from the
upload is written to disk. Allowed values and patterns come from the concept dictionary.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

from synthgen.core.models import COLUMN_TYPES, ColumnSpec, SchemaProfile
from synthgen.schema.dictionary import ConceptDictionary
from synthgen.schema.rules_matcher import detect_type

_TRUE = {"true", "yes", "y", "1"}
_FALSE = {"false", "no", "n", "0"}


def _present(sample: pd.DataFrame | None, name: str) -> pd.Series:
    if sample is None or name not in sample.columns:
        return pd.Series(dtype=str)
    values = sample[name].dropna().astype(str).str.strip()
    return values[values != ""]


def build_column(
    name: str,
    concept: str | None,
    dictionary: ConceptDictionary,
    sample: pd.DataFrame | None = None,
    inferred_by: dict[str, float] | None = None,
) -> ColumnSpec:
    """Build a draft column spec from the chosen concept and the shape of the example rows."""
    entry = dictionary.get(concept) if concept else None
    present = _present(sample, name)
    detected = detect_type(present) if len(present) else None

    col_type = entry.type if entry else detected or "string"
    if col_type not in COLUMN_TYPES:
        col_type = "string"

    allowed = list(entry.allowed_values) if entry and entry.allowed_values else None
    if allowed and not present.isin(allowed).all():
        allowed = None  # the dictionary list doesn't cover this schema; leave it to the user

    pattern = entry.pattern if entry else None
    if pattern and not present.str.fullmatch(pattern).all():
        pattern = None

    decimals = None
    if col_type == "decimal":
        places = present.str.extract(r"\.(\d+)$")[0].dropna().str.len()
        decimals = int(places.max()) if len(places) else 2

    n_rows = 0 if sample is None or name not in sample.columns else len(sample)
    return ColumnSpec(
        name=name,
        concept=concept,
        type=col_type,
        pattern=pattern,
        decimals=decimals,
        allowed_values=allowed,
        nullable=n_rows > 0 and len(present) < n_rows,
        inferred_by=dict(inferred_by or {}),
    )


def coerce_sample(sample: pd.DataFrame, columns: list[ColumnSpec]) -> pd.DataFrame:
    """Convert example rows (read as text) to each column's type so rules can be checked."""
    out = sample.copy()
    for col in columns:
        if col.name not in out.columns:
            continue
        values = out[col.name]
        if col.type == "date":
            # ISO first; dayfirst would otherwise swap 2025-07-02 into 7 February
            iso = pd.to_datetime(values, errors="coerce", format="ISO8601")
            uk = pd.to_datetime(values, errors="coerce", format="mixed", dayfirst=True)
            out[col.name] = iso.fillna(uk)
        elif col.type in {"integer", "decimal"}:
            out[col.name] = pd.to_numeric(values, errors="coerce")
        elif col.type == "boolean":
            lowered = values.astype(str).str.strip().str.lower()
            out[col.name] = lowered.map(
                lambda v: True if v in _TRUE else False if v in _FALSE else None
            )
    return out


def profile_id_from_filename(filename: str) -> str:
    """``Civil Claims_schema.csv`` -> ``civil-claims-schema``."""
    stem = Path(filename).stem.lower()
    return re.sub(r"[^a-z0-9]+", "-", stem).strip("-") or "profile"


def next_version(store: str | Path, profile_id: str) -> int:
    folder = Path(store) / profile_id
    versions = [
        int(m.group(1))
        for p in folder.glob("v*.yaml")
        if (m := re.fullmatch(r"v(\d+)\.yaml", p.name))
    ]
    return max(versions, default=0) + 1


def save_profile(profile: SchemaProfile, changes: list[dict], store: str | Path) -> Path:
    """Save an approved profile as ``<store>/<profile_id>/v<version>.yaml``.

    The change log of what the user altered is saved next to it (used to measure RQ1).
    An existing version is never overwritten.
    """
    folder = Path(store) / profile.profile_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"v{profile.version}.yaml"
    if path.exists():
        raise FileExistsError(f"Profile {profile.profile_id} version {profile.version} exists")
    profile.save(path)
    log = folder / f"v{profile.version}.changes.yaml"
    log.write_text(yaml.safe_dump({"changes": changes}, sort_keys=False))
    return path
