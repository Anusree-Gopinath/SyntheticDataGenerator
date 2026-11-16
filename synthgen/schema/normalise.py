"""Header normalisation: lower case, split words, expand common abbreviations."""

from __future__ import annotations

import re

ABBREVIATIONS = {
    "dt": "date",
    "no": "number",
    "num": "number",
    "nbr": "number",
    "ref": "reference",
    "amt": "amount",
    "val": "value",
    "hrg": "hearing",
    "trk": "track",
    "alloc": "allocation",
    "cd": "code",
    "desc": "description",
    "id": "identifier",
    "crt": "court",
    "def": "defendant",
    "clmt": "claimant",
}


def normalise_header(header: str) -> str:
    """Turn a raw header such as ``hrgDt`` or ``HRG_DT`` into ``hearing date``."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", header.strip())
    tokens = re.split(r"[\s_\-.]+", text.lower())
    expanded = [ABBREVIATIONS.get(t, t) for t in tokens if t]
    return " ".join(expanded)
