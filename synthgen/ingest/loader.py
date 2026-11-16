"""Load an uploaded CSV into headers plus an in-memory sample.

Example rows are kept in memory only. This module never writes them to disk.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

import pandas as pd

MAX_COLUMNS = 50
MAX_ROWS = 500


@dataclass
class UploadedSchema:
    headers: list[str]
    sample: pd.DataFrame  # may be empty when only headers were uploaded

    @property
    def has_rows(self) -> bool:
        return not self.sample.empty


def load_csv(content: bytes | str) -> UploadedSchema:
    """Parse CSV content (bytes or text). Raises ValueError if limits are exceeded."""
    text = content.decode("utf-8-sig") if isinstance(content, bytes) else content
    first_line = next(csv.reader(io.StringIO(text)), [])
    raw_headers = [h.strip() for h in first_line]
    if len(set(raw_headers)) != len(raw_headers):
        raise ValueError("Column names must be unique")
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False, na_values=[""])
    headers = [h.strip() for h in df.columns]
    if len(headers) > MAX_COLUMNS:
        raise ValueError(f"Too many columns: {len(headers)} (limit {MAX_COLUMNS})")
    if len(set(headers)) != len(headers):
        raise ValueError("Column names must be unique")
    if len(df) > MAX_ROWS:
        raise ValueError(f"Too many example rows: {len(df)} (limit {MAX_ROWS})")
    df.columns = headers
    return UploadedSchema(headers=headers, sample=df)
