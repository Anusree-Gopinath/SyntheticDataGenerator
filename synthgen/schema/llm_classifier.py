"""Method 3 of 3: local LLM for column understanding and rule proposals.

Prompts list the allowed concepts and ask for a fixed JSON structure. Few-shot variants
add worked examples of fictional schemas. Each prompt runs three times at low
temperature; agreement between runs is recorded as a consistency score.
"""

from __future__ import annotations

import pandas as pd

from synthgen.gateway import LLMGateway, LLMRequest
from synthgen.schema.dictionary import ConceptDictionary
from synthgen.schema.rules_matcher import ColumnInference

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "columns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "concept": {"type": ["string", "null"]},
                    "type": {"type": "string"},
                    "format": {"type": ["string", "null"]},
                    "confidence": {"type": "number"},
                },
                "required": ["column", "concept", "type", "confidence"],
            },
        }
    },
    "required": ["columns"],
}

SYSTEM_PROMPT = (
    "You identify what each column in a test data schema means. "
    "Choose only from the allowed concepts, or null if none fit. Reply with JSON only."
)


class LLMClassifier:
    method = "llm"

    def __init__(self, gateway: LLMGateway, dictionary: ConceptDictionary, model: str):
        self.gateway = gateway
        self.dictionary = dictionary
        self.model = model

    def build_prompt(self, headers: list[str], sample: pd.DataFrame | None = None) -> str:
        concepts = "\n".join(f"- {c.concept}: {c.description}" for c in self.dictionary)
        lines = [f"Allowed concepts:\n{concepts}", "", "Columns:"]
        for h in headers:
            if sample is not None and h in sample.columns:
                values = "; ".join(sample[h].dropna().astype(str).head(5))
                lines.append(f"- {h} | examples: {values}")
            else:
                lines.append(f"- {h}")
        return "\n".join(lines)

    def infer_all(
        self, headers: list[str], sample: pd.DataFrame | None = None, seed: int = 0
    ) -> list[ColumnInference]:
        request = LLMRequest(
            model=self.model,
            system=SYSTEM_PROMPT,
            prompt=self.build_prompt(headers, sample),
            json_schema=RESPONSE_SCHEMA,
            seed=seed,
        )
        data = self.gateway.complete_json(request)
        known = {c.concept for c in self.dictionary}
        results = []
        for item in data.get("columns", []):
            concept = item.get("concept") if item.get("concept") in known else None
            results.append(
                ColumnInference(
                    column=item["column"],
                    concept=concept,
                    type=item.get("type"),
                    confidence=float(item.get("confidence", 0.0)),
                    method=self.method,
                )
            )
        return results
