# Fictional test schemas

30 fictional CSV schemas for testing SynthGen's schema understanding (RQ1) and rule proposals.
**All values are made up.** Identifiers are pseudonymous codes (for example `J-123`, `P-10293`),
and no names, addresses, phone numbers, emails or National Insurance numbers appear.
Court and region names are public place names.

Suggested location in the repository: `examples/test_schemas/` (committed to git, unlike `data/`).

## Contents

| File | What it holds |
| --- | --- |
| `index.csv` | One row per schema: domain, header style, number of example rows, date format, boolean encoding, train/validation/test split |
| `schemas/S01_...csv` to `S30_...csv` | The schemas. 25 include 4 to 5 made-up rows; 5 are headers only |
| `labels/column_labels.csv` | The correct answer for every column (192): concept, type, format, nullable, unique |
| `labels/rule_labels.csv` | The expected cross-column rules for each schema (60), written in terms of concepts |
| `concept_dictionary_extended.yaml` | 38 concepts covering every labelled column. Replaces `config/concept_dictionary.yaml` |

## What makes them a useful test

- **Header styles vary:** snake_case, camelCase, UPPER_ABBREV, Title Case, abbreviated (`hrg_dt`),
  cryptic (`dt1`, `FLD03`) and mixed within one file (S30).
- **Domains vary:** civil claims, listings, tribunals, criminal listing, enforcement, family,
  fees, documents, audit, performance.
- **Formats vary:** ISO and UK (`DD/MM/YYYY`) dates; booleans as `true/false`, `Y/N` or `1/0`.
  The `1/0` booleans look like integers, so type detection has to work for it.
- **Unknown columns:** 8 columns (for example `XREF_CD`, `txn_batch`, `lang_cd`) are not in the
  dictionary. Their labels have an empty `concept` and `in_dictionary = no`; the correct answer
  is "unknown, flag for review".
- **Headers-only schemas** (S03, S10, S17, S24, S27) test inference from names alone.
- **Rule evidence:** in S30 the last row deliberately breaks `hearing_date > issue_date`, so the
  rule proposer should report "fails in 1 of N rows" for that rule.

## Splits

Use the `split` column in `index.csv` so no schema is used for both training and testing:
18 train, 6 validation (for choosing weights and thresholds), 6 test (for final results only).

## Rule labels

Rules are written using **concept names**, not column names, so they can be compared with
proposals whatever the headers are called. The `columns` field gives the matching headers
in order. To check a rule against a schema's rows, rename columns to their concepts and parse
dates using the schema's `date_format` first.

## Before using them

- Review the labels yourself. They are your ground truth, so you should agree with every one.
- Ask a second person to label a sample (about 20% of columns) and report agreement (Cohen's kappa).
- If you add `received_date`-style concepts or new types, update the dictionary and tests.
- Record in `docs/ai-use-log.md` that these schemas were generated with AI assistance and then reviewed.
