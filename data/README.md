# data/

Generated locally and never committed (see `.gitignore`). Rebuild everything with `make reference`.

| File | Contents | Readable by |
| --- | --- | --- |
| `reference/reference.csv` | Full reference world: 2,000 fictional cases (set `n_cases` in the config to change), 19 columns | Store builders, evaluators |
| `calibration/calibration_stats.yaml` | Aggregate statistics generators may see | Generators, rule proposer |
| `calibration/scenario2_sample.csv` | 200-row sample for Scenario 2 | Generators (Scenario 2 only), privacy evaluator |
| `evaluation/held_back_stats.yaml` | Subgroup and multi-column patterns, never given to generators | Evaluators only |
| `evaluation/candidate_pool.csv` | 400 rows, half from the sample (`member = True`) | Privacy evaluator only |
| `runs/` | Generated datasets, run records and reports | Everyone |

## Reference dataset columns

`case_number`, `claim_type`, `claim_amount`, `track`, `issue_date`, `court`, `region`, `defended`,
`claimant_represented`, `defendant_represented`, `interpreter_required`, `hearing_date`,
`hearing_type`, `case_status`, `outcome`, `judgment_type`, `judgment_date`, `judgment_amount`,
`warrant_issue_date`. Every case number starts with `SYN`.

Never put real or personal data here.
