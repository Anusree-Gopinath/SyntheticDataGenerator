# SynthGen: LLM-Guided Synthetic Data Generation for Automated Software Testing

SynthGen understands an unfamiliar data schema and generates realistic, rule-compliant
synthetic test data **without access to real operational records**. Justice case management
systems are the case study.

> **Central research question:** To what extent can a hybrid AI system combining large
> language models, semantic schema classification and statistical generative methods produce
> realistic, constraint-compliant and privacy-conscious synthetic tabular data without access
> to real operational records?

## How it works

1. **Upload** a fictional CSV schema (headers, optional made-up rows).
2. **Identify** each column's meaning and type (rules, embeddings, local LLM).
3. **Propose** cross-column business rules.
4. **Review and approve** the profile (a person always approves).
5. **Generate** records with several approaches.
6. **Validate** every row against the approved rules.
7. **Report** a comparison of realism, rule compliance, privacy risk and run time.

## Design principles

- **No real data.** Inputs are fictional schemas, documented assumptions and a synthetic reference world.
- **Everything runs locally.** All LLMs run through Ollama on `localhost`; nothing is sent to an external AI service.
- **People approve, AI proposes.** Only approved profiles drive generation.
- **One rule engine for all generators**, so comparisons are fair.
- **Evaluation data is kept apart** from generators.
- **Reproducible by default.** Profiles, configuration, seeds and model versions are recorded.

See [docs/architecture.md](docs/architecture.md) for the full design.

## Generation approaches

| Approach | Scenario 1 (no rows) | Scenario 2 (200 sample rows) | Built with |
| --- | --- | --- | --- |
| Knowledge-based (baseline) | Yes | Yes | NumPy, pandas |
| Gaussian copula | Yes | Yes | NumPy, SciPy (written in this project) |
| CTGAN | No | Yes | PyTorch, scikit-learn (written in this project) |
| TVAE | No | Yes | PyTorch, scikit-learn (written in this project) |
| LLM (`llama3.1:8b`) | Yes | Yes | Ollama, running locally |

The copula, CTGAN and TVAE are written here rather than taken from SDV or synthcity: those
libraries need pandas 2, and SDV and CTGAN use a source-available licence (BUSL-1.1).
See [docs/decisions.md](docs/decisions.md).

## Project structure

```text
synthgen/
  app/               Streamlit app: step bar and shared look (main.py, ui.py), screens in
                     app_pages/, the Inter font in static/ (theme: .streamlit/config.toml)
  core/              Shared data model: profiles, rules, run records
  ingest/            CSV parsing and personal data screening
  schema/            Header normalisation, rules / embedding / LLM inference, combining results
  rules/             Safe rule language, validator, contradiction check
  generators/        knowledge_based + dependencies, copula, ctgan + tvae + deep, llm; fields
  gateway/           LLM gateway (local models only)
  reference_world/   Simulator for the synthetic benchmark and the stores built from it
  evaluation/        Realism metrics, privacy tests, report builder
  orchestration/     Approaches, run orchestrator (reject and regenerate), experiment plan
config/              Concept dictionary, reference world, experiment plan and its profile, thresholds
data/                Generated locally, never committed (see .gitignore)
examples/            Fictional sample schemas
tests/               pytest test suite
notebooks/           Analysis notebooks for the report (none yet)
docs/                Setup guide, architecture, decisions, model licences, AI use log
```

## Getting started

Requires Python 3.11 or later and, for the LLM features, [Ollama](https://ollama.com).
For a step-by-step setup from a fresh clone (Windows commands, what each extra installs,
troubleshooting), see [docs/setup-guide.md](docs/setup-guide.md).

```bash
# 1. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install the core package and development tools
pip install -e ".[dev]"

# 3. Optional extras
pip install -e ".[app]"         # Streamlit review app
pip install -e ".[embeddings]"  # sentence-transformers
pip install -e ".[deep]"        # CTGAN and TVAE (PyTorch and scikit-learn)

# 4. Build the synthetic reference world and the stores the app reads
#    (data/reference, data/calibration, data/evaluation)
make reference

# 5. Pull the local model and start Ollama (needed for the LLM approach only)
ollama pull llama3.1:8b
ollama serve

# 6. Run the tests
make test

# 7. Start the app (http://localhost:8501)
make app
```

## Using the app

The bar under the header holds the four steps. Only **Upload** is open at first; each later
step opens once the one before it is done (a schema uploaded, a profile approved, runs
finished). Finished steps show a tick and the current step is highlighted.

- **Upload:** a CSV of headers, optionally with a few made-up rows. Values that look like
  personal data must be confirmed as fictional before columns are identified.
- **Review and approve:** columns on the left, the selected column on the right (its meaning,
  how sure each method was, example values and the rules that use it). Below are the
  cross-column rules and the approval, with a trial run of 1,000 rows that flags rules that
  contradict each other. In rules, a column name with spaces goes in backticks:
  `` unique(`Case Reference`) ``.
- **Generate:** pick a scenario, approaches, rows and seeds. The LLM approach and the LLM help
  on the Review screen need Ollama running.
- **Quality report:** compare approaches on rule compliance, realism, privacy risk and run
  time, with paired significance tests across seeds.

The light theme is a soft blue-grey page with white cards; the dark choice (Settings →
Theme) is a grey page with black text. The app loads nothing from the internet: the Inter
font is served from `synthgen/app/static/fonts/`.

## Running the experiment plan

`make experiments` runs [config/experiment_plan.yaml](config/experiment_plan.yaml) (pinned to
the approved profile `s01-civil-money-claims` version 7, kept in `config/profiles/` so it runs
from a fresh clone) in one go: every scenario, approach and seed through the same orchestrator as the app, then every
run is evaluated. Each scenario becomes one batch in `data/runs/` that the Report screen can
open, with an `experiment_report/` folder holding every run's metrics (`runs.csv`), the mean
and 95% confidence interval per approach (`summary.csv`), paired significance tests
(`significance.csv`) and all three on one page (`report.html`).

Only approved profiles are used. Options override the plan without changing it, for example
a quick check:

```bash
python -m synthgen.orchestration.run_experiments --profile s01-civil-money-claims \
    --seeds 1 2 3 --rows 1000 --approaches knowledge_based gaussian_copula ctgan
```

`--profile` takes a YAML path, `id@version` or `id` (latest saved version). The LLM approach
is skipped, with the reason, when Ollama or its model isn't available.

## Status

| Component | Status |
| --- | --- |
| Data model, rule language and validator | Working first version |
| Personal data screening (regex) | Working first version |
| Header normalisation and rules-based column matching | Working first version |
| LLM gateway (local only) | Working first version |
| Knowledge-based generator (reads approved rules as dependencies; draws within groups) | Working first version |
| Realism metrics (KS, Jensen-Shannon) | Working first version |
| Embedding and LLM schema inference | Working first version |
| Gaussian copula, CTGAN, TVAE and LLM generators | Working first version |
| Reference world simulator and stores | Working first version |
| Privacy evaluation (distance to closest record, membership attacks) | Working first version |
| Streamlit app (Upload, Review, Generate, Report) | Working first version |
| Trial-generation check before approval | Working first version |
| Comparative report (mean and 95% CI across seeds, paired significance tests) | Working first version |
| One-command experiment run (`make experiments`) | Working first version |

## Data and governance

- Never commit real or personal data. The `data/` folder is ignored by git.
- Example rows uploaded to the app stay in memory and are never written to disk.
- Record each model's licence in [docs/model-licences.md](docs/model-licences.md).
- Record use of AI coding assistants in [docs/ai-use-log.md](docs/ai-use-log.md).
