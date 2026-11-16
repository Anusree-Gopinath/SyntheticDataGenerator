# Architecture

The full design is in *Architecture and Design: LLM-Guided Synthetic Data Generation System*
(add the PDF to this folder, or link to it). This page is a short map from the design to the code.

## Layers and modules

| Layer | Component | Module |
| --- | --- | --- |
| Interface | Review app: step bar (steps open in order), Upload, two-pane Review, Generate, Report | `synthgen/app/` (`main.py`, `ui.py`, `app_pages/`) |
| Application | Ingestion and screening | `synthgen/ingest/` |
| Application | Profile manager | `synthgen/core/models.py`, `synthgen/core/profile_manager.py` (draft columns, versions and change log in `data/profiles/`) |
| Application | Run orchestrator | `synthgen/orchestration/` |
| AI and generation | Schema inference (rules, embeddings, LLM) | `synthgen/schema/` |
| AI and generation | Rule proposer | `synthgen/rules/proposer.py` |
| AI and generation | Generators: knowledge-based (rules read as dependencies), Gaussian copula, CTGAN, TVAE, LLM | `synthgen/generators/` |
| AI and generation | Rule engine, validator and trial-generation contradiction check | `synthgen/rules/` |
| Evaluation | Realism evaluator | `synthgen/evaluation/realism.py` |
| Evaluation | Privacy evaluator | `synthgen/evaluation/privacy.py` |
| Evaluation | Report builder | `synthgen/evaluation/report.py` |
| Shared services | LLM gateway (local only) | `synthgen/gateway/` |
| Benchmark | Reference world simulator and stores | `synthgen/reference_world/` |

## Safeguards enforced in code

- `LLMGateway` refuses any non-localhost model server.
- `tests/test_architecture.py` fails if any module outside `gateway/` imports `requests`
  or an LLM client, and if any generator module imports the evaluation code.
- Rules are parsed into a restricted expression tree and never run with `eval`.
- `data/` is git-ignored; uploaded rows live only in Streamlit session state.
- The app loads nothing from external services: its font is served from the app itself.
- Generation is reachable only after a profile is approved: the step bar keeps later steps
  locked, and a locked page opened by its address returns to the furthest open step.

## Traceability to research questions

| Research question | Modules |
| --- | --- |
| RQ1: schema understanding | `schema/`, `rules/proposer.py` |
| RQ2: realism and rule compliance | `generators/`, `rules/`, `reference_world/`, `evaluation/realism.py` |
| RQ3: privacy risk | `evaluation/privacy.py`, `reference_world/stores.py` |
