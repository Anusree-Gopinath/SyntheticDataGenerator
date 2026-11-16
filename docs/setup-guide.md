# Setup guide: running SynthGen on your own machine

This guide takes you from a fresh clone of the repository to the running app, the test suite
and the experiment plan. Every step was checked on a fresh copy of the project on 13 November
2026 (macOS 14, Apple silicon, Python 3.11): all 116 tests passed and the app started.

Everything runs on your machine. The internet is needed only to install packages, to download
the embedding model the first time the Review screen opens, and to pull the optional local LLM.

## 1. What you need

| Item | Required? | Notes |
| --- | --- | --- |
| Python 3.11 or later | Yes | Check with `python3 --version`. Tested on 3.11.12. |
| Git | Yes | To clone the repository. |
| Disk space | Yes | About 1.5 GB for the full install (PyTorch alone is about 630 MB). |
| Memory | Yes | 8 GB is enough without the local LLM; 16 GB or more with it (the model takes about 7 GB when loaded). |
| [Ollama](https://ollama.com) | Optional | Runs the local LLM. Without it, every feature works except the LLM approach and the LLM suggestions on the Review screen. |
| `make` | Optional | Shortcuts for the commands below. On Windows, use the plain commands in [section 9](#9-commands-without-make). |

macOS is the tested platform. The code is plain Python, so Linux and Windows should work, but
they have not been checked.

## 2. Get the code

```bash
git clone <repository URL>
cd SyntheticDataGenerator
```

## 3. Create a virtual environment

macOS and Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

Keep this environment active for every command below. Your prompt shows `(.venv)` while it is.

## 4. Install the packages

Install everything:

```bash
pip install --upgrade pip
pip install -e ".[dev,app,embeddings,deep]"
```

Or install only what you need:

| Extra | Installs | Needed for |
| --- | --- | --- |
| (none) | NumPy, pandas, SciPy, PyYAML, requests | The core package |
| `dev` | pytest, hypothesis, ruff | Running the tests and the linter |
| `app` | Streamlit 1.67 or later | The app |
| `embeddings` | sentence-transformers (with PyTorch and scikit-learn) | The embedding classifier on the Review screen |
| `deep` | PyTorch, scikit-learn | The CTGAN and TVAE generators |

Versions tested: Python 3.11.12, Streamlit 1.67.0, pandas 3.1.1, NumPy 2.4.6, PyTorch 2.15.0,
scikit-learn 1.9.1, sentence-transformers 6.3.0.

## 5. Build the reference data

The `data/` folder is not in the repository; it is built on your machine. This creates the
fictional reference world (2,000 cases) and the stores the app reads, in a few seconds:

```bash
make reference
```

It always produces the same data, so results can be reproduced. See [data/README.md](../data/README.md)
for what each file holds.

## 6. Run the tests

```bash
make test
```

Expect `116 passed` with every extra installed. Tests that need something you haven't set up
are skipped with the reason, not failed: four without the reference data from step 5, three
without PyTorch (`deep`), one without scikit-learn. `pytest -rs` lists the reasons.

## 7. Start the app

```bash
make app
```

Open <http://localhost:8501>. Only **Upload** is open at first; each later step opens once the
one before it is done. To try it, upload one of the fictional schemas in
[examples/test_schemas/](../examples/test_schemas/), for example `S01_civil_money_claims.csv`.

The first time the Review screen opens, the embedding model `all-MiniLM-L6-v2` (about 90 MB) is
downloaded from Hugging Face and then used offline. Without internet access at that moment, the
Review screen still works with the other methods and says the embedding classifier is not used.

Stop the app with `Ctrl+C` in the terminal.

## 8. Optional: the local LLM

1. Install Ollama from <https://ollama.com/download> and open it (on macOS it runs from the menu
   bar), or run `ollama serve` in a separate terminal.
2. Pull the model the project uses (about 4.9 GB):

   ```bash
   ollama pull llama3.1:8b
   ```

3. Reload the app. The LLM appears under **Inference settings** on the Review screen and as an
   approach on the Generate screen.

The app talks to Ollama only at `http://localhost:11434` and refuses any other address. On the
Mac this was tested on, the LLM produced about one row every 1.4 seconds, so keep runs small (a
few hundred rows) when trying it.

## 9. Commands without `make`

| `make` command | Plain command |
| --- | --- |
| `make install` | `pip install -e ".[dev]"` |
| `make reference` | `python -m synthgen.reference_world.simulator --config config/reference_world.yaml` then `python -m synthgen.reference_world.stores --config config/reference_world.yaml` |
| `make test` | `pytest` |
| `make lint` | `ruff check synthgen tests` |
| `make app` | `streamlit run synthgen/app/main.py` |
| `make experiments` | `python -m synthgen.orchestration.run_experiments --plan config/experiment_plan.yaml` |

## 10. Optional: the experiment plan

`make experiments` runs every approach, scenario and seed in
[config/experiment_plan.yaml](../config/experiment_plan.yaml) on the approved profile kept in
[config/profiles/](../config/profiles/), and writes a report for each scenario under `data/runs/`.
The full plan is 5 seeds of 10,000 rows. CTGAN and TVAE train for about 45 and 22 seconds
per seed; the LLM adds about 40 hours. Try a short run first (about 2 minutes 20 seconds when
checked):

```bash
python -m synthgen.orchestration.run_experiments --seeds 1 2 --rows 100 \
    --approaches knowledge_based gaussian_copula ctgan tvae
```

CTGAN and TVAE need sample rows, so they run in Scenario 2 only.

The LLM approach is skipped, with the reason printed, when Ollama or its model isn't available.

## 11. Troubleshooting

| Problem | Fix |
| --- | --- |
| `streamlit: command not found` or `No module named synthgen` | The virtual environment isn't active: run the activate command from section 3. |
| `requires a different Python` during install | Install Python 3.11 or later and recreate the virtual environment with it. |
| Generate shows "The calibration store is missing" | Run `make reference`. |
| Port 8501 is already in use | `streamlit run synthgen/app/main.py --server.port 8502` |
| No LLM options in the app | Start Ollama and pull `llama3.1:8b` (section 8), then reload the page. |
| The page looks unstyled or uses another font | Stop the app and start it again: theme changes are read at start-up. |
| Want to start over in the app | Reload the browser tab: uploads and drafts live only in that session. |
