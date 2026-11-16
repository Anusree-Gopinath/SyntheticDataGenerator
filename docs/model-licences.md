# Model and library licences

Check and record the licence of every model and library before use.

| Item | Version / tag | Licence | Conditions (for example attribution) | Checked on |
| --- | --- | --- | --- | --- |
| Llama 3.1 8B | llama3.1:8b | Meta Llama community licence | TBC | |
| all-MiniLM-L6-v2 | sentence-transformers/all-MiniLM-L6-v2 | Apache 2.0 (per the model card; confirm) | Weights downloaded once from Hugging Face, then used offline | |
| PyTorch | 2.x (installed with sentence-transformers) | BSD-3-Clause (confirm) | Used for this project's own CTGAN and TVAE implementations | |
| scikit-learn | 1.x | BSD-3-Clause (confirm) | Gaussian mixtures for CTGAN and TVAE; logistic regression for the embedding classifier | |
| Streamlit | | TBC | | |
| Inter (font) | 4.x, variable, Latin subset (via @fontsource-variable/inter 5.1.0) | SIL Open Font License 1.1 | Served from the app (`synthgen/app/static/fonts/`, licence alongside); downloaded once, never fetched at run time | 2026-11-13 |
