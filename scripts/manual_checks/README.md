# Manual checks

Exploratory / debugging scripts that are **not** part of the automated `pytest`
suite. They typically require a running API, live data files, or a configured
MLflow server, and use `print()`-based assertions rather than pytest.

Run them by hand when investigating a specific area:

| Script | Purpose |
|--------|---------|
| `debug_preprocessing.py` | Inspect intermediate preprocessing output |
| `test_ab_system.py` | Smoke-check the A/B experiment routing against a live API |
| `test_dtype_fix.py` | Reproduce/verify the prediction dtype handling |
| `test_evidently_api.py` | Sanity-check the installed Evidently API version |
| `test_drift_retrain_logic.py` | Integration walkthrough of the drift-retrain decision logic |

The automated suite lives in `../../tests/` and is what CI / `RUN_TESTS.sh` run.
