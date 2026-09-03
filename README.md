# Secure Mobile Banking Fraud Detection System

Implementation code for the MSc dissertation of the same name. Every
design decision here is pinned to a specific section of the dissertation -
see "Where things live" below if you're checking the code against a
particular chapter.

## Requirements

- Python 3.10+ (developed against Python 3 on Windows 11, per section 4.1)
- The real dataset - see `data/README.md` for how to get it (not included;
  ~150 MB and requires a free Kaggle account)

## Setup

```
python -m venv venv
venv\Scripts\activate          (Windows)   or   source venv/bin/activate   (macOS/Linux)
pip install -r requirements.txt
```

Or just double-click `setup_project.bat` on Windows, which does the above
and also creates `data/`, `models/`, `outputs/` if they're missing.

## Running the pipeline

```
python -m src.data_prep      # Table 4 & Table 5: cleaning and split stats
python -m src.evaluate       # trains both models, prints Table 9,
                              # writes models/*.joblib and outputs/figure_*.png
python -m src.make_samples   # builds data/sample_transactions.csv from the
                              # TEST partition, for the web app's demo path
```

## Running the web app

```
set FLASK_SECRET_KEY=change-this-in-production      (Windows)
export FLASK_SECRET_KEY=change-this-in-production    (macOS/Linux)
python webapp/app.py
```

Then open http://127.0.0.1:5000. A default admin account is created
automatically the first time the app runs:

```
username: admin
password: admin123
```

Change this password (or delete `webapp/app.db` and register fresh
accounts) before treating this as anything other than a local demo -
see section 6.4, "Recommendations for Future Enhancement", for what a
production hardening pass would need to add (this default credential
is exactly the kind of thing that section flags).

## Running the tests

```
pip install pytest
pytest tests/ -v
```

`tests/test_functional.py` covers Table 8's T1-T8 and T10. `tests/test_consistency.py`
covers T9 - the numerical check that the web app and the training pipeline
produce identical probabilities for the same input (tolerance 1e-12,
section 5.1).

## Where things live

| Dissertation section | Code |
|---|---|
| 3.1.2 - dataset, class weighting formula | `src/config.py`, `src/data_prep.py` |
| 3.1.2 - LR/RF equations | `src/train.py` (the models); the equations themselves are in the dissertation text, not reproduced as code comments here |
| 4.2.1 - data preparation | `src/data_prep.py` |
| 4.2.2 - model training (Table 6) | `src/train.py`, `LOGREG_PARAMS` / `RF_PARAMS` in `src/config.py` |
| 4.2.2 - threshold selection | `src/evaluate.py: best_threshold_by_f1()` |
| 4.2.3 - prediction interface | `src/predict.py: score_transaction()` |
| 4.2.4 - the web application | `webapp/app.py`, `webapp/auth.py`, `webapp/db.py`, `webapp/templates/` |
| 4.3.1 - security | `webapp/auth.py` (hashing, generic login errors), `webapp/db.py` (parameterised queries) |
| 4.3.2 - performance (model cached once) | `src/predict.py: _load_cache()` |
| 4.3.3 - scalability (swap to MySQL/Postgres) | only `webapp/db.py: get_connection()` needs to change |
| 5.1-5.3 - evaluation, Table 9, Figures 9-10 | `src/evaluate.py` |
| Table 8 (T1-T10) | `tests/test_functional.py`, `tests/test_consistency.py` |
| Gap 2 (section 2.4) - threshold recorded per decision | `webapp/db.py: record_prediction()` stores `threshold_used` on every row |

## Project structure

```
src/            preprocessing, training, evaluation, prediction interface
webapp/         Flask application and database access
webapp/templates/   HTML templates
data/           dataset (not included - see data/README.md)
models/         serialised model and fitted scaler (generated)
outputs/        evaluation figures (generated)
tests/          functional and consistency tests (Table 8)
```

## Honesty about what's been tested here

This code has been run end-to-end - data prep, training, evaluation,
figure generation, the full web app (auth, both scoring paths, dashboard,
history), and all ten Table 8 test cases - against a small synthetic
stand-in dataset with the same 31 columns as the real file, since this
build environment has no network access to Kaggle. That confirms the
*code* runs correctly and the *logic* matches the dissertation exactly
(same split ratios, same class weights, same scaler, same threshold
procedure, same metrics). It does not and cannot confirm the specific
reported numbers (492 fraud cases, 1,081 duplicates, PR-AUC 0.8168, etc.)
- those will reproduce once you run this same code against the real
`creditcard.csv` on your own machine, which is exactly what section 4.1
describes doing.
