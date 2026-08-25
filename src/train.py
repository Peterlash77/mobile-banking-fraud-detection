"""
Train and compare the candidate models.

Two models are trained on identical data, with identical preprocessing and an
identical evaluation protocol, so that the comparison is fair and any
difference in performance can be attributed to the model rather than to the
pipeline around it.

  1. Logistic Regression - a linear baseline. Simple, fast and interpretable.
     It establishes the level of performance a straightforward statistical
     method reaches, which is the reference point the proposed model has to
     beat to justify its added complexity.

  2. Random Forest - the proposed model. A non-linear ensemble of decision
     trees. Tree ensembles are the established strong performer on tabular
     data of this size and are widely reported in the literature on this
     dataset, including the credit card studies reviewed in the report.

Class imbalance is handled by class weighting during training and by threshold
tuning on the validation set. No synthetic oversampling is applied to the
validation or test sets. Resampling evaluation data would change the class
distribution the model is supposed to cope with and would make the reported
precision meaningless.
"""

import json
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import config
import data_prep
import evaluate


def build_models():
    """
    Define the candidate models.

    Hyperparameters are deliberately modest rather than exhaustively tuned.
    The purpose at this stage is a fair comparison between model families, not
    to squeeze the last fraction of a percent out of any single one.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            # Re-weights the loss so that the 492 fraud cases are not simply
            # ignored in favour of the overwhelming majority class.
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        ),
    }


def main():
    print("=" * 78)
    print("PREPARING DATA")
    print("=" * 78)
    data = data_prep.prepare()

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    results = {}

    print("\n" + "=" * 78)
    print("TRAINING")
    print("=" * 78)

    for name, model in build_models().items():
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        # Probability of the positive (fraud) class.
        scores_val = model.predict_proba(X_val)[:, 1]
        scores_test = model.predict_proba(X_test)[:, 1]

        # Threshold chosen on validation data, then applied unchanged to test.
        threshold = evaluate.choose_threshold(y_val, scores_val)
        metrics = evaluate.evaluate(y_test, scores_test, threshold)
        metrics["train_seconds"] = round(elapsed, 1)

        results[name] = metrics
        joblib.dump(model, config.MODEL_DIR / f"{name.lower().replace(' ', '_')}.pkl")

        print(f"  {name:<22} trained in {elapsed:6.1f}s  "
              f"(threshold={threshold:.4f})")

    joblib.dump(data["scaler"], config.MODEL_DIR / "scaler.pkl")

    print("\n" + "=" * 78)
    print("RESULTS ON HELD-OUT TEST SET")
    print("=" * 78)
    print(evaluate.TABLE_HEADER)
    print("-" * 78)
    for name, m in results.items():
        print(evaluate.format_row(name, m))

    print(f"\nTest set: {len(y_test):,} transactions, "
          f"{int(y_test.sum())} of them fraudulent.")

    with open(config.OUTPUT_DIR / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nMetrics written to {config.OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
