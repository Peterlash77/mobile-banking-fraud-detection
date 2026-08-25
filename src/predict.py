"""
The single point of contact between the trained model and the web application.

This module exists so that the Flask application never has to know anything
about machine learning. It loads the saved model once, applies exactly the
same preprocessing that was used during training, and returns a decision.

If the model is ever retrained or replaced, nothing in the web application
changes -- only the files this module loads.
"""

import joblib
import numpy as np
import pandas as pd

import config

# The feature order the model was trained on. Order matters: the model
# identifies features by position, not by name, so a reordered input would
# silently produce nonsense rather than raising an error.
FEATURE_ORDER = (["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"])

# Loaded once at import rather than on every request. Loading a model from
# disk takes far longer than making a prediction, so doing it per request
# would dominate the response time.
_MODEL = None
_SCALER = None


def _load():
    """Lazily load the model and scaler into module-level cache."""
    global _MODEL, _SCALER
    if _MODEL is None:
        _MODEL = joblib.load(config.MODEL_DIR / "random_forest.pkl")
        _SCALER = joblib.load(config.MODEL_DIR / "scaler.pkl")
    return _MODEL, _SCALER


def score_transaction(features: dict, threshold: float = None):
    """
    Score a single transaction.

    Parameters
    ----------
    features : dict
        Must contain every key in FEATURE_ORDER: 'Time', 'V1'...'V28', 'Amount'.
    threshold : float, optional
        Probability above which a transaction is flagged. Defaults to the
        value tuned on the validation set during training.

    Returns
    -------
    dict with the fraud probability, the binary decision and the threshold used.

    The same scaler object fitted during training is applied here. Re-fitting a
    scaler on incoming data would shift the feature distribution away from what
    the model was trained on and quietly degrade every prediction.
    """
    model, scaler = _load()

    if threshold is None:
        threshold = get_default_threshold()

    missing = [f for f in FEATURE_ORDER if f not in features]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    row = pd.DataFrame([{f: float(features[f]) for f in FEATURE_ORDER}])
    row[config.COLUMNS_TO_SCALE] = scaler.transform(row[config.COLUMNS_TO_SCALE])

    probability = float(model.predict_proba(row)[0][1])

    return {
        "probability": probability,
        "is_fraud": bool(probability >= threshold),
        "threshold": float(threshold),
        # A coarse band, used only for presentation in the interface.
        "risk_band": _risk_band(probability, threshold),
    }


def _risk_band(probability, threshold):
    """Translate a probability into a label a non-technical user can act on."""
    if probability >= threshold:
        return "HIGH"
    if probability >= threshold / 2:
        return "MEDIUM"
    return "LOW"


def get_default_threshold():
    """
    Read the tuned decision threshold produced by training.

    Falls back to 0.5 only if the results file is absent, which would mean the
    model has not yet been trained.
    """
    import json
    path = config.OUTPUT_DIR / "results.json"
    if not path.exists():
        return 0.5
    with open(path) as fh:
        results = json.load(fh)
    return results.get("Random Forest", {}).get("threshold", 0.5)
