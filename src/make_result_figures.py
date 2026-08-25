"""
Generate the evaluation figures used in Chapter Five of the project report.

These are produced from the saved models and the held-out test set, so every
figure in the report traces back to the same run that produced results.json.
Regenerating them cannot silently disagree with the numbers in the text.

Run:  python src/make_result_figures.py
"""

import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

import config
import data_prep
import evaluate

FIG_DIR = config.OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

MODELS = {
    "Logistic Regression": ("logistic_regression.pkl", "#8b9dc3"),
    "Random Forest": ("random_forest.pkl", "#c9575c"),
}


def _save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.name}")


def load_scores():
    """Score the held-out test set with each saved model."""
    data = data_prep.prepare(verbose=False)
    out = {"y_test": data["y_test"], "y_val": data["y_val"]}
    for name, (fname, _) in MODELS.items():
        model = joblib.load(config.MODEL_DIR / fname)
        out[name] = {
            "test": model.predict_proba(data["X_test"])[:, 1],
            "val": model.predict_proba(data["X_val"])[:, 1],
        }
    return out


# ---------------------------------------------------------------------------
# Figure 7 - Precision-recall curves
# ---------------------------------------------------------------------------

def fig_pr_curves(s):
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    y = s["y_test"]

    for name, (_, colour) in MODELS.items():
        scores = s[name]["test"]
        precision, recall, _ = precision_recall_curve(y, scores)
        ap = average_precision_score(y, scores)
        ax.plot(recall, precision, color=colour, linewidth=1.6,
                label=f"{name} (PR-AUC = {ap:.4f})")

    # A classifier with no skill sits at the positive class rate.
    baseline = y.mean()
    ax.axhline(baseline, color="#999999", linestyle="--", linewidth=1.0,
               label=f"No-skill classifier ({baseline:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    ax.grid(linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    _save(fig, "fig7_pr_curves.png")


# ---------------------------------------------------------------------------
# Figure 8 - ROC curves
# ---------------------------------------------------------------------------

def fig_roc_curves(s):
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    y = s["y_test"]

    for name, (_, colour) in MODELS.items():
        scores = s[name]["test"]
        fpr, tpr, _ = roc_curve(y, scores)
        auc = roc_auc_score(y, scores)
        ax.plot(fpr, tpr, color=colour, linewidth=1.6,
                label=f"{name} (ROC-AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1.0,
            label="Random classifier")

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.grid(linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    _save(fig, "fig8_roc_curves.png")


# ---------------------------------------------------------------------------
# Figure 9 - Confusion matrices at the selected thresholds
# ---------------------------------------------------------------------------

def fig_confusion(s):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    y = s["y_test"]

    for ax, (name, _) in zip(axes, MODELS.items()):
        threshold = evaluate.choose_threshold(s["y_val"], s[name]["val"])
        pred = (s[name]["test"] >= threshold).astype(int)
        cm = confusion_matrix(y, pred)

        disp = ConfusionMatrixDisplay(cm, display_labels=["Legitimate", "Fraud"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=",d")
        ax.set_title(f"{name}\n(threshold = {threshold:.4f})", fontsize=9.5)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
        for txt in disp.text_.ravel():
            txt.set_fontsize(9)

    fig.tight_layout()
    _save(fig, "fig9_confusion.png")


# ---------------------------------------------------------------------------
# Figure 10 - Metric comparison
# ---------------------------------------------------------------------------

def fig_comparison(s):
    y = s["y_test"]
    metrics = ["PR-AUC", "Precision", "Recall", "F1"]
    values = {}

    for name in MODELS:
        threshold = evaluate.choose_threshold(s["y_val"], s[name]["val"])
        m = evaluate.evaluate(y, s[name]["test"], threshold)
        values[name] = [m["pr_auc"], m["precision"], m["recall"], m["f1"]]

    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 4.0))

    for i, (name, (_, colour)) in enumerate(MODELS.items()):
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, values[name], width, label=name,
                      color=colour, edgecolor="#444444", linewidth=0.7)
        for bar, v in zip(bars, values[name]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                    f"{v:.3f}", ha="center", fontsize=7.5)

    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    _save(fig, "fig10_comparison.png")


# ---------------------------------------------------------------------------
# Figure 11 - Predicted probability distributions
# ---------------------------------------------------------------------------

def fig_probability_distributions(s):
    """
    Shows how each model distributes its predicted probabilities.

    This figure exists to make a specific point discussed in Chapter Five:
    class weighting drives logistic regression's outputs to the extremes, so
    its probabilities rank transactions correctly but cannot be read as
    calibrated likelihoods.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    y = s["y_test"]
    bins = np.linspace(0, 1, 41)

    for ax, (name, _) in zip(axes, MODELS.items()):
        scores = s[name]["test"]
        ax.hist(scores[y == 0], bins=bins, color="#7fa8d4",
                edgecolor="none", alpha=0.85, label="Legitimate")
        ax.hist(scores[y == 1], bins=bins, color="#c9575c",
                edgecolor="none", alpha=0.85, label="Fraud")
        ax.set_yscale("log")
        ax.set_title(name, fontsize=9.5)
        ax.set_xlabel("Predicted probability of fraud", fontsize=8.5)
        ax.set_ylabel("Count (log scale)", fontsize=8.5)
        ax.legend(fontsize=8)
        ax.grid(linestyle=":", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.tight_layout()
    _save(fig, "fig11_probability_distributions.png")


if __name__ == "__main__":
    print(f"Writing evaluation figures to {FIG_DIR}")
    s = load_scores()
    fig_pr_curves(s)
    fig_roc_curves(s)
    fig_confusion(s)
    fig_comparison(s)
    fig_probability_distributions(s)
    print("Done.")
