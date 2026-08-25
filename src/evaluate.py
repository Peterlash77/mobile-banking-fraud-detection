"""
Evaluation metrics for a highly imbalanced classification problem.

The central argument of this module: accuracy is meaningless here. A model
that labels every single transaction as legitimate scores 99.83% accuracy on
this dataset while catching no fraud at all. Any evaluation built on accuracy
would therefore reward exactly the behaviour the system exists to prevent.

The metrics used instead are described in the docstrings below.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_threshold(y_val, scores_val, target="f1"):
    """
    Select the probability cut-off that separates 'fraud' from 'legitimate'.

    The default 0.5 cut-off is arbitrary and performs poorly under severe
    imbalance, because the model's probabilities are pushed towards zero by the
    overwhelming majority class. The threshold is therefore chosen on the
    validation set by sweeping every candidate cut-off and keeping the one with
    the best F1 score.

    This is done on validation data, never on test data. Choosing the threshold
    on the test set and then reporting the resulting score would mean the
    reported figure had been optimised on the data it claims to be measured on.
    """
    precision, recall, thresholds = precision_recall_curve(y_val, scores_val)

    # precision_recall_curve returns one more precision/recall value than
    # thresholds, so the final element is dropped to align the arrays.
    precision, recall = precision[:-1], recall[:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )

    best = int(np.nanargmax(f1))
    return float(thresholds[best])


def evaluate(y_true, scores, threshold):
    """
    Compute the reported metrics for one model.

    PR-AUC (average precision) is the headline metric. It summarises the
    trade-off between precision and recall across all thresholds, and unlike
    ROC-AUC it is not flattered by the huge number of true negatives. With a
    0.17% positive rate, ROC-AUC can look excellent for a model of limited
    practical use, so it is reported for comparability with published work but
    is not the basis for model selection.

    The confusion matrix is reported in raw counts because, for a bank, the
    absolute number of missed frauds and false alarms is the operational
    reality, not a ratio.
    """
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "pr_auc": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def format_row(name, m):
    """One-line summary used for the console comparison table."""
    return (f"{name:<22} {m['pr_auc']:.4f}  {m['roc_auc']:.4f}  "
            f"{m['precision']:.4f}  {m['recall']:.4f}  {m['f1']:.4f}   "
            f"{m['tp']:>3} {m['fp']:>5} {m['fn']:>3}")


TABLE_HEADER = (f"{'Model':<22} {'PR-AUC':>6}  {'ROC-AUC':>7}  {'Prec':>6}  "
                f"{'Recall':>6}  {'F1':>6}    {'TP':>3} {'FP':>5} {'FN':>3}")
