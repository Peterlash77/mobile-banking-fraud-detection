"""
Data loading, cleaning and splitting for the ULB credit card dataset.

The design decisions in this module are the ones most likely to be questioned
in a viva, so each is documented with the reasoning behind it rather than just
the mechanics of what the code does.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

import config


def load_raw():
    """Read the raw CSV exactly as downloaded from Kaggle."""
    df = pd.read_csv(config.RAW_DATA_FILE)
    # The Class column is quoted in the raw file, so pandas may read it as text.
    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].astype(int)
    return df


def clean(df, verbose=True):
    """
    Remove exact duplicate rows.

    Why this matters: the raw file contains 1,081 rows that are byte-for-byte
    identical to another row. Because the train/test split is random, an
    identical pair can be separated so that the model is tested on a record it
    has already memorised. That is a form of data leakage and it makes the
    reported performance optimistic. Removing them costs a small amount of data
    and buys an honest evaluation.
    """
    before = len(df)
    if config.DROP_DUPLICATES:
        df = df.drop_duplicates().reset_index(drop=True)
    if verbose:
        removed = before - len(df)
        print(f"[clean] removed {removed:,} duplicate rows "
              f"({before:,} -> {len(df):,})")
    return df


def split(df, verbose=True):
    """
    Produce train / validation / test sets.

    Stratified sampling is used at every split. With only 492 fraud cases in
    roughly 285,000 records, an unstratified random split could easily produce
    a test set whose fraud rate differs substantially from the population,
    which would make the results unstable and hard to defend.

    Three-way split rather than two:
      - train      : fits the model
      - validation : chooses the decision threshold and stops training early
      - test       : used once, at the end, for the reported figures

    Tuning the threshold on the test set and then reporting scores from that
    same set is a common mistake in student projects. It is avoided here.
    """
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_SEED,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=config.VALIDATION_SIZE,
        stratify=y_temp,
        random_state=config.RANDOM_SEED,
    )

    if verbose:
        for name, yy in (("train", y_train), ("val", y_val), ("test", y_test)):
            print(f"[split] {name:5s} n={len(yy):>7,}  fraud={int(yy.sum()):>4,}  "
                  f"rate={100 * yy.mean():.4f}%")

    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_scaler(X_train):
    """
    Fit a scaler on the training data only.

    Only 'Time' and 'Amount' are scaled. V1-V28 are principal components and
    are already on a comparable scale, so rescaling them adds nothing.

    RobustScaler is used rather than StandardScaler because 'Amount' is heavily
    right-skewed (median 22, maximum 25,691). StandardScaler would let those
    extreme values dominate the mean and standard deviation. RobustScaler
    centres on the median and scales by the interquartile range, so the large
    transactions do not distort the transformation.

    The scaler is fitted on the training set alone. Fitting it on the full
    dataset would let information about the test distribution reach the model
    before evaluation, which is leakage.
    """
    scaler = RobustScaler()
    scaler.fit(X_train[config.COLUMNS_TO_SCALE])
    return scaler


def apply_scaler(scaler, X):
    """Apply a previously fitted scaler, returning a new DataFrame."""
    X = X.copy()
    X[config.COLUMNS_TO_SCALE] = scaler.transform(X[config.COLUMNS_TO_SCALE])
    return X


def prepare(verbose=True):
    """Run the whole preparation pipeline and return everything downstream needs."""
    df = load_raw()
    df = clean(df, verbose=verbose)
    X_train, X_val, X_test, y_train, y_val, y_test = split(df, verbose=verbose)

    scaler = fit_scaler(X_train)
    X_train = apply_scaler(scaler, X_train)
    X_val = apply_scaler(scaler, X_val)
    X_test = apply_scaler(scaler, X_test)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler,
        "feature_names": list(X_train.columns),
    }


if __name__ == "__main__":
    data = prepare()
    print(f"\n[prepare] {len(data['feature_names'])} features ready")
