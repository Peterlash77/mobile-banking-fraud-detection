"""
Export a small set of real transactions for the web application to demonstrate.

Why this exists
---------------
The model consumes V1-V28, which are anonymised principal components. A human
being cannot invent plausible values for them, so the interface cannot
realistically ask a user to type them in.

In a production bank, those features would be computed by the core banking
system and handed to the fraud service automatically. This project has no such
system, so a sample of genuine held-out transactions is exported instead. The
interface can then load a real transaction and score it, which demonstrates the
model faithfully without pretending a user could have supplied the vector.

This is a stated limitation of the project, not a workaround being hidden.
Crucially the sample is drawn from the TEST set only -- records the model never
saw during training -- so every demonstration is an honest out-of-sample
prediction.
"""

import numpy as np
import pandas as pd

import config
import data_prep

N_FRAUD = 20
N_LEGIT = 60


def main():
    data = data_prep.prepare(verbose=False)

    # Re-read the raw (unscaled) test rows. The web application applies the
    # scaler itself, so the samples must be exported in their original units --
    # otherwise the scaling would be applied twice.
    df = data_prep.clean(data_prep.load_raw(), verbose=False)
    test_index = data["X_test"].index
    test_rows = df.loc[test_index]

    rng = np.random.RandomState(config.RANDOM_SEED)

    fraud = test_rows[test_rows[config.TARGET_COLUMN] == 1]
    legit = test_rows[test_rows[config.TARGET_COLUMN] == 0]

    sample = pd.concat([
        fraud.sample(min(N_FRAUD, len(fraud)), random_state=config.RANDOM_SEED),
        legit.sample(N_LEGIT, random_state=config.RANDOM_SEED),
    ])
    sample = sample.sample(frac=1, random_state=config.RANDOM_SEED)  # shuffle

    out = config.PROJECT_ROOT / "webapp" / "sample_transactions.csv"
    out.parent.mkdir(exist_ok=True)
    sample.to_csv(out, index=False)

    print(f"Exported {len(sample)} sample transactions to {out}")
    print(f"  fraudulent : {int(sample[config.TARGET_COLUMN].sum())}")
    print(f"  legitimate : {int((sample[config.TARGET_COLUMN] == 0).sum())}")
    print("\nAll drawn from the held-out test set (never seen during training).")


if __name__ == "__main__":
    main()
