"""
Flask application for the fraud detection system.

Architecture
------------
This is the application layer of a three-tier design:

    Presentation  ->  HTML templates rendered by Flask (login, dashboard,
                      transaction scoring, prediction history)
    Application   ->  this file: routing, session handling, input validation,
                      and the call into the model
    Data          ->  database.py (SQLite) for users and the prediction audit
                      log; the trained model file on disk

The important separation: this file contains no machine learning code at all.
It calls src/predict.py, which owns everything about the model. That boundary
means the model can be retrained or swapped without touching the web layer.
"""

import csv
import sys
from functools import wraps
from pathlib import Path

from flask import (
    Flask, flash, redirect, render_template, request, session, url_for
)

# Make the src/ package importable so the web layer can reach the model.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import predict  # noqa: E402
import database  # noqa: E402

app = Flask(__name__)

# Signs the session cookie so it cannot be tampered with by the browser.
# In a deployed system this would be read from an environment variable rather
# than written in source.
app.secret_key = "change-this-in-production-4f8a9c2e1b7d"

SAMPLES_FILE = Path(__file__).resolve().parent / "sample_transactions.csv"


def login_required(view):
    """
    Protect a route so that only authenticated users can reach it.

    Without this, anyone could open /dashboard directly by typing the URL and
    bypass the login page entirely.
    """
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def load_samples():
    """Read the exported held-out transactions used for demonstration."""
    if not SAMPLES_FILE.exists():
        return []
    with open(SAMPLES_FILE, newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return redirect(url_for("dashboard" if "user_id" in session else "login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = database.verify_user(
            request.form.get("username", "").strip(),
            request.form.get("password", ""),
        )
        if user:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("dashboard"))
        # Deliberately vague: naming which field was wrong would tell an
        # attacker which usernames exist.
        flash("Invalid username or password.", "danger")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if len(username) < 3 or len(password) < 6:
            flash("Username must be 3+ characters and password 6+.", "danger")
        elif database.create_user(username, password,
                                  request.form.get("full_name", "").strip()):
            flash("Account created. You can now log in.", "success")
            return redirect(url_for("login"))
        else:
            flash("That username is already taken.", "danger")
    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template(
        "dashboard.html",
        stats=database.dashboard_stats(),
        recent=database.recent_predictions(limit=10),
        threshold=predict.get_default_threshold(),
    )


@app.route("/check", methods=["GET", "POST"])
@login_required
def check():
    """Score a transaction and record the decision."""
    samples = load_samples()
    result = None
    submitted = None

    if request.method == "POST":
        try:
            source = request.form.get("source", "manual")

            if source == "sample":
                # Score a real held-out transaction chosen from the sample set.
                idx = int(request.form.get("sample_index", 0))
                row = samples[idx]
                features = {f: float(row[f]) for f in predict.FEATURE_ORDER}
                actual = int(float(row["Class"]))
            else:
                # Manual entry. Any V-feature left blank defaults to zero,
                # which is the mean of a PCA component.
                features = {}
                for f in predict.FEATURE_ORDER:
                    raw = request.form.get(f, "").strip()
                    features[f] = float(raw) if raw else 0.0
                actual = None

            result = predict.score_transaction(features)
            submitted = features

            database.log_prediction(
                user_id=session["user_id"],
                amount=features["Amount"],
                time_seconds=features["Time"],
                result=result,
                actual_class=actual,
                source=source,
            )
            result["actual_class"] = actual

        except (ValueError, IndexError, KeyError) as exc:
            flash(f"Could not score that transaction: {exc}", "danger")

    return render_template(
        "check.html",
        samples=samples,
        result=result,
        submitted=submitted,
        features=predict.FEATURE_ORDER,
    )


@app.route("/history")
@login_required
def history():
    return render_template("history.html",
                           predictions=database.recent_predictions(limit=100))


database.init_db()

if __name__ == "__main__":
    print("Database ready. Default login: admin / admin123")
    app.run(debug=True, port=5000)
