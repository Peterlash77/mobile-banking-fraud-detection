"""
Generate the diagrams used in the project proposal and final report.

All figures are produced with matplotlib so that they can be regenerated
exactly, rather than drawn by hand in a diagramming tool and lost. Each figure
is written to outputs/figures/ as a PNG at 200 dpi, which is sufficient for
printing inside a Word document.

Run:  python src/make_figures.py
"""

import matplotlib
matplotlib.use("Agg")          # no display needed; write straight to file

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

import config

FIG_DIR = config.OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# A restrained palette. Academic documents are often printed in greyscale, so
# the fills are distinguishable by lightness as well as by hue.
INK = "#1a1a1a"
BOX_PRESENTATION = "#dce6f2"
BOX_APPLICATION = "#d9e8d4"
BOX_DATA = "#f2e2d0"
BOX_NEUTRAL = "#eeeeee"
EDGE = "#555555"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
})


def _box(ax, x, y, w, h, text, fc, fontsize=9, weight="normal"):
    """Draw a rounded box with centred text."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=fc, edgecolor=EDGE, linewidth=0.9,
    ))
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            color=INK, weight=weight, linespacing=1.4)


def _arrow(ax, start, end, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=11,
        linewidth=0.9, color=EDGE, linestyle=ls,
        shrinkA=2, shrinkB=2,
    ))


def _save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.name}")


# ---------------------------------------------------------------------------
# Figure 1 - Three-tier system architecture
# ---------------------------------------------------------------------------

def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    # Tier bands
    for y, label, colour in [
        (7.4, "PRESENTATION LAYER", BOX_PRESENTATION),
        (4.2, "APPLICATION LAYER", BOX_APPLICATION),
        (1.0, "DATA LAYER", BOX_DATA),
    ]:
        ax.add_patch(Rectangle((0.3, y), 9.4, 2.4, facecolor=colour,
                               edgecolor=EDGE, linewidth=1.1, alpha=0.35))
        ax.text(0.5, y + 2.15, label, fontsize=8.5, weight="bold",
                color=EDGE, va="center")

    # Presentation
    for i, t in enumerate(["Login /\nRegistration", "Dashboard",
                           "Transaction\nScoring", "Prediction\nHistory"]):
        _box(ax, 0.7 + i * 2.25, 7.7, 2.0, 1.5, t, BOX_PRESENTATION)

    # Application
    _box(ax, 0.7, 4.5, 2.5, 1.7, "Flask\nRouting &\nSessions", BOX_APPLICATION)
    _box(ax, 3.6, 4.5, 2.6, 1.7, "Input Validation\n&\nPreprocessing", BOX_APPLICATION)
    _box(ax, 6.6, 4.5, 2.7, 1.7, "Prediction Service\n(trained model\n+ scaler)",
         BOX_APPLICATION, weight="bold")

    # Data
    _box(ax, 0.9, 1.3, 2.6, 1.6, "Users\nTable", BOX_DATA)
    _box(ax, 3.9, 1.3, 2.6, 1.6, "Predictions\nAudit Log", BOX_DATA)
    _box(ax, 6.9, 1.3, 2.4, 1.6, "Serialised\nModel Files", BOX_DATA)

    # Flows
    _arrow(ax, (5.0, 7.6), (5.0, 6.3), style="<|-|>")
    _arrow(ax, (5.0, 4.4), (5.0, 3.0), style="<|-|>")
    _arrow(ax, (3.2, 5.35), (3.55, 5.35))
    _arrow(ax, (6.25, 5.35), (6.55, 5.35))

    ax.text(5.15, 6.95, "HTTP request / response", fontsize=7.5,
            color=EDGE, style="italic")
    ax.text(5.15, 3.75, "read / write", fontsize=7.5,
            color=EDGE, style="italic")

    _save(fig, "fig1_architecture.png")


# ---------------------------------------------------------------------------
# Figure 2 - Data flow through the prediction pipeline
# ---------------------------------------------------------------------------

def fig_dataflow():
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.set_xlim(-0.2, 12.6); ax.set_ylim(0.2, 6.3); ax.axis("off")

    steps = [
        (0.1, "Transaction\nsubmitted", BOX_PRESENTATION),
        (2.55, "Assemble\nfeature vector\n(Time, V1-V28,\nAmount)", BOX_APPLICATION),
        (5.0, "Apply saved\nscaler to\nTime & Amount", BOX_APPLICATION),
        (7.45, "Model returns\nfraud\nprobability", BOX_APPLICATION),
        (9.9, "Compare to\nthreshold", BOX_APPLICATION),
    ]
    for x, t, c in steps:
        _box(ax, x, 4.1, 2.1, 1.9, t, c, fontsize=8)
        if x < 9.9:
            _arrow(ax, (x + 2.12, 5.05), (x + 2.43, 5.05))

    # Outcomes
    _box(ax, 8.15, 1.7, 1.8, 1.2, "FLAGGED", "#f6d5d5", fontsize=8.5, weight="bold")
    _box(ax, 10.4, 1.7, 1.8, 1.2, "CLEARED", "#d8ecd8", fontsize=8.5, weight="bold")
    _arrow(ax, (10.6, 4.05), (9.35, 2.95))
    _arrow(ax, (11.3, 4.05), (11.3, 2.95))
    ax.text(9.62, 3.42, "≥ threshold", fontsize=7.5, color=EDGE, ha="right")
    ax.text(11.45, 3.42, "< threshold", fontsize=7.5, color=EDGE)

    # Audit log
    _box(ax, 4.3, 0.5, 2.7, 1.1, "Prediction written\nto audit log", BOX_DATA,
         fontsize=8)
    _arrow(ax, (8.1, 1.9), (7.05, 1.25), ls="--")
    _arrow(ax, (10.35, 1.8), (7.05, 1.05), ls="--")

    _save(fig, "fig2_dataflow.png")


# ---------------------------------------------------------------------------
# Figure 3 - Use case diagram
# ---------------------------------------------------------------------------

def fig_usecase():
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 9); ax.axis("off")

    # System boundary
    ax.add_patch(Rectangle((2.9, 0.6), 5.6, 7.8, fill=False,
                           edgecolor=EDGE, linewidth=1.2))
    ax.text(5.7, 8.05, "Fraud Detection System", fontsize=9,
            weight="bold", ha="center", color=EDGE)

    # Actor (stick figure)
    ax.plot([1.5], [5.3], marker="o", markersize=11, color=INK,
            markerfacecolor="white", markeredgewidth=1.2)
    ax.plot([1.5, 1.5], [4.15, 5.0], color=INK, linewidth=1.2)
    ax.plot([0.95, 2.05], [4.65, 4.65], color=INK, linewidth=1.2)
    ax.plot([1.5, 1.0], [4.15, 3.4], color=INK, linewidth=1.2)
    ax.plot([1.5, 2.0], [4.15, 3.4], color=INK, linewidth=1.2)
    ax.text(1.5, 2.75, "Bank Officer /\nAnalyst", ha="center", va="top",
            fontsize=8.5, color=INK)

    cases = ["Register account", "Log in", "Score a transaction",
             "View dashboard", "View prediction history", "Log out"]
    for i, c in enumerate(cases):
        y = 7.1 - i * 1.15
        e = plt.matplotlib.patches.Ellipse((5.7, y + 0.25), 4.4, 0.85,
                                           facecolor=BOX_NEUTRAL,
                                           edgecolor=EDGE, linewidth=0.9)
        ax.add_patch(e)
        ax.text(5.7, y + 0.25, c, ha="center", va="center", fontsize=8.5)
        ax.plot([2.1, 3.5], [5.0, y + 0.25], color=EDGE, linewidth=0.7)

    _save(fig, "fig3_usecase.png")


# ---------------------------------------------------------------------------
# Figure 4 - Entity relationship diagram
# ---------------------------------------------------------------------------

def fig_erd():
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")

    def table(x, y, title, rows, colour):
        h = 0.52
        ax.add_patch(Rectangle((x, y), 4.3, h, facecolor=colour,
                               edgecolor=EDGE, linewidth=1.0))
        ax.text(x + 2.15, y + h / 2, title, ha="center", va="center",
                fontsize=9, weight="bold")
        for i, (col, typ) in enumerate(rows):
            yy = y - (i + 1) * h
            ax.add_patch(Rectangle((x, yy), 4.3, h, facecolor="white",
                                   edgecolor=EDGE, linewidth=0.6))
            ax.text(x + 0.14, yy + h / 2, col, va="center", fontsize=7.8)
            ax.text(x + 4.16, yy + h / 2, typ, va="center", ha="right",
                    fontsize=7.2, color="#777777")

    table(0.4, 6.6, "users", [
        ("id  (PK)", "INTEGER"),
        ("username  (UNIQUE)", "TEXT"),
        ("password_hash", "TEXT"),
        ("full_name", "TEXT"),
        ("created_at", "TIMESTAMP"),
    ], BOX_PRESENTATION)

    table(7.2, 6.6, "predictions", [
        ("id  (PK)", "INTEGER"),
        ("user_id  (FK)", "INTEGER"),
        ("amount", "REAL"),
        ("time_seconds", "REAL"),
        ("probability", "REAL"),
        ("is_fraud", "INTEGER"),
        ("risk_band", "TEXT"),
        ("threshold", "REAL"),
        ("actual_class", "INTEGER"),
        ("source", "TEXT"),
        ("created_at", "TIMESTAMP"),
    ], BOX_DATA)

    # One-to-many relationship
    ax.annotate("", xy=(7.15, 5.9), xytext=(4.75, 5.9),
                arrowprops=dict(arrowstyle="-", color=EDGE, linewidth=1.0))
    ax.text(5.95, 6.12, "1", fontsize=8.5, ha="center", weight="bold")
    ax.text(6.95, 6.12, "N", fontsize=8.5, ha="center", weight="bold")
    ax.text(5.95, 5.55, "generates", fontsize=7.5, ha="center",
            style="italic", color=EDGE)

    _save(fig, "fig4_erd.png")


# ---------------------------------------------------------------------------
# Figure 5 - Project timeline (Gantt)
# ---------------------------------------------------------------------------

def fig_gantt():
    tasks = [
        ("Requirements & literature review", 0, 3),
        ("Dataset acquisition & exploration", 2, 3),
        ("Data preprocessing pipeline", 4, 3),
        ("Model training & tuning", 6, 4),
        ("Model evaluation & selection", 9, 3),
        ("Web application development", 10, 5),
        ("Model–application integration", 14, 3),
        ("System testing & validation", 16, 3),
        ("Documentation & report writing", 17, 4),
    ]

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    colours = plt.cm.Blues(np.linspace(0.35, 0.72, len(tasks)))

    for i, (name, start, dur) in enumerate(tasks):
        y = len(tasks) - i - 1
        ax.barh(y, dur, left=start, height=0.55,
                color=colours[i], edgecolor=EDGE, linewidth=0.7)
        ax.text(start + dur + 0.25, y, f"{dur}w",
                va="center", fontsize=7.5, color=EDGE)

    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t[0] for t in reversed(tasks)], fontsize=8.5)
    ax.set_xlabel("Week", fontsize=9)
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 23, 2))
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    _save(fig, "fig5_gantt.png")


# ---------------------------------------------------------------------------
# Figure 6 - Class distribution
# ---------------------------------------------------------------------------

def fig_class_distribution():
    """
    Illustrates the class imbalance in the dataset.

    This describes the *data*, not an experimental result, so it is appropriate
    for the proposal stage. Two panels are used because on a linear scale the
    fraud bar is invisible -- which is itself the point being made.
    """
    import data_prep

    df = data_prep.clean(data_prep.load_raw(), verbose=False)
    counts = df[config.TARGET_COLUMN].value_counts().sort_index()
    legit, fraud = int(counts[0]), int(counts[1])
    total = legit + fraud

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8))

    for ax, log in zip(axes, (False, True)):
        bars = ax.bar(["Legitimate", "Fraudulent"], [legit, fraud],
                      color=["#7fa8d4", "#c9575c"],
                      edgecolor=EDGE, linewidth=0.8, width=0.55)
        if log:
            ax.set_yscale("log")
            ax.set_xlabel("Logarithmic scale", fontsize=9, labelpad=8)
        else:
            ax.set_xlabel("Linear scale", fontsize=9, labelpad=8)
        for bar, v in zip(bars, (legit, fraud)):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * (1.12 if log else 1.0) +
                    (0 if log else total * 0.015),
                    f"{v:,}\n({100 * v / total:.3f}%)",
                    ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_ylabel("Number of transactions", fontsize=8.5)

    fig.tight_layout()
    _save(fig, "fig6_class_distribution.png")


if __name__ == "__main__":
    print(f"Writing figures to {FIG_DIR}")
    fig_architecture()
    fig_dataflow()
    fig_usecase()
    fig_erd()
    fig_gantt()
    fig_class_distribution()
    print("Done.")
