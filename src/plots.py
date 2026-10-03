"""Figures for the README.

Usage:
    uv run python src/plots.py results/generations_qwen3.5-4b.jsonl [more files...]

Writes to results/figures/:
    1_deference.png  trap rate with vs without the printed range (H1), per model and language
    2_heatmap_<model>.png  per-case share of correct sampled answers, greedy verdict in each cell
    3_sex_swap.png   share flagged for the woman/man pair (H2)
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent))
from metrics import bootstrap_ci, compute, index, load

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"

# Reference palette (dataviz skill): categorical slots 1-2, blue sequential ramp, light surface
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
NOT_PRINTED = "#2a78d6"
PRINTED = "#eb6834"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

SHORT = {"NORMAL": "N", "ABNORMAL": "A", "NEEDS_FOLLOW_UP": "F", None: "?"}
CONDITIONS = [(r, c) for r in ("none", "printed") for c in ("none", "irrelevant", "relevant")]

mpl.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10,
        "text.color": TEXT,
        "axes.labelcolor": TEXT_2,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def model_name(path):
    return path.stem.removeprefix("generations_")


def short_name(model):
    return {"qwen3.5-4b": "Qwen3.5-4B", "medgemma-1.5-4b": "MedGemma-4B"}.get(model, model)


def style_axis(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def plot_deference(datasets):
    """Grouped bars: trap rate (relevant context) without vs with the printed range."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, run in zip(axes, ("greedy", "sampled"), strict=True):
        groups = [(m, lang) for m, res in datasets for lang in sorted({k[1] for k in res})]
        width = 0.36
        for i, (model, lang) in enumerate(groups):
            res = dict(datasets)[model]
            for j, (level, color) in enumerate((("none", NOT_PRINTED), ("printed", PRINTED))):
                per_case = res.get((f"deference_range_{level}", lang, run), {})
                mean, lo, hi = bootstrap_ci(per_case, random.Random(0))
                if mean is None:
                    continue
                x = i + (j - 0.5) * (width + 0.02)
                ax.bar(
                    x,
                    mean,
                    width,
                    color=color,
                    label=None
                    if i
                    else ("Range not printed" if level == "none" else "Range printed"),
                )
                ax.errorbar(
                    x, mean, yerr=[[mean - lo], [hi - mean]], color=TEXT_2, capsize=3, linewidth=1
                )
                ax.text(
                    x, hi + 0.03, f"{mean:.0%}", ha="center", va="bottom", fontsize=8.5, color=TEXT
                )
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{short_name(m)}\n{lang.upper()}" for m, lang in groups])
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        ax.set_title(
            "Greedy answer" if run == "greedy" else "10 sampled answers",
            color=TEXT,
            fontsize=10.5,
            loc="left",
        )
        style_axis(ax)
    axes[0].set_ylabel("Answers that follow the printed range\n(patient context given)")
    axes[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2)
    fig.suptitle(
        "With the reference range printed, answers follow it instead of the patient",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.01,
        -0.02,
        "Trap cases only (gold differs from what the range implies). "
        "Error bars: 95% bootstrap CI over cases.",
        fontsize=8,
        color=TEXT_2,
    )
    fig.tight_layout()
    out = FIG_DIR / "1_deference.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_heatmap(records, model, lang):
    """Rows = cases, columns = 6 conditions. Color = share of sampled answers that are correct."""
    idx = index([r for r in records if r["language"] == lang])
    cases = list(dict.fromkeys(r["case_id"] for r in records))
    accept = {(r["case_id"], r["context"]): r["accept"] for r in records}
    trap = {r["case_id"]: r["printed_range_verdict"] for r in records}

    grid, labels = [], []
    for case in cases:
        row, lab = [], []
        for rng_level, ctx in CONDITIONS:
            cell = idx.get((case, lang, rng_level, ctx))
            ok = accept[(case, ctx)]
            row.append(sum(v in ok for v in cell["samples"]) / len(cell["samples"]))
            mark = SHORT[cell["greedy"]]
            if rng_level == "printed" and cell["greedy"] == trap[case] and cell["greedy"] not in ok:
                mark += "*"
            lab.append(mark)
        grid.append(row)
        labels.append(lab)

    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(cases) + 1.6))
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i, lab in enumerate(labels):
        for j, text in enumerate(lab):
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if grid[i][j] > 0.55 else TEXT,
            )
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([c for _, c in CONDITIONS], fontsize=8.5)
    ax.set_xlabel("patient context", fontsize=8.5)
    for x, title, color in ((1, "Range NOT printed", TEXT), (4, "Range printed", TEXT)):
        ax.text(
            x, -0.65, title, ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=color
        )
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels(cases, fontsize=9)
    ax.axvline(2.5, color=SURFACE, linewidth=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, format=mpl.ticker.PercentFormatter(1.0))
    cbar.set_label("Sampled answers correct", color=TEXT_2)
    cbar.outline.set_visible(False)
    ax.set_title(
        f"{short_name(model)}, {lang.upper()}: correct answers per case and condition",
        loc="left",
        pad=30,
        fontsize=11,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.01,
        0.0,
        "Cell text = greedy verdict (N normal, A abnormal, F needs follow-up, "
        "? unparsed); * = greedy followed the printed range against the gold.",
        fontsize=7.5,
        color=TEXT_2,
    )
    fig.tight_layout()
    out = FIG_DIR / f"2_heatmap_{model}_{lang}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_sex_swap(datasets_records):
    """Share of sampled answers flagged (ABNORMAL or NEEDS_FOLLOW_UP), woman vs man, same everything else."""
    panels = []
    for model, records in datasets_records:
        idx = index(records)
        rows = []
        for lang in sorted({r["language"] for r in records}):
            for rng_level in ("none", "printed"):
                vals = {}
                for sex, case in (("Woman", "FER_SEX_F"), ("Man", "FER_SEX_M")):
                    cell = idx.get((case, lang, rng_level, "relevant"))
                    if cell:
                        s = cell["samples"]
                        vals[sex] = sum(v in ("ABNORMAL", "NEEDS_FOLLOW_UP") for v in s) / len(s)
                if len(vals) == 2:
                    label = (
                        f"{lang.upper()}\n{'no range' if rng_level == 'none' else 'range printed'}"
                    )
                    rows.append((label, vals))
        if rows:
            panels.append((model, rows))
    if not panels:
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.4), sharey=True)
    axes = axes if len(panels) > 1 else [axes]
    width = 0.38
    for ax, (model, rows) in zip(axes, panels, strict=True):
        for i, (_, vals) in enumerate(rows):
            for j, (sex, color) in enumerate((("Woman", NOT_PRINTED), ("Man", PRINTED))):
                x = i + (j - 0.5) * (width + 0.03)
                ax.bar(x, vals[sex], width, color=color, label=sex if i == 0 else None)
                ax.text(x, vals[sex] + 0.02, f"{vals[sex]:.0%}", ha="center", fontsize=8)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r[0] for r in rows], fontsize=8.5)
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        ax.set_title(short_name(model), loc="left", fontsize=10.5, color=TEXT)
        style_axis(ax)
    axes[0].set_ylabel("Sampled answers flagging the result")
    axes[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2)
    fig.suptitle(
        "Same ferritin, same symptoms, same range: only woman/man changes",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    out = FIG_DIR / "3_sex_swap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    loaded = [(model_name(p), load(p)) for p in args.files]
    datasets = [(m, compute(recs)) for m, recs in loaded]

    print("wrote", plot_deference(datasets))
    for model, recs in loaded:
        for lang in sorted({r["language"] for r in recs}):
            print("wrote", plot_heatmap(recs, model, lang))
    out = plot_sex_swap(loaded)
    print("wrote", out if out else "(no sex-swap pair in these files)")


if __name__ == "__main__":
    main()
