"""Compare harnesses: does a better setup remove the trap without adding false alarms?

For each model x harness x language (10 sampled answers per prompt):
  trap rate      trap cases, relevant context, range printed: share of answers
                 equal to the printed-range verdict (the plain metrics.py definition)
  false alarms   prompts without relevant context whose gold is NORMAL: share of
                 answers that flag them (ABNORMAL or NEEDS_FOLLOW_UP) when the
                 accept set does not allow that flag
  correct        all prompts: share of answers in the accept set
For the agent harness, also the tool-call rate and the trap rate with and
without a tool call. 95% bootstrap CIs over cases.

Usage:
    uv run python src/harness.py results/qwen3.5-4b/generations_qwen3.5-4b.jsonl \
        results/qwen3.5-4b/*/generations_*.jsonl
Writes results/harness_summary.csv and results/figures/4_harnesses.png.
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from metrics import bootstrap_ci, load
from plots import FIG_DIR, GRID, PRINTED, TEXT, TEXT_2, short_name, style_axis

ROOT = Path(__file__).resolve().parent.parent
HARNESS_ORDER = ["plain", "instruction", "guidelines", "agent"]
FALSE_ALARM = "#1baf7a"  # categorical slot 3; bars carry value labels (contrast relief)
FLAGGED = ("ABNORMAL", "NEEDS_FOLLOW_UP")


def model_and_harness(path, records):
    harness = records[0].get("harness", "plain")
    model = path.stem.removeprefix("generations_").removesuffix(f"_{harness}")
    return model, harness


def per_case(records, select, hit):
    """{case: share of selected sampled answers where hit(record)}"""
    groups = defaultdict(list)
    for r in records:
        if r["run"] == "sample" and select(r):
            groups[r["case_id"]].append(hit(r))
    return {case: sum(v) / len(v) for case, v in groups.items()}


def is_trap_prompt(r):
    return r["gold_relevant"] != r["printed_range_verdict"] and r["context"] == "relevant"


def summarise(path):
    records = load(path)
    model, harness = model_and_harness(path, records)
    # gold with relevant context, per case (defines which cases can test the trap)
    gold_rel = {r["case_id"]: r["gold"] for r in records if r["context"] == "relevant"}
    for r in records:
        r["gold_relevant"] = gold_rel[r["case_id"]]

    rows = []
    for lang in sorted({r["language"] for r in records}):
        recs = [r for r in records if r["language"] == lang]
        measures = {
            "trap_rate": per_case(
                recs,
                lambda r: is_trap_prompt(r) and r["range"] == "printed",
                lambda r: r["verdict"] == r["printed_range_verdict"],
            ),
            "false_alarm_rate": per_case(
                recs,
                lambda r: r["context"] != "relevant" and r["gold"] == "NORMAL",
                # a flag counts as a false alarm only if the accept set does not allow it
                lambda r: r["verdict"] in FLAGGED and r["verdict"] not in r["accept"],
            ),
            "correct": per_case(recs, lambda r: True, lambda r: r["verdict"] in r["accept"]),
        }
        if harness == "agent":
            measures["tool_call_rate"] = per_case(recs, lambda r: True, lambda r: r["tool_called"])
            for called in (True, False):
                measures[f"trap_rate_tool_{'called' if called else 'not_called'}"] = per_case(
                    recs,
                    lambda r, c=called: (
                        is_trap_prompt(r) and r["range"] == "printed" and r["tool_called"] == c
                    ),
                    lambda r: r["verdict"] == r["printed_range_verdict"],
                )
        for name, values in measures.items():
            mean, lo, hi = bootstrap_ci(values, random.Random(0))
            rows.append(
                {
                    "model": model,
                    "harness": harness,
                    "language": lang,
                    "measure": name,
                    "n_cases": len(values),
                    "mean": None if mean is None else round(mean, 3),
                    "ci_low": None if lo is None else round(lo, 3),
                    "ci_high": None if hi is None else round(hi, 3),
                }
            )
    return rows


def plot(rows):
    models = list(dict.fromkeys(r["model"] for r in rows))
    langs = sorted({r["language"] for r in rows})
    fig, axes = plt.subplots(
        len(models),
        len(langs),
        figsize=(5.2 * len(langs), 3.4 * len(models)),
        sharey=True,
        squeeze=False,
    )
    width = 0.38
    for i, model in enumerate(models):
        for j, lang in enumerate(langs):
            ax = axes[i][j]
            harnesses = [
                h
                for h in HARNESS_ORDER
                if any(r["model"] == model and r["harness"] == h for r in rows)
            ]
            for k, harness in enumerate(harnesses):
                for m, (measure, color) in enumerate(
                    (("trap_rate", PRINTED), ("false_alarm_rate", FALSE_ALARM))
                ):
                    row = next(
                        (
                            r
                            for r in rows
                            if (r["model"], r["harness"], r["language"], r["measure"])
                            == (model, harness, lang, measure)
                        ),
                        None,
                    )
                    if not row or row["mean"] is None:
                        continue
                    x = k + (m - 0.5) * (width + 0.03)
                    label = (
                        None
                        if (i, j, k) != (0, 0, 0)
                        else (
                            "Trap rate (follows the printed range)"
                            if m == 0
                            else "False alarms (flags a normal result)"
                        )
                    )
                    ax.bar(x, row["mean"], width, color=color, label=label)
                    ax.text(
                        x,
                        row["mean"] + 0.02,
                        f"{row['mean']:.0%}",
                        ha="center",
                        fontsize=8,
                        color=TEXT,
                    )
            ax.set_xticks(range(len(harnesses)))
            ax.set_xticklabels(harnesses, fontsize=9)
            ax.set_ylim(0, 1.1)
            ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
            ax.set_title(
                f"{short_name(model)}, {lang.upper()}", loc="left", fontsize=10, color=TEXT
            )
            style_axis(ax)
            ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    axes[0][0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.12), ncol=2, fontsize=9)
    fig.suptitle(
        "Harnesses: fewer traps, but at what cost?",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.01,
        -0.02,
        "10 sampled answers per prompt. Trap rate: trap cases with patient context and the range "
        "printed. False alarms: prompts without relevant context whose correct answer is NORMAL, flagged with "
        "a verdict that is not an accepted answer.",
        fontsize=7.5,
        color=TEXT_2,
    )
    fig.tight_layout()
    out = FIG_DIR / "4_harnesses.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()

    rows = [row for path in args.files for row in summarise(path)]
    out_csv = ROOT / "results" / "harness_summary.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for r in rows:
        ci = "" if r["ci_low"] is None else f"  [{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
        mean = "n/a" if r["mean"] is None else f"{r['mean']:.2f}"
        print(
            f"{r['model']:16} {r['harness']:12} {r['language']} {r['measure']:28} n={r['n_cases']:2}  {mean}{ci}"
        )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("wrote", out_csv.relative_to(ROOT), "and", plot(rows).relative_to(ROOT))


if __name__ == "__main__":
    main()
