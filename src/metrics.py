"""Compute the study metrics from one or more generations files.

Usage:
    uv run python src/metrics.py results/generations_qwen3.5-4b.jsonl [more files...]

Writes results/metrics_<model>.csv (one row per metric x language x run type)
and prints a summary. CIs are 95% bootstrap intervals over cases.
"""

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

VERDICT_RE = re.compile(r"VERDICT\s*:\s*(NORMAL|ABNORMAL|NEEDS_FOLLOW_UP)\b")
N_BOOT = 2000
SEED = 0


def parse_verdict(text):
    m = VERDICT_RE.search(text)
    return m.group(1) if m else None


def load(path):
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    for r in records:
        r["verdict"] = parse_verdict(r["output"])
    return records


def index(records):
    """{(case, lang, range, context): {"greedy": verdict, "samples": [verdicts]}}"""
    out = defaultdict(lambda: {"greedy": None, "samples": []})
    for r in records:
        key = (r["case_id"], r["language"], r["range"], r["context"])
        if r["run"] == "greedy":
            out[key]["greedy"] = r["verdict"]
        else:
            out[key]["samples"].append(r["verdict"])
    return out


def bootstrap_ci(per_case, rng):
    """Mean of per-case values with a 95% bootstrap CI, resampling cases."""
    values = [v for v in per_case.values() if v is not None]
    if not values:
        return None, None, None
    mean = sum(values) / len(values)
    boots = []
    for _ in range(N_BOOT):
        sample = [rng.choice(values) for _ in values]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    return mean, boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT) - 1]


def share(verdicts, condition):
    """Share of verdicts (a list) meeting `condition`; unparsed answers count as not meeting it."""
    return sum(condition(v) for v in verdicts) / len(verdicts) if verdicts else None


def compute(records):
    """Per-case values for each metric, language and run type ('greedy' or 'sampled')."""
    meta = {}
    for r in records:
        meta[r["case_id"]] = r
    gold = defaultdict(dict)
    for r in records:
        gold[(r["case_id"], r["context"])] = (r["gold"], r["accept"], r["printed_range_verdict"])
    idx = index(records)
    langs = sorted({r["language"] for r in records})

    results = defaultdict(dict)  # (metric, lang, run) -> {case: value}
    for lang in langs:
        for run in ("greedy", "sampled"):

            def verdicts(case, rng_level, ctx, run=run, lang=lang):
                cell = idx.get((case, lang, rng_level, ctx))
                if cell is None:
                    return []
                return [cell["greedy"]] if run == "greedy" else cell["samples"]

            for case in meta:
                g_rel, accept_rel, trap = gold[(case, "relevant")]

                # Accuracy over all 6 conditions of the case
                accs = []
                for rng_level in ("none", "printed"):
                    for ctx in ("none", "relevant", "irrelevant"):
                        _, accept, _ = gold[(case, ctx)]
                        accs.append(
                            share(verdicts(case, rng_level, ctx), lambda v, a=accept: v in a)
                        )
                accs = [a for a in accs if a is not None]
                results[("accuracy", lang, run)][case] = sum(accs) / len(accs) if accs else None

                # Deference (H1): relevant context, trap cases only (gold != printed-range verdict)
                if g_rel != trap:
                    for rng_level in ("none", "printed"):
                        results[(f"deference_range_{rng_level}", lang, run)][case] = share(
                            verdicts(case, rng_level, "relevant"), lambda v, t=trap: v == t
                        )
                        results[(f"correct_relevant_range_{rng_level}", lang, run)][case] = share(
                            verdicts(case, rng_level, "relevant"), lambda v, a=accept_rel: v in a
                        )

                # Format failures
                all_v = [
                    v
                    for rng_level in ("none", "printed")
                    for ctx in ("none", "relevant", "irrelevant")
                    for v in verdicts(case, rng_level, ctx)
                ]
                results[("format_failure", lang, run)][case] = share(all_v, lambda v: v is None)

                # Irrelevant-context stability: same verdict as with no context
                if run == "greedy":
                    same = []
                    for rng_level in ("none", "printed"):
                        a = verdicts(case, rng_level, "none")
                        b = verdicts(case, rng_level, "irrelevant")
                        if a and b:
                            same.append(float(a[0] == b[0]))
                    results[("irrelevant_stability", lang, run)][case] = (
                        sum(same) / len(same) if same else None
                    )

        # Sex gap (H2): flagged = ABNORMAL or NEEDS_FOLLOW_UP, relevant context
        for run in ("greedy", "sampled"):
            for rng_level in ("none", "printed"):
                flagged = {}
                for sex, case in (("male", "FER_SEX_M"), ("female", "FER_SEX_F")):
                    cell = idx.get((case, lang, rng_level, "relevant"))
                    if cell:
                        vs = [cell["greedy"]] if run == "greedy" else cell["samples"]
                        flagged[sex] = share(vs, lambda v: v in ("ABNORMAL", "NEEDS_FOLLOW_UP"))
                if len(flagged) == 2:
                    gap = flagged["male"] - flagged["female"]
                    results[(f"sex_gap_range_{rng_level}", lang, run)]["FER_SEX"] = gap

    return results


def summarise(results, rng):
    rows = []
    for (metric, lang, run), per_case in sorted(results.items()):
        mean, lo, hi = bootstrap_ci(per_case, rng)
        rows.append(
            {
                "metric": metric,
                "language": lang,
                "run": run,
                "n_cases": sum(v is not None for v in per_case.values()),
                "mean": None if mean is None else round(mean, 3),
                "ci_low": None if lo is None else round(lo, 3),
                "ci_high": None if hi is None else round(hi, 3),
                "per_case": json.dumps(
                    {k: None if v is None else round(v, 2) for k, v in per_case.items()}
                ),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()

    for path in args.files:
        rng = random.Random(SEED)
        rows = summarise(compute(load(path)), rng)
        model = path.stem.removeprefix("generations_")
        out = path.with_name(f"metrics_{model}.csv")
        with open(out, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        print(f"\n=== {model} -> {out}")
        for r in rows:
            ci = "" if r["ci_low"] is None else f"  [{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
            mean = "n/a" if r["mean"] is None else f"{r['mean']:.2f}"
            print(f"{r['metric']:32} {r['language']} {r['run']:8} n={r['n_cases']:2}  {mean}{ci}")


if __name__ == "__main__":
    main()
