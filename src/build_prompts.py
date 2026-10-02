"""Build prompt variants: cases x range x context x language -> data/prompts.jsonl.

Usage:
    uv run python src/build_prompts.py
    uv run python src/build_prompts.py --languages en fr
"""

import argparse
import json
from itertools import product
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RANGES = ["none", "printed"]
CONTEXTS = ["none", "relevant", "irrelevant"]


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_user_message(template, text, range_level, context_level):
    context = ""
    if context_level != "none":
        context = template["context"].format(sentence=text[context_level])

    range_part = ""
    if range_level == "printed":
        range_part = template["range"].format(low=text["low"], high=text["high"])

    return template["user"].format(
        context=context,
        analyte=text["analyte"],
        value=text["value"],
        unit=text["unit"],
        range=range_part,
    )


def build_variants(cases, templates, languages):
    variants = []
    for case, lang in product(cases, languages):
        text = case[lang]
        template = templates[lang]
        for range_level, context_level in product(RANGES, CONTEXTS):
            # "none" and "irrelevant" context share the context-free gold
            gold = case["gold"]["relevant" if context_level == "relevant" else "no_context"]
            variants.append(
                {
                    "prompt_id": f"{case['id']}|{lang}|range={range_level}|context={context_level}",
                    "case_id": case["id"],
                    "language": lang,
                    "range": range_level,
                    "context": context_level,
                    "system": template["system"],
                    "user": build_user_message(template, text, range_level, context_level),
                    "gold": gold["verdict"],
                    "accept": gold["accept"],
                    "printed_range_verdict": case["printed_range_verdict"],
                }
            )
    return variants


def is_single_insertion(short, long):
    """True if `long` is `short` with exactly one piece of text inserted somewhere."""
    if len(long) <= len(short):
        return False
    p = 0
    while p < len(short) and short[p] == long[p]:
        p += 1
    return long.endswith(short[p:])


def check_one_factor(variants):
    """Each variant must differ from its 'none' twin by one inserted piece only."""
    by_id = {v["prompt_id"]: v for v in variants}
    n_checked = 0
    for v in variants:
        twins = []
        if v["range"] == "printed":
            twins.append(v["prompt_id"].replace("range=printed", "range=none"))
        if v["context"] != "none":
            twins.append(v["prompt_id"].replace(f"context={v['context']}", "context=none"))
        for twin_id in twins:
            twin = by_id[twin_id]
            assert twin["system"] == v["system"], f"system prompt differs: {v['prompt_id']}"
            assert is_single_insertion(twin["user"], v["user"]), (
                f"{v['prompt_id']} differs from {twin_id} by more than one insertion"
            )
            n_checked += 1
    return n_checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=ROOT / "data" / "cases.yaml", type=Path)
    parser.add_argument("--templates", default=ROOT / "data" / "templates.yaml", type=Path)
    parser.add_argument("--out", default=ROOT / "data" / "prompts.jsonl", type=Path)
    parser.add_argument("--languages", nargs="+", default=["en"])
    args = parser.parse_args()

    cases = load_yaml(args.cases)["cases"]
    templates = load_yaml(args.templates)
    variants = build_variants(cases, templates, args.languages)

    n_checked = check_one_factor(variants)
    assert len({v["prompt_id"] for v in variants}) == len(variants), "duplicate prompt_id"

    with open(args.out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(v, ensure_ascii=False) + "\n" for v in variants)

    print(f"{len(cases)} cases x {len(args.languages)} language(s) -> {len(variants)} prompts")
    print(f"One-factor check passed on {n_checked} pairs. Wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
