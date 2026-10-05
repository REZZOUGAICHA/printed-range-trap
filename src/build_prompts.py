"""Build prompt variants: cases x range x context x language -> data/prompts.jsonl.

A harness only changes the system prompt; user messages, prompt ids (and so the
sampling seeds) stay the same, so answers can be compared prompt by prompt.
  plain        the original prompts (data/prompts.jsonl)
  instruction  + one sentence: a printed range describes a general population
  guidelines   + the guideline thresholds for the test (data/guidelines.yaml)

Usage:
    uv run python src/build_prompts.py --languages en fr
    uv run python src/build_prompts.py --languages en fr --harness guidelines
"""

import argparse
import json
from itertools import product
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RANGES = ["none", "printed"]
CONTEXTS = ["none", "relevant", "irrelevant"]
HARNESSES = ["plain", "instruction", "guidelines"]


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


def analyte_key(case):
    """'Fasting glucose' -> 'fasting_glucose', the key used in guidelines.yaml."""
    return case["en"]["analyte"].lower().replace(" ", "_")


def build_system(template, harness, guideline_text):
    system = template["system"]
    if harness == "instruction":
        system += "\n\n" + template["instruction"]
    elif harness == "guidelines":
        system += "\n\n" + template["guidelines"].format(guidelines=guideline_text)
    return system


def build_variants(cases, templates, languages, harness="plain", guidelines=None):
    variants = []
    for case, lang in product(cases, languages):
        text = case[lang]
        template = templates[lang]
        guideline_text = guidelines[analyte_key(case)][lang] if guidelines else None
        system = build_system(template, harness, guideline_text)
        for range_level, context_level in product(RANGES, CONTEXTS):
            # "none" and "irrelevant" context share the context-free gold
            gold = case["gold"]["relevant" if context_level == "relevant" else "no_context"]
            variant = {
                "prompt_id": f"{case['id']}|{lang}|range={range_level}|context={context_level}",
                "case_id": case["id"],
                "language": lang,
                "range": range_level,
                "context": context_level,
                "system": system,
                "user": build_user_message(template, text, range_level, context_level),
                "gold": gold["verdict"],
                "accept": gold["accept"],
                "printed_range_verdict": case["printed_range_verdict"],
            }
            # plain prompts keep their original fields, so data/prompts.jsonl is unchanged
            if harness != "plain":
                variant["harness"] = harness
            variants.append(variant)
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
    parser.add_argument("--guidelines", default=ROOT / "data" / "guidelines.yaml", type=Path)
    parser.add_argument(
        "--out", type=Path, help="default: data/prompts.jsonl or data/prompts_<harness>.jsonl"
    )
    parser.add_argument("--languages", nargs="+", default=["en"])
    parser.add_argument("--harness", default="plain", choices=HARNESSES)
    args = parser.parse_args()
    suffix = "" if args.harness == "plain" else f"_{args.harness}"
    args.out = args.out or ROOT / "data" / f"prompts{suffix}.jsonl"

    cases = load_yaml(args.cases)["cases"]
    templates = load_yaml(args.templates)
    guidelines = load_yaml(args.guidelines) if args.harness == "guidelines" else None
    variants = build_variants(cases, templates, args.languages, args.harness, guidelines)

    n_checked = check_one_factor(variants)
    assert len({v["prompt_id"] for v in variants}) == len(variants), "duplicate prompt_id"

    with open(args.out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(v, ensure_ascii=False) + "\n" for v in variants)

    print(f"{len(cases)} cases x {len(args.languages)} language(s) -> {len(variants)} prompts")
    print(f"One-factor check passed on {n_checked} pairs. Wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
