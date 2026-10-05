"""Agent harness: the model may call a guideline-lookup tool before answering.

Same prompts, prompt ids and seeds as the plain harness. The only difference is
that a tool, lookup_guideline(test), is available; the model decides whether
to use it. The tool returns the text of data/guidelines.yaml for that test, in
the prompt's language. At most one tool call per answer:
turn 1 -> (tool call -> lookup -> turn 2) -> final verdict.

Qwen only: MedGemma's chat template has no tool support.

Usage (GPU):
    python src/run_agent.py --out results/qwen3.5-4b/agent/generations_qwen3.5-4b_agent.jsonl
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_model import (
    MAX_NEW_TOKENS,
    SAMPLING,
    THINK_OFF_SUFFIX,
    check_finite,
    load_done,
    load_prompts,
    seed_for,
    write_meta,
)

ROOT = Path(__file__).resolve().parent.parent

TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_guideline",
        "description": "Return the clinical guideline thresholds for a laboratory test.",
        "parameters": {
            "type": "object",
            "properties": {
                "test": {
                    "type": "string",
                    "description": "Name of the laboratory test, e.g. ferritin",
                }
            },
            "required": ["test"],
        },
    },
}

# Words that identify each test in the tool argument (English and French)
ALIASES = {
    "ferritin": r"ferritin",
    "creatinine": r"cr[ée]atinin",
    "fasting_glucose": r"glucose|glyc[ée]mie",
    "hemoglobin": r"h(?:a?e|é)moglobin",  # hemoglobin, haemoglobin, hémoglobine
    "alkaline_phosphatase": r"phosphatase|\balp\b|\bpal\b",
}
NOT_FOUND = "No guideline found for this test."

TOOL_CALL_RE = re.compile(r"<tool_call>\s*<function=([\w.-]+)>(.*?)</function>", re.DOTALL)
PARAM_RE = re.compile(r"<parameter=([\w-]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def lookup(test, lang, guidelines):
    for key, pattern in ALIASES.items():
        if re.search(pattern, test, re.IGNORECASE):
            return guidelines[key][lang]
    return NOT_FOUND


def parse_tool_call(text):
    """(name, args, text before the call) for the first tool call in `text`, or None."""
    m = TOOL_CALL_RE.search(text)
    if not m:
        return None
    args = dict(PARAM_RE.findall(m.group(2)))
    return m.group(1), args, text[: text.find("<tool_call>")].strip()


def chat_text(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages, tools=[TOOL], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    assert text.endswith(THINK_OFF_SUFFIX), "Thinking is ON: check enable_thinking"
    return text


def generate_batch(model, tokenizer, texts, n_per_text=1, sample=False, seed=None):
    """One generate() call over several conversations; returns (text, truncated) per output."""
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    if sample:
        torch.manual_seed(seed)
        kwargs = {"do_sample": True, "num_return_sequences": n_per_text, **SAMPLING}
    else:
        kwargs = {"do_sample": False}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, **kwargs)

    new_tokens = output[:, inputs["input_ids"].shape[1] :]
    texts_out = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    eos = model.generation_config.eos_token_id
    stop_ids = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.pad_token_id}
    truncated = [
        new_tokens.shape[1] == MAX_NEW_TOKENS and row[-1].item() not in stop_ids
        for row in new_tokens
    ]
    return list(zip(texts_out, truncated, strict=True))


def run_prompt(model, tokenizer, prompt, guidelines, n_samples):
    """All answers for one prompt: greedy first, then n_samples sampled conversations."""
    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"]},
    ]
    first_text = chat_text(tokenizer, messages)
    seed = seed_for(prompt["prompt_id"])

    runs = [("greedy", None, False)] + [("sample", k, True) for k in range(n_samples)]
    turn1 = generate_batch(model, tokenizer, [first_text])
    if n_samples:
        turn1 += generate_batch(model, tokenizer, [first_text], n_samples, sample=True, seed=seed)

    # Second turn for the conversations that called the tool
    pending, records = [], []
    for (run, k, sampled), (text, trunc) in zip(runs, turn1, strict=True):
        record = {
            **prompt,
            "harness": "agent",
            "run": run,
            "sample_idx": k,
            "seed": seed if sampled else None,
            "turn1": text,
            "tool_called": False,
            "tool_arg": None,
            "tool_found": None,
            "output": text,
            "truncated": trunc,
        }
        call = parse_tool_call(text)
        if call and call[0] == TOOL["function"]["name"]:
            name, args, before = call
            arg = args.get("test", "")
            result = lookup(arg, prompt["language"], guidelines)
            record.update(tool_called=True, tool_arg=arg, tool_found=result != NOT_FOUND)
            followup = [
                *messages,
                {
                    "role": "assistant",
                    "content": before,
                    "tool_calls": [
                        {"type": "function", "function": {"name": name, "arguments": args}}
                    ],
                },
                {"role": "tool", "content": result},
            ]
            pending.append((record, chat_text(tokenizer, followup), sampled))
        records.append(record)

    for sampled in (False, True):
        batch = [(r, t) for r, t, s in pending if s == sampled]
        if not batch:
            continue
        outs = generate_batch(
            model, tokenizer, [t for _, t in batch], sample=sampled, seed=seed + 1
        )
        for (record, _), (text, trunc) in zip(batch, outs, strict=True):
            record.update(output=text, truncated=trunc)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default=ROOT / "data" / "prompts.jsonl", type=Path)
    parser.add_argument("--guidelines", default=ROOT / "data" / "guidelines.yaml", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--meta", type=Path, help="default: <out>_meta.json next to --out")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--n-samples", default=10, type=int)
    parser.add_argument("--limit", type=int, help="only the first N prompts (quick test)")
    args = parser.parse_args()
    args.meta = args.meta or args.out.with_name(args.out.stem + "_meta.json")
    args.suppress_tokens = []

    with open(args.guidelines, encoding="utf-8") as f:
        guidelines = yaml.safe_load(f)
    prompts = load_prompts(args.prompts, args.limit)
    done = load_done(args.out)
    todo = [p for p in prompts if p["prompt_id"] not in done]
    print(f"{len(prompts)} prompts, {len(done)} already done, {len(todo)} to run")
    if not todo:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    tokenizer.padding_side = "left"  # needed to batch conversations of different lengths
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        dtype=getattr(torch, args.dtype),
        device_map=args.device_map,
    )
    model.eval()
    first = [
        {"role": "system", "content": todo[0]["system"]},
        {"role": "user", "content": todo[0]["user"]},
    ]
    check_finite(model, tokenizer, chat_text(tokenizer, first))
    write_meta(args.meta, args, model)
    with open(args.meta, encoding="utf-8") as f:
        meta = json.load(f)
    meta.update(harness="agent", tool=TOOL, max_tool_calls=1, guidelines_file=str(args.guidelines))
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for i, prompt in enumerate(todo, 1):
        start = time.time()
        records = run_prompt(model, tokenizer, prompt, guidelines, args.n_samples)
        seconds = round(time.time() - start, 1)
        for r in records:
            r["prompt_seconds"] = seconds
        with open(args.out, "a", encoding="utf-8") as f:
            f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))

        n_calls = sum(r["tool_called"] for r in records)
        greedy = records[0]["output"].strip().splitlines()
        print(
            f"[{i}/{len(todo)}] {prompt['prompt_id']}  {seconds}s  tool calls {n_calls}/{len(records)}"
            f"  greedy: {greedy[0] if greedy else '(empty)'}"
        )


if __name__ == "__main__":
    main()
