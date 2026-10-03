"""Run a model on data/prompts.jsonl: 1 greedy + N sampled answers per prompt.

Resumable: each prompt's answers are appended to the output file as soon as
they are generated; prompts already in the file are skipped on the next run.

Usage (on a GPU machine, e.g. Kaggle):
    python src/run_model.py --out results/generations_qwen3.5-4b.jsonl --limit 2
    python src/run_model.py --out results/generations_qwen3.5-4b.jsonl
"""

import argparse
import json
import platform
import time
import zlib
from datetime import UTC, datetime
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent

# Qwen3.5 model card, non-thinking mode, general tasks.
# presence_penalty=1.5 is also recommended, but generate() does not support it (decision D12).
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}
MAX_NEW_TOKENS = 128
THINK_OFF_SUFFIX = "<think>\n\n</think>\n\n"


def load_prompts(path, limit=None):
    with open(path, encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f]
    return prompts[:limit] if limit else prompts


def load_done(out_path):
    """prompt_ids already in the output file. A half-written last line is ignored."""
    done = set()
    if not out_path.exists():
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["prompt_id"])
            except json.JSONDecodeError:
                print("Skipping a broken line (interrupted write); that prompt will be redone.")
    return done


def seed_for(prompt_id):
    """Stable seed from the prompt's name: same prompt -> same seed, whatever the run order."""
    return zlib.crc32(prompt_id.encode("utf-8"))


def build_prompt_text(tokenizer, prompt):
    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"]},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    if "enable_thinking" in (tokenizer.chat_template or ""):
        assert text.endswith(THINK_OFF_SUFFIX), "Thinking is ON: check enable_thinking"
    return text


def generate(model, tokenizer, prompt_text, n_samples=0, seed=None, suppress_ids=None):
    """Greedy answer if n_samples == 0, else n_samples sampled answers from one call.

    suppress_ids: token ids that may never be generated (e.g. MedGemma's thinking start).
    """
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    if n_samples:
        torch.manual_seed(seed)
        kwargs = {"do_sample": True, "num_return_sequences": n_samples, **SAMPLING}
    else:
        kwargs = {"do_sample": False}
    if suppress_ids:
        kwargs["suppress_tokens"] = suppress_ids

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, **kwargs)

    new_tokens = output[:, inputs["input_ids"].shape[1] :]
    texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    # Truncated = used all MAX_NEW_TOKENS without reaching an end-of-answer token
    eos = model.generation_config.eos_token_id
    stop_ids = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.pad_token_id}
    truncated = [
        new_tokens.shape[1] == MAX_NEW_TOKENS and row[-1].item() not in stop_ids
        for row in new_tokens
    ]
    return list(zip(texts, truncated, strict=True))


def check_finite(model, tokenizer, prompt_text):
    """Stop early if the chosen dtype overflows (NaN/inf logits), e.g. Gemma in float16."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    assert torch.isfinite(logits).all(), (
        "NaN/inf in logits: this dtype is broken for this model. "
        "Retry with --dtype float32 --device-map auto (uses both T4 GPUs)."
    )


def write_meta(meta_path, args, model):
    meta = {
        "model_id": args.model_id,
        "revision": args.revision,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "n_parameters_loaded": sum(p.numel() for p in model.parameters()),
        "thinking": "off",
        "suppress_tokens": args.suppress_tokens,
        "greedy": True,
        "n_samples": args.n_samples,
        "sampling": SAMPLING,
        "sampling_note": "presence_penalty=1.5 (model card) not supported by transformers generate(); omitted (D12)",
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": "zlib.crc32(prompt_id)",
        "prompts_file": str(args.prompts),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default=ROOT / "data" / "prompts.jsonl", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--meta", type=Path, help="default: <out>_meta.json next to --out")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--device-map", default="cuda:0", help='"auto" splits the model over all GPUs'
    )
    parser.add_argument("--n-samples", default=10, type=int)
    parser.add_argument("--limit", type=int, help="only the first N prompts (quick test)")
    parser.add_argument(
        "--suppress-tokens",
        nargs="*",
        default=[],
        help='tokens the model may never generate, e.g. "<unused94>" (MedGemma thinking start)',
    )
    args = parser.parse_args()
    args.meta = args.meta or args.out.with_name(args.out.stem + "_meta.json")

    prompts = load_prompts(args.prompts, args.limit)
    done = load_done(args.out)
    todo = [p for p in prompts if p["prompt_id"] not in done]
    print(f"{len(prompts)} prompts, {len(done)} already done, {len(todo)} to run")
    if not todo:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        dtype=getattr(torch, args.dtype),
        device_map=args.device_map,
    )
    model.eval()
    suppress_ids = tokenizer.convert_tokens_to_ids(args.suppress_tokens)
    assert tokenizer.unk_token_id not in suppress_ids, f"unknown token in {args.suppress_tokens}"
    check_finite(model, tokenizer, build_prompt_text(tokenizer, todo[0]))
    write_meta(args.meta, args, model)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for i, prompt in enumerate(todo, 1):
        start = time.time()
        prompt_text = build_prompt_text(tokenizer, prompt)
        seed = seed_for(prompt["prompt_id"])

        answers = [
            ("greedy", None, *generate(model, tokenizer, prompt_text, suppress_ids=suppress_ids)[0])
        ]
        if args.n_samples:
            sampled = generate(model, tokenizer, prompt_text, args.n_samples, seed, suppress_ids)
            answers += [("sample", k, text, trunc) for k, (text, trunc) in enumerate(sampled)]

        seconds = round(time.time() - start, 1)
        records = [
            {
                **prompt,
                "run": run,
                "sample_idx": k,
                "seed": seed if run == "sample" else None,
                "output": text,
                "truncated": trunc,
                "prompt_seconds": seconds,
            }
            for run, k, text, trunc in answers
        ]
        # One write per prompt, so a crash can't leave a prompt half-saved
        with open(args.out, "a", encoding="utf-8") as f:
            f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))

        greedy_line = answers[0][2].strip().splitlines()[0] if answers[0][2].strip() else "(empty)"
        print(f"[{i}/{len(todo)}] {prompt['prompt_id']}  {seconds}s  greedy: {greedy_line}")


if __name__ == "__main__":
    main()
