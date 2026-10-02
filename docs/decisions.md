# Decision log

One entry per design decision: what, why, what a reviewer might question.
Newest at the bottom.

---

## D1 — Sex-swap pair changes only the sex word (2026-10-02)

**Decision.** H2 is tested on a new pair, FER_SEX_F / FER_SEX_M: ferritin
25 µg/L, printed range 15–150, symptoms "fatigue, hair loss" — identical except
"woman" / "man". The original FER_F / FER_M pair is kept as a separate,
descriptive comparison of sex-specific printed ranges.

**Why.** FER_F and FER_M differed in three things at once (sex, printed range,
symptoms: heavy periods vs blood in stool). Any verdict gap could not be
attributed to sex. A counterfactual design requires one factor per change.

**Reviewer question.** "Is a shared 15–150 range realistic for a man?" —
Open (`TODO_VERIFY`). The pair isolates the model's sex bias, not realism of
the report; the realistic setting is covered by FER_F vs FER_M.

## D2 — FER_M context-free gold is ABNORMAL (2026-10-02)

**Decision.** Gold for FER_M at none/irrelevant context changed from
NEEDS_FOLLOW_UP to ABNORMAL (accept ABNORMAL, NEEDS_FOLLOW_UP).

**Why.** 25 µg/L is below the printed male lower limit (30), so the
context-free reading is "low". Because the printed range agrees with the gold,
FER_M is not a trap case and is excluded from the H1 deference metric (the
metric's definition already excludes it; stated explicitly in the metrics definition).

## D3 — Separate accept sets per context level (2026-10-02)

**Decision.** The case table has two accept columns: one for none/irrelevant
context, one for relevant context. A single-label accept set equals the gold.

**Why.** FER_F's context-free gold was "NORMAL or NEEDS_FOLLOW_UP", but the
single accept column only covered the relevant-context gold, so scoring was
ambiguous. For in-range ferritin of 25 with no context, both NORMAL and
NEEDS_FOLLOW_UP are defensible; gold is NORMAL, accept set includes both.

**Reviewer question.** "Do lenient accept sets inflate accuracy?" — Yes,
slightly; we also report strict accuracy (gold label only) in Phase 4.

## D4 — uv for dependency management (2026-10-02)

**Decision.** `pyproject.toml` + committed `uv.lock`; run scripts with `uv run`.

**Why.** A lockfile pins every transitive dependency, so the "Reproduce"
section is honest. Alternatives: plain `requirements.txt` (no lock of
transitive deps unless hand-frozen), conda (slow, heavy for a small project).
On Kaggle we export from the lock and keep Kaggle's preinstalled CUDA torch.

## D5 — Main model: Qwen/Qwen3.5-4B, thinking off (2026-10-02)

**Decision.** `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`,
text-only use, `enable_thinking=False` for the main run.

**Why.** Current frontier small-model family, strong FR support, Apache-2.0,
not gated. 4.66B parameters (including the vision encoder) is within the
0.6–6B limit. Thinking is on by default in its chat template; the main run
turns it off to match typical chatbot use.

**Reviewer question.** "Does the unused vision encoder matter?" — No for
behaviour (text-only input), but the parameter count we report includes it.

## D6 — Dependency groups; transformers pinned to a release (2026-10-02)

**Decision.** Python 3.13 (matches the Kaggle image). Base deps = local
tooling (pyyaml); `inference` group = transformers 5.18.0 + accelerate,
locked but not installed locally; `dev` group = ruff.

**Why.** Inference only runs on Kaggle, and accelerate pulls in torch, which
is several hundred MB we never use locally. The model card suggests installing
transformers from GitHub main, but 5.18.0 already ships `qwen3_5`; a release
pin is reproducible, main is not. On Kaggle we keep the preinstalled CUDA torch.

## D7 — Load text-only in fp16, with a NaN check (2026-10-02)

**Decision.** Load with `AutoModelForCausalLM` (text part only) in float16,
pinned revision, and check that the logits are finite before generating.

**Why.** We never send images, so the vision encoder is not needed. T4/P100
have no bfloat16 support; the model was trained in bfloat16, and fp16 can
overflow into NaN. The check makes that failure visible instead of silent.
Fallbacks if it fails: float32 split over both T4s, or 8-bit.

**Reviewer question.** "Could fp16 change the model's answers compared to
bf16?" — Possibly, slightly. We report the precision used in `run_meta.json`.

## D8 — Greedy + 10 samples with the model's recommended settings (2026-10-02)

**Decision.** Each prompt gets 1 greedy run and 10 sampled runs (fixed seeds)
with Qwen's recommended non-thinking settings (temperature 0.7, top_p 0.8,
top_k 20, presence_penalty 1.5). Greedy and sampled results are reported side
by side. Replaces the earlier "3 samples at temperature 0.7".

**Why.** Greedy is the standard primary result (reproducible; usually scores
higher than sampling — Song et al. 2024, arXiv 2407.10457), but real chatbot
users receive sampled answers, and medical LLM outputs vary across runs
(e.g. medRxiv 10.1101/2025.06.04.25328288). With 10 samples we can report a
trap *rate* per prompt instead of a single yes/no. Using the model card's
settings avoids an arbitrary choice.

**Reviewer question.** "Why 10?" — A cost/precision trade-off: short answers
make 10 cheap, and it gives rates in steps of 10%. "Is presence_penalty
supported in transformers?" — To verify when writing run_model.py.

## D9 — FER_CRP: CRP moves from the lab line to the relevant context (2026-10-02)

**Decision.** The FER_CRP lab line shows ferritin only. "CRP 60 mg/L" is part
of the relevant context sentence (rheumatoid arthritis flare).

**Why.** If CRP 60 is always on the lab line, the model sees the inflammation
even with no context, so the no-context gold (NORMAL) would be unfair and two
things would change between variants. Moving CRP into the context keeps one
factor per change.

## D10 — Same identity in relevant and irrelevant context (2026-10-02)

**Decision.** Relevant and irrelevant sentences start with the same identity
("I am a 28-year-old woman ...") and have the same word count (±1).
CREA_TREND gets "55-year-old man" (not specified in the original plan; the
KDIGO creatinine-rise criterion does not depend on age or sex).

**Why.** Then relevant vs irrelevant differ only in medical information. If
only the relevant sentence had age/sex, a verdict change could be caused by
the demographics or by sentence length, not by the clinical facts.

**Reviewer question.** "The 'none' level has no identity at all — is it
comparable?" — It is the realistic "just the report" setting; the clean
comparison for context effects is relevant vs irrelevant.

## D11 — Pilot uses greedy + 10 samples (2026-10-02)

**Decision.** The pilot (4 cases × 6 variants = 24 prompts, EN) runs greedy
and 10 sampled answers per prompt, not greedy only.

**Why.** In the Phase 0 smoke test, greedy escaped the trap on FER_F with
relevant context. A greedy-only pilot could miss a trap that appears in a
fraction of sampled answers. Extra cost is small (short answers).

## D12 — presence_penalty omitted; seeds from prompt ids (2026-10-02)

**Decision.** Sampled runs use temperature 0.7, top_p 0.8, top_k 20, min_p 0.0
from the Qwen3.5 model card, but not presence_penalty=1.5. Each prompt's seed
is `zlib.crc32(prompt_id)`, and its 10 samples come from one generate() call.

**Why.** transformers 5.18.0 `generate()` has no presence_penalty (checked in
its GenerationConfig). Writing our own version risks not matching the serving
frameworks' definition, which would look identical but not be. Its purpose is
to stop repetition in long outputs; our answers are 2 lines and the verdict
comes first, so the effect on verdicts should be small. Seeds tied to the
prompt id (not its position) stay the same when a run is resumed or reordered.

**Reviewer question.** "Are sampled runs exactly what chatbot users get?" —
Close, not identical: no presence_penalty, and GPU arithmetic is not fully
deterministic. Stated in run_meta.json and the limitations.

## D13 — Pilot shows a signal: proceed to Phase 2 (2026-10-02)

**Result.** Qwen3.5-4B, 4 cases, EN, greedy + 10 samples (264 answers, 0 parse
failures, 0 truncated, 4.2 GPU-minutes). Relevant context, range not printed vs
printed — correct samples out of 10:
HB_PREG 10 → 0, FER_CRP 10 → 1, FER_F 10 → 6, CREA_TREND: no trap (the model
spots the AKI but answers NEEDS_FOLLOW_UP with or without the range).
Greedy fell into the trap on HB_PREG and FER_CRP; on FER_F only samples did (4/10).

**Notable outputs.** HB_PREG printed: the model cites the 11.0 g/dL pregnancy
cutoff and still says 11.3 is below it. FER_CRP printed: it notes ferritin is
raised by inflammation and still answers NORMAL because 80 is within 15–150.

**Decision.** Proceed. Added to the plan: clinician review by 2–3 doctors,
2–3 prompt wordings for the range line, second model (MedGemma) required.

**Caveats.** 4 cases only: shows the trap exists, not how often. Two gold
answers are debatable (CREA_TREND accept set; FER_F no-context gold), sent to
the doctors. Expected reviewer objection: "deferring to a lab's own range is
sometimes right" — answer: in our cases the printed range is known not to
apply (non-pregnant range in pregnancy, ferritin under inflammation).
