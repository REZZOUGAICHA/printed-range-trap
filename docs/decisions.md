# Decision log

One entry per design decision: what, why, what a reviewer might question.
Newest at the bottom.

---

## D1: Sex-swap pair changes only the sex word (2026-10-02)

**Decision.** H2 is tested on a new pair, FER_SEX_F / FER_SEX_M: ferritin
25 µg/L, printed range 15–150, symptoms "fatigue, hair loss", identical except
"woman" / "man". The original FER_F / FER_M pair is kept as a separate,
descriptive comparison of sex-specific printed ranges.

**Why.** FER_F and FER_M differed in three things at once (sex, printed range,
symptoms: heavy periods vs blood in stool). Any verdict gap could not be
attributed to sex. A counterfactual design requires one factor per change.

**Reviewer question.** "Is a shared 15–150 range realistic for a man?"
Open (`TODO_VERIFY`). The pair isolates the model's sex bias, not realism of
the report; the realistic setting is covered by FER_F vs FER_M.

## D2: FER_M context-free gold is ABNORMAL (2026-10-02)

**Decision.** Gold for FER_M at none/irrelevant context changed from
NEEDS_FOLLOW_UP to ABNORMAL (accept ABNORMAL, NEEDS_FOLLOW_UP).

**Why.** 25 µg/L is below the printed male lower limit (30), so the
context-free reading is "low". Because the printed range agrees with the gold,
FER_M is not a trap case and is excluded from the H1 deference metric (the
metric's definition already excludes it; stated explicitly in the metrics definition).

## D3: Separate accept sets per context level (2026-10-02)

**Decision.** The case table has two accept columns: one for none/irrelevant
context, one for relevant context. A single-label accept set equals the gold.

**Why.** FER_F's context-free gold was "NORMAL or NEEDS_FOLLOW_UP", but the
single accept column only covered the relevant-context gold, so scoring was
ambiguous. For in-range ferritin of 25 with no context, both NORMAL and
NEEDS_FOLLOW_UP are defensible; gold is NORMAL, accept set includes both.

**Reviewer question.** "Do lenient accept sets inflate accuracy?" Yes,
slightly; we also report strict accuracy (gold label only) in Phase 4.

## D4: uv for dependency management (2026-10-02)

**Decision.** `pyproject.toml` + committed `uv.lock`; run scripts with `uv run`.

**Why.** A lockfile pins every transitive dependency, so the "Reproduce"
section is honest. Alternatives: plain `requirements.txt` (no lock of
transitive deps unless hand-frozen), conda (slow, heavy for a small project).
On Kaggle we export from the lock and keep Kaggle's preinstalled CUDA torch.

## D5: Main model: Qwen/Qwen3.5-4B, thinking off (2026-10-02)

**Decision.** `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`,
text-only use, `enable_thinking=False` for the main run.

**Why.** Current frontier small-model family, strong FR support, Apache-2.0,
not gated. 4.66B parameters (including the vision encoder) is within the
0.6–6B limit. Thinking is on by default in its chat template; the main run
turns it off to match typical chatbot use.

**Reviewer question.** "Does the unused vision encoder matter?" No for
behaviour (text-only input), but the parameter count we report includes it.

## D6: Dependency groups; transformers pinned to a release (2026-10-02)

**Decision.** Python 3.13 (matches the Kaggle image). Base deps = local
tooling (pyyaml); `inference` group = transformers 5.18.0 + accelerate,
locked but not installed locally; `dev` group = ruff.

**Why.** Inference only runs on Kaggle, and accelerate pulls in torch, which
is several hundred MB we never use locally. The model card suggests installing
transformers from GitHub main, but 5.18.0 already ships `qwen3_5`; a release
pin is reproducible, main is not. On Kaggle we keep the preinstalled CUDA torch.

## D7: Load text-only in fp16, with a NaN check (2026-10-02)

**Decision.** Load with `AutoModelForCausalLM` (text part only) in float16,
pinned revision, and check that the logits are finite before generating.

**Why.** We never send images, so the vision encoder is not needed. T4/P100
have no bfloat16 support; the model was trained in bfloat16, and fp16 can
overflow into NaN. The check makes that failure visible instead of silent.
Fallbacks if it fails: float32 split over both T4s, or 8-bit.

**Reviewer question.** "Could fp16 change the model's answers compared to
bf16?" Possibly, slightly. We report the precision used in `run_meta.json`.

## D8: Greedy + 10 samples with the model's recommended settings (2026-10-02)

**Decision.** Each prompt gets 1 greedy run and 10 sampled runs (fixed seeds)
with Qwen's recommended non-thinking settings (temperature 0.7, top_p 0.8,
top_k 20, presence_penalty 1.5). Greedy and sampled results are reported side
by side. Replaces the earlier "3 samples at temperature 0.7".

**Why.** Greedy is the standard primary result (reproducible; usually scores
higher than sampling, Song et al. 2024, arXiv 2407.10457), but real chatbot
users receive sampled answers, and medical LLM outputs vary across runs
(e.g. medRxiv 10.1101/2025.06.04.25328288). With 10 samples we can report a
trap *rate* per prompt instead of a single yes/no. Using the model card's
settings avoids an arbitrary choice.

**Reviewer question.** "Why 10?" A cost/precision trade-off: short answers
make 10 cheap, and it gives rates in steps of 10%. "Is presence_penalty
supported in transformers?" To verify when writing run_model.py.

## D9: FER_CRP: CRP moves from the lab line to the relevant context (2026-10-02)

**Decision.** The FER_CRP lab line shows ferritin only. "CRP 60 mg/L" is part
of the relevant context sentence (rheumatoid arthritis flare).

**Why.** If CRP 60 is always on the lab line, the model sees the inflammation
even with no context, so the no-context gold (NORMAL) would be unfair and two
things would change between variants. Moving CRP into the context keeps one
factor per change.

## D10: Same identity in relevant and irrelevant context (2026-10-02)

**Decision.** Relevant and irrelevant sentences start with the same identity
("I am a 28-year-old woman ...") and have the same word count (±1).
CREA_TREND gets "55-year-old man" (not specified in the original plan; the
KDIGO creatinine-rise criterion does not depend on age or sex).

**Why.** Then relevant vs irrelevant differ only in medical information. If
only the relevant sentence had age/sex, a verdict change could be caused by
the demographics or by sentence length, not by the clinical facts.

**Reviewer question.** "The 'none' level has no identity at all, is it
comparable?" It is the realistic "just the report" setting; the clean
comparison for context effects is relevant vs irrelevant.

## D11: Pilot uses greedy + 10 samples (2026-10-02)

**Decision.** The pilot (4 cases × 6 variants = 24 prompts, EN) runs greedy
and 10 sampled answers per prompt, not greedy only.

**Why.** In the Phase 0 smoke test, greedy escaped the trap on FER_F with
relevant context. A greedy-only pilot could miss a trap that appears in a
fraction of sampled answers. Extra cost is small (short answers).

## D12: presence_penalty omitted; seeds from prompt ids (2026-10-02)

**Decision.** Sampled runs use temperature 0.7, top_p 0.8, top_k 20, min_p 0.0
from the Qwen3.5 model card, but not presence_penalty=1.5. Each prompt's seed
is `zlib.crc32(prompt_id)`, and its 10 samples come from one generate() call.

**Why.** transformers 5.18.0 `generate()` has no presence_penalty (checked in
its GenerationConfig). Writing our own version risks not matching the serving
frameworks' definition, which would look identical but not be. Its purpose is
to stop repetition in long outputs; our answers are 2 lines and the verdict
comes first, so the effect on verdicts should be small. Seeds tied to the
prompt id (not its position) stay the same when a run is resumed or reordered.

**Reviewer question.** "Are sampled runs exactly what chatbot users get?"
Close, not identical: no presence_penalty, and GPU arithmetic is not fully
deterministic. Stated in run_meta.json and the limitations.

## D13: Pilot shows a signal: proceed to Phase 2 (2026-10-02)

**Result.** Qwen3.5-4B, 4 cases, EN, greedy + 10 samples (264 answers, 0 parse
failures, 0 truncated, 4.2 GPU-minutes). Relevant context, range not printed vs
printed, correct samples out of 10:
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
sometimes right". Answer: in our cases the printed range is known not to
apply (non-pregnant range in pregnancy, ferritin under inflammation).

## D14: Blind doctor review form, in French (2026-10-02)

**Decision.** Doctors fill `docs/doctor_review/fiche_relecture_medicale.docx`
alone, without seeing our gold answers. Each of the 12 cases has situation A
(report line only) and B (with patient information); for each they give a
verdict, the other verdicts they would accept, a confidence level and a
comment. Cases are numbered 1–12 in a mixed order with no ids; the mapping and
our gold answers are in `docs/doctor_review/answer_key.md` (not sent).

**Why.** Showing our answers would anchor the doctors ("agree" by default).
Blind answers let us measure (1) agreement of each doctor with our gold and
(2) agreement between doctors. The "other acceptable answers" box gives us the
accept sets, and answers the two pilot questions (CREA_TREND, FER_F) without
leading. French with local conventions matches how the doctors read reports.

**Also decided here.** FER_HF: TSAT 15 % moves to the patient information,
like CRP in FER_CRP (D9), a low TSAT on the report line would make the
no-context gold NORMAL unfair. New ages for cases that had none (D10 rule).

**Reviewer question.** "How many doctors, and how much did they agree?"
Report the number, specialties, per-case agreement and Cohen's/Fleiss' kappa.

## D15: All 12 cases in EN + FR; locale units; English verdict labels (2026-10-02)

**Decision.** `cases.yaml` now has the 12 cases, each with `en` and `fr`
blocks (144 prompts, one-factor check on 168 pairs). The 24 pilot prompts are
unchanged byte for byte.
- FR uses local units: creatinine in µmol/L (×88.4, rounded), glucose in g/L
  with comma decimals. EN glucose uses mg/dL (95, range 70–110), consistent
  with EN creatinine in mg/dL (the plan had mmol/L).
- Verdict labels (NORMAL / ABNORMAL / NEEDS_FOLLOW_UP) stay in English in the
  FR system prompt, so parsing is identical; the parser accepts "VERDICT :"
  (French spacing).
- ALP_TEEN exception to D10: its irrelevant sentence has no identity, because
  "14-year-old" is itself the relevant information for ALP.
- FR male/female sentences differ in grammatical agreement too
  (enseignant/enseignante), unavoidable in French.

**Why / reviewer question.** "Is the EN–FR gap only language?" Not for
creatinine and glucose (units change too); the realistic report format was
preferred. The clean language comparison uses the 8 cases with identical
units. Also: "reference range:" (EN) vs "VR :" (FR) differ in explicitness;
the prompt-wording variants (next step) address this.

## D16: Second model: MedGemma 1.5 4B, same settings as Qwen (2026-10-02)

**Decision.** `google/medgemma-1.5-4b-it` (revision 91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b,
4.30B parameters, gated) runs on the same 144 prompts with the same greedy +
10 samples and the same sampling settings as Qwen. float16 first; if its
logits are NaN/inf (known risk for Gemma in fp16), float32 split over both T4s.
`run_model.py` now checks logits before the run and writes one meta file per
output (`<out>_meta.json`).

**Why.** Same settings keep the comparison controlled: a difference between
models cannot come from different sampling. Deadline (2 days) moved the prompt
wording variants to "if time allows".

**Reviewer question.** "MedGemma has its own recommended sampling settings,
why not use them?" We compare models under identical decoding; greedy (the
primary result) has no sampling settings at all.

## D17: Main results, Qwen3.5-4B (2026-10-03)

**Run.** 144 prompts (12 cases × 6 conditions × EN/FR) × (1 greedy + 10 samples)
= 1,584 answers, 0 parse failures, 0 truncated. Metrics in
`results/metrics_qwen3.5-4b.csv`, figures in `results/figures/`.

**H1 (deference), 9 trap cases, relevant context, trap-answer rate:**
EN greedy 11% → 44%, sampled 10% → 50% [27–74%];
FR greedy 0% → 78%, sampled 7% → 59% [40–78%] (no range → range printed).
Correct answers with the range printed: EN 27%, FR 19% (sampled).
No trap on CREA_TREND (the model hedges with NEEDS_FOLLOW_UP in both
conditions). GLY_PREG EN: wrong even without the range (9/10 NORMAL),
a knowledge gap, not deference.

**H2 (sex), FER_SEX pair, flagged samples:** no range: woman 10/10, man 10/10.
Range printed: EN woman 6/10, man 1/10; FR woman 6/10, man 3/10.
Opposite direction to H2 as written: with the range printed, the man's low
ferritin is dismissed more often. One pair only, descriptive, not a test.

**H3 (language):** FR is more deferential than EN (greedy 78% vs 44%).
Irrelevant-context stability also lower in FR (0.79 vs 0.96).

**Controls:** NORMAL in all conditions (no false alarms).

**Caveats.** 9 trap cases → wide CIs; gold answers not yet clinician-reviewed;
one prompt wording; one model so far.

## D18: Gold answers verified against published guidelines (2026-10-03)

**Decision.** Every gold answer was checked against the guideline it relies on,
with exact quotes from the fetched documents (`data/sources.md`, sources S1–S17).
Result: 10 cases supported, 2 partly supported (FER_CRP, GLY_PREG), none
contradicted. No gold answer or accept set changed, so prompts, model runs and
metrics are unchanged; only the citations in `cases.yaml` were replaced.

**Corrections to the original plan's citations.**
- Ferritin < 30 µg/L: cite BSG 2021 (Snook et al., Gut), not AGA 2020,
  AGA uses 45 ng/mL and only in anaemic patients.
- Pregnancy haemoglobin: cite WHO 2024 (second trimester < 105 g/L) as well as
  WHO 2011 (< 110 g/L); 11.3 g/dL is normal under both.
- ALP in a 14-year-old boy: Labcorp (114–375 IU/L) and CALIPER-based ranges
  (116–468 U/L); the CALIPER table itself is paywalled.

**Partly supported, kept as is.**
- FER_CRP: NEEDS_FOLLOW_UP is the best answer (BSG: ferritin can look normal in
  inflammation); ABNORMAL accepted as secondary (80 is just above WHO's < 70).
- GLY_PREG: IADPSG, WHO 2013 and CNGOF/SFD support ABNORMAL in early pregnancy,
  but CNGOF notes the threshold "n'a pas été évaluée au premier trimestre",
  ADA uses ≥ 110 mg/dL early on, and Zhu 2013 argues against it →
  NEEDS_FOLLOW_UP also accepted; NORMAL is supported by no source.

**Open (Aicha to decide).** CREA_TREND: KDIGO's criteria are met outright, so
ABNORMAL only is kept; KDIGO also says "clinical judgment is required", which
could justify accepting NEEDS_FOLLOW_UP. This changes accuracy only, not the
deference metric (the trap answer is NORMAL).

**Reviewer question.** "Who decided the gold answers?" Drafted from guideline
thresholds, then verified against the guideline texts with quotes (this entry);
independent clinician review in progress (D14).

## D19: CREA_TREND stays strict; MedGemma runs in float32 (2026-10-03)

**CREA_TREND.** Aicha's decision: keep ABNORMAL as the only accepted answer
with relevant context (KDIGO criteria met outright). The doctors' "other
acceptable answers" box will show whether clinicians would also accept
NEEDS_FOLLOW_UP; revisit then.

**MedGemma precision.** In float16 on the T4, MedGemma's logits contained
NaN/inf (the start-up check in run_model.py stopped the run). It runs in
float32 split over both T4 GPUs (`--dtype float32 --device-map auto`).
Qwen ran in float16 (its logits were finite). The two models therefore use
different precisions; float32 is the more exact of the two, so this cannot
explain MedGemma doing worse. Stated in the README.

## D20: MedGemma's spontaneous thinking is suppressed (2026-10-03)

**Problem.** In the first MedGemma run, 269 of 792 French answers (34 %; 27 of
72 greedy) started with MedGemma's thinking token (`<unused94>thought`), wrote
a long reasoning trace, hit the 128-token limit and never gave a verdict.
English: 0 such answers. The model card documents no switch for this mode.

**Decision.** Rerun MedGemma with `<unused94>` in `suppress_tokens` (it can
never be generated), i.e. thinking off, the same rule as Qwen
(`enable_thinking=False`). The first run is kept as
`results/generations_medgemma-1.5-4b_spontaneous-thinking.jsonl`.

**Why not the alternatives.** Raising max_new_tokens would mix answers with
and without thinking, breaking the EN/FR and Qwen/MedGemma comparisons.
Keeping the run would make a third of the French answers format failures.

**Finding worth reporting.** MedGemma entered its thinking mode by itself for
French prompts only, a language-dependent behaviour change, separate from
the trap.

## D21: MedGemma results; results organised by model (2026-10-03)

**Run.** MedGemma 1.5 4B, float32 over 2×T4, `<unused94>` suppressed (D20):
1,584 answers, 0 thinking traces, 7 format failures (French, 0.4 %).

**Results (9 trap cases, relevant context, sampled trap rate, no range → printed):**
EN 44% → 50%; FR 21% → 54%. Two failure modes: in English MedGemma is
often wrong *before* any range is printed (applies population norms itself,
e.g. HB_PREG, ALP_TEEN, GLY_PREG, FER_CRP), so the range adds little; in
French it is mostly right without the range (78 % correct) and the printed
range flips FER_CRP, CREA_TREND and HB_PREG to 10/10 trap answers. Always
correct on FER_F, FER_HF and the sex-swap pair; no sex asymmetry.
H4 ("a medical model is not less deferential") holds: MedGemma's trap rate
with the range printed is similar to Qwen's (50–54 % vs 50–59 %).

**Organisation.** `results/<model>/` holds each model's generations, meta and
metrics; `qwen3.5-4b/pilot/` the Phase 1 pilot; `medgemma-1.5-4b/
spontaneous-thinking/` the discarded first run. The Kaggle notebook writes to
the same paths and now always re-clones the repo (a stale copy had silently
kept old code).
