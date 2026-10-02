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
