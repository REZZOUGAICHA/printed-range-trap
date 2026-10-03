# Doctor review: private answer key (do NOT send to the doctors)

Maps the case numbers in `fiche_relecture_medicale.docx` to our case ids and
current draft gold answers, to compare with the doctors' blind answers.

Situation A = context `none` (report line only). Situation B = context `relevant`.

| Cas | id | Gold A (accept) | Gold B (accept) |
|---|---|---|---|
| 1 | CTRL_FER | NORMAL | NORMAL |
| 2 | CREA_TREND | NORMAL | ABNORMAL (pilot question: accept NEEDS_FOLLOW_UP too?) |
| 3 | FER_SEX_F | NORMAL (NORMAL, NEEDS_FOLLOW_UP) | ABNORMAL (ABNORMAL, NEEDS_FOLLOW_UP) |
| 4 | GLY_PREG | NORMAL | ABNORMAL (ABNORMAL, NEEDS_FOLLOW_UP) |
| 5 | FER_CRP | NORMAL | NEEDS_FOLLOW_UP (NEEDS_FOLLOW_UP, ABNORMAL) |
| 6 | ALP_TEEN | ABNORMAL | NORMAL |
| 7 | FER_M | ABNORMAL (ABNORMAL, NEEDS_FOLLOW_UP) | ABNORMAL (ABNORMAL, NEEDS_FOLLOW_UP) |
| 8 | HB_PREG | ABNORMAL | NORMAL |
| 9 | CTRL_CREA | NORMAL | NORMAL |
| 10 | FER_HF | NORMAL | ABNORMAL |
| 11 | FER_SEX_M | NORMAL (NORMAL, NEEDS_FOLLOW_UP) | ABNORMAL (ABNORMAL, NEEDS_FOLLOW_UP) |
| 12 | FER_F | NORMAL (NORMAL, NEEDS_FOLLOW_UP); pilot question: is NORMAL right? | ABNORMAL (ABNORMAL, NEEDS_FOLLOW_UP) |

Label mapping: Normal = NORMAL, Anormal = ABNORMAL, À contrôler = NEEDS_FOLLOW_UP.

## What the form shows that is new (not yet in cases.yaml)

- French units: creatinine in µmol/L. Conversions (×88.4, rounded): 1.1 mg/dL = 97,
  0.7 = 62, 0.5 = 44, 0.8 = 71, 0.6 = 53, 1.2 = 106 µmol/L.
- FER_HF: TSAT 15 % is given in the patient information (situation B), not on
  the report line, same reasoning as FER_CRP (D9). See D14.
- Ages/sex added where the plan had none (D10 rule): GLY_PREG 29-year-old woman,
  CTRL_FER 35-year-old man, CTRL_CREA 50-year-old man, FER_HF 65-year-old man.
- Order of cases is mixed so that the woman/man pair (cas 3 and 11) is not adjacent.
