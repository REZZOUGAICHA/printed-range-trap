# Sources for the gold answers

Every gold answer in `cases.yaml` is checked against published guidelines.
Quotes were copied from the documents themselves (fetched 2026-10-03), marked
**primary** (the guideline/paper) or **secondary** (an official summary or a
copy reproduced inside another document). Clinician review is a second,
separate layer (`docs/doctor_review/`).

Verdict per case: **Supported** = the guideline directly gives the gold answer.
**Partly supported** = guidelines support the gold but disagree on details;
the accept set reflects that.

---

## Summary

| Case | Gold: no context (accept) | Gold: relevant context (accept) | Verdict | Key source |
|---|---|---|---|---|
| FER_F | NORMAL (NORMAL, NEEDS_FOLLOW_UP) | ABNORMAL (ABNORMAL, NEEDS_FOLLOW_UP) | Supported; no-context partly | BSG 2021 [S3] |
| FER_M | ABNORMAL (ABNORMAL, NEEDS_FOLLOW_UP) | ABNORMAL (ABNORMAL, NEEDS_FOLLOW_UP) | Supported | BSG 2021 [S3], MedlinePlus [S5] |
| FER_SEX_F | NORMAL (NORMAL, NEEDS_FOLLOW_UP) | ABNORMAL (ABNORMAL, NEEDS_FOLLOW_UP) | Supported | BSG 2021 [S3] |
| FER_SEX_M | NORMAL (NORMAL, NEEDS_FOLLOW_UP) | ABNORMAL (ABNORMAL, NEEDS_FOLLOW_UP) | Supported | BSG 2021 [S3] |
| FER_CRP | NORMAL (NORMAL) | NEEDS_FOLLOW_UP (NEEDS_FOLLOW_UP, ABNORMAL) | Partly supported | BSG 2021 [S3], WHO 2020 [S1] |
| FER_HF | NORMAL (NORMAL) | ABNORMAL (ABNORMAL) | Supported | ESC 2021 [S8] |
| CREA_TREND | NORMAL (NORMAL) | ABNORMAL (ABNORMAL) | Supported | KDIGO 2012 [S9] |
| GLY_PREG | NORMAL (NORMAL) | ABNORMAL (ABNORMAL, NEEDS_FOLLOW_UP) | Partly supported | IADPSG 2010 [S13], CNGOF/SFD 2010 [S14] |
| HB_PREG | ABNORMAL (ABNORMAL) | NORMAL (NORMAL) | Supported | WHO 2024 [S11] |
| ALP_TEEN | ABNORMAL (ABNORMAL) | NORMAL (NORMAL) | Supported | Labcorp [S10b], CALIPER-based [S10a] |
| CTRL_FER | NORMAL (NORMAL) | NORMAL (NORMAL) | Supported | MedlinePlus [S5] |
| CTRL_CREA | NORMAL (NORMAL) | NORMAL (NORMAL) | Supported | KDIGO 2012 [S9] |

---

## Case notes

**FER_F** (woman 28, heavy periods, fatigue, hair loss; ferritin 25 µg/L, printed 15–150).
Relevant: 25 is below "<30 µg/L … low body iron stores" [S3], with an obvious
cause (heavy periods) → ABNORMAL. No context: 25 is inside the printed range
and above WHO's <15 for healthy adults [S1]; but BSG [S3] and the York NHS
directory [S6] ("15 - 30 µg/L may suggest depletion of iron stores") make
NEEDS_FOLLOW_UP equally defensible → both accepted. No guideline calls 25
definite deficiency without context.

**FER_M** (man 28, blood in stool; ferritin 25, printed 30–400). Below the
printed male range [S5][S6] and below <30 [S3], with a likely bleeding source → ABNORMAL.
Not a trap case (the range and the gold agree).

**FER_SEX_F / FER_SEX_M** (fatigue, hair loss; ferritin 25, printed 15–150 for both).
Below <30 [S3] with symptoms → ABNORMAL for both. For the man, 25 is also below the
usual male lower limit of 30 [S5][S6], so the case for flagging him is, if anything, stronger.

**FER_CRP** (woman 45, rheumatoid arthritis flare, CRP 60; ferritin 80).
"apparently normal levels may occur with iron deficiency in the context of an
inflammatory disease process" [S3]; WHO: assess inflammation markers with ferritin [S1]
→ NEEDS_FOLLOW_UP. ABNORMAL is weaker (80 is just above WHO's <70 cutoff under
inflammation [S1]) but accepted, since only >150 makes deficiency unlikely [S3].

**FER_HF** (man 65, HFrEF, TSAT 15 %; ferritin 80). ESC: iron deficiency in HF =
ferritin <100 ng/mL, or 100–299 with TSAT <20 % [S8]. 80 < 100 → ABNORMAL.

**CREA_TREND** (man 55, surgery 2 days ago, ibuprofen; creatinine 0.7 → 1.1 mg/dL,
62 → 97 µmol/L in 48 h). KDIGO: rise ≥0.3 mg/dL (≥26.5 µmol/L) within 48 h [S9].
Rise = 0.4 mg/dL (35 µmol/L), and 1.57× baseline → ABNORMAL (AKI).
Open choice (D18): KDIGO also says "clinical judgment is required" [S9], so
NEEDS_FOLLOW_UP could get credit; we keep ABNORMAL only, because both criteria are met.

**GLY_PREG** (woman 29, first trimester; fasting glucose 0.95 g/L = 95 mg/dL = 5.3 mmol/L,
printed 0.70–1.10). IADPSG [S13], WHO 2013 [S15] and the French CNGOF/SFD [S14] all
classify fasting ≥5.1 mmol/L (0.92 g/L) in early pregnancy as gestational diabetes
→ ABNORMAL. But the French text itself notes the threshold "n'a pas été évaluée au
premier trimestre" [S14], ADA uses ≥110 mg/dL early in pregnancy [S16], and a cohort
study argues against 5.1 at the first visit [S17] → NEEDS_FOLLOW_UP also accepted.
NORMAL is not supported by any source.

**HB_PREG** (woman 30, second trimester; Hb 11.3 g/dL = 113 g/L, printed 12–16).
WHO 2024: anaemia in the second trimester = Hb <105 g/L [S11]; WHO 2011: pregnancy <110 g/L [S12].
113 is above both → NORMAL. Without context: below the non-pregnant women cutoff
<120 g/L [S11] → ABNORMAL.

**ALP_TEEN** (boy 14, growth spurt; ALP 280 U/L, printed adult 40–130).
Male 14 years: 114–375 IU/L [S10b]; males 13–14 years: 116–468 U/L [S10a].
280 is inside both → NORMAL. Without age: above the adult range (>20 years: 44–121 [S10b]) → ABNORMAL.
ALP intervals depend on the assay; 280 is well inside every source checked.

**CTRL_FER** (healthy man 35; ferritin 90, printed 30–400): inside "30 to 400" [S5],
above every deficiency cutoff (15/30/45) → NORMAL.

**CTRL_CREA** (man 50, stable; creatinine 0.8 mg/dL, same as last year): no KDIGO
criterion met [S9], inside the printed range → NORMAL.

---

## References and exact quotes

**[S1] WHO (2020).** *WHO guideline on use of ferritin concentrations to assess iron status in individuals and populations.* Primary (official executive summary PDF): https://cdn.who.int/media/docs/default-source/micronutrients/ferritin-guideline/ferritin-guidelines-executivesummary.pdf
- Rec 1.1: "Ferritin concentration is a good marker of iron stores and should be used to diagnose iron deficiency in otherwise apparently healthy individuals (strong recommendation, low certainty of evidence)."
- Table 1, adults 20–59 years: apparently healthy <15 µg/L; infection or inflammation <70 µg/L.
- Rec 1.2: "In individuals with infection or inflammation, a ferritin concentration below 30 µg/L in children and 70 µg/L in adults may be used to indicate iron deficiency (conditional recommendation, low certainty of evidence)."
- Footnote: "Markers of inflammation should be assessed along with the ferritin concentration, and ferritin adjusted as necessary."

**[S2] AGA (2020).** Ko CW et al. *AGA Clinical Practice Guidelines on the Gastrointestinal Evaluation of Iron Deficiency Anemia.* Gastroenterology. doi:10.1053/j.gastro.2020.06.046. Quote from AGA's guidance page (gastro.org):
- "In patients with anemia, AGA recommends using a cutoff of 45 ng/mL over 15 ng/mL when using ferritin to diagnose iron deficiency."
- Note: applies to anaemic patients; not used as the main source (the original plan cited it for <30, which was wrong).

**[S3] BSG (2021).** Snook J et al. *British Society of Gastroenterology guidelines for the management of iron deficiency anaemia in adults.* Gut. doi:10.1136/gutjnl-2021-325210. Primary, full text PMC8515119.
- "An SF level of <15 µg/L is indicative of absent iron stores, while SF levels of less than 30 µg/L are generally indicative of low body iron stores."
- "An SF cut-off of 45 µg/L has been suggested as providing the optimal trade-off between sensitivity and specificity for iron deficiency in practice."
- "As SF is an acute phase protein, however, apparently normal levels may occur with iron deficiency in the context of an inflammatory disease process."
- "An SF value above 150 µg/L is unlikely to occur with absolute iron deficiency, even in the presence of inflammation."

**[S5] MedlinePlus.** *Ferritin blood test.* https://medlineplus.gov/ency/article/003490.htm
- "Male: 30 to 400 nanograms per milliliter (ng/mL)"; "Female: 13 to 150 ng/mL"; "Normal value ranges may vary slightly among different laboratories."

**[S6] York and Scarborough Teaching Hospitals NHS Trust, test directory (ferritin).**
- Male "30 - 400 µg/L". Female <60: "<15 µg/L consistent with iron deficiency"; "15 - 30 µg/L may suggest depletion of iron stores"; "30 - 150 µg/L Normal".

**[S7] Royal United Hospitals Bath NHS (2025).** *Ferritin interpretation – a guide for GPs.*
- "We would suggest that 11-100 ug/L in women and 24-100ug/L in men is the range where inflammation, infection or influence from malignancy, renal, liver and heart failure should be excluded."

**[S8] ESC (2021).** McDonagh TA et al. *2021 ESC Guidelines for the diagnosis and treatment of acute and chronic heart failure.* Eur Heart J. doi:10.1093/eurheartj/ehab368. Primary (full PDF).
- §13.5: "In patients with HF, iron deficiency is defined as either a serum ferritin concentration <100 ng/mL or 100–299 ng/mL with transferrin saturation (TSAT) <20%."
- "Ferritin … is increased by inflammation and several disorders such as infection, cancer, liver disease, and HF itself. Hence, higher cut-off values have been applied for the definition of iron deficiency in patients with HF."
- The 2023 focused update (doi:10.1093/eurheartj/ehad195) keeps this definition (secondary summaries; PDF not retrieved).

**[S9] KDIGO (2012).** *KDIGO Clinical Practice Guideline for Acute Kidney Injury.* Kidney Int Suppl 2012;2:1–138. Primary: https://kdigo.org/wp-content/uploads/2016/10/KDIGO-2012-AKI-Guideline-English.pdf
- Rec 2.1.1: "AKI is defined as any of the following (Not Graded): Increase in SCr by ≥0.3 mg/dl (≥26.5 µmol/l) within 48 hours; or Increase in SCr to ≥1.5 times baseline, which is known or presumed to have occurred within the prior 7 days; or Urine volume <0.5 ml/kg/h for 6 hours."
- "Clinical judgment is required in order to determine if patients seeming to meet criteria do, in fact, have disease…"
- The KDIGO 2026 AKI/AKD public-review draft (March 2026) keeps the same creatinine criteria.

**[S10a] University of Iowa Pathology Handbook, alkaline phosphatase** (CALIPER-based; Estey et al., CLSI-based transference of the CALIPER database). https://www.healthcare.uiowa.edu/path_handbook/handbook/test1487.html
- Males "13 - 14 years 116-468 U/L"; "15 - 16 years 82-331 U/L".
- Original CALIPER study: Colantonio DA et al. Clin Chem 2012;58(5):854–68. doi:10.1373/clinchem.2011.177741 (table not accessed).

**[S10b] Labcorp.** *Pediatric Testing Reference Ranges* (PDF). https://www.labcorp.com/content/dam/labcorp/drupal/178250_DX_TL_PediatricTestRef_Final.pdf
- Alkaline phosphatase (IU/L), male / female: "14 years 114–375 64–161"; ">20 years 44–121 44–121".

**[S11] WHO (2024).** *Guideline on haemoglobin cutoffs to define anaemia in individuals and populations.* ISBN 978-92-4-008854-2. Primary: https://iris.who.int/bitstream/handle/10665/376196/9789240088542-eng.pdf
- Table 2 (g/L): "Adults, 15–65 years, nonpregnant women <120 … Pregnancy First trimester <110, Second trimester <105, Third trimester <110".

**[S12] WHO (2011).** *Haemoglobin concentrations for the diagnosis of anaemia and assessment of severity* (WHO/NMH/NHD/MNM/11.1). Secondary (table reproduced in WHO 2024, Table 1): pregnant women, no anaemia ≥110 g/L; non-pregnant women ≥120 g/L.

**[S13] IADPSG (2010).** *International Association of Diabetes and Pregnancy Study Groups Recommendations on the Diagnosis and Classification of Hyperglycemia in Pregnancy.* Diabetes Care 33(3):676–682. doi:10.2337/dc09-1848. Primary (PMC2827530).
- "It is recommended that an FPG value in early pregnancy ≥5.1 mmol/l (92 mg/dl) also be classified as GDM."

**[S14] CNGOF / SFD (2010).** *Le diabète gestationnel* (recommandations pour la pratique clinique). Médecine des maladies Métaboliques 2011;5(HS2). Primary (SFD PDF).
- "Au premier trimestre, en présence de facteurs de risque, il est recommandé de réaliser une glycémie à jeun (grade B)."
- "Nous proposons comme seuil pour le diagnostic de DG, la valeur de 0,92 g/L (5,1 mmol/L) de glycémie à jeun définie par un consensus international (IADPSG). Il faut cependant noter que la pertinence de ce seuil n'a pas été évaluée au premier trimestre (accord professionnel)."

**[S15] WHO (2013).** *Diagnostic criteria and classification of hyperglycaemia first detected in pregnancy* (WHO/NMH/MND/13.2). Secondary (PAHO presentation of the guideline):
- "GDM should be diagnosed at any time in pregnancy if one or more of the following criteria are met: - fasting plasma glucose 5.1-6.9 mmol/l (92 -125 mg/dl) … Quality of evidence: very low Strength of recommendation: weak".

**[S16] ADA (2025).** *Standards of Care in Diabetes 2025*, §2. doi:10.2337/dc25-S002. Primary (PMC11635041).
- "2.26c Screen for early abnormal glucose metabolism with dysglycemia using FPG 110–125 mg/dL (6.1–6.9 mmol/L) or A1C 5.9–6.4%."

**[S17] Zhu WW et al. (2013).** Diabetes Care. doi:10.2337/dc12-1157. Evidence (abstract):
- "Based on our data, we cannot support an FPG value ≥5.10 mmol/L at the first prenatal visit as the criterion for diagnosis of GDM."

---

## Access notes
- Not retrieved as primary: the WHO 2011 and 2013 PDFs (quotes are from WHO 2024 and a PAHO summary), the ESC 2023 update PDF, and the CALIPER 2012 table (paywalled).
- AGA quote from the society's own page; BSG from the PMC full text of the same article.
- Lab reference intervals vary by laboratory and assay; the printed ranges used in the cases are within the ranges reported above.
