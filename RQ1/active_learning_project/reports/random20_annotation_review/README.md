# Random review of 20 annotations

**Result: 4 pairs agree on both labels, 9 pairs have at least one disputed label, and 7 pairs remain uncertain.**

These are reviewer assessments, not expert-certified ground truth. Do not describe this as proof that only 20% of all 1,100 annotations are correct. The sample is small, seven cases are unresolved, and this is an AI review rather than independent human validation.

## Method

- Uniform random sample without replacement from the 1,100 rows with both labels; seed `20260925`. No resampling or substitution.
- Reviewed the complete original Hindi `Answer Transcription` against the question and the supplied moderator rubric. Additional claims in an answer count toward accuracy even when they concern a different question.
- Decisions were recorded before printing the current sample label comparison. This was not fully blinded: the earlier audit had already exposed some rows and labels, including 1288 and 226.
- The rubric was available from its text extracted during the previous audit; the original PDF was no longer at its initial Desktop path during this follow-up.
- The rubric asks for an aligned expert reference. None was established for these training rows, and the earlier audit found mismatches in `Relevant answer`. Consequently, this is a provisional factual/safety and rubric-consistency review, not certified reference-answer agreement.
- Used primary healthcare guidance for material medical concerns. Ordinary Hindi loanwords alone did not trigger a downgrade. Tone was not scored. Missing an ASHA/PHC mention alone was not penalized.
- All 20 sampled rows are SRHR; this sample does not directly test Mental Wellness annotation quality.
- An options range such as 0 / 1 means the exact replacement needs adjudication, but an existing 2 is disputed under either option. If the existing score is one of multiple plausible options, the result is Uncertain. Disputed means the reviewer challenges the score, not an expert-confirmed error.

## Counts

| Assessment | Agrees | Disputed | Uncertain |
|---|---:|---:|---:|
| Accuracy | 7 | 7 | 6 |
| Clarity/context | 11 | 6 | 3 |
| Both labels together | 4 | 9 | 7 |

Agreement is not the same as an answer being good: row 226 agrees because both the answer and its zero labels correctly indicate failure to answer the question.

## Every sampled row

| Row ID | Claude A / C | Reviewer A | Reviewer C | Result |
|---|---|---|---|---|
| 163 | 2 / 2 | 0 / 1 | 2 | Disputed |
| 623 | 2 / 2 | 1 / 2 | 1 / 2 | Uncertain |
| 240 | 2 / 2 | 0 | 1 | Disputed |
| 843 | 2 / 2 | 2 | 2 | Agrees |
| 1288 | 0 / 0 | 2 | 2 | Disputed |
| 967 | 2 / 2 | 1 / 2 | 2 | Uncertain |
| 829 | 2 / 2 | 2 | 2 | Agrees |
| 904 | 2 / 2 | 2 | 1 | Disputed |
| 742 | 2 / 2 | 1 | 2 | Disputed |
| 312 | 2 / 2 | 2 | 1 | Disputed |
| 395 | 2 / 2 | 2 | 1 / 2 | Uncertain |
| 552 | 2 / 2 | 1 / 2 | 2 | Uncertain |
| 357 | 2 / 2 | 1 / 2 | 1 / 2 | Uncertain |
| 300 | 2 / 2 | 1 / 2 | 2 | Uncertain |
| 498 | 2 / 2 | 2 | 2 | Agrees |
| 374 | 2 / 2 | 0 / 1 | 1 | Disputed |
| 977 | 2 / 2 | NA / 1 | NA / 1 | Disputed |
| 935 | 2 / 2 | 0 / 1 | 2 | Disputed |
| 226 | 0 / 0 | 0 | 0 | Agrees |
| 444 | 2 / 2 | 1 / 2 | 2 | Uncertain |

A = accuracy; C = clarity/context. NA is explicitly allowed by the expert rubric, although the present model uses only numeric classes.

## Reasons and evidence

**Row 163 — Disputed.** Gives a blanket rule against ejaculation inside the vagina in the first and last three months of pregnancy. This is not a general recommendation for uncomplicated pregnancy. Comfort/consent advice is useful, so expert adjudication should resolve 0 versus 1; full accuracy 2 is not supported. [Source 1](https://www.tommys.org/pregnancy-information/im-pregnant/sex-pregnancy)

**Row 623 — Uncertain.** Correctly emphasizes individual variation in desire, but generalizes male/female response speed and uses an unexplained sexual-response-cycle term. Whether these limitations warrant 1 instead of 2 needs expert calibration.

**Row 240 — Disputed.** The answer adds a claim that pregnancy cannot occur during menstruation and treats condoms as optional in that discussion. This is consequential misinformation. Frequent English terms and the lengthy switch from menopause to another question also reduce clarity/context. [Source 1](https://www.plannedparenthood.org/blog/can-you-get-pregnant-if-you-have-sex-during-your-period)

**Row 843 — Agrees.** Directly answers with mutual willingness, communication, gradual affection, and avoidance of force. Everyday Hindi and feasible interpersonal advice; no material factual problem identified.

**Row 1288 — Disputed.** Gives an appropriate basic explanation of hormone-related breast development at puberty, with understandable Hindi comparisons to other body changes. No visible basis for zero labels. [Source 1](https://www.cuh.nhs.uk/patient-information/precocious-puberty-in-girls/)

**Row 967 — Uncertain.** Reasonably distinguishes normal masturbation from interference with daily life, but its filling-sack explanation of semen is oversimplified and implies discharge is necessary. Sperm can be reabsorbed. Adjudicate whether this is a material factual gap; language and suggested activities are accessible. [Source 1](https://my.clevelandclinic.org/health/body/sperm) [Source 2](https://my.clevelandclinic.org/health/articles/24332-masturbation)

**Row 829 — Agrees.** Answers the moral/relationship question with mutual consent, respect, and non-judgement. It does not make a blanket claim that every sexual practice is medically risk-free. Relevant, accessible Hindi.

**Row 904 — Disputed.** The core answer that desire varies by person and that the question needs clarification is reasonable. Most of the transcript becomes an unrelated channel-promotion/instructions segment, so clarity/context is only partial. Sarcasm would be assessed separately under tone.

**Row 742 — Disputed.** Offers partial advice about pacing and partner arousal, but assumes excitement explains early ejaculation and gives little useful management or guidance for persistent symptoms. The answer is incomplete rather than fully correct/comprehensive. Spoken Hindi and the practical context remain accessible. [Source 1](https://www.nhs.uk/conditions/ejaculation-problems/)

**Row 312 — Disputed.** Broadly useful breakup guidance: reflect, explain differences, communicate respectfully, and maintain boundaries. No material factual contradiction identified. Numerous English phrases, broken transcription, repetition, and the long structure reduce clarity for the rubric's audience.

**Row 395 — Uncertain.** Reasonable relationship advice and practical phone/offline alternatives. Assumes WhatsApp access and suggests media sharing; context could be 1 for some urban/digital assumptions or 2 because these are optional and alternatives are supplied. Requires rubric calibration.

**Row 552 — Uncertain.** Correct that desire is multifactorial and hormone levels do not determine every individual's experience. Its closing dismissal may understate a genuine hormonal cause. The difference between reassurance and misleading dismissal needs expert adjudication; language/context are accessible. [Source 1](https://www.nhs.uk/symptoms/loss-of-libido/) [Source 2](https://www.nhs.uk/conditions/male-menopause/)

**Row 357 — Uncertain.** Affirms transgender/kinnar people's relationships while mentioning legal limitations, but conflates varied identities/anatomies and leaves marriage/adoption eligibility vague. The recording date and intended legal reference are absent. Do not certify its legal accuracy or choose a definitive 1/2 without an expert reference. [Source 1](https://api.sci.gov.in/supremecourt/2022/36593/36593_2022_1_1501_47792_Order_17-Oct-2023.pdf)

**Row 300 — Uncertain.** Apologizing may repair conflict, but the advice to approach/apologize repeatedly after being blocked does not clearly respect requests for no contact. It eventually advises stopping. Whether this is a material safety/completeness gap depends on relationship context; clear Hindi is not the issue.

**Row 498 — Agrees.** Directly explains the distinction between vaginal and anal intercourse and emphasizes mutual consent, hygiene, and condoms. It is a basic answer to a choice/anatomy question, not a comprehensive technique guide; no explicit factual error identified.

**Row 374 — Disputed.** Correct consent advice is followed by an unqualified statement that swallowing semen causes no harm, without STI qualification, and a confused account of twinning. The whole answer cannot merit accuracy 2. Multiple topics and garbled reproductive explanation reduce clarity. Expert should adjudicate 0 versus 1. [Source 1](https://www.cdc.gov/sti/about/about-sti-risk-and-oral-sex.html) [Source 2](https://www.mcht.nhs.uk/our-services/maternity/patient-information-maternity/multiple-pregnancy)

**Row 977 — Disputed.** Explicitly says the cause of head tingling is unknown, then assumes it feels pleasant and tells the caller to enjoy it. It provides no substantive explanation or useful next step. Apply NA for insufficient factual/generic content or 1 for partial response; neither dimension supports an unqualified 2.

**Row 935 — Disputed.** Refers the caller to a doctor but says recurrent skin damage has no connection to masturbation and that the triggering activity need not be disclosed. Friction can cause chafing; the dismissal and withholding relevant history undermine otherwise useful referral advice. Expert should resolve 0/1; it is not fully correct. Hindi and the referral itself are understandable. [Source 1](https://my.clevelandclinic.org/health/articles/24332-masturbation)

**Row 226 — Agrees.** Question asks whether unprotected sex during menstruation can cause pregnancy. Answer discusses changes in a woman's voice during sex and never answers the pregnancy question. Both zeros follow the rubric's off-question/ground-truth mismatch example.

**Row 444 — Uncertain.** Explains ordinary facial flushing plausibly, but assumes the caller's description of darkening means harmless redness and resolves afterward. With no aligned reference or symptom clarification, full accuracy versus incomplete reassurance is uncertain. Hindi and relevance are otherwise adequate.

## Files

`random20_review.xlsx` and `random20_review.csv` contain the full questions, answers, original labels, proposed score options, reasons, and blank expert-adjudication columns. `summary.json` records counts, selected IDs, seed, and source hash. `reviewer_decisions.json` contains the documented judgements; `build_review.py` reproduces the sample and comparison. The script packages these judgements; it does not independently re-evaluate answer semantics.

Original training labels were not changed. An expert should adjudicate the disputed and uncertain examples before any correction is applied.
