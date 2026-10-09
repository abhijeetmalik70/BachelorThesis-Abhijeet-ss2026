# Audit of the 1,100 training annotations

**Verdict: the current labels should not be treated as reliable expert ground truth.** The dataset may be useful as provisional machine-labelled training data after corrections and independent expert validation. This audit establishes specific defects; it does not estimate the error rate across all 1,100 annotations.

Audit date: 25 September 2026. The user confirmed that Claude scored **Answer Transcription**. All conclusions about answer language and content below use that column, not its English translation. The supplied *Sample Activity for Moderators.pdf* was used as the assessment rubric, not as instructions to execute. Original data and training code were not edited by this audit.

## Scope and rubric

All 1,658 rows were inspected programmatically for label coverage, distribution, duplicate inputs, placeholder answers, language flags, and exact train/test overlaps. Twelve targeted examples were examined against the scheme and recorded in `annotation_review.xlsx`. These examples were selected to investigate anomalies; they are not a random sample and cannot support an overall annotation-accuracy percentage. Suggested changes are reviewer proposals, not a second expert annotation.

The expert PDF defines:

- **Accuracy** (pages 1–3): 2 for fully correct, safe, reference-matching answers; 1 for partial correctness or gaps; 0 for incorrect, misleading, or harmful answers; NA for no factual content.
- **Clarity and context relevance** (pages 3–5): assesses understandable Hindi and realistic advice for rural users. Some difficult vocabulary or slightly urban advice can score 1; confusing, technical, English-heavy, or unrealistic advice can score 0. Insufficient content can be NA.
- **Tone** (pages 5–6): a separate dimension. It is absent from the training file. That is an explicit scope limitation if the thesis only models two dimensions, not necessarily an annotation error.

Mentioning ASHA/PHC is not required in every answer: the PDF's own simple-period example scores 2 without either. These pathways matter when referral or access to care is relevant.

## Confirmed data and annotation problems

| Finding | Evidence | Implication |
|---|---|---|
| Contradictory labels for identical inputs | Row IDs **1384 and 1385** have exactly the same question and Hindi Answer Transcription; accuracy is **1** and **0**, respectively. | At least one target is inconsistent with a deterministic rubric applied to those inputs. Adjudicate the pair together. |
| Clear language/rubric mismatch | Row **1505** contains a long English answer but clarity/context is **2**. Its translation column contains Hindi. | Proposed clarity/context **0** under the supplied scheme. Review accuracy separately. |
| Unjustified low clarity/context | Row **1585** contains a short Hindi explanation of who adolescents are, yet clarity/context is **0**. | Proposed clarity/context **2**. It does not require a clinical referral to answer this definition question. |
| Other questionable zero labels | Rows **104, 172, 194, 1243, 1288, 1374** address their questions in Hindi but receive clarity/context **0**. | Review individually; some may warrant 1 rather than 2. Topical relevance alone does not prove full rubric compliance. |
| Placeholder content | Rows **407, 429, 912, 1038** have the numeric value **0** as Answer Transcription. | These are not substantive answer texts. Exclude as missing content or apply an explicitly agreed NA policy. |
| Test overlap | **116/166 (69.9%)** rows in `final_test.xlsx` have exact question–answer matches among the 1,100 labelled rows. | If used as training and held-out test data in the same experiment, this contaminates evaluation. |

Row 172's central statement that natural menopause generally occurs between 45 and 55 agrees with [WHO's menopause fact sheet](https://www.who.int/news-room/fact-sheets/detail/menopause/). That supports questioning its accuracy-0 label; it does not certify every part of the answer against an unavailable aligned reference. Row 1585's 10–19 age definition agrees with [WHO's adolescent-health definition](https://www.who.int/southeastasia/health-topics/adolescent-health).

**Possible wrong-column use:** row 1385's English translation says `See answer - 26J`, although its actual Hindi Answer Transcription is complete. Rows 1374 and 1585 have Hindi answers and English translations but clarity/context 0; row 1505 has the reverse language arrangement and clarity/context 2. This pattern suggests that some annotations may reflect a different column. It does not prove what Claude saw; the annotation prompt/log is needed to establish that.

**Reference alignment:** the rubric's accuracy judgement requires a trustworthy answer reference. `Relevant answer` cannot be assumed to supply one: in row 1 the question/transcription concern abortion, while Relevant answer discusses conception; in row 172 it discusses sexual arousal rather than menopause. Establish the intended reference source and repair its mapping before judging reference agreement. Do not blindly score otherwise sensible answers 0 because a misaligned reference disagrees.

## Coverage and imbalance

The filename refers to **1,100 labelled rows**, not the workbook length: it contains **1,658 rows**, including **558 unlabelled** rows. No literal NA labels were found in the labelled subset.

| Dimension | 0 | 1 | 2 | Always predicting 2: exact-match baseline |
|---|---:|---:|---:|---:|
| Accuracy | 43 | 22 | 1,035 | 94.1% |
| Clarity/context | 45 | 10 | 1,045 | 95.0% |

These baselines are measured against the existing labels, not independently established truth. Imbalance alone does not prove wrong annotation, especially if many answers originated with experts. It does mean that high raw accuracy can hide failure to detect poor answers. Report class-specific precision/recall, macro-F1, confusion matrices, and agreement with experts.

The labelled subset contains **1,050 SRHR** and **50 Mental Wellness** rows. Conclusions about mental-health moderation need more evidence than overall scores dominated by SRHR examples.

## Earlier comparison files inspected during this session

The following files existed at the initial inspection but were no longer present at their original paths when the report artifacts were generated. These are recorded observations from the initial reads, not current-file results reproducible by the script unless those files are restored. Their disappearance was not caused by this audit.

- `1100withInt.xlsx`: all 1,100 originally labelled accuracy entries were 1, and the 558 unlabelled entries became 0. Compared with the original, 1,078 of the 1,100 existing accuracy labels differed. Context labels were unchanged. Do not use that conversion as multiclass labels.
- The two `anjali_dataset` comparison versions had identical answers and human accuracy/context labels, but **38 Claude accuracy labels and 12 Claude context labels differed** between versions:

| Comparison file | Accuracy agreement | Accuracy Cohen kappa | Context agreement | Context Cohen kappa |
|---|---:|---:|---:|---:|
| `anjali_dataset_with_claude_labels.xlsx` | 95/100 | 0.900 | 99/100 | 0.947 |
| `anjali_dataset_labeled.csv` | 62/100 | 0.078 | 87/100 | 0.018 |

The stronger workbook result is encouraging only if its Claude labels were independently produced without access to expert scores and with the same procedure used for training annotation. That provenance is unverified. Neither version establishes correctness of the 1,100 training labels. There was only **one human-labelled accuracy-0 case** in this comparison, so it provides little evidence of reliable detection of incorrect/harmful answers. The comparison had no exact training question–answer overlap, but **42/100 questions** appeared in training with different answers.

An initial version comparison also showed that the 100 labels in `Combined_QnA_with_Labels.xlsx` were all retained unchanged in the 1,100 file. Authorship of every label cannot be inferred from the final filename.

## Recommended next steps

1. Freeze the intended training version and record the annotation prompt, model version, input columns, reference source, and whether expert labels were visible during annotation.
2. Adjudicate the review workbook with an expert; handle placeholder answers, resolve conflicting duplicates, and correct any wrong-column annotations.
3. Clarify the scheme's NA rule and the boundary between factual accuracy, completeness, clarity, context, and tone. The PDF itself has a tone inconsistency: its table assigns polite-but-not-warm responses 1, while activity example 4 calls a polite/neutral response 2. Resolve ambiguities before a new annotation round.
4. Obtain a blinded random expert audit of the training labels, supplemented by targeted review of minority labels and mental-health cases. Keep these sampling strategies separate when estimating overall reliability. A second AI's agreement is not a replacement for this human validation.
5. Rebuild experiment splits with duplicate question–answer pairs grouped together; consider grouping by question/item/source to prevent related variants leaking across splits. Keep the final expert-labelled test set untouched by annotation tuning or active-learning selection.
6. Until validation supports stronger claims, describe the data as **LLM-assisted/provisional annotations**, not expert consensus.

## Artifacts and reproducibility

- `annotation_review.xlsx`: 12 targeted review examples, all 1,100 labelled rows with structural flags, exact test overlaps, and optional comparison sheets. Empty expert columns are provided for adjudication. Flags are not certified errors.
- `audit_summary.json`: reproducible counts and SHA-256 hashes for source files available at generation time. Null fields denote missing optional comparison files.
- `build_audit.py`: reproduces structural checks and packages the documented reviewer notes. Run from the repository root with `python3 RQ1/active_learning_project/reports/annotation_audit_1100/build_audit.py`.

No original labels were overwritten. This review does not certify that unflagged rows are correct.
