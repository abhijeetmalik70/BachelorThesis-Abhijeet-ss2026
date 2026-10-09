# Full fresh-minority generation

User approved the 12-example draft on 9 October 2026 and authorized full generation,
explicitly withholding authorization to start training.

The eligible original pool has Accuracy counts 42 / 50 / 1,558. Add 1,516 new
class-0 pairs and 1,508 new class-1 pairs, including six approved draft pairs per
class. The remaining generation targets are 1,510 / 1,502. The eventual combined
pool has 4,674 pairs, with 1,558 per Accuracy class. Context & Relevance is labelled
separately and is not forced to be balanced.

`../../generate_fresh_minority.py` performs inference only with
Qwen2.5-32B-Instruct by default, using 4-bit NF4 quantization and batch size 1.
The script supports Qwen through AutoModelForCausalLM, and optionally Gemma 3
through --loader gemma3. It never loads the collapsed trained adapter. It drafts new Hindi questions
and answers based on 1,416 distinct eligible training questions. Source questions
matching the fixed validation questions are excluded. The original class-2 answers
provide drafting context; their inherited labels do not establish medical validity.
Generated rationales and labels are provisional model annotations. Qwen drafting and eventual Gemma scoring use different model families, but a larger
generator does not establish valid labels or replace independent review.
No expert or independent teacher validation is claimed.

The generator varies mistake types, question components and proposed context labels.
It rejects malformed records, requested-label mismatches, exact pair duplicates and
exact validation-question overlaps. These checks cannot establish factual correctness,
semantic novelty, label validity, near-duplicate independence, or equal topic coverage.
Related source topics and multiple variants per source remain correlated. Review those
properties before interpreting a training result or calling the data fully validated.

Generation is resumable using `accepted.jsonl`, with raw outputs and rejections in
`generation_audit.jsonl` and counts in `generation_status.json`. The completed outputs
are `synthetic_examples.csv` and `augmented_train.csv`. Only the twelve draft rows have
been reviewed by the user. Full generation has not yet started while SSH is unavailable.

Run generation only from a CUDA environment. Model downloading is enabled unless
--local-files-only is supplied. The default 32B loader refuses a free-memory budget
below 20 GiB before downloading weights or writing accepted examples. The known
16 GB RTX A4000 cannot fit this 32B configuration; use a larger GPU or explicitly
select Qwen/Qwen2.5-14B-Instruct with batch size 1 and smoke-check memory first.
Actual memory fit for either model is not verified while SSH is unavailable.

Generation command (requires a sufficiently large GPU):

```bash
python -u RQ1/experiments/generate_fresh_minority.py \
  --data-dir RQ1/experiments/synthetic_data/fresh_minority_20261009
```

`planned_training_config.json` is a future configuration only. It specifies unweighted
loss with no oversampling and preserves the prior two-epoch, effective-batch-24 recipe.
Neither generation nor this document executes the training runner. Do not launch
training until the user authorizes it after generation.
