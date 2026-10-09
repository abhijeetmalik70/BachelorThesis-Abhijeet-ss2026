# RQ1 experiments

One JSON-configured runner for baseline moderation and two-dimension LoRA/QLoRA training. Only RQ1 is implemented. RQ2 generation and the final test set are excluded.

## Data

Paths in configs are relative to the repository root, independent of the shell's working directory.

- Training: `active_learning_project/data/trainset/trainset.xlsx`. The current workbook contains 1,653 rows; 1,652 have both labels present. Each config selects its requested training size from these rows. Uses `Sanitized Question`, `Answer Transcription`, `Accuracy`, and `Context & Relevance`.
- Baseline and post-training validation: all 100 rows of `active_learning_project/data/validationset/anjali_dataset_labeled.csv`. Uses `question`, `model_answer`, **Claude** `claude_accuracy` and `claude_context_relevance`. The `ground_truth` answer is not shown to the model.
- Config mappings are explicit. Change them if you choose original caller transcripts instead of sanitized questions. Do not combine incompatible representations silently.
- Exact question–answer overlap raises an error. Shared questions with different answers are counted in the manifest. Existing data are not question-disjoint. Statistical comparisons need an independence check or grouped analysis.
- Training label provenance is inherited from the workbook; the runner does not reannotate or certify the teacher source.

## Setup and execution

Run on a CUDA GPU with Hugging Face access to the selected model. Authentication uses the existing Hugging Face cache or `HF_TOKEN`; do not put credentials in configs.

```bash
pip install -r RQ1/experiments/requirements.txt

# Local data validation; no GPU or model download needed
python RQ1/experiments/run_experiment.py --config RQ1/experiments/configs/gemma_train_500.json --check-data

# Six training examples; evaluates the complete fixed 100-row validation set
python RQ1/experiments/run_experiment.py --config RQ1/experiments/configs/smoke.json

# Baseline only
python RQ1/experiments/run_experiment.py --config RQ1/experiments/configs/gemma_baseline.json

# Baseline, training on 500 examples, then validation
python RQ1/experiments/run_experiment.py --config RQ1/experiments/configs/gemma_train_500.json

# Repeat with an isolated output directory
python RQ1/experiments/run_experiment.py --config RQ1/experiments/configs/gemma_train_500.json --output-dir /tmp/gemma_500_repeat

python RQ1/experiments/compare_results.py
```

Use a Python executable from your installed environment. The repository's current `env` does not necessarily contain the GPU dependencies. Package ranges are starting requirements, not a verified CUDA lockfile. Run the smoke config on the target GPU before submitting the research matrix.

Each config runs one experiment. Copy a config, change its `name` and the relevant settings, then pass it with `--config`. Nonempty result folders are refused. Failed jobs can be rerun in a fresh directory; automatic checkpoint resume is not implemented.

## Config groups

| Files | Purpose |
|---|---|
| `*_baseline.json` | Baseline on the same 100 Claude-labelled examples, no training |
| `gemma_train_100.json` … `gemma_train_1100.json` | Cumulative nested subsets; each run starts a fresh adapter from the base model |
| `truncation_256/512/1024.json` | Train and validate at different sequence lengths, same 500 training examples |
| `train_batch_1/2/4/6.json` | Training batch sizes with accumulation 24/12/6/4; effective batch size stays 24 |
| `balance_none/class_weighted.json` | Unweighted versus class-weighted example loss; no oversampling |
| `balance_sampling_accuracy.json` | Balanced Accuracy presentations with unweighted loss, same 500-row pool and 500 presentations per epoch |
| `balance_sampling_full.json` | Full eligible pool; minority oversampling to the majority count, unweighted loss; excludes conflicting duplicate IDs 1384/1385 |
| `qwen_train_500.json`, `llama_train_500.json`, `gemma_train_500.json` | Example under-8B model comparison after training on the same 500 rows |
| `smoke.json` | Runtime check; not a thesis research result |

The alternative models are explicit starting candidates, not a claim that they lead your existing benchmark. Change identifiers after reviewing the baseline shortlist. Gemma uses the multimodal Gemma 3 loader for text-only inputs; text models use `AutoModelForCausalLM`. Models with different architectures may need different LoRA target modules. Revision defaults to the provider's current default; pin an immutable revision for research runs. Resolved model revision is saved in `model_metadata.json` where exposed by Transformers.

Default input policy preserves the rubric and full question, removing answer tokens from the end until the input plus JSON target fits the maximum sequence length. If the rubric/question/target alone exceeds the budget, the run fails explicitly. Inference uses the same input budget and reserves the target length, so training and validation see the same preprocessing. `max_answer_tokens` optionally caps the answer before fitting; leave it null to isolate sequence length. `full` bypasses truncation and sequence-length limiting; use only with a model/hardware context that can fit it. `tail` is an experimental control that may remove the rubric/question. It requires a positive sequence budget and is not recommended as the primary condition.

The 256-token condition may fail when the preserved rubric and question do not fit. That is a documented feasibility result; do not silently remove the question. Try a frozen shorter rubric or a larger length, and apply that choice to all comparable conditions.

All training runs evaluate baseline and adapted predictions separately for **both** dimensions. Targets contain both labels in one JSON object. Class weighting uses the mean of each dimension's inverse-frequency class weights as an example weight, applied to the mean target-token loss. It is a joint-label approximation, not independent label-token weighting; it does not normalize by the minibatch sum of weights. Prompt tokens are excluded from the loss.

Balanced sampling is an alternative to weighting. It creates a fixed presentation list of the same length as the selected pool, with counts differing by at most one across the chosen dimension's classes. Each class's rows are shuffled without replacement before cycling; minority rows repeat and some majority rows are omitted. The same list is shuffled by Trainer each epoch. All loss weights are 1. `training_ids.csv` records the eligible pool; `sampling_presentations.csv` records the actual presentation list, including repeated IDs and both labels. `sampling.json` records distinct presented rows, counts per dimension per epoch and the sampling seed. Only `training.sampling_dimension` is deliberately balanced; the other dimension's counts are reported. This keeps the presentation/update budget fixed, rather than oversampling minorities to the majority count and increasing training duration.

The default training recipe uses two epochs and selects the **final-epoch adapter**, not a validation-loss-selected checkpoint. This explicit rule differs from the draft PDF; update the thesis protocol accordingly or implement a separate validation-loss selection experiment. Model selection across configs can use the mean of the two validation macro-F1 scores. No final-test-based selection is implemented.

Three inference repetitions use greedy decoding by default. They measure reproducibility under those settings and may be identical. Set `do_sample` and `temperature` explicitly if studying stochastic inference. Repeated inference is not repeated model training; use separate seed configs for training robustness.

## Saved outputs and comparisons

Each run saves the exact config, input hashes, selected source rows, environment, token lengths, status, model metadata, raw outputs and parsed predictions. Metrics include both macro-F1 scores, per-class precision/recall/F1, confusion matrices, exact agreement, JSON failures, inter-run disagreement, inference time and GPU allocation. Invalid output is -1 and counts as a failure; macro-F1 averages only classes 0/1/2. The parser accepts code fences, surrounding prose around one unambiguous object, single-quoted dictionaries, and common context-label key variants. Invalid labels, duplicate fields, extra fields and multiple objects are rejected. Strict-format failure rate is reported separately from parsing failure rate. Training runs additionally save adapter/tokenizer, final checkpoint, loss logs, effective batch size, training duration and peak GPU allocation.

GPU allocation measures PyTorch allocated memory, not total process VRAM. Batch duration per example is an amortized measurement, not individual request latency. No health-category breakdown is generated because the validation file lacks a verified category column.

`compare_results.py` reads only complete runs, refuses mixed validation hashes, and writes `all_runs.csv`, `summary.csv`, a comparison figure, and learning-curve figures when matching curve runs exist. Standard deviations describe inference repetitions, not training uncertainty. Compare matched conditions; different input policies or training settings are not interchangeable.

Optional development statistics for a single paired baseline/adapted run:

```bash
python RQ1/experiments/compare_results.py --output-dir /tmp/rq1_comparison_stats --paired-run RQ1/experiments/results/gemma_train_500 --assume-independent --samples 10000
```

This computes paired bootstrap confidence intervals, approximate randomization, exact McNemar and Holm adjustment across the four p-values. It uses the preselected inference repeat (default 0) and the validation set; it is **development evidence, not final-test significance**. Use `--assume-independent` only after verifying validation rows are independent. Grouped resampling is not implemented; if questions/users repeat, do not use these row-wise tests.

## Local checks

```bash
python -m unittest discover -s RQ1/experiments/tests
```

Local checks validate data/config wiring, invalid-output handling, metric conventions and comparison generation. They do not establish that model loading, CUDA training or every batch size fits the target GPU.

Reparse saved generations without retraining (original outputs and metrics are preserved):

```bash
python RQ1/experiments/rescore_outputs.py --run-dir RQ1/experiments/results/gemma_train_full_20261005_v3
```

Rescored outputs are saved in `rescored_v1/`; the comparison script continues to read the original `results.json`, so use the separate rescored metrics when reporting parser-corrected scores.

## Validation label change — 8 October 2026

Future configs score the Claude columns on the same 100 validation examples. These scores measure teacher agreement. Claude supports are Accuracy 1/3/96 and Context & Relevance 0/4/96 for classes 0/1/2; macro-F1 still averages all three predefined classes, including absent class 0. Completed human-scored artifacts remain unchanged. Config run names now end in `_claude_validation` to keep new outputs separate. Comparison refuses mixed label-column mappings even when the validation CSV hash matches. This change does not launch training or rescore old runs.
