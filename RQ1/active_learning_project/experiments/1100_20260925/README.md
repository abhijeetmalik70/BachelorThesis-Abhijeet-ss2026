# 1,100-example Gemma 3 4B experiment

**Status: prepared locally; training has not started.** The configured university-server connection failed authentication, and the retry was declined. This machine has no available CUDA GPU. No remote job was submitted.

The training labels are accepted unchanged for this experiment. The default task is **Accuracy only**, matching the existing cluster runner. A question about training both dimensions remains open; this package does not train Clarity/Context.

## Frozen inputs and protocol

- 1,100 labelled rows, with accuracy counts 43 / 22 / 1,035 for classes 0 / 1 / 2.
- 100 human-labelled validation rows recovered from the exact Git revision recorded in `data_manifest.json`. These are isolated copies, not restored/deleted working-tree files.
- Zero exact question–answer overlap between these training and validation copies. There are 42 shared questions with different answers; this is not a question-disjoint validation set.
- `final_test.xlsx` is not used in this run.
- Gemma 3 4B instruction model, LoRA rank 8, alpha 16, dropout 0.05, learning rate 1e-5, one epoch, seed 42, batch 1, accumulation 4. These preserve the previous cluster experiment settings, not the separate batch-6 refined notebook settings.
- Existing oversampling repeats minority-class rows to 1,035 entries per class: **3,105 training presentations from 1,100 original rows**. No new annotations are created. The inherited normalized class weighting cancels at batch size 1; oversampling is the effective balancing method.
- Complete question and answer text; no silent truncation to the first 120 answer tokens. Full sequences can require more GPU memory.
- Baseline and trained generation are compared against the same human labels. Invalid output counts as an error in accuracy and macro-F1; parse rate is reported separately.
- Adapter, checkpoint, training logs, confusion matrix (including invalid predictions), raw outputs, errors and baseline/final metrics are saved. Output directories must be empty to avoid accidental result reuse.

The legacy input filename `anjali_dataset_with_claude_labels.xlsx` is retained to match the runner's expected path, but **its Accuracy (0-2) column contains the original human accuracy labels**, not Claude's labels. This mapping is recorded in the manifest.

## University GPU server

Transfer the self-contained `experiment_1100_bundle.zip` to the server, extract it, and enter its directory. With the established thesis environment and Hugging Face access:

```bash
~/venvs/thesis/bin/python -u input/scripts/run_experiment.py --output-dir smoke --smoke
~/venvs/thesis/bin/python -u input/scripts/run_experiment.py --output-dir full
```

The smoke run trains six rows (two per accuracy class) and validates on three. It is a runtime check, not a research result. The full run uses all 1,100 source rows. Add `--four-bit` for QLoRA on smaller GPUs. Use a fresh output directory for each attempt. No GPU smoke test has yet been executed.

For a persistent server session after smoke succeeds:

```bash
nohup ~/venvs/thesis/bin/python -u input/scripts/run_experiment.py --output-dir full > full.log 2>&1 &
tail -f full.log
```

Use the existing server environment if compatible. `requirements-gpu.txt` provides compatible version ranges for a new environment; actual installed versions are recorded by each run. Hugging Face authentication uses its normal cache or `HF_TOKEN`; no credentials are bundled.

## Google Colab

Open `Run_1100_Colab.ipynb` in a GPU runtime. It uploads the experiment ZIP, authenticates interactively to Hugging Face, runs the smoke check, then starts the full QLoRA run. Keep the runtime connected and download results before it expires. This notebook was prepared locally; it has not run on Colab.

## Local validation and implementation notes

The package generation and data checks passed: all 1,100 labels are valid, the 100-row human validation mapping is exact, and exact train/validation pair overlap is zero. The generated Python source compiles. The supervised-position loss was compared with a full-logit reference on a small synthetic batch, including gradients; this checks alignment without downloading model weights. It does not establish CUDA compatibility or successful model training.

The small loss adaptation computes vocabulary logits only at positions supervising JSON/EOS tokens, using the documented [Gemma 3 logits_to_keep interface](https://huggingface.co/docs/transformers/v4.50.0/model_doc/gemma3). LoRA construction follows the [PEFT API](https://huggingface.co/docs/peft/quicktour). Other training settings come from the existing project runner.

The source workbook contains 1,658 rows in total, so it currently has **558 unlabelled rows**, not another 1,600. A separate forthcoming corpus can be added in a later versioned run.
