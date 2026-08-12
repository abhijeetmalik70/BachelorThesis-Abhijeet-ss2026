# Active Learning Project (Human-in-the-Loop)

Fine-tune a Hugging Face model on ~1600 rows using an iterative loop:
train → predict on unlabeled pool → surface the model's failures/low-confidence
points → you manually label them → merge back into training data → repeat.

## 0. Setup

```bash
pip install -r requirements.txt
```

Put your full dataset (with or without labels) at:
```
data/raw/full_dataset_1600.csv
```
It needs at minimum a text column (default name: `text`). If you already have
some ground-truth labels for a seed set, put those in `label` too.

Edit `configs/round_config.yaml`:
- `model.base_model_name` → the HF model id you're downloading (e.g. a Gemma/Qwen/Llama variant)
- `data.text_column` / `data.label_column` → match your actual column names
- `model.num_labels` → number of classes

## 1. Bootstrap round 0 (seed set)

Split your raw data into an initial labeled seed set and the rest as the
unlabeled pool. If you already have some labels, this is as simple as:

```python
import pandas as pd
df = pd.read_csv("data/raw/full_dataset_1600.csv")
seed = df[df["label"].notna()]        # whatever subset you've already labeled
pool = df[df["label"].isna()]
seed.to_csv("data/master_labeled_dataset.csv", index=False)
pool.to_csv("data/unlabeled_pool.csv", index=False)
```

If you have zero labels to start, hand-label a small random seed batch
(50–100 rows) first — every active learning loop needs an initial model to
generate the first round of predictions.

## 2. The loop (repeat each round)

```bash
# 1. Fine-tune on everything labeled so far
python scripts/train.py

# 2. Run the fresh checkpoint on the remaining unlabeled pool
python scripts/infer.py

# 3. Pick the rows the model is least confident about
python scripts/select_failures.py
#    -> writes errors/round_XX_failures.csv

# 4. YOU manually open errors/round_XX_failures.csv, fill in `manual_label`
#    for each row, then save it as:
#    data/labeled/round_XX.csv

# 5. Merge your new labels into the training set
python scripts/merge_labels.py

# 6. Score this round's checkpoint on a held-out validation split
python scripts/evaluate.py

# 7. Bump `round:` in configs/round_config.yaml and go back to step 1
```

`metrics/training_curve.csv` accumulates Accuracy / Precision / Recall / F1 /
mIoU (macro-averaged) after every round, so you can plot how performance
improves as you add manually-corrected labels — useful directly for your
thesis results section.

## Notes

- **Selection strategy**: `select_failures.py` defaults to least-confidence
  sampling (lowest max softmax probability). You can swap in margin or
  entropy sampling later by extending `active_learning.strategy` in the
  config — the hook is already there.
- **mIoU on classification labels**: computed here as macro Jaccard over
  predicted vs. true label sets, consistent with how you're already scoring
  Ollama Cloud SLMs in your thesis eval script.
- **Reproducibility**: every round's checkpoint, predictions, failures, and
  metrics are kept in separate round-numbered files — nothing gets
  overwritten, so you can trace exactly which manual labels drove which
  performance jump.
- If your task is closer to *generative* labeling (free-text output rather
  than a fixed label set, like your `RQ2_output1.csv` scoring tool), the
  same round structure still applies — just swap `train.py`/`infer.py` for
  a generation + scoring version (`AutoModelForCausalLM` + your existing
  JSON/regex label-parsing logic from `ollama_cloud_slm_moderation_eval.py`)
  and keep everything else (selection, merge, evaluate) the same. Happy to
  build that variant if that's the setup you actually need.
