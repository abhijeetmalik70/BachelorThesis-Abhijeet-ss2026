Cluster run of `active_learning_2_merged_experiment-2.ipynb`

Server: `idisc-09.cs.uni-saarland.de` (hostname `jianlong-cluster`).
Python: `/home/abma00001/venvs/thesis/bin/python`.
Run root: `/home/abma00001/BachelorThesis-Abhijeet-ss2026/cluster_runs/merged_20260919`.

The standalone script uses the local notebook's effective settings: Gemma 3 4B,
bfloat16, LoRA rank 8 / alpha 16, dropout 0.05, learning rate 1e-5,
batch size 1, accumulation 4, one epoch, seed 42. The 290 training rows
are oversampled to 555 rows (185 per class), as in notebook cell 11.
The notebook's introductory rank 16 / alpha 32 description is stale.

Cluster adaptation removes Colab setup, embedded authentication, exploratory
cells, and the pre-training prediction pass over the training set. It retains
training, epoch validation, final validation, adapter/checkpoint saves, plots,
and metrics. Outputs are isolated from existing project results. Invalid
predictions return None. Loss computation selects supervised positions to
reduce memory, and explicitly enables Trainer's gradient-accumulation scaling.
The supplied class-weight formula is retained; at batch size 1 its normalized
weight cancels, so oversampling provides the effective class balancing.

Epoch validation uses the XLSX `Accuracy (0-2)` labels; final evaluation uses
the CSV `accuracy` labels, matching the notebook. These are different label
sources and should not be treated as interchangeable in thesis reporting.

Run from the run root:

```bash
~/venvs/thesis/bin/python -u input/scripts/run_merged_experiment_cluster.py \
  --output-dir full
```

Add `--smoke` with a separate output directory for a six-example training test.
`--four-bit` is optional; the initial run uses the notebook's bfloat16 setting.
Authentication comes from the existing Hugging Face cache or HF_TOKEN.
`manifest.json` records package versions and SHA-256 hashes of copied inputs.

Monitor:

```bash
tail -f ~/BachelorThesis-Abhijeet-ss2026/cluster_runs/merged_20260919/full.log
```

Corrected dataset run: `merged_1100_20260919/full/` on the same server.
Uses `Combined_QnA_with_Labels_annotated_1100.xlsx`: 1,658 total rows,
1,100 labeled rows (0: 43, 1: 22, 2: 1,035), 558 unlabeled rows excluded.
The `Accuracy` column is mapped to `Accuracy (0-2)`. Existing oversampling
produces 3,105 training entries. Notebook paths and script default updated.
The previous 290-row run remains in `merged_20260919`.
