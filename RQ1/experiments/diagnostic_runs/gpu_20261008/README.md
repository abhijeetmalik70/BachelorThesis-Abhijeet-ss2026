# Balanced overfit diagnostic — 8 October 2026

Completed on idisc-09 (reported hostname jianlong-cluster), NVIDIA RTX A4000 16 GB. Training took 459 seconds for 30 epochs / 300 optimizer steps. A fresh Gemma 3 4B QLoRA adapter was trained on 30 unique examples, 10 per Accuracy class, and evaluated on the same examples.

| Dimension | Baseline macro-F1 | Adapted macro-F1 | Adapted correct |
|---|---:|---:|---:|
| Accuracy | 0.2843 | 1.0000 | 30/30 |
| Context & Relevance | 0.2874 | 1.0000 | 30/30 |

All three classes have recall 1.0 in both dimensions after training. No invalid JSON outputs occurred. Context & Relevance supports are 11/7/12; only Accuracy was deliberately balanced.

This verifies that this fresh-adapter training setup can memorize all three classes. It does not establish validation generalization or isolate the cause of the previous class-2 collapse. This run changes data composition, exposure and potentially software relative to previous runs; it does not prove earlier adapters or scripts were correct.

Metrics, raw predictions, training logs, token lengths, model metadata and exact runtime config are saved here. The full adapter and checkpoint remain on the GPU host at `/home/abma00001/rq1_overfit_20261008/RQ1/experiments/diagnostic_results/diagnostic_balanced_overfit_30/adapter`.

`selected_30.csv` preserves the original workbook source rows. The remote runner reselects all 30 exported rows deterministically, so prediction `source_row` values refer to exported CSV rows; join by `record_id` to recover original workbook rows. `selection_manifest.json` records original workbook provenance; `data_manifest.json` records the uploaded CSV provenance. Human validation was used solely for exact-pair exclusion, and the test set was not used.

No subsequent experiment was launched.
