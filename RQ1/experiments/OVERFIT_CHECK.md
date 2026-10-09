# Experiment 1: balanced overfit check

This checks whether the training pipeline can memorize 30 examples. It does not measure generalization. Run only this experiment before deciding the next step.

Selection uses the current `trainset.xlsx`, seed 42 and 10 unique question–answer pairs per **Accuracy** class. Exact pairs appearing in human validation are excluded. The same selected rows are evaluated before and after training. Both dimensions are trained and reported; only Accuracy is deliberately balanced. The checked Context & Relevance counts are 11/7/12.

The diagnostic uses a fresh Gemma 3 4B adapter, unweighted target-only JSON loss, no oversampling, full inputs, learning rate 2e-4, 30 epochs and effective batch size 3. Full inputs avoid losing information through truncation; long examples may require substantial GPU memory. Thirty epochs are a deliberate memorization attempt, not a recommended general training duration.

From the repository root on a CUDA machine with model access:

```bash
python -m pip install -r RQ1/experiments/requirements.txt
python RQ1/experiments/run_experiment.py --config RQ1/experiments/configs/diagnostic_balanced_overfit_30.json --check-data
python RQ1/experiments/run_experiment.py --config RQ1/experiments/configs/diagnostic_balanced_overfit_30.json
```

Results go to `RQ1/experiments/diagnostic_results/diagnostic_balanced_overfit_30/`, separate from validation research runs. `training_ids.csv` and `validation_ids.csv` identify the same 30 rows. The latter filename is inherited from the shared runner and refers to training-set evaluation here. `data_manifest.json` explicitly records this scope.

Inspect `adapted_metrics.json`: Accuracy recall and F1 for each class, `prediction_counts`, and exact agreement. Inspect `adapted_predictions.jsonl` for raw generations and `training_log.json` for loss. Near-perfect training-set predictions across all classes support that the pipeline can learn these examples. Continued class-2-only predictions mean we should investigate optimization, labels, targets and adapter loading before proceeding. Intermediate results can mean the memorization budget was insufficient; this diagnostic alone does not prove a bug.

Local data selection and automated checks can run without CUDA. Model training requires CUDA, PyTorch and the other GPU dependencies; it has not been executed in the local macOS environment.
