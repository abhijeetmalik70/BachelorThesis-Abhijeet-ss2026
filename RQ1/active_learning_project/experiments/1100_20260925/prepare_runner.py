"""Package the existing cluster experiment with frozen data and human validation.

Does not change the original runner or launch a job. Run from any directory.
"""
from pathlib import Path
import hashlib
import json
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
source = PROJECT / 'scripts/run_merged_experiment_cluster.py'
code = source.read_text()

def replace(old, new):
    global code
    assert code.count(old) == 1, f'Expected one match: {old[:100]}'
    code = code.replace(old, new)

# Keep the prior cluster settings: accuracy, r8/alpha16, lr1e-5, one epoch,
# batch1/accumulation4, seed42 and class oversampling. Use source text in full.
replace('options = parser.parse_args()', '''options = parser.parse_args()
if options.output_dir.exists() and any(options.output_dir.iterdir()):
    raise RuntimeError("Output directory is not empty; choose a new run directory.")''')
replace('MODEL_ID = "google/gemma-3-4b-it"', '''MODEL_ID = "google/gemma-3-4b-it"
if not torch.cuda.is_available():
    raise RuntimeError("A CUDA GPU is required. No training has started.")
COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16''')
code = code.replace('bnb_4bit_compute_dtype=torch.bfloat16', 'bnb_4bit_compute_dtype=COMPUTE_DTYPE')
code = code.replace('torch_dtype=torch.bfloat16', 'torch_dtype=COMPUTE_DTYPE')
replace('''        valid_indices.append(idx)''', '''        valid_indices.append(idx)
        if len(predictions) % 20 == 0 or len(predictions) == len(dataframe):
            print(f"Validation generation: {len(predictions)}/{len(dataframe)}", flush=True)''')
replace('    bf16=True,', '    bf16=(COMPUTE_DTYPE == torch.bfloat16),\n    fp16=(COMPUTE_DTYPE == torch.float16),')
replace('''        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        shift_labels = labels[:, 1:]
        valid_tokens = shift_labels.ne(-100)
        logits = outputs.logits[:, :-1, :][valid_tokens].float()''', '''        # Only project positions needed to predict supervised JSON/EOS tokens.
        shift_labels = labels[:, 1:]
        positions = shift_labels.ne(-100).any(dim=0).nonzero(as_tuple=True)[0]
        outputs = model(
            **{k: v for k, v in inputs.items() if k != "labels"},
            logits_to_keep=positions,
        )
        shift_labels = shift_labels[:, positions]
        valid_tokens = shift_labels.ne(-100)
        logits = outputs.logits[valid_tokens].float()''')
replace('''"accuracy": accuracy_score(y_true_valid, y_pred_valid),''', '''"accuracy": accuracy_score(y_true, [-1 if p is None else p for p in predictions]),''')
replace('''            y_true_valid,
            y_pred_valid,
            average="macro",''', '''            y_true,
            [-1 if p is None else p for p in predictions],
            average="macro",''')
replace('''# -----------------------------
# Callback: evaluate every epoch''', '''# Baseline uses the same human-labelled validation pairs and prompt.
# Fresh LoRA is initialized to leave the base model output unchanged.
model.config.use_cache = True
baseline_metrics, baseline_predictions, baseline_raw = evaluate_generation(model, val_df)
model.config.use_cache = False
baseline_df = val_df.copy()
baseline_df["gemma_accuracy"] = baseline_predictions
baseline_df["gemma_raw_output"] = baseline_raw
baseline_df.to_excel(RUN_DIR / "metrics/baseline_predictions.xlsx", index=False)
with open(RUN_DIR / "metrics/baseline_metrics.json", "w") as handle:
    json.dump(baseline_metrics, handle, indent=2)
print("BASELINE HUMAN VALIDATION:", baseline_metrics, flush=True)

# -----------------------------
# Callback: evaluate every epoch''')
replace("models/gemma3_4b_lora_round1_adapter_200", "models/gemma3_4b_lora_1100_adapter")
marker = '# Notebook cell 17\n'
assert code.count(marker) == 1
code = code.split(marker)[0] + '''# Final generation metrics use the same human validation labels as baseline.
model.config.use_cache = True
final_metrics, predictions, raw_outputs = evaluate_generation(model, val_df)
final_df = val_df.copy()
final_df["gemma_accuracy"] = predictions
final_df["gemma_raw_output"] = raw_outputs
final_df.to_excel(RUN_DIR / "metrics/final_predictions.xlsx", index=False)
y_true = val_df[CLAUDE_ACC_COL].astype(int).tolist()
y_pred = [-1 if p is None else int(p) for p in predictions]
final_df[[a != b for a, b in zip(y_true, y_pred)]].to_excel(
    RUN_DIR / "metrics/final_errors.xlsx", index=False)
final_metrics["total_examples"] = len(val_df)
final_metrics["parsed_examples"] = sum(p != -1 for p in y_pred)
final_metrics["prediction_distribution"] = {
    str(k): int(v) for k, v in pd.Series(y_pred).value_counts().items()}
final_metrics["classification_report"] = classification_report(
    y_true, y_pred, labels=[0, 1, 2], zero_division=0, output_dict=True)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, -1])[:3, :]
pd.DataFrame(cm, index=[0, 1, 2], columns=[0, 1, 2, "invalid"]).to_csv(
    RUN_DIR / "metrics/final_confusion_matrix.csv", index_label="human_label")
with open(RUN_DIR / "metrics/final_metrics.json", "w") as handle:
    json.dump(final_metrics, handle, indent=2)
with open(RUN_DIR / "metrics/baseline_vs_trained.json", "w") as handle:
    json.dump({"baseline": baseline_metrics, "trained": final_metrics}, handle, indent=2)
print("FINAL HUMAN VALIDATION:", final_metrics, flush=True)
(RUN_DIR / "COMPLETED").write_text("Training and human validation completed.\\n")
'''
# Persist exact package versions before expensive model loading.
replace('''MODEL_ID = "google/gemma-3-4b-it"''', '''import importlib.metadata
with open(RUN_DIR / "configs/environment.json", "w") as handle:
    json.dump({name: importlib.metadata.version(name) for name in
               ["torch", "transformers", "peft", "accelerate", "pandas", "scikit-learn"]}, handle, indent=2)
MODEL_ID = "google/gemma-3-4b-it"''')
compile(code, 'run_experiment.py', 'exec')
script = HERE/'input/scripts/run_experiment.py'
script.parent.mkdir(parents=True, exist_ok=True)
script.write_text(code)
train = pd.read_csv(HERE/'train.csv')
val = pd.read_csv(HERE/'validation_human.csv')
mapping = {'question': 'Sanitized Question', 'answer': 'Answer Transcription', 'accuracy': 'Accuracy (0-2)'}
train_path = HERE/'input/data/trainset/Combined_QnA_with_Labels_annotated_1100.xlsx'
val_path = HERE/'input/data/varification set/anjali_dataset_with_claude_labels.xlsx'
for p in [train_path, val_path]:
    p.parent.mkdir(parents=True, exist_ok=True)
train.rename(columns=mapping).to_excel(train_path, index=False)
# Legacy filename is required by the old runner, but this copy contains HUMAN
# labels mapped to its expected name, not Claude validation labels.
val.rename(columns=mapping).to_excel(val_path, index=False)
manifest = json.loads((HERE/'data_manifest.json').read_text())
manifest['task'] = 'accuracy only'
manifest['training_settings'] = dict(epochs=1, learning_rate=1e-5, batch_size=1,
    gradient_accumulation_steps=4, lora_r=8, lora_alpha=16, lora_dropout=0.05,
    seed=42, oversampling=True, effective_training_entries=3105,
    four_bit='optional CLI flag; use for Colab', truncation=False)
manifest['validation_mapping'] = 'Legacy XLSX filename; Accuracy (0-2) is copied from HUMAN accuracy column'
manifest['source_runner_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
manifest['input_sha256'] = {str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [script, train_path, val_path]}
(HERE/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Prepared accuracy-only runner, 1100 training rows, 100 human validation rows. Not launched.')
