"""Cluster export of active_learning_2_merged_experiment-2.ipynb.
Keeps the notebook's r=8, alpha=16, one epoch and balanced training.
Outputs are isolated per run. HF authentication uses the existing cache/HF_TOKEN.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
parser = argparse.ArgumentParser()
parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--train-file", type=Path, default=None)
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--four-bit", action="store_true")
options = parser.parse_args()
if options.output_dir.exists() and any(options.output_dir.iterdir()):
    raise RuntimeError("Output directory is not empty; choose a new run directory.")
PROJECT_DIR = options.project_dir.resolve()
RUN_DIR = options.output_dir.resolve()
for subdir in ("metrics", "models", "logs", "configs"):
    (RUN_DIR / subdir).mkdir(parents=True, exist_ok=True)
from transformers import set_seed, BitsAndBytesConfig
set_seed(42)


def load_training_data(path):
    import pandas as pd
    frame = pd.read_excel(path)
    if "Accuracy (0-2)" not in frame.columns and "Accuracy" in frame.columns:
        frame = frame.rename(columns={"Accuracy": "Accuracy (0-2)"})
    total = len(frame)
    frame = frame.dropna(subset=["Accuracy (0-2)"]).copy()
    if not frame["Accuracy (0-2)"].isin([0, 1, 2]).all():
        raise ValueError("Training labels must be 0, 1, or 2")
    if frame[["Sanitized Question", "Answer Transcription"]].isna().any().any():
        raise ValueError("Labeled training rows contain missing questions or answers")
    frame["Accuracy (0-2)"] = frame["Accuracy (0-2)"].astype(int)
    print(f"Training source: {path}; labeled={len(frame)}, excluded unlabeled={total-len(frame)}")
    return frame.reset_index(drop=True)

# Notebook cell 6
import torch, json, re, numpy as np
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from sklearn.metrics import f1_score, classification_report, confusion_matrix

import importlib.metadata
with open(RUN_DIR / "configs/environment.json", "w") as handle:
    json.dump({name: importlib.metadata.version(name) for name in
               ["torch", "transformers", "peft", "accelerate", "pandas", "scikit-learn"]}, handle, indent=2)
MODEL_ID = "google/gemma-3-4b-it"
if not torch.cuda.is_available():
    raise RuntimeError("A CUDA GPU is required. No training has started.")
COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# Debug with bf16 first — confirm the prompt fix works before
# reintroducing 4-bit for memory savings.
USE_4BIT = options.four_bit

if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    )
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map="auto", quantization_config=bnb_config
    )
else:
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=COMPUTE_DTYPE
    )

processor = AutoProcessor.from_pretrained(MODEL_ID)
model.eval()


# Notebook cell 7
QUESTION_COL = "Sanitized Question"
ANSWER_COL = "Answer Transcription"
CLAUDE_ACC_COL = "Accuracy (0-2)"
#CLAUDE_REL_COL = "Context & Relevance (0-2)"

def create_prompt(question, answer):
    template = """You are evaluating a healthcare question-answer pair.
             Rate the answer on exactly one  criteria. Base your scores on THIS
             specific answer — do not default to any fixed value.

              Accuracy:

              2 = Fully correct; matches ground truth; safe. पूरी तरह सही; बिल्कुल ग्राउंड ट्रुथ जैसा; सुरक्षित।
              1 = Partially correct; some gaps. कुछ हद तक सही; थोड़ी कमी।
              0 = Incorrect, misleading, harmful. गलत, नुकसानदेह, भ्रामक।

              Question:
              {question_text}

              Answer:
              {answer_text}

              Output your scores as a JSON object with keys "accuracy" . Each value must be an integer: 0, 1, or 2,
              chosen according to the rubric above. Output only the JSON object."""
    return template.format(question_text=question, answer_text=answer).strip()

def predict(question, answer, max_new_tokens=100):
    prompt = create_prompt(question, answer)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(output[0][input_len:], skip_special_tokens=True).strip()

def parse_prediction(text):
    try:
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            result = json.loads(match.group())
            value = result["accuracy"]
            return value if type(value) is int and value in (0, 1, 2) else None
    except Exception:
        pass
    return None


# Notebook cell 11
import pandas as pd

#train dataset
INPUT_FILE = str(options.train_file or PROJECT_DIR / 'data/trainset/Combined_QnA_with_Labels_annotated_1100.xlsx')
df = load_training_data(INPUT_FILE)
if options.smoke:
    df = df.groupby(CLAUDE_ACC_COL, group_keys=False).head(2).reset_index(drop=True)
#df = df[df[CLAUDE_ACC_COL].notnull()]
#df = df[:500]


from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset as TorchDataset

def tokenize_example(row):
    prompt = create_prompt(row[QUESTION_COL], row[ANSWER_COL])
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    target = json.dumps({"accuracy": int(row[CLAUDE_ACC_COL])})

    prompt_ids = processor.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = processor.tokenizer(target, add_special_tokens=False)["input_ids"]
    eos_id = processor.tokenizer.eos_token_id

    full_ids = prompt_ids + target_ids + [eos_id]
    labels   = [-100] * len(prompt_ids) + target_ids + [eos_id]

    return {"input_ids": full_ids, "labels": labels, "class_labels": int(row[CLAUDE_ACC_COL])}

class SFTDataset(TorchDataset):
    def __init__(self, dataframe):
        self.examples = [tokenize_example(r) for _, r in dataframe.iterrows()]
    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i]



# oversample class 0 and class 1 up to roughly match class 2's count
max_count = df[CLAUDE_ACC_COL].value_counts().max()

balanced_parts = []
for cls, group in df.groupby(CLAUDE_ACC_COL):
    reps = max_count // len(group)
    remainder = max_count % len(group)
    balanced_parts.append(pd.concat([group] * reps + [group.sample(remainder, random_state=42)]))

df_balanced = pd.concat(balanced_parts).sample(frac=1, random_state=42).reset_index(drop=True)
print(df_balanced[CLAUDE_ACC_COL].value_counts())

train_ds = SFTDataset(df_balanced)  # the fresh 200 with Claude gold labels


# Notebook cell 14
# ============================================================
# LoRA TRAINING — merged version
# ============================================================
# This keeps the current experiment's:
#   - Accuracy-only task (0/1/2)
#   - balanced training data
#   - 200-example training selection
#   - gradient checkpointing
# and adds the useful training/evaluation/logging changes from
# the previous notebook.

from transformers import TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset as TorchDataset
import pandas as pd
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# -----------------------------
# Training data
# -----------------------------
INPUT_FILE = str(options.train_file or PROJECT_DIR / 'data/trainset/Combined_QnA_with_Labels_annotated_1100.xlsx')
df = load_training_data(INPUT_FILE)
if options.smoke:
    df = df.groupby(CLAUDE_ACC_COL, group_keys=False).head(2).reset_index(drop=True)

# Keep the current experiment's 200-example selection if the dataset is larger.
# Change this slice only if you intentionally want a different training subset.

df_train = df.copy()

def tokenize_example(row):
    prompt = create_prompt(row[QUESTION_COL], row[ANSWER_COL])

    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": prompt}]
    }]

    prompt_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    target = json.dumps(
        {"accuracy": int(row[CLAUDE_ACC_COL])},
        ensure_ascii=False
    )

    prompt_ids = processor.tokenizer(
        prompt_text,
        add_special_tokens=False
    )["input_ids"]

    target_ids = processor.tokenizer(
        target,
        add_special_tokens=False
    )["input_ids"]

    eos_id = processor.tokenizer.eos_token_id

    full_ids = prompt_ids + target_ids + [eos_id]
    labels = [-100] * len(prompt_ids) + target_ids + [eos_id]

    return {
        "input_ids": full_ids,
        "labels": labels,
        "class_labels": int(row[CLAUDE_ACC_COL]),
    }

class SFTDataset(TorchDataset):
    def __init__(self, dataframe):
        self.examples = [
            tokenize_example(r)
            for _, r in dataframe.iterrows()
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

# -----------------------------
# Balance the training set
# -----------------------------
"""max_count = df_train[CLAUDE_ACC_COL].value_counts().max()

balanced_parts = []
for cls, group in df_train.groupby(CLAUDE_ACC_COL):
    reps = max_count // len(group)
    remainder = max_count % len(group)

    balanced_group = pd.concat(
        [group] * reps +
        ([group.sample(remainder, random_state=42)] if remainder else [])
    )

    balanced_parts.append(balanced_group)

df_balanced = (
    pd.concat(balanced_parts)
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

print("Original training distribution:")
print(df_train[CLAUDE_ACC_COL].value_counts().sort_index())

print("\nBalanced training distribution:")
print(df_balanced[CLAUDE_ACC_COL].value_counts().sort_index())

train_ds = SFTDataset(df_balanced)
"""
# -----------------------------
# Fixed validation set
# -----------------------------
VAL_FILE = (
    str(PROJECT_DIR / 'data/varification set/anjali_dataset_with_claude_labels.xlsx')
)

val_df = pd.read_excel(VAL_FILE)
if options.smoke:
    val_df = val_df.head(3).copy()
val_ds = SFTDataset(val_df)

print(f"\nTraining examples after balancing: {len(train_ds)}")
print(f"Validation examples: {len(val_ds)}")

# -----------------------------
# LoRA
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

if options.four_bit:
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Memory-saving settings for the T4.
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False

model.print_trainable_parameters()

# -----------------------------
# Collator
# -----------------------------
def collate(batch):
    max_len = max(len(b["input_ids"]) for b in batch)

    pad_id = (
        processor.tokenizer.pad_token_id
        if processor.tokenizer.pad_token_id is not None
        else processor.tokenizer.eos_token_id
    )

    input_ids, attention_masks, labels, class_labels = [], [], [], []

    for b in batch:
        pad = max_len - len(b["input_ids"])

        input_ids.append(
            b["input_ids"] + [pad_id] * pad
        )

        attention_masks.append(
            [1] * len(b["input_ids"]) + [0] * pad
        )

        labels.append(
            b["labels"] + [-100] * pad
        )

        class_labels.append(b["class_labels"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "class_labels": torch.tensor(class_labels, dtype=torch.long),
    }



#####custom weighted trainer

import torch
import torch.nn.functional as F

class ClassWeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        # This loss is a microbatch mean; Trainer must scale accumulation.
        self.model_accepts_loss_kwargs = False

        self.class_weights = torch.tensor(
            class_weights,
            dtype=torch.float32
        )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        class_labels = inputs.pop("class_labels")
        labels = inputs["labels"]

        # Only project positions needed to predict supervised JSON/EOS tokens.
        shift_labels = labels[:, 1:]
        positions = shift_labels.ne(-100).any(dim=0).nonzero(as_tuple=True)[0]
        outputs = model(
            **{k: v for k, v in inputs.items() if k != "labels"},
            logits_to_keep=positions,
        )
        shift_labels = shift_labels[:, positions]
        valid_tokens = shift_labels.ne(-100)
        logits = outputs.logits[valid_tokens].float()
        token_loss = F.cross_entropy(logits, shift_labels[valid_tokens], reduction="none")
        weights = self.class_weights.to(class_labels.device)[class_labels]
        token_weights = weights.unsqueeze(1).expand_as(shift_labels)[valid_tokens]
        weighted_loss = (token_loss * token_weights).sum() / token_weights.sum().clamp_min(1e-8)

        return (
            weighted_loss,
            outputs
        ) if return_outputs else weighted_loss


#####weights


counts = df_train[CLAUDE_ACC_COL].value_counts().sort_index()

class_weights = (
    len(df_train) /
    (len(counts) * counts)
).to_dict()

class_weights = [
    class_weights.get(0, 1.0),
    class_weights.get(1, 1.0),
    class_weights.get(2, 1.0),
]

print("Class weights:", class_weights)


# -----------------------------
# Direct generation evaluation
# -----------------------------
# Trainer's standard eval loss is useful, but for this generative
# classification task we also want actual 0/1/2 predictions.
def evaluate_generation(model, dataframe):
    was_training = model.training
    model.eval()

    predictions = []
    raw_outputs = []
    valid_indices = []

    for idx, row in dataframe.iterrows():
        out = predict(row[QUESTION_COL], row[ANSWER_COL])
        pred = parse_prediction(out)

        predictions.append(pred)
        raw_outputs.append(out)
        valid_indices.append(idx)
        if len(predictions) % 20 == 0 or len(predictions) == len(dataframe):
            print(f"Validation generation: {len(predictions)}/{len(dataframe)}", flush=True)

    y_true = dataframe[CLAUDE_ACC_COL].astype(int).tolist()

    valid_mask = [
        p is not None and not pd.isna(p)
        for p in predictions
    ]

    y_true_valid = [
        y for y, keep in zip(y_true, valid_mask) if keep
    ]
    y_pred_valid = [
        int(p) for p, keep in zip(predictions, valid_mask) if keep
    ]

    if not y_pred_valid:
        metrics = {
            "accuracy": 0,
            "macro_f1": 0,
            "parse_rate": 0,
        }
        model.train(was_training)
        return metrics, predictions, raw_outputs

    metrics = {
        "accuracy": accuracy_score(y_true, [-1 if p is None else p for p in predictions]),
        "macro_f1": f1_score(
            y_true,
            [-1 if p is None else p for p in predictions],
            average="macro",
            labels=[0, 1, 2],
            zero_division=0,
        ),
        "parse_rate": len(y_pred_valid) / len(predictions),
    }

    model.train(was_training)
    return metrics, predictions, raw_outputs

# Baseline uses the same human-labelled validation pairs and prompt.
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
# Callback: evaluate every epoch
# -----------------------------
class GenerationEvalCallback(TrainerCallback):
    def __init__(self, validation_df):
        self.validation_df = validation_df
        self.history = []

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control

        metrics, predictions, raw_outputs = evaluate_generation(
            model,
            self.validation_df
        )

        epoch = state.epoch

        print("\n" + "=" * 60)
        print(f"Generation evaluation — epoch {epoch:.2f}")
        print(f"Accuracy:   {metrics['accuracy']:.4f}")
        print(f"Macro F1:   {metrics['macro_f1']:.4f}")
        print(f"Parse rate: {metrics['parse_rate']:.4f}")
        print("Prediction distribution:")
        print(pd.Series(predictions).value_counts(dropna=False).sort_index())
        print("=" * 60)

        # Log to TensorBoard through Trainer's logging mechanism.
        state.log_history.append({
            "eval_gen_accuracy": metrics["accuracy"],
            "eval_gen_macro_f1": metrics["macro_f1"],
            "eval_parse_rate": metrics["parse_rate"],
            "epoch": epoch,
        })

        self.history.append({
            "epoch": epoch,
            **metrics,
        })

        # Save epoch-level predictions for later inspection.
        out = self.validation_df.copy()
        out["gemma_accuracy"] = predictions
        out["gemma_raw_output"] = raw_outputs

        os.makedirs(
            str(RUN_DIR / 'metrics'),
            exist_ok=True
        )

        out.to_excel(
            f"{RUN_DIR}/metrics/"
            f"validation_epoch_{int(round(epoch))}.xlsx",
            index=False
        )

        return control

# -----------------------------
# Training arguments
# -----------------------------
OUTPUT_DIR = (
    str(RUN_DIR / 'models/gemma3_4b_lora_r1')
)

LOG_DIR = (
    str(RUN_DIR / 'logs/gemma3_4b_lora_r1')
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,

    num_train_epochs=1,
    learning_rate=1e-5,

    logging_steps=5,
    logging_first_step=True,

    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,

    bf16=(COMPUTE_DTYPE == torch.bfloat16),
    fp16=(COMPUTE_DTYPE == torch.float16),

    report_to="tensorboard",
    run_name="gemma3_4b_lora_r1",

    remove_unused_columns=False,
    label_names=["labels"],
    prediction_loss_only=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # Keep the best checkpoint according to Trainer's token-level eval loss.
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Save LoRA configuration for reproducibility.
config_path = (
    str(RUN_DIR / 'configs/gemma3_4b_lora_r1.json')
)
os.makedirs(os.path.dirname(config_path), exist_ok=True)

with open(config_path, "w") as f:
    json.dump(
        {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "task_type": "CAUSAL_LM",
            "learning_rate": 1e-5,
            "epochs": 1,
            "gradient_accumulation_steps": 4,
            "balanced_training": True,
        },
        f,
        indent=4,
    )

# -----------------------------
# Trainer
# -----------------------------
generation_callback = GenerationEvalCallback(val_df)

trainer = ClassWeightedTrainer(
    class_weights=class_weights,
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate,
    callbacks=[generation_callback],
)

# -----------------------------
# Train
# -----------------------------
trainer.train()

# -----------------------------
# Save LoRA adapter
# -----------------------------
ADAPTER_DIR = (
    str(RUN_DIR / 'models/gemma3_4b_lora_1100_adapter')
)

model.save_pretrained(ADAPTER_DIR)
processor.save_pretrained(ADAPTER_DIR)

print(f"LoRA adapter saved to: {ADAPTER_DIR}")

# -----------------------------
# Save Trainer log history
# -----------------------------
with open(
    str(RUN_DIR / 'metrics/gemma3_4b_lora_r1_log_history.json'),
    "w"
) as f:
    json.dump(trainer.state.log_history, f, indent=2)

# -----------------------------
# Plot training loss
# -----------------------------
logs = trainer.state.log_history

train_steps = []
train_losses = []

for log in logs:
    if "loss" in log and "step" in log:
        train_steps.append(log["step"])
        train_losses.append(log["loss"])

plt.figure(figsize=(8, 5))
plt.plot(train_steps, train_losses, marker="o")
plt.xlabel("Training Steps")
plt.ylabel("Training Loss")
plt.title("Gemma 3 4B + LoRA — Training Loss")
plt.grid(True)
plt.tight_layout()

loss_path = (
    str(RUN_DIR / 'metrics/gemma3_4b_lora_r1_training_loss.png')
)
plt.savefig(loss_path, dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# Plot generation metrics
# -----------------------------
if generation_callback.history:
    hist = pd.DataFrame(generation_callback.history)

    plt.figure(figsize=(8, 5))
    plt.plot(hist["epoch"], hist["macro_f1"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("Validation Macro F1")
    plt.grid(True)
    plt.tight_layout()

    f1_path = (
        str(RUN_DIR / 'metrics/gemma3_4b_lora_r1_macro_f1.png')
    )
    plt.savefig(f1_path, dpi=300, bbox_inches="tight")
    plt.close()

    hist.to_csv(
        str(RUN_DIR / 'metrics/gemma3_4b_lora_r1_generation_metrics.csv'),
        index=False
    )


# Notebook cell 15
#logging the history of training
import json

with open(f"{LOG_DIR}/trainer_log_history.json", "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)

#viewing the training loss
trainer.state.log_history


# Notebook cell 16
import torch
torch.cuda.empty_cache()
print(torch.cuda.memory_allocated() / 1e9, "GB allocated")


# Final generation metrics use the same human validation labels as baseline.
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
(RUN_DIR / "COMPLETED").write_text("Training and human validation completed.\n")
