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

MODEL_ID = "google/gemma-3-4b-it"

# Debug with bf16 first — confirm the prompt fix works before
# reintroducing 4-bit for memory savings.
USE_4BIT = options.four_bit

if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map="auto", quantization_config=bnb_config
    )
else:
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
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

        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        shift_labels = labels[:, 1:]
        valid_tokens = shift_labels.ne(-100)
        logits = outputs.logits[:, :-1, :][valid_tokens].float()
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
        "accuracy": accuracy_score(y_true_valid, y_pred_valid),
        "macro_f1": f1_score(
            y_true_valid,
            y_pred_valid,
            average="macro",
            labels=[0, 1, 2],
            zero_division=0,
        ),
        "parse_rate": len(y_pred_valid) / len(predictions),
    }

    model.train(was_training)
    return metrics, predictions, raw_outputs

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

    bf16=True,

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
    str(RUN_DIR / 'models/gemma3_4b_lora_round1_adapter_200')
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


# Notebook cell 17
# ============================================================
# FINAL VALIDATION EVALUATION
# ============================================================

model.eval()
model.config.use_cache = True

VAL_FILE = (
    str(PROJECT_DIR / 'data/varification set/anjali_dataset_labeled.csv')
)
val_df = pd.read_csv(VAL_FILE)
if options.smoke:
    val_df = val_df.head(3).copy()

val_acc, val_raw = [], []

for i, row in val_df.iterrows():
    out = predict(row["question"], row["model_answer"])
    a = parse_prediction(out)

    val_acc.append(a)
    val_raw.append(out)

    print(
        f"{i + 1}/{len(val_df)} | "
        f"Accuracy={a}"
    )

val_df["gemma_accuracy"] = val_acc
val_df["gemma_raw_output"] = val_raw

validation_output = (
    str(RUN_DIR / 'metrics/gemma3_lora_round1_validation.xlsx')
)
val_df.to_excel(validation_output, index=False)

print(f"\nSaved predictions to: {validation_output}")


# Notebook cell 18
# ============================================================
# FINAL METRICS
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import numpy as np
import os

valid_val = val_df.dropna(subset=["gemma_accuracy"]).copy()

y_true = valid_val["accuracy"].astype(int)
y_pred = valid_val["gemma_accuracy"].astype(int)

final_accuracy = accuracy_score(y_true, y_pred)
final_macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    labels=[0, 1, 2],
    zero_division=0,
)

print(f"Parsed successfully: {len(valid_val)}/{len(val_df)}")
print(f"Parse rate: {len(valid_val) / len(val_df):.3f}")
print(f"Accuracy: {final_accuracy:.3f}")
print(f"Macro-F1: {final_macro_f1:.3f}")

print("\nClassification report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=["Class 0", "Class 1", "Class 2"],
        zero_division=0,
    )
)

# Prediction distribution — critical for detecting class collapse.
print("\nGround-truth distribution:")
print(y_true.value_counts().sort_index())

print("\nPrediction distribution:")
print(y_pred.value_counts().sort_index())

# Confusion matrix.
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1, 2],
)

print("\nConfusion matrix:")
print(cm)

metrics_dir = str(RUN_DIR / 'metrics')
os.makedirs(metrics_dir, exist_ok=True)

np.savetxt(
    f"{metrics_dir}/gemma3_4b_lora_r1_confusion_matrix.csv",
    cm,
    delimiter=",",
    fmt="%d",
)

# Plot confusion matrix.
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm)

ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(["0", "1", "2"])
ax.set_yticklabels(["0", "1", "2"])

ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Gemma 3 4B + LoRA — Accuracy Confusion Matrix")

for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center")

fig.colorbar(im, ax=ax)
fig.tight_layout()

cm_path = (
    f"{metrics_dir}/gemma3_4b_lora_r1_confusion_matrix.png"
)
fig.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.close()

# Save a compact final metrics file.
final_metrics = {
    "parsed_examples": int(len(valid_val)),
    "total_examples": int(len(val_df)),
    "parse_rate": float(len(valid_val) / len(val_df)),
    "accuracy": float(final_accuracy),
    "macro_f1": float(final_macro_f1),
    "prediction_distribution": {
        str(k): int(v)
        for k, v in y_pred.value_counts().sort_index().items()
    },
}

with open(
    f"{metrics_dir}/gemma3_4b_lora_r1_final_metrics.json",
    "w"
) as f:
    json.dump(final_metrics, f, indent=2)

# Save errors for qualitative analysis.
errors = valid_val[y_true != y_pred].copy()

errors.to_excel(
    f"{metrics_dir}/gemma3_4b_lora_r1_errors.xlsx",
    index=False,
)

print(f"\nSaved metrics and confusion matrix under: {metrics_dir}")
