"""
Fine-tunes the HF model on the current master_labeled_dataset.csv and
saves a checkpoint under models/checkpoints/round_XX.

Supports:
  - Multiple text columns (config: data.text_columns), combined into a
    single input string using data.text_combine_strategy / text_separator.
  - Multiple label columns (config: data.label_columns), each trained as
    its own classification head (multi-task), sharing one text encoder.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/round_config.yaml
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset as TorchDataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from utils import load_config, full_path, ensure_id_column, checkpoint_dir_for_round


# ------------------------------------------------------------------
# Text combination
# ------------------------------------------------------------------
def combine_text_columns(df: pd.DataFrame, text_columns, strategy: str, separator: str) -> pd.Series:
    """Combine multiple text columns into a single string per row."""
    cols = [df[c].fillna("").astype(str) for c in text_columns]
    if strategy == "template":
        # "Column A: ... [SEP] Column B: ..." style, useful if you want the
        # model to see which field each piece of text came from.
        parts = [f"{name}: {series}" for name, series in zip(text_columns, cols)]
    else:  # "concat" (default)
        parts = cols
    combined = parts[0]
    for p in parts[1:]:
        combined = combined.str.cat(p, sep=separator)
    return combined


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class MultiTaskTextDataset(TorchDataset):
    def __init__(self, encodings, labels_dict):
        self.encodings = encodings
        self.labels_dict = labels_dict  # {task_name: np.array of int labels}
        self.task_names = list(labels_dict.keys())

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(
            [self.labels_dict[t][idx] for t in self.task_names], dtype=torch.long
        )
        return item


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        labels = torch.stack([b.pop("labels") for b in batch])
        padded = tokenizer.pad(batch, return_tensors="pt")
        padded["labels"] = labels
        return padded
    return collate_fn


# ------------------------------------------------------------------
# Multi-task model: shared encoder + one linear head per label column
# ------------------------------------------------------------------
class MultiTaskSequenceClassifier(nn.Module):
    def __init__(self, base_model_name: str, task_names, num_labels: int):
        super().__init__()
        self.task_names = list(task_names)
        self.backbone = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(getattr(self.backbone.config, "hidden_dropout_prob", 0.1))
        self.heads = nn.ModuleDict(
            {name: nn.Linear(hidden_size, num_labels) for name in self.task_names}
        )
        self.config = self.backbone.config  # so Trainer/save_pretrained-style code has something to inspect

    def _pool(self, last_hidden_state, attention_mask):
        # Mean-pool over non-padding tokens (robust across encoder architectures,
        # unlike assuming a BERT-style [CLS] token at position 0).
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(outputs.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)

        logits = {name: self.heads[name](pooled) for name in self.task_names}
        stacked_logits = torch.stack([logits[name] for name in self.task_names], dim=1)
        # shape: (batch, num_tasks, num_labels)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = 0.0
            for i, name in enumerate(self.task_names):
                loss = loss + loss_fct(logits[name], labels[:, i])

        return {"loss": loss, "logits": stacked_logits}

    def save_pretrained(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(out_dir, "pytorch_model.bin"))
        self.backbone.config.save_pretrained(out_dir)


# ------------------------------------------------------------------
# Metrics (averaged across tasks, plus per-task breakdown printed)
# ------------------------------------------------------------------
def make_compute_metrics(task_names):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred  # logits: (batch, num_tasks, num_labels), labels: (batch, num_tasks)
        preds = np.argmax(logits, axis=-1)  # (batch, num_tasks)

        metrics = {}
        f1s = []
        for i, name in enumerate(task_names):
            acc = accuracy_score(labels[:, i], preds[:, i])
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels[:, i], preds[:, i], average="macro", zero_division=0
            )
            safe_name = name.replace(" ", "_").replace("&", "and")
            metrics[f"{safe_name}_accuracy"] = acc
            metrics[f"{safe_name}_precision_macro"] = precision
            metrics[f"{safe_name}_recall_macro"] = recall
            metrics[f"{safe_name}_f1_macro"] = f1
            f1s.append(f1)

        metrics["f1_macro"] = float(np.mean(f1s))  # overall metric Trainer selects "best model" on
        return metrics
    return compute_metrics


def main(config_path: str = None):
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    master_path = full_path(data_cfg["master_labeled_dataset"])
    df = pd.read_csv(master_path) if str(master_path).endswith(".csv") else pd.read_excel(master_path)
    df = ensure_id_column(df, data_cfg["id_column"])

    text_columns = data_cfg["text_columns"]        # list, e.g. 3 columns
    label_columns = data_cfg["label_columns"]       # list, e.g. 2 columns
    combine_strategy = data_cfg.get("text_combine_strategy", "concat")
    separator = data_cfg.get("text_separator", " [SEP] ")

    # Row is only usable if ALL text columns and ALL label columns are present.
    df = df.dropna(subset=text_columns + label_columns).reset_index(drop=True)

    df["combined_text"] = combine_text_columns(df, text_columns, combine_strategy, separator)
    for lc in label_columns:
        df[lc] = df[lc].astype(int)

    # Stratify on a single combined key since train_test_split only accepts one stratify array.
    # With small label sets some combinations may have <2 members, which stratification can't
    # handle — degrade gracefully instead of hard-failing.
    def split_with_fallback(df, label_columns, test_size, seed):
        combined_key = df[label_columns].astype(str).agg("_".join, axis=1)
        if combined_key.value_counts().min() >= 2:
            return train_test_split(df, test_size=test_size, random_state=seed, stratify=combined_key)

        print(
            "[train.py] Combined label-stratification key has a class with <2 members "
            "(likely too little labeled data yet) — falling back to stratifying on "
            f"'{label_columns[0]}' alone."
        )
        single_key = df[label_columns[0]].astype(str)
        if single_key.value_counts().min() >= 2:
            return train_test_split(df, test_size=test_size, random_state=seed, stratify=single_key)

        print(
            "[train.py] Even single-column stratification has a class with <2 members — "
            "falling back to a plain random split (no stratification) for this round."
        )
        return train_test_split(df, test_size=test_size, random_state=seed)

    train_df, val_df = split_with_fallback(df, label_columns, data_cfg["val_split"], train_cfg["seed"])

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model_name"])

    def build_dataset(sub_df):
        enc = tokenizer(
            sub_df["combined_text"].tolist(),
            truncation=True,
            max_length=model_cfg["max_length"],
            padding=False,
        )
        labels_dict = {lc: sub_df[lc].to_numpy() for lc in label_columns}
        return MultiTaskTextDataset(enc, labels_dict)

    train_ds = build_dataset(train_df)
    val_ds = build_dataset(val_df)

    model = MultiTaskSequenceClassifier(
        base_model_name=model_cfg["base_model_name"],
        task_names=label_columns,
        num_labels=model_cfg["num_labels"],
    )

    out_dir = checkpoint_dir_for_round(cfg)

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        learning_rate=train_cfg["learning_rate"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_dir=full_path(f"logs/{os.path.basename(out_dir)}"),
        seed=train_cfg["seed"],
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=make_collate_fn(tokenizer),
        compute_metrics=make_compute_metrics(label_columns),
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"[round {cfg['round']}] validation metrics: {metrics}")

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved checkpoint to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)