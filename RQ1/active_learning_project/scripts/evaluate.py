"""
Evaluates the current round's checkpoint on a held-out validation split
of master_labeled_dataset.csv and appends the result to
metrics/training_curve.csv so you can plot progress across rounds.

Usage:
    python scripts/evaluate.py
"""
import argparse
import json
import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, jaccard_score
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import load_config, full_path, ensure_id_column, checkpoint_dir_for_round, round_tag


def main(config_path: str = None):
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    round_num = cfg["round"]
    text_col, label_col = data_cfg["text_column"], data_cfg["label_column"]

    df = pd.read_csv(full_path(data_cfg["master_labeled_dataset"]))
    df = ensure_id_column(df, data_cfg["id_column"]).dropna(subset=[text_col, label_col])

    _, val_df = train_test_split(
        df, test_size=data_cfg["val_split"], random_state=cfg["training"]["seed"], stratify=df[label_col]
    )

    ckpt_dir = checkpoint_dir_for_round(cfg, round_num)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    texts = val_df[text_col].tolist()
    true_labels = val_df[label_col].astype(int).tolist()
    preds = []

    with torch.no_grad():
        for i in range(0, len(texts), 16):
            batch = texts[i:i + 16]
            enc = tokenizer(batch, truncation=True, padding=True, max_length=cfg["model"]["max_length"],
                             return_tensors="pt").to(device)
            logits = model(**enc).logits
            batch_preds = torch.argmax(logits, dim=-1).cpu().tolist()
            preds.extend(batch_preds)

    acc = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, preds, average="macro", zero_division=0
    )
    miou = jaccard_score(true_labels, preds, average="macro")

    metrics = {
        "round": round_num,
        "n_val": len(val_df),
        "n_train_total": len(df) - len(val_df),
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "miou_macro": miou,
    }

    metrics_dir = full_path("metrics")
    with open(os.path.join(metrics_dir, f"{round_tag(round_num)}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    curve_path = os.path.join(metrics_dir, "training_curve.csv")
    row_df = pd.DataFrame([metrics])
    if os.path.exists(curve_path):
        existing = pd.read_csv(curve_path)
        existing = existing[existing["round"] != round_num]  # overwrite if re-running same round
        combined = pd.concat([existing, row_df], ignore_index=True).sort_values("round")
    else:
        combined = row_df
    combined.to_csv(curve_path, index=False)

    print(f"[round {round_num}] {metrics}")
    print(f"Appended to: {curve_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
