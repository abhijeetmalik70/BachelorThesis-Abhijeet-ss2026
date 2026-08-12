"""
Runs the current round's fine-tuned checkpoint over the unlabeled pool
and writes predictions + confidence scores to predictions/round_XX_preds.csv.

Usage:
    python scripts/infer.py
"""
import argparse
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import load_config, full_path, ensure_id_column, checkpoint_dir_for_round, round_tag


def main(config_path: str = None):
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    round_num = cfg["round"]

    pool_path = full_path(data_cfg["unlabeled_pool"])
    df = pd.read_csv(pool_path)
    df = ensure_id_column(df, data_cfg["id_column"])
    text_col = data_cfg["text_column"]

    ckpt_dir = checkpoint_dir_for_round(cfg, round_num)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    preds, confidences = [], []
    batch_size = 16
    texts = df[text_col].fillna("").tolist()

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, truncation=True, padding=True, max_length=cfg["model"]["max_length"],
                             return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            conf, pred = torch.max(probs, dim=-1)
            preds.extend(pred.cpu().tolist())
            confidences.extend(conf.cpu().tolist())

    df["predicted_label"] = preds
    df["confidence"] = confidences

    out_path = full_path(f"predictions/{round_tag(round_num)}_preds.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} predictions to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
