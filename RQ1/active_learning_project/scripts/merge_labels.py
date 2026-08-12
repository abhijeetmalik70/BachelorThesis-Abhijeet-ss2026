"""
After you've manually filled in `manual_label` in
errors/round_XX_failures.csv and saved it as data/labeled/round_XX.csv,
this script:
  1. Appends the newly labeled rows into data/master_labeled_dataset.csv
  2. Removes those same rows (by id) from data/unlabeled_pool.csv

Usage:
    python scripts/merge_labels.py
"""
import argparse
import pandas as pd

from utils import load_config, full_path, ensure_id_column, round_tag


def main(config_path: str = None):
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    round_num = cfg["round"]
    id_col = data_cfg["id_column"]
    label_col = data_cfg["label_column"]
    text_col = data_cfg["text_column"]

    labeled_path = full_path(f"data/labeled/{round_tag(round_num)}.csv")
    new_labels = pd.read_csv(labeled_path)

    if "manual_label" not in new_labels.columns:
        raise ValueError("Expected a 'manual_label' column — did you export this from select_failures.py?")

    missing = new_labels["manual_label"].isna() | (new_labels["manual_label"] == "")
    if missing.any():
        raise ValueError(f"{missing.sum()} rows still have an empty manual_label — finish labeling before merging.")

    # normalize the manually labeled rows to match master schema
    new_labels = new_labels.rename(columns={"manual_label": label_col})
    keep_cols = [id_col, text_col, label_col]
    new_labels = new_labels[keep_cols]

    master_path = full_path(data_cfg["master_labeled_dataset"])
    master = pd.read_csv(master_path)
    master = ensure_id_column(master, id_col)

    combined = pd.concat([master, new_labels], ignore_index=True)
    combined = combined.drop_duplicates(subset=[id_col], keep="last")
    combined.to_csv(master_path, index=False)

    pool_path = full_path(data_cfg["unlabeled_pool"])
    pool = pd.read_csv(pool_path)
    pool = ensure_id_column(pool, id_col)
    pool_remaining = pool[~pool[id_col].isin(new_labels[id_col])]
    pool_remaining.to_csv(pool_path, index=False)

    print(f"Merged {len(new_labels)} newly labeled rows into: {master_path} (now {len(combined)} rows)")
    print(f"Removed them from unlabeled pool: {pool_path} (now {len(pool_remaining)} rows remaining)")
    print(f"Next: bump `round` in configs/round_config.yaml to {round_num + 1} and re-run train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
