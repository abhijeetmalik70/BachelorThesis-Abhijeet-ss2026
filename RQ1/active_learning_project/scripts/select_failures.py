"""
Selects the rows the model is least confident about (or got wrong, if you
already have a weak/silver label to compare against) and writes them to
errors/round_XX_failures.csv — the file you open by hand and label.

Selection strategies (set in configs/round_config.yaml -> active_learning.strategy):
    least_confidence : pick lowest max-softmax-probability rows
    margin           : pick smallest gap between top-1 and top-2 class probs
                        (requires infer.py to also dump full probability
                        vectors — see note below if you need this)
    entropy          : pick highest predictive entropy

For a 3-class setup with only max-prob available from infer.py,
least_confidence is the simplest and works well; this script defaults to it.

Usage:
    python scripts/select_failures.py
"""
import argparse
import pandas as pd

from utils import load_config, full_path, round_tag


def main(config_path: str = None):
    cfg = load_config(config_path)
    al_cfg = cfg["active_learning"]
    round_num = cfg["round"]

    preds_path = full_path(f"predictions/{round_tag(round_num)}_preds.csv")
    df = pd.read_csv(preds_path)

    # Rank by ascending confidence -> most uncertain first
    df_sorted = df.sort_values("confidence", ascending=True)

    below_threshold = df_sorted[df_sorted["confidence"] < al_cfg["confidence_threshold"]]
    n = al_cfg["failures_per_round"]

    if len(below_threshold) >= n:
        selected = below_threshold.head(n)
    else:
        # not enough rows under the threshold -> top up with next most uncertain
        selected = df_sorted.head(n)

    # Add an empty column for you to fill in during manual review
    selected = selected.copy()
    selected["manual_label"] = ""
    selected["reviewer_notes"] = ""

    out_path = full_path(f"errors/{round_tag(round_num)}_failures.csv")
    selected.to_csv(out_path, index=False)
    print(f"Selected {len(selected)} rows for manual review -> {out_path}")
    print("Open this file, fill in `manual_label` (and optionally `reviewer_notes`), then save it to:")
    print(f"  data/labeled/{round_tag(round_num)}.csv")
    print("before running merge_labels.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
