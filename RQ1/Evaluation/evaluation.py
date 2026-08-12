"""
SLM Content-Moderation Labeling: Ollama Cloud Free Models vs Ground Truth
==========================================================================

What this script does
----------------------
For every (question, answer) pair in your labeled dataset, each free Ollama Cloud
model is prompted to generate its OWN "Accuracy" and "Context and Relevance" labels
(0-2 each) -- i.e. the small language model plays the role of the moderator. Those
generated labels are then compared against your human ground-truth labels using
standard classification metrics: Accuracy, Precision, Recall, F1, and mIoU
(macro-averaged Jaccard index). No LLM judge is used -- only Ollama Cloud.

How to use this in Google Colab
--------------------------------
1. Open a new Colab notebook (colab.research.google.com).
2. Paste this entire script into a single cell and run it (Shift+Enter).
   - It will prompt you for your Ollama Cloud API key (https://ollama.com/settings/keys).
   - It will prompt you to upload your labeled .xlsx / .csv file.
3. Progress is checkpointed to disk (predicted_labels.csv), so if a run gets
   interrupted (rate limits, timeouts), just re-run the cell -- it skips
   (model, row) pairs that are already done.
4. At the end it saves + downloads: predicted_labels.csv, metrics_by_model_dimension.csv,
   model_leaderboard.csv, and f1_by_model_dimension.png.

You can also run this as a normal local script: `python ollama_cloud_slm_moderation_eval.py`
(it detects it's not in Colab and falls back to a local file-path prompt instead of the
upload widget, and just leaves output files on disk instead of triggering a browser download).
"""

import os
import re
import time
import json
from getpass import getpass

import requests
import pandas as pd
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Detect whether we're running inside Google Colab. This lets the same script
# work both pasted into a Colab cell (file upload/download widgets available)
# and run locally (falls back to plain file paths).
# ---------------------------------------------------------------------------
try:
    from google.colab import files as colab_files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


# ===========================================================================
# CONFIG -- edit these if your column names / sheet name differ from the
# defaults, which match "V1_master_with_labels.xlsx" (sheet "All").
# ===========================================================================
SHEET_NAME = "All"                              # only used if the file is .xlsx
QUESTION_COL = "Sanitized Question"
ANSWER_COL = "Answer Transcription"
LABEL_COLS = {
    "accuracy": "Accuracy",
    "context_relevance": "Context and Relevance",
}

AUTO_DETECT_MODELS = False                    # pull live free-model list from Ollama Cloud
MANUAL_MODEL_LIST = [                           # used if AUTO_DETECT_MODELS = False, or as fallback


"gemma4:e4b",

]


PRED_PATH = "predicted_labels.csv"              # checkpoint file for generated labels
# Columns the checkpoint file must have. Used to detect stale/incompatible
# checkpoints left over from an older version of this script (different
# column names, dropped "tone" dimension, etc.) so we don't crash with a
# KeyError when re-running against an old predicted_labels.csv.
PRED_COLUMNS = ["row_id", "model", "pred_accuracy", "pred_context_relevance", "raw_output"]
CHECKPOINT_EVERY = 10                           # rows between checkpoint saves
MAX_RETRIES = 3                                 # per model call, before giving up on a row
RETRY_DELAY_SEC = 5

# The rubric given to each SLM. Keep it short -- small models follow short,
# rigid instructions better than long ones.
MODERATION_SYSTEM_PROMPT = '''You are a content moderator reviewing answers given on a health helpline.
You will be given a QUESTION and an ANSWER. Score the ANSWER on exactly two dimensions,
each as an integer 0, 1, or 2:

accuracy: 0 = factually incorrect, 1 = partially correct, 2 = fully correct
context_relevance: 0 = irrelevant/off-topic, 1 = partially relevant, 2 = fully relevant and appropriate

Respond with ONLY a JSON object, nothing else, no markdown fences:
{"accuracy": <0|1|2>, "context_relevance": <0|1|2>}'''


# ===========================================================================
# STEP 1: Ollama Cloud API key
# ===========================================================================
def get_api_key():
    """Prompt for the Ollama Cloud API key without echoing it to the screen."""
    key = getpass("Ollama Cloud API key (https://ollama.com/settings/keys): ")
    os.environ["OLLAMA_API_KEY"] = key
    return key


# ===========================================================================
# STEP 2: Load the labeled dataset
# ===========================================================================
def load_dataset():
    """
    In Colab: shows the upload widget.
    Locally: asks for a file path on disk.
    Returns a cleaned dataframe with a stable 'row_id' column.
    """
    if IN_COLAB:
        uploaded = colab_files.upload()
        data_path = list(uploaded.keys())[0]
    else:
        data_path = input("Path to your .xlsx or .csv dataset: ").strip()

    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path, sheet_name=SHEET_NAME)

    # Keep only the columns we need, drop rows missing question/answer text.
    keep_cols = [QUESTION_COL, ANSWER_COL] + list(LABEL_COLS.values())
    df = df[keep_cols].dropna(subset=[QUESTION_COL, ANSWER_COL]).reset_index(drop=True)
    df["row_id"] = df.index

    # Ground-truth labels must be plain integers (0/1/2) for sklearn metrics later.
    for col in LABEL_COLS.values():
        df[col] = df[col].astype(int)

    print(f"Loaded {len(df)} labeled rows")
    return df


# ===========================================================================
# STEP 3: Decide which free Ollama Cloud models to test
# ===========================================================================
def detect_cloud_models(api_key):
    """
    Pulls the live model catalog from ollama.com/api/tags and filters down to
    cloud-hosted models. Falls back to MANUAL_MODEL_LIST if the request fails
    (e.g. network hiccup, API changes) so the script never hard-crashes here.
    """
    try:
        resp = requests.get(
            "https://ollama.com/api/tags",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        names = [m["name"] for m in resp.json().get("models", [])]
        cloud_models = [n for n in names if "cloud" in n or ":" in n]
        return sorted(set(cloud_models))
    except Exception as e:
        print(f"Auto-detect failed ({e}), falling back to manual list")
        return MANUAL_MODEL_LIST


def get_models_to_test(api_key):
    models = detect_cloud_models(api_key) if AUTO_DETECT_MODELS else MANUAL_MODEL_LIST
    print(f"Testing {len(models)} models:")
    for m in models:
        print(" -", m)
    return models


# ===========================================================================
# STEP 4: Query a single model for a single (question, answer) pair
# ===========================================================================
def parse_labels(text):
    """
    Extract {accuracy, context_relevance} from a model's raw text output.
    Small models don't always emit clean JSON, so this tries strict JSON first,
    then falls back to a regex scan for 'key: digit' patterns anywhere in the
    text. Anything that still can't be parsed becomes -1, which downstream
    code treats as a miss and excludes from metric computation (rather than
    silently guessing a value, which would bias the scores).
    """
    text = text.strip()
    # strip markdown code fences if the model added them despite instructions
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()

    try:
        obj = json.loads(text)
        return int(obj["accuracy"]), int(obj["context_relevance"])
    except Exception:
        pass  # fall through to regex fallback

    out = {}
    for key in ["accuracy", "context_relevance"]:
        m = re.search(rf'{key}["\']?\s*[:=]\s*["\']?([0-2])', text, re.IGNORECASE)
        out[key] = int(m.group(1)) if m else -1
    return out["accuracy"], out["context_relevance"]


def query_model(client, model_name, question, answer):
    """
    Sends one (question, answer) pair to one Ollama Cloud model and returns
    the parsed (accuracy, context_relevance, raw_text) tuple.
    Retries transient errors (network, rate limit) up to MAX_RETRIES times.
    """
    user_msg = f"QUESTION:\n{question}\n\nANSWER:\n{answer}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": MODERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                stream=False,
            )
            raw = resp["message"]["content"]
            acc, ctx = parse_labels(raw)
            return acc, ctx, raw
        except Exception as e:
            if attempt == MAX_RETRIES:
                return -1, -1, f"[ERROR: {e}]"
            time.sleep(RETRY_DELAY_SEC)


# ===========================================================================
# STEP 5: Run every model over every row, with checkpointing
# ===========================================================================
def load_checkpoint():
    """
    Loads PRED_PATH if it exists, validating that it has the columns this
    version of the script expects.

    This guards against the KeyError: 'model' crash that happens when an old
    predicted_labels.csv (written by an earlier/different version of this
    script -- e.g. different column names, or a dimension that's since been
    added/removed) gets picked up on a fresh run. Instead of blowing up on
    `pred_df["model"]`, we detect the mismatch up front, move the old file
    aside, and start a clean checkpoint.
    """
    if not os.path.exists(PRED_PATH):
        return pd.DataFrame(columns=PRED_COLUMNS)

    pred_df = pd.read_csv(PRED_PATH)
    missing = [c for c in PRED_COLUMNS if c not in pred_df.columns]
    if missing:
        backup_path = PRED_PATH + ".bak"
        print(
            f"Found existing {PRED_PATH} but it's missing column(s) {missing} "
            f"(looks like it's from an older run/schema). Moving it to "
            f"{backup_path} and starting a fresh checkpoint instead of crashing."
        )
        os.replace(PRED_PATH, backup_path)
        return pd.DataFrame(columns=PRED_COLUMNS)

    return pred_df


def generate_predictions(client, df, models):
    """
    Loops over every (row, model) combination that hasn't been done yet
    (based on what's already saved in PRED_PATH), queries the model, and
    saves progress to disk every CHECKPOINT_EVERY rows. Safe to interrupt
    and re-run -- it will pick up where it left off.
    """
    pred_df = load_checkpoint()

    done_pairs = set(zip(pred_df["row_id"], pred_df["model"])) if len(pred_df) else set()

    pairs = [
        (row.row_id, model, row[QUESTION_COL], row[ANSWER_COL])
        for _, row in df.iterrows()
        for model in models
        if (row.row_id, model) not in done_pairs
    ]

    new_rows = []
    for row_id, model, question, answer in tqdm(pairs, desc="Generating labels"):
        acc, ctx, raw = query_model(client, model, question, answer)
        new_rows.append({
            "row_id": row_id, "model": model,
            "pred_accuracy": acc, "pred_context_relevance": ctx,
            "raw_output": raw,
        })
        if len(new_rows) % CHECKPOINT_EVERY == 0:
            pred_df = pd.concat([pred_df, pd.DataFrame(new_rows)], ignore_index=True)
            pred_df.to_csv(PRED_PATH, index=False)
            new_rows = []

    if new_rows:
        pred_df = pd.concat([pred_df, pd.DataFrame(new_rows)], ignore_index=True)
    pred_df.to_csv(PRED_PATH, index=False)

    n_unparsed = (pred_df[["pred_accuracy", "pred_context_relevance"]] == -1).any(axis=1).sum()
    print(f"Total predictions: {len(pred_df)} | unparsed/excluded: {n_unparsed}")

    # If everything (or almost everything) failed to parse, the raw_output
    # column almost always tells you why (e.g. an actual API/model error
    # rather than a genuinely malformed JSON response). Surface a few
    # samples so the real cause is visible instead of silently proceeding
    # to an empty metrics table.
    if len(pred_df) > 0 and n_unparsed == len(pred_df):
        print(
            "\nWARNING: every prediction was unparsed/excluded. This usually means "
            "the model calls themselves are failing (bad model name, auth error, "
            "rate limit, etc.) rather than the model just returning malformed JSON. "
            "Here are a few raw outputs to help diagnose:"
        )
        for raw in pred_df["raw_output"].head(5):
            print(" -", raw)

    return pred_df


# ===========================================================================
# STEP 6: Score predictions against ground truth
# ===========================================================================
def compute_metrics(df, pred_df, models):
    """
    For each model and each of the two label dimensions, computes:
      - accuracy
      - precision (macro-averaged across classes 0/1/2)
      - recall (macro-averaged)
      - f1 (macro-averaged)
      - mIoU (macro-averaged Jaccard index -- the standard IoU-style score
        for multi-class classification; treats each class as a "set" and
        averages IoU across classes 0/1/2)

    Rows where a model's output could not be parsed (-1) are excluded from
    that dimension's metrics for that model, and the exclusion count is
    reported alongside the scores for transparency.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

    merged = pred_df.merge(df, on="row_id", how="left")

    dimensions = {
        "accuracy": ("pred_accuracy", LABEL_COLS["accuracy"]),
        "context_relevance": ("pred_context_relevance", LABEL_COLS["context_relevance"]),
    }

    results = []
    for model in models:
        model_df = merged[merged["model"] == model]
        for dim, (pred_col, gt_col) in dimensions.items():
            valid = model_df[model_df[pred_col] != -1]
            if len(valid) == 0:
                continue
            y_true = valid[gt_col].astype(int)
            y_pred = valid[pred_col].astype(int)
            results.append({
                "model": model,
                "dimension": dim,
                "n_scored": len(valid),
                "n_excluded": len(model_df) - len(valid),
                "accuracy": accuracy_score(y_true, y_pred),
                "precision_macro": precision_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0),
                "recall_macro": recall_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0),
                "f1_macro": f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0),
                "mIoU": jaccard_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0),
            })

    # Build with an explicit column list so that even a fully-empty result
    # (e.g. every model/dimension had zero valid rows) still produces a
    # DataFrame with a "model" column, instead of a columnless empty
    # DataFrame that causes a KeyError: 'model' downstream in plotting /
    # leaderboard code.
    metrics_columns = [
        "model", "dimension", "n_scored", "n_excluded",
        "accuracy", "precision_macro", "recall_macro", "f1_macro", "mIoU",
    ]
    metrics_df = pd.DataFrame(results, columns=metrics_columns).round(3)
    metrics_df.to_csv("metrics_by_model_dimension.csv", index=False)

    if metrics_df.empty:
        print(
            "\nWARNING: metrics_by_model_dimension.csv is empty -- no model had any "
            "valid (parsed) predictions for any dimension. Check the raw_output "
            "samples printed above, or inspect predicted_labels.csv directly, "
            "before trusting the leaderboard/chart."
        )

    return metrics_df


def build_leaderboard(metrics_df):
    """Averages the per-dimension scores into one overall ranking per model."""
    if metrics_df.empty:
        print("Skipping leaderboard: metrics_by_model_dimension is empty.")
        return pd.DataFrame(
            columns=["accuracy", "precision_macro", "recall_macro", "f1_macro", "mIoU"]
        )

    leaderboard = (
        metrics_df.groupby("model")[["accuracy", "precision_macro", "recall_macro", "f1_macro", "mIoU"]]
        .mean()
        .round(3)
        .sort_values("f1_macro", ascending=False)
    )
    leaderboard.to_csv("model_leaderboard.csv")
    return leaderboard


# ===========================================================================
# STEP 7: Chart -- grouped bar chart, models on x-axis, one bar per
# dimension (Accuracy / Context & Relevance), F1 score (%) on y-axis.
# Styled to match a reference "benchmark leaderboard" chart look.
# ===========================================================================
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def plot_f1_by_model_dimension(metrics_df, save_path="f1_by_model_dimension.png"):
    """
    Grouped bar chart:
      x-axis = model
      2 bars per model -- one for 'accuracy' F1, one for 'context_relevance' F1
      y-axis = F1 score (%), with the value labeled in bold above each bar.
    """
    if metrics_df.empty:
        print(f"Skipping chart ({save_path}): metrics_by_model_dimension is empty.")
        return None

    dims = ["accuracy", "context_relevance"]
    dim_labels = {"accuracy": "Accuracy", "context_relevance": "Context & Relevance"}
    models = metrics_df["model"].unique().tolist()

    # pivot so rows=model, cols=dimension, values=f1_macro (as a percentage)
    pivot = metrics_df.pivot(index="model", columns="dimension", values="f1_macro").reindex(columns=dims)
    pivot = pivot.reindex(models) * 100  # keep original model order, scale 0-1 -> 0-100

    n_models = len(models)
    n_dims = len(dims)
    x = np.arange(n_models)
    bar_width = 0.32

    fig, ax = plt.subplots(figsize=(max(9, n_models * 2.2), 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7f7f9")

    colors = ["#c9b6f0", "#a8c8f0"]  # soft purple (accuracy), soft blue (context & relevance)

    for i, dim in enumerate(dims):
        offset = (i - (n_dims - 1) / 2) * bar_width
        values = pivot[dim].values
        bars = ax.bar(
            x + offset, values, width=bar_width,
            label=dim_labels[dim], color=colors[i],
            edgecolor="white", linewidth=0.5, zorder=3,
        )
        for bar, val in zip(bars, values):
            if np.isnan(val):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{val:.1f}",
                ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#222222",
            )

    ax.set_ylabel("F1 Score (%)", fontsize=12, fontweight="bold")
    ax.set_title("SLM Moderation Performance by Model", fontsize=16, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold", rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.grid(axis="y", linestyle="-", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=2, frameon=False, fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


# ===========================================================================
# STEP 8: Deliver output files
# ===========================================================================
def deliver_outputs():
    output_files = [
        "predicted_labels.csv",
        "metrics_by_model_dimension.csv",
        "model_leaderboard.csv",
        "f1_by_model_dimension.png",
    ]
    if IN_COLAB:
        for f in output_files:
            if os.path.exists(f):
                colab_files.download(f)
    else:
        print("Output files saved to disk:")
        for f in output_files:
            if os.path.exists(f):
                print(" -", os.path.abspath(f))


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    from ollama import Client

    api_key = get_api_key()
    df = load_dataset()
    models = get_models_to_test(api_key)

    client = Client(host="https://ollama.com", headers={"Authorization": f"Bearer {api_key}"})

    pred_df = generate_predictions(client, df, models)
    metrics_df = compute_metrics(df, pred_df, models)
    plot_f1_by_model_dimension(metrics_df)
    leaderboard = build_leaderboard(metrics_df)

    print("\n=== Per-model, per-dimension metrics ===")
    print(metrics_df.sort_values(["dimension", "f1_macro"], ascending=[True, False]).to_string(index=False))

    print("\n=== Overall leaderboard ===")
    print(leaderboard.to_string())

    deliver_outputs()


if __name__ == "__main__":
    main()