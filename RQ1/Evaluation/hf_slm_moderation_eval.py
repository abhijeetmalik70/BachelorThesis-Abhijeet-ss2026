"""
=================================================================================
  Hugging Face SLM Moderation Evaluator  (v1)
=================================================================================

WHAT THIS DOES
--------------
Given ONE Excel file that already contains:
  - the caller's question
  - the virtual moderator's answer  ("Answer Model" column — the thing being judged)
  - a human reference / gold answer ("Relevant answer" column)
  - human ground-truth labels        ("Accuracy" and "Context and Relevance" columns, 0-2)

...this script:
  1. Loads one or more small open-weight moderation/instruction models FROM
     HUGGING FACE HUB (no Ollama involved at all).
  2. Prompts each model to judge the "Answer Model" text against the question
     and the reference answer, asking it to output JSON scores for
     accuracy and context_relevance (0-2 each — same scale as the human labels).
  3. Parses the JSON, compares it to the human ground truth per row.
  4. Computes accuracy, macro-precision, macro-recall, macro-F1 and mean IoU
     for each model x dimension.
  5. Prints a comparison table across all models tested and saves a grouped
     bar chart (F1-macro by model, one group per dimension) + an annotated
     Excel file with every model's predictions next to the ground truth.

WHY THESE MODELS
-----------------
"Guard"-style models are purpose-built to classify/moderate content and tend
to be much better at this exact task (structured judgment, not open chat)
than general chat SLMs of the same size:

  - meta-llama/Llama-Guard-3-1B / -8B   -> Meta's safety/moderation classifier
  - ibm-granite/granite-guardian-3.1-2b -> IBM's guardrail/risk classifier
  - Qwen/Qwen3Guard-Gen-4B (or Qwen2.5-1.5B-Instruct as a lighter fallback)
  - microsoft/Phi-3-mini-4k-instruct    -> strong small general instruct model
  - google/gemma-2-2b-it / gemma-3-4b-it-> strong small general instruct model

Guard-family models (Llama-Guard, Granite-Guardian, Qwen-Guard) are trained
for binary/categorical safety classification, not 0-2 quality scoring, so for
THIS task (quality/accuracy/relevance scoring) the general instruct models
(Phi-3-mini, Qwen2.5-Instruct, Gemma-it) usually parse into valid JSON more
reliably. The Guard models are included because the user asked for
moderation-oriented SLMs, and they're worth benchmarking against the
instruct models directly on parse-rate and agreement.

Most of these are GATED on Hugging Face (Llama, Gemma, Granite). Before
running:
    huggingface-cli login
  or export HF_TOKEN=hf_xxx
and make sure you've accepted the model license on huggingface.co for any
gated model you want to use.

USAGE
-----
  # Sanity check on 5 rows with one model, see raw output before trusting it
  python hf_slm_moderation_eval.py --input V1_master_with_labels.xlsx \
      --models qwen1.5b --rows 5 --debug

  # Full run, compare several models
  python hf_slm_moderation_eval.py --input V1_master_with_labels.xlsx \
      --models qwen1.5b,phi3,gemma2b,llamaguard1b --batch 8

  # 4-bit quantization (needed for the 8B Guard models on small GPUs)
  python hf_slm_moderation_eval.py --input V1_master_with_labels.xlsx \
      --models llamaguard8b --quant4

REQUIREMENTS
------------
  pip install transformers torch pandas openpyxl accelerate tqdm scikit-learn matplotlib
  pip install bitsandbytes   # optional, for --quant4 / --quant8
  huggingface-cli login      # required for gated models (Llama, Gemma, Granite)
=================================================================================
"""

import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

# ─── CONFIG ─────────────────────────────────────────────────────────────────

# Small models from Hugging Face Hub known to work well for moderation /
# structured-judgment tasks. Keys are short CLI aliases; values are the
# real HF repo ids. Feel free to add more.
MODEL_REGISTRY = {
    # general small instruct models (best JSON-following / scoring reliability)
    "qwen0.5b":     "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen1.5b":     "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3b":       "Qwen/Qwen2.5-3B-Instruct",
    "phi3":         "microsoft/Phi-3-mini-4k-instruct",
    "gemma2b":      "google/gemma-2-2b-it",
    "gemma4b":      "google/gemma-3-4b-it",
    "tinyllama":    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",

    # moderation / guardrail-specific models (gated, need license acceptance)
    "llamaguard1b": "meta-llama/Llama-Guard-3-1B",
    "llamaguard8b": "meta-llama/Llama-Guard-3-8B",
    "granite2b":    "ibm-granite/granite-guardian-3.1-2b",
    "granite8b":    "ibm-granite/granite-guardian-3.1-8b",
    "qwenguard4b":  "Qwen/Qwen3Guard-Gen-4B",
}

# Columns in the SAME excel file that already carries ground truth
DEFAULT_Q_COL      = "Sanitized Question English Translation"
DEFAULT_ANSMOD_COL = "Answer Model"                        # answer being judged
DEFAULT_REF_COL    = "Relevant answer (translation in English)"  # gold reference
DEFAULT_GT_ACC_COL = "Accuracy"
DEFAULT_GT_REL_COL = "Context and Relevance"

SCORE_MIN, SCORE_MAX = 0, 2
PARSE_FAILURE        = -1
SCORE_SCHEMA         = '{"accuracy":<0-2>,"context_relevance":<0-2>}'

SYSTEM_PROMPT = (
    "You are a strict content-moderation evaluator for a healthcare helpline. "
    "You will be given a caller's question, a virtual moderator's answer, and "
    "a human-approved reference answer. Judge the moderator's answer on two "
    "dimensions, each scored 0 (poor), 1 (partially correct/relevant), or 2 (fully correct/relevant):\n"
    "  - accuracy: is the moderator's answer medically/factually correct relative to the reference?\n"
    "  - context_relevance: does it directly and appropriately address the caller's question?\n"
    f"Return ONLY valid JSON in this exact format: {SCORE_SCHEMA}. "
    "No explanation, no markdown, no extra keys."
)


# ─── MODEL LOADER ───────────────────────────────────────────────────────────

def load_model(model_key: str, quant4: bool = False, quant8: bool = False, hf_token: str = None):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        raise ImportError("Run: pip install transformers torch accelerate")

    model_id = MODEL_REGISTRY.get(model_key, model_key)  # allow raw HF id too
    print(f"⏳ Loading model: {model_id}")

    token = hf_token or os.environ.get("HF_TOKEN")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    print(f"   Device : {device}  |  dtype : {dtype}")

    bnb_config = None
    if quant4 or quant8:
        try:
            from transformers import BitsAndBytesConfig
            if quant4:
                bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                print("   Quantisation : 4-bit")
            else:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                print("   Quantisation : 8-bit")
        except ImportError:
            print("   ⚠️  bitsandbytes not installed — skipping quantisation.")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side    = "left"
    tokenizer.truncation_side = "left"

    load_kwargs = dict(trust_remote_code=True, device_map="auto", token=token)
    if bnb_config:
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()

    print(f"✅ Model ready: {model_id}\n")
    return tokenizer, model, device


# ─── PROMPT BUILDER ─────────────────────────────────────────────────────────

def build_user_message(question: str, answer_model: str, reference: str,
                        max_q: int = 800, max_a: int = 1200, max_r: int = 1200) -> str:
    return (
        f"Return ONLY this JSON: {SCORE_SCHEMA}\n\n"
        f"Caller question: {str(question)[:max_q]}\n\n"
        f"Moderator's answer (to judge): {str(answer_model)[:max_a]}\n\n"
        f"Reference / gold answer: {str(reference)[:max_r]}\n\n"
        "JSON:"
    )


def apply_chat_template(tokenizer, user_msg: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        combined = f"{SYSTEM_PROMPT}\n\n{user_msg}"
        return f"<s>[INST] {combined} [/INST]"


# ─── JSON EXTRACTOR ─────────────────────────────────────────────────────────

def _validate_scores(data: dict) -> dict:
    def _coerce(val):
        if val is None:
            return None
        try:
            v = int(float(val))
            return v if SCORE_MIN <= v <= SCORE_MAX else None
        except (TypeError, ValueError):
            return None
    return {
        "accuracy":          _coerce(data.get("accuracy")),
        "context_relevance": _coerce(data.get("context_relevance") or data.get("relevance")),
    }


def extract_scores(text: str) -> dict:
    text = re.sub(r"```(?:json)?|```", "", text).strip()

    try:
        return _validate_scores(json.loads(text))
    except (json.JSONDecodeError, ValueError):
        pass

    m = re.search(r"\{[^}]*\}", text, re.DOTALL)
    if m:
        try:
            return _validate_scores(json.loads(m.group()))
        except (json.JSONDecodeError, ValueError):
            pass

    acc_m = re.search(r'"?accuracy"?\s*[=:]\s*([012])', text, re.IGNORECASE)
    rel_m = re.search(r'"?(?:context_relevance|relevance)"?\s*[=:]\s*([012])', text, re.IGNORECASE)
    if acc_m or rel_m:
        return {
            "accuracy":          int(acc_m.group(1)) if acc_m else None,
            "context_relevance": int(rel_m.group(1)) if rel_m else None,
        }

    return {"accuracy": None, "context_relevance": None}


# ─── DEBUG HELPER ───────────────────────────────────────────────────────────

def run_debug(tokenizer, model, device, prompts, max_new_tokens, max_length, n_samples=3):
    import torch
    print("\n" + "=" * 70)
    print(f"  DEBUG MODE — showing first {n_samples} raw model outputs")
    print("=" * 70)

    for i, prompt in enumerate(prompts[:n_samples]):
        print(f"\n{'─'*70}\n  SAMPLE {i+1}\n{'─'*70}")
        print(f"\n[PROMPT TAIL — last 300 chars]:\n...{prompt[-300:]}\n")

        enc = tokenizer([prompt], return_tensors="pt", padding=True,
                         truncation=True, max_length=max_length).to(device)
        n_input_tokens = enc["input_ids"].shape[1]
        print(f"[INPUT TOKEN COUNT]: {n_input_tokens}")

        with torch.inference_mode():
            out_ids = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        new_decoded = tokenizer.decode(out_ids[0][n_input_tokens:], skip_special_tokens=True)
        print(f"\n[RAW NEW TOKENS ONLY]:\n{repr(new_decoded)}\n")
        print(f"[PARSED SCORES]: {extract_scores(new_decoded)}")

        if len(new_decoded.strip()) == 0:
            print("⚠️  WARNING: empty output — raise --tokens")
        elif "Caller question:" in new_decoded or "Moderator's answer" in new_decoded:
            print("⚠️  WARNING: model is REPEATING the prompt — chat template may be wrong")
        elif new_decoded.count("{") == 0:
            print("⚠️  WARNING: no '{' in output — model ignored JSON instruction")

    print("\n" + "=" * 70 + "\n  END DEBUG\n" + "=" * 70 + "\n")


# ─── BATCH SCORER ───────────────────────────────────────────────────────────

def score_dataframe(df, tokenizer, model, device,
                     n_rows=None, batch_size=8, max_new_tokens=32, max_length=768,
                     question_col=DEFAULT_Q_COL, answer_col=DEFAULT_ANSMOD_COL,
                     reference_col=DEFAULT_REF_COL, debug=False, debug_samples=3):
    import torch

    sample = (df.head(n_rows).copy().reset_index(drop=True) if n_rows else df.copy().reset_index(drop=True))
    n = len(sample)

    print(f"📝 Building prompts for {n} rows …")
    prompts = [
        apply_chat_template(tokenizer, build_user_message(
            row.get(question_col, ""), row.get(answer_col, ""), row.get(reference_col, ""),
        ))
        for _, row in sample.iterrows()
    ]

    lengths = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    n_over = sum(1 for l in lengths if l > max_length)
    print(f"   Prompt lengths — min:{min(lengths)} max:{max(lengths)} mean:{sum(lengths)/len(lengths):.0f} truncated:{n_over}/{n}")
    if n_over:
        print(f"   ⚠️  {n_over} prompts exceed {max_length} tokens; raise --max-len if this matters.")

    if debug:
        run_debug(tokenizer, model, device, prompts, max_new_tokens, max_length, n_samples=debug_samples)
        ans = input("Continue with full scoring? [y/N]: ").strip().lower()
        if ans != "y":
            print("Exiting. Fix the issues shown above and re-run.")
            return [PARSE_FAILURE] * n, [PARSE_FAILURE] * n

    print(f"\n🚀 Scoring in batches of {batch_size} …\n")
    raw_scores = []
    t0 = time.time()
    n_parsed = 0

    for start in tqdm(range(0, n, batch_size), unit="batch", desc="Scoring"):
        batch_prompts = prompts[start:start + batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                         truncation=True, max_length=max_length).to(device)

        with torch.inference_mode():
            out_ids = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )

        prompt_len = enc["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(out_ids[:, prompt_len:], skip_special_tokens=True)

        for text in decoded:
            scores = extract_scores(text)
            raw_scores.append(scores)
            if scores["accuracy"] is not None:
                n_parsed += 1

    elapsed = time.time() - t0
    parse_rate = 100 * n_parsed / n if n else 0
    print(f"\n⏱  {n} rows in {elapsed:.1f}s  ({n/elapsed:.1f} rows/s)")
    print(f"   Parse success rate: {n_parsed}/{n} ({parse_rate:.1f}%)")
    if parse_rate == 0:
        print("\n❌ 0% parse rate — run with --debug to inspect raw model output.")

    accuracy_vector = [s["accuracy"] if s["accuracy"] is not None else PARSE_FAILURE for s in raw_scores]
    relevance_vector = [s["context_relevance"] if s["context_relevance"] is not None else PARSE_FAILURE for s in raw_scores]
    return accuracy_vector, relevance_vector


# ─── METRICS ────────────────────────────────────────────────────────────────

def mean_iou(y_true, y_pred, labels=(0, 1, 2)):
    """Mean IoU across ordinal classes, computed from a confusion matrix.
    IoU_c = TP_c / (TP_c + FP_c + FN_c). Rows where prediction failed to
    parse should already be excluded before calling this."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    ious = []
    for i in range(len(labels)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = tp + fp + fn
        if denom > 0:
            ious.append(tp / denom)
    return float(np.mean(ious)) if ious else 0.0


def compute_metrics(y_true, y_pred):
    """Excludes rows where the model failed to parse (PARSE_FAILURE) from
    the metric computation, but reports how many were dropped."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid = y_pred != PARSE_FAILURE
    n_dropped = int((~valid).sum())

    if valid.sum() == 0:
        return {"accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0,
                "f1_macro": 0.0, "miou": 0.0, "n_dropped": n_dropped, "n_scored": 0}

    yt, yp = y_true[valid], y_pred[valid]
    acc = accuracy_score(yt, yp)
    precision, recall, f1, _ = precision_recall_fscore_support(
        yt, yp, labels=[0, 1, 2], average="macro", zero_division=0
    )
    iou = mean_iou(yt, yp, labels=(0, 1, 2))

    return {
        "accuracy": acc, "precision_macro": precision, "recall_macro": recall,
        "f1_macro": f1, "miou": iou, "n_dropped": n_dropped, "n_scored": int(valid.sum()),
    }


def print_metrics_table(all_results: dict):
    """all_results: {model_name: {"accuracy": {...metrics}, "context_relevance": {...metrics}}}"""
    print("\n" + "=" * 100)
    print("  MODEL COMPARISON — moderation scoring vs. human ground truth")
    print("=" * 100)
    header = f"{'Model':<16}{'Dim':<20}{'Acc':>8}{'Prec(M)':>10}{'Rec(M)':>10}{'F1(M)':>10}{'mIoU':>10}{'Dropped':>10}"
    print(header)
    print("-" * 100)
    for model_name, dims in all_results.items():
        for dim_name, m in dims.items():
            print(f"{model_name:<16}{dim_name:<20}{m['accuracy']*100:>7.1f}%{m['precision_macro']*100:>9.1f}%"
                  f"{m['recall_macro']*100:>9.1f}%{m['f1_macro']*100:>9.1f}%{m['miou']*100:>9.1f}%{m['n_dropped']:>10}")
    print("=" * 100)


def plot_f1_comparison(all_results: dict, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(all_results.keys())
    dims = list(next(iter(all_results.values())).keys()) if models else []
    x = np.arange(len(models))
    width = 0.8 / max(len(dims), 1)

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.6), 5))
    for i, dim in enumerate(dims):
        f1s = [all_results[m][dim]["f1_macro"] * 100 for m in models]
        ax.bar(x + i * width, f1s, width, label=dim)

    ax.set_ylabel("F1 (macro) %")
    ax.set_title("F1-macro by model and evaluation dimension")
    ax.set_xticks(x + width * (len(dims) - 1) / 2)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n📊 Saved comparison chart → {out_path}")


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Score moderator answers with HF small LLMs and benchmark against human ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Excel file containing data AND ground truth columns")
    parser.add_argument("--models", default="qwen1.5b", help="Comma-separated list of registry keys or raw HF ids")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=768)
    parser.add_argument("--quant4", action="store_true")
    parser.add_argument("--quant8", action="store_true")
    parser.add_argument("--hf-token", default=None, help="Overrides HF_TOKEN env var, needed for gated models")
    parser.add_argument("--q-col", default=DEFAULT_Q_COL)
    parser.add_argument("--a-col", default=DEFAULT_ANSMOD_COL)
    parser.add_argument("--r-col", default=DEFAULT_REF_COL)
    parser.add_argument("--gt-acc-col", default=DEFAULT_GT_ACC_COL)
    parser.add_argument("--gt-rel-col", default=DEFAULT_GT_REL_COL)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-samples", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_excel(args.input)
    print(f"✅ Loaded {len(df)} rows from {args.input}")

    n_rows = args.rows or len(df)
    sample = df.head(n_rows).copy().reset_index(drop=True)
    gt_accuracy = sample[args.gt_acc_col].astype(int).tolist()
    gt_relevance = sample[args.gt_rel_col].astype(int).tolist()

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    all_results = {}
    annotated = sample.copy()

    for model_key in model_keys:
        print(f"\n{'#'*70}\n#  MODEL: {model_key}\n{'#'*70}")
        tokenizer, model, device = load_model(
            model_key, quant4=args.quant4, quant8=args.quant8, hf_token=args.hf_token
        )

        accuracy_vector, relevance_vector = score_dataframe(
            sample, tokenizer, model, device,
            n_rows=None,  # sample already truncated to n_rows
            batch_size=args.batch, max_new_tokens=args.tokens, max_length=args.max_len,
            question_col=args.q_col, answer_col=args.a_col, reference_col=args.r_col,
            debug=args.debug, debug_samples=args.debug_samples,
        )

        annotated[f"{model_key}_accuracy_pred"] = accuracy_vector
        annotated[f"{model_key}_context_relevance_pred"] = relevance_vector

        acc_metrics = compute_metrics(gt_accuracy, accuracy_vector)
        rel_metrics = compute_metrics(gt_relevance, relevance_vector)
        all_results[model_key] = {"accuracy": acc_metrics, "context_relevance": rel_metrics}

        # free GPU memory before loading the next model
        del model, tokenizer
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    print_metrics_table(all_results)

    chart_path = os.path.join(args.output_dir, "f1_comparison.png")
    plot_f1_comparison(all_results, chart_path)

    out_xlsx = os.path.join(args.output_dir, "moderation_eval_annotated.xlsx")
    annotated.to_excel(out_xlsx, index=False)
    print(f"💾 Saved annotated file (all models + ground truth) → {out_xlsx}")

    out_json = os.path.join(args.output_dir, "moderation_eval_metrics.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"💾 Saved metrics JSON → {out_json}")


if __name__ == "__main__":
    main()
