"""
=============================================================
  Small LLM Scorer for Accuracy & Relevance  (v6 — f1_macro eval)
=============================================================

The model is shown ONLY the question and the model's answer. It never sees
the ground-truth "Relevant answer" or the human-assigned "Accuracy " /
"Context & Relevance " labels — those live in the SAME input dataset and
are used purely afterwards to score the model's predictions.

USAGE:
  # Always run --debug first on a small sample to see raw model output
  python llm_scorer_gemma.py --model gemma4b --input data.xlsx --rows 5 --debug

  # Once output looks correct, score everything and evaluate against the
  # ground truth columns already present in data.xlsx
  python llm_scorer_gemma.py --model gemma4b --input data.xlsx --rows 100 --batch 32

  # 4-bit quantization on low-VRAM GPU
  python llm_scorer_gemma.py --model gemma4b --input data.xlsx --quant4

  # Custom column names, if your sheet differs from the defaults
  python llm_scorer_gemma.py --model gemma4b --input data.xlsx \
      --q-col "Sanitized Question" --a-col "Answer Transcription" \
      --gt-acc-col "Accuracy " --gt-rel-col "Context & Relevance "

RETURN VALUE (score_dataframe):
  Returns (accuracy_vector, relevance_vector) — two plain Python lists of
  integers, one entry per dataset row, in the same order as the input.
  Scores are 0 (worst) … 2 (best).  Parse failures are stored as -1.

EVALUATION:
  After scoring, accuracy_vector / relevance_vector are compared against
  the dataset's own "Accuracy " and "Context & Relevance " columns
  (ground truth). The headline metric reported is f1_macro
  (sklearn.metrics.f1_score, average="macro") for each of the two tasks,
  plus exact-agreement % and a full per-class classification report.

REQUIREMENTS:
  pip install transformers torch pandas openpyxl accelerate tqdm scikit-learn
  pip install bitsandbytes   # optional, for --quant4 / --quant8
"""

import argparse
import json
import re
import time

import pandas as pd
from tqdm import tqdm

# ─── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "gemma":     "google/gemma-3-1b-it",
    "gemma4b":   "google/gemma-3-4b-it",
    "phi3":      "microsoft/Phi-3-mini-4k-instruct",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "qwen":      "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen1.5b":  "Qwen/Qwen2.5-1.5B-Instruct",
    "mistral":   "mistralai/Mistral-7B-Instruct-v0.3",

    "llama" : "meta-llama/Llama-Guard-3-8B",
    "llama1b" : "meta-llama/Llama-Guard-3-1B",
    "llanav2" : "Llama-Nematron-Safety-Guard-V2",
    "llamav3" : "nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3",
    "granite" : "ibm-granite/granite-guardian-3.2-5",
    "qwen3" : "Qwen/Qwen3-Guard-Stream-8B",








}

DEFAULT_Q_COL   = "Sanitized Question"     # input to the model
DEFAULT_ANS_COL = "Answer Transcription"   # input to the model

# Ground-truth label columns — ALREADY PRESENT in the input dataset.
# These are used only for evaluation after inference. They are never
# passed to the model (the model only ever sees DEFAULT_Q_COL / DEFAULT_ANS_COL).
DEFAULT_GT_ACC_COL = "Accuracy "
DEFAULT_GT_REL_COL = "Context & Relevance "

SCORE_MIN, SCORE_MAX = 0, 2
PARSE_FAILURE        = -1                        # sentinel for failed parse rows
SCORE_SCHEMA         = '{"accuracy":<0-2>,"relevance":<0-2>}'

SYSTEM_PROMPT = (
    "You are a strict evaluator. "
    "You will be given a QUESTION and an ANSWER given by a person. "
    "Judge the answer on its own merits, using only your general "
    "knowledge — you are NOT given and must NOT assume any reference or "
    "ground-truth answer. "
    "Score 'accuracy' (is the answer factually correct / well-formed?) and "
    "'relevance' (does the answer actually address the question?) each on "
    "a 0-2 scale (0=worst, 2=best), then "
    f"return ONLY valid JSON in this exact format: {SCORE_SCHEMA}. "
    "No explanation, no markdown, no extra keys."
)


# ─── MODEL LOADER ──────────────────────────────────────────────────────────────

def load_model(model_key: str, quant4: bool = False, quant8: bool = False):
    try:
        import torch
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        raise ImportError("Run: pip install transformers torch accelerate")

    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        AutoModelForImageTextToText = None  # older transformers — see warning below

    try:
        from transformers import Gemma3ForConditionalGeneration
    except ImportError:
        Gemma3ForConditionalGeneration = None  # older transformers, no Gemma-3 support yet

    model_id = MODEL_REGISTRY.get(model_key, model_key)
    print(f"⏳ Loading model: {model_id}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Gemma (and most modern instruct models) are trained in bfloat16. Running
    # them in float16 is a well-known source of NaN activations — Gemma's
    # attention uses logit soft-capping with values that overflow fp16's
    # ~65504 max range but fit comfortably in bf16's much wider range. Once
    # activations go NaN, generation degenerates to picking the same token
    # (often id 0 / <pad>) at every single position, regardless of prompt —
    # exactly the symptom of "always <pad>, even on trivial input".
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"   Device : {device}  |  dtype : {dtype}")

    # ── Detect multimodal (vision+text) checkpoints, e.g. Gemma-3 4B/12B/27B ──
    # Their native class is *ForConditionalGeneration, not a plain causal LM.
    # Loading these with AutoModelForCausalLM does NOT raise an error — it
    # silently fails to map the nested `language_model.*` weight keys, so
    # key layers (often the embeddings/LM head) end up randomly initialized.
    # Symptom: generation collapses to one repeated token (often <pad>) for
    # every position, since the weights were never actually loaded.
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    architectures = getattr(config, "architectures", []) or []
    is_multimodal = any(
        ("ConditionalGeneration" in a) or ("ImageTextToText" in a)
        for a in architectures
    )

    if is_multimodal and "Gemma3ForConditionalGeneration" in architectures and Gemma3ForConditionalGeneration is not None:
        # Exact class for the checkpoint's declared architecture — the most
        # reliable option when we know for certain it's a Gemma-3 model.
        print(f"   Detected {architectures} → loading with Gemma3ForConditionalGeneration "
              f"directly (text-only use is fine, just don't pass an image).")
        model_cls = Gemma3ForConditionalGeneration
    elif is_multimodal and AutoModelForImageTextToText is not None:
        # Generic fallback — covers any other *ForConditionalGeneration /
        # *ImageTextToText checkpoint that might get added to the registry.
        print(f"   Detected multimodal architecture {architectures} → "
              f"loading with AutoModelForImageTextToText (text-only use is fine, "
              f"just don't pass an image).")
        model_cls = AutoModelForImageTextToText
    elif is_multimodal:
        print(f"   ⚠️  Detected multimodal architecture {architectures}, but this "
              f"transformers version can't load it correctly (no matching class "
              f"available). Run: pip install -U transformers  — otherwise this "
              f"model's weights will NOT load correctly and generation will be garbage.")
        model_cls = AutoModelForCausalLM
    else:
        model_cls = AutoModelForCausalLM

    bnb_config = None
    if quant4 or quant8:
        try:
            from transformers import BitsAndBytesConfig
            if quant4:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
                )
                print("   Quantisation : 4-bit")
            else:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                print("   Quantisation : 8-bit")
        except ImportError:
            print("   ⚠️  bitsandbytes not installed — skipping quantisation.")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side    = "left"
    tokenizer.truncation_side = "left"

    load_kwargs = dict(trust_remote_code=True, device_map="auto")
    if bnb_config:
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["dtype"] = dtype

    model = model_cls.from_pretrained(model_id, **load_kwargs)
    model.eval()

    print(f"✅ Model ready: {model_id}\n")

    _self_test(tokenizer, model, device)

    return tokenizer, model, device


def _self_test(tokenizer, model, device):
    """
    Quick post-load sanity check: generate a couple of tokens for a trivial,
    neutral prompt. If the model collapses to a single repeated special
    token (the classic sign of a weight-loading / architecture mismatch,
    see load_model() above), warn loudly right away instead of letting the
    user burn a full --debug run on their real data before noticing.
    """
    import torch

    try:
        msg = [{"role": "user", "content": "Reply with exactly: OK"}]
        prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = tokenizer([prompt], return_tensors="pt", add_special_tokens=False).to(device)
        with torch.inference_mode():
            out = model.generate(
                **enc, max_new_tokens=8, min_new_tokens=2, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_ids = out[0][enc["input_ids"].shape[1]:]
        decoded_clean = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        decoded_raw   = tokenizer.decode(new_ids, skip_special_tokens=False)

        if len(decoded_clean) == 0:
            print("⚠️  SELF-TEST FAILED: model produced no real tokens for a trivial prompt.")
            print(f"   Raw output: {decoded_raw!r}")
            print("   This usually means the model weights loaded incorrectly (wrong "
                  "model class for this checkpoint) rather than a content issue — "
                  "double-check the warnings printed above.")
        else:
            print(f"   Self-test OK — model responded: {decoded_clean!r}\n")
    except Exception as e:
        print(f"   ⚠️  Self-test skipped (error: {e})\n")


# ─── PROMPT BUILDER ────────────────────────────────────────────────────────────

def build_user_message(question: str, answer: str,
                       max_q: int = 1500, max_a: int = 1500) -> str:
    """Builds the prompt from ONLY the question + answer — no reference/ground truth."""
    return (
        f"Return ONLY this JSON: {SCORE_SCHEMA}\n\n"
        f"Question: {question[:max_q]}\n\n"
        f"Answer: {answer[:max_a]}\n\n"
        "JSON:"
    )


def apply_chat_template(tokenizer, user_msg: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        combined = f"{SYSTEM_PROMPT}\n\n{user_msg}"
        return f"<s>[INST] {combined} [/INST]"


# ─── JSON EXTRACTOR ────────────────────────────────────────────────────────────

def extract_scores(text: str) -> dict:
    """
    Parse a model response and return {"accuracy": int|None, "relevance": int|None}.
    None means the field could not be extracted.
    """
    text = re.sub(r"```(?:json)?|```", "", text).strip()

    # 1. Direct JSON parse
    try:
        return _validate_scores(json.loads(text))
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. First {...} block
    m = re.search(r"\{[^}]*\}", text, re.DOTALL)
    if m:
        try:
            return _validate_scores(json.loads(m.group()))
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Loose key:value regex
    acc_m = re.search(r'"?accuracy"?\s*[=:]\s*([012])', text, re.IGNORECASE)
    rel_m = re.search(r'"?relevance"?\s*[=:]\s*([012])', text, re.IGNORECASE)
    if acc_m or rel_m:
        return {
            "accuracy":  int(acc_m.group(1)) if acc_m else None,
            "relevance": int(rel_m.group(1)) if rel_m else None,
        }

    return {"accuracy": None, "relevance": None}


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
        "accuracy":  _coerce(data.get("accuracy")),
        "relevance": _coerce(data.get("relevance")),
    }


# ─── DEBUG HELPER ──────────────────────────────────────────────────────────────

def run_debug(tokenizer, model, device, prompts: list, max_new_tokens: int,
              max_length: int, n_samples: int = 3, min_new_tokens: int = 6):
    import torch

    print("\n" + "=" * 70)
    print(f"  DEBUG MODE — showing first {n_samples} raw model outputs")
    print("=" * 70)

    for i, prompt in enumerate(prompts[:n_samples]):
        print(f"\n{'─'*70}")
        print(f"  SAMPLE {i+1}")
        print(f"{'─'*70}")

        print(f"\n[PROMPT TAIL — last 300 chars]:\n...{prompt[-300:]}\n")

        enc = tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,  # prompt already has <bos>/<start_of_turn>/etc.
                                        # from apply_chat_template — adding them again
                                        # double-BOS's the input and breaks Gemma generation
        ).to(device)

        n_input_tokens = enc["input_ids"].shape[1]
        print(f"[INPUT TOKEN COUNT]: {n_input_tokens}")

        with torch.inference_mode():
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,  # forbid emitting EOS/<end_of_turn> as
                                                 # the very first token — see note below
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_ids     = out_ids[0][n_input_tokens:]
        new_decoded = tokenizer.decode(new_ids, skip_special_tokens=True)
        raw_with_specials = tokenizer.decode(new_ids, skip_special_tokens=False)

        print(f"\n[RAW NEW TOKENS ONLY]:\n{repr(new_decoded)}\n")
        print(f"[RAW INCLUDING SPECIAL TOKENS]:\n{repr(raw_with_specials)}\n")
        print(f"[PARSED SCORES]: {extract_scores(new_decoded)}")

        if len(new_decoded.strip()) == 0:
            print("⚠️  WARNING: model generated EMPTY output after stripping special tokens.")
            print("   Check '[RAW INCLUDING SPECIAL TOKENS]' above:")
            print("   - If it's just <end_of_turn>/<eos> repeated, the model is emitting its")
            print("     stop token immediately — this often means the model is REFUSING")
            print("     (silently) rather than running out of budget. Common on Gemma for")
            print("     sexual-health / reproductive-health content, even in an evaluator role.")
            print("     Try: a less safety-aggressive model (--model qwen1.5b / mistral),")
            print("     or raise --min-tokens further to force real content out.")
        elif "Question:" in new_decoded or "Answer:" in new_decoded:
            print("⚠️  WARNING: model is REPEATING the prompt — chat template may be wrong")
        elif new_decoded.count("{") == 0:
            print("⚠️  WARNING: no '{' in output — model ignored JSON instruction")

    print("\n" + "=" * 70)
    print("  END DEBUG — fix any warnings above before running full scoring")
    print("=" * 70 + "\n")


# ─── BATCH SCORER ──────────────────────────────────────────────────────────────

def score_dataframe(
    df: pd.DataFrame,
    tokenizer,
    model,
    device: str,
    n_rows: int         = None,
    batch_size: int     = 16,
    max_new_tokens: int = 32,
    max_length: int     = 512,
    question_col: str   = DEFAULT_Q_COL,
    answer_col: str     = DEFAULT_ANS_COL,
    debug: bool         = False,
    debug_samples: int  = 3,
    min_new_tokens: int = 6,
) -> tuple[list[int], list[int]]:
    """
    Score every row in *df* and return two integer vectors:

        accuracy_vector, relevance_vector = score_dataframe(df, ...)

    Each vector has one entry per scored row, in dataset order.
    Valid scores are 0, 1, 2.  Parse failures are stored as -1.

    IMPORTANT: only `question_col` and `answer_col` are ever read here and
    fed to the model. No ground-truth / reference-answer column is used
    for inference — those are only consulted later, during evaluation.
    """
    import torch

    sample = (df.head(n_rows).copy().reset_index(drop=True)
              if n_rows else df.copy().reset_index(drop=True))
    n = len(sample)

    print(f"📝 Building prompts for {n} rows …")
    prompts = [
        apply_chat_template(
            tokenizer,
            build_user_message(
                str(row.get(question_col, "")),
                str(row.get(answer_col, "")),
            ),
        )
        for _, row in sample.iterrows()
    ]

    # Prompt length diagnostic
    lengths = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    n_over  = sum(1 for l in lengths if l > max_length)
    print(f"   Prompt lengths — min:{min(lengths)}  max:{max(lengths)}  "
          f"mean:{sum(lengths)/len(lengths):.0f}  truncated:{n_over}/{n}")
    if n_over:
        print(f"   ⚠️  {n_over} prompts exceed {max_length} tokens. "
              f"Truncating from LEFT. Use --max-len to raise the limit.")

    # ── Debug mode ──────────────────────────────────────────────────────────
    if debug:
        run_debug(tokenizer, model, device, prompts, max_new_tokens,
                  max_length, n_samples=debug_samples, min_new_tokens=min_new_tokens)
        answer = input("Continue with full scoring? [y/N]: ").strip().lower()
        if answer != "y":
            print("Exiting. Fix the issues shown above and re-run.")
            # Return PARSE_FAILURE vectors matching the sample length
            return [PARSE_FAILURE] * n, [PARSE_FAILURE] * n

    print(f"\n🚀 Scoring in batches of {batch_size} …\n")

    # Collect raw parsed dicts first
    raw_scores: list[dict] = []
    t0 = time.time()
    n_parsed = 0

    for start in tqdm(range(0, n, batch_size), unit="batch", desc="Scoring"):
        batch_prompts = prompts[start : start + batch_size]

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,  # see note in run_debug() — avoid double-BOS
        ).to(device)

        with torch.inference_mode():
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_len = enc["input_ids"].shape[1]
        decoded    = tokenizer.batch_decode(
            out_ids[:, prompt_len:], skip_special_tokens=True
        )

        for text in decoded:
            scores = extract_scores(text)
            raw_scores.append(scores)
            if scores["accuracy"] is not None:
                n_parsed += 1

    elapsed    = time.time() - t0
    parse_rate = 100 * n_parsed / n
    print(f"\n⏱  {n} rows in {elapsed:.1f}s  ({n/elapsed:.1f} rows/s)")
    print(f"   Parse success rate: {n_parsed}/{n}  ({parse_rate:.1f}%)")

    if parse_rate == 0:
        print("\n❌ 0% parse rate — run with --debug to inspect raw model output.")
        print("   Common causes:")
        print("   1. Model repeating the prompt  → chat template issue")
        print("   2. Model outputting empty text → raise --tokens")
        print("   3. Model ignoring JSON format  → try --model qwen1.5b instead")

    # ── Build the two integer vectors ───────────────────────────────────────
    # None (parse failure) is stored as PARSE_FAILURE (-1) so both vectors
    # stay pure-integer lists with one entry per row.
    accuracy_vector  = [
        s["accuracy"]  if s["accuracy"]  is not None else PARSE_FAILURE
        for s in raw_scores
    ]
    relevance_vector = [
        s["relevance"] if s["relevance"] is not None else PARSE_FAILURE
        for s in raw_scores
    ]

    return accuracy_vector, relevance_vector





# ─── VECTOR PRINTER ────────────────────────────────────────────────────────────

def print_vectors(accuracy_vector: list[int], relevance_vector: list[int]):
    """Print both score vectors and a concise per-value breakdown."""
    n         = len(accuracy_vector)
    failures  = sum(1 for v in accuracy_vector if v == PARSE_FAILURE)

    print("\n" + "=" * 60)
    print("  SCORE VECTORS  (one integer per dataset row, in order)")
    print("=" * 60)
    print(f"\n  accuracy_vector  ({n} entries):")
    print(f"  {accuracy_vector}")
    print(f"\n  relevance_vector ({n} entries):")
    print(f"  {relevance_vector}")

    print("\n" + "-" * 60)
    print("  VALUE BREAKDOWN")
    print("-" * 60)
    for label, vec in [("Accuracy ", accuracy_vector), ("Relevance", relevance_vector)]:
        valid = [v for v in vec if v != PARSE_FAILURE]
        counts = {s: valid.count(s) for s in (0, 1, 2)}
        print(f"  {label}  |  "
              f"0:{counts[0]:>4}  1:{counts[1]:>4}  2:{counts[2]:>4}  "
              f"(parse failures: {n - len(valid)})")
    print("=" * 60)
    if failures:
        print(f"\n  ⚠️  {failures} row(s) could not be parsed — stored as {PARSE_FAILURE}.")
        print("     Re-run with --debug to inspect raw model output for those rows.")


# ─── EVALUATION (vs. ground truth already present in the dataset) ─────────────

def _score_one_task(name: str, y_true_raw, y_pred_raw, labels: list[int]) -> float:
    """
    Compares one prediction vector against its ground-truth column.
    Prints exact-agreement %, a classification report, and returns f1_macro.

    Parse failures (PARSE_FAILURE / -1) in y_pred are kept as-is: since -1
    isn't one of `labels`, they simply never count as a correct prediction
    for any class — i.e. they are scored as wrong, not silently dropped.
    Rows with a missing/unparseable ground-truth value ARE dropped (can't
    score against a label that doesn't exist).
    """
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    y_true = pd.to_numeric(pd.Series(y_true_raw), errors="coerce").reset_index(drop=True)
    y_pred = pd.Series(y_pred_raw).reset_index(drop=True)

    valid = y_true.notna()
    n_dropped = int((~valid).sum())
    y_true = y_true[valid].astype(int)
    y_pred = y_pred[valid].astype(int)

    print(f"\n{'-'*60}")
    print(f"  {name}")
    print(f"{'-'*60}")
    if n_dropped:
        print(f"  ⚠️  {n_dropped} row(s) skipped — missing/non-numeric ground truth")
    if len(y_true) == 0:
        print("  ❌ No valid rows to evaluate.")
        return float("nan")

    exact_agreement = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    print(f"  Rows evaluated   : {len(y_true)}")
    print(f"  Exact agreement  : {exact_agreement * 100:.2f}%")
    print(f"  f1_macro         : {f1_macro:.4f}")
    print(f"\n  Per-class report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    return f1_macro


def evaluate_predictions(
    sample: pd.DataFrame,
    accuracy_vector: list[int],
    relevance_vector: list[int],
    gt_acc_col: str = DEFAULT_GT_ACC_COL,
    gt_rel_col: str = DEFAULT_GT_REL_COL,
) -> dict:
    """
    Evaluates model predictions against the ground truth that already lives
    in `sample` (columns `gt_acc_col` / `gt_rel_col`). Returns
    {"f1_macro_accuracy": float, "f1_macro_relevance": float}.
    """
    labels = list(range(SCORE_MIN, SCORE_MAX + 1))  # [0, 1, 2]

    print("\n" + "=" * 60)
    print("  EVALUATION vs. GROUND TRUTH (already in the dataset)")
    print("=" * 60)

    f1_acc = _score_one_task("Accuracy", sample[gt_acc_col], accuracy_vector, labels)
    f1_rel = _score_one_task("Context & Relevance", sample[gt_rel_col], relevance_vector, labels)

    print("\n" + "=" * 60)
    print(f"  f1_macro (Accuracy)             : {f1_acc:.4f}")
    print(f"  f1_macro (Context & Relevance)  : {f1_rel:.4f}")
    print("=" * 60)

    return {"f1_macro_accuracy": f1_acc, "f1_macro_relevance": f1_rel}


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Score accuracy & relevance with a small LLM — returns two integer vectors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model",   default="gemma",
                        choices=list(MODEL_REGISTRY.keys()) + ["demo"])
    parser.add_argument("--input",   required=True,
                        help="Dataset file (.xlsx/.xls/.csv). Must contain the "
                             "question/answer columns AND the ground-truth "
                             "'Accuracy '/'Context & Relevance ' columns.")
    parser.add_argument("--output",  default=None)
    parser.add_argument("--rows",    type=int, default=None)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--tokens",  type=int, default=32,
                        help="Max new tokens per row (default 32)")
    parser.add_argument("--min-tokens", type=int, default=6,
                        help="Min new tokens before the model is allowed to stop "
                             "(default 6). Prevents an instant <end_of_turn>/EOS — "
                             "the usual signature of a silent safety refusal — from "
                             "producing empty output.")
    parser.add_argument("--max-len", type=int, default=512,
                        help="Max prompt token length before truncation (default 512)")
    parser.add_argument("--quant4",  action="store_true")
    parser.add_argument("--quant8",  action="store_true")
    parser.add_argument("--q-col",   default=DEFAULT_Q_COL,
                        help="Column with the question (fed to the model)")
    parser.add_argument("--a-col",   default=DEFAULT_ANS_COL,
                        help="Column with the answer (fed to the model)")
    parser.add_argument("--gt-acc-col", default=DEFAULT_GT_ACC_COL,
                        help="Ground-truth accuracy column, already in the dataset "
                             "(used only for evaluation, NEVER shown to the model)")
    parser.add_argument("--gt-rel-col", default=DEFAULT_GT_REL_COL,
                        help="Ground-truth relevance column, already in the dataset "
                             "(used only for evaluation, NEVER shown to the model)")
    parser.add_argument("--debug",         action="store_true",
                        help="Print raw model output for first N rows before full scoring")
    parser.add_argument("--debug-samples", type=int, default=3,
                        help="Number of samples to show in debug mode (default 3)")
    args = parser.parse_args()

    # ── Load dataset (ground truth already lives in this same file) ────────
    if args.input.lower().endswith(".csv"):
        df = pd.read_csv(args.input)
    else:
        df = pd.read_excel(args.input)
    print(f"✅ Loaded {len(df)} rows from {args.input}")

    required_cols = [args.q_col, args.a_col, args.gt_acc_col, args.gt_rel_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(
            f"❌ Missing column(s) {missing} in {args.input}.\n"
            f"   Available columns: {list(df.columns)}"
        )

    tokenizer, model, device = load_model(
        args.model, quant4=args.quant4, quant8=args.quant8
    )

    # NOTE: only question_col / answer_col are passed here — the model never
    # sees the ground-truth "Accuracy "/"Context & Relevance " labels or the
    # "Relevant answer" reference text.
    accuracy_vector, relevance_vector = score_dataframe(
        df, tokenizer, model, device,
        n_rows=args.rows,
        batch_size=args.batch,
        max_new_tokens=args.tokens,
        max_length=args.max_len,
        question_col=args.q_col,
        answer_col=args.a_col,
        debug=args.debug,
        debug_samples=args.debug_samples,
        min_new_tokens=args.min_tokens,
    )

    print_vectors(accuracy_vector, relevance_vector)

    # ── Save annotated file ─────────────────────────────────────────────────
    sample = (df.head(args.rows).copy().reset_index(drop=True)
              if args.rows else df.copy().reset_index(drop=True))
    sample["accuracy_score"]  = accuracy_vector
    sample["relevance_score"] = relevance_vector

    base = re.sub(r"\.(xlsx|xls|csv)$", "", args.input, flags=re.IGNORECASE)
    out_path = args.output or f"{base}_scored_{args.model}.xlsx"
    sample.to_excel(out_path, index=False)
    print(f"\n💾 Saved annotated file → {out_path}")

    # ── Evaluation vs. ground truth ALREADY in the dataset (f1_macro) ──────
    evaluate_predictions(
        sample,
        accuracy_vector=accuracy_vector,
        relevance_vector=relevance_vector,
        gt_acc_col=args.gt_acc_col,
        gt_rel_col=args.gt_rel_col,
    )

if __name__ == "__main__":
    main()
