"""Generate provisional minority examples only. This script never trains a model."""
import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path

import pandas as pd


def norm(value):
    return re.sub(r'\s+', ' ', str(value)).strip().casefold()


def save_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def parse_example(raw, label, context, forbidden_questions, forbidden_pairs):
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip())
    obj = json.loads(text)
    for k in ('question', 'answer', 'accuracy_rationale', 'context_rationale'):
        if not isinstance(obj.get(k), str) or not obj[k].strip():
            raise ValueError('Missing text: ' + k)
    for k, expected in [('accuracy', label), ('context_relevance', context)]:
        if type(obj.get(k)) is not int or obj[k] != expected:
            raise ValueError('Requested label mismatch: ' + k)
    if len(obj['answer'].split()) < 5:
        raise ValueError('Answer too short')
    if norm(obj['question']) in forbidden_questions:
        raise ValueError('Validation question overlap')
    pair = (norm(obj['question']), norm(obj['answer']))
    if pair in forbidden_pairs:
        raise ValueError('Duplicate/overlapping question-answer pair')
    if norm(obj['answer']) in {'i cannot help with that', 'मैं मदद नहीं कर सकता'}:
        raise ValueError('Refusal')
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--model', default='Qwen/Qwen2.5-32B-Instruct')
    parser.add_argument('--loader', choices=['auto', 'gemma3'], default='auto')
    parser.add_argument('--local-files-only', action='store_true', help='Require cached weights; otherwise allow model download')
    parser.add_argument('--revision', help='Optional immutable model revision')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--limit', type=int, help='Optional accepted-row limit for a generation smoke check')
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    import torch
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              Gemma3ForConditionalGeneration, BitsAndBytesConfig, set_seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for generation; no training is performed')
    # The known RTX A4000 has 16 GB. 32B at 4-bit needs more than that once
    # quantization metadata, nonquantized weights, activations and KV cache are included.
    free_bytes, _ = torch.cuda.mem_get_info()
    if '32b' in args.model.lower() and free_bytes < 20 * 1024**3:
        raise RuntimeError('32B generation needs a larger free GPU memory budget (at least 20 GiB '
                           'for this loader; runtime fit still needs a smoke check). Use a 24 GB+ GPU '
                           'or explicitly select --model Qwen/Qwen2.5-14B-Instruct on the 16 GB A4000. '
                           'No generation or training was started.')
    d = args.data_dir
    base = pd.read_csv(d / 'original_eligible.csv', dtype={'record_id': str})
    sources = pd.read_csv(d / 'generation_sources.csv', dtype={'record_id': str})
    validation = pd.read_csv(d / 'validation_exclusion.csv')
    majority = int(base.accuracy.value_counts()[2])
    targets = {c: majority - int((base.accuracy == c).sum()) for c in (0, 1)}
    forbidden_questions = set(validation.question.map(norm))
    forbidden_pairs = set(zip(base.question.map(norm), base.answer.map(norm)))
    forbidden_pairs.update(zip(validation.question.map(norm), validation.answer.map(norm)))
    accepted_path = d / 'accepted.jsonl'
    rows = [json.loads(x) for x in accepted_path.read_text().splitlines()] if accepted_path.exists() else []
    if not rows and (d / 'draft_review.csv').exists():
        for row in pd.read_csv(d / 'draft_review.csv').to_dict('records'):
            rows.append(dict(row, origin='synthetic', label_status='user_reviewed_draft',
                             author='Codex assistant', independent_label_review=False))
        with accepted_path.open('w') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    for row in rows:
        pair = (norm(row['question']), norm(row['answer']))
        if pair in forbidden_pairs or norm(row['question']) in forbidden_questions:
            raise ValueError('Existing accepted row violates exclusion/uniqueness')
        forbidden_pairs.add(pair)
    counts = {c: sum(row['accuracy'] == c for row in rows) for c in (0, 1)}
    if any(counts[c] > targets[c] for c in counts):
        raise ValueError('Accepted examples exceed target')
    progress_path = d / 'generation_status.json'
    previous = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    existing_authors = {r.get('author') for r in rows if r.get('label_status') == 'provisional_model_authored'}
    if existing_authors - {args.model}:
        raise ValueError('Cannot mix generators in a resumed dataset; use a separate data directory')
    if previous.get('generator') and previous['generator'] != args.model and previous.get('attempted', 0):
        raise ValueError('Cannot change generator after attempts; use a separate data directory')
    attempted = previous.get('attempted', 0)
    started = time.time()
    metadata = {'status': 'loading_generator', 'synthetic_targets': targets,
                'accepted_counts': counts, 'training_started': False,
                'generator': args.model, 'label_status': 'provisional_model_authored',
                'loader': args.loader, 'requested_revision': args.revision, 'quantization': 'bitsandbytes_nf4_4bit',
                'independent_label_review': False, 'attempted': attempted,
                'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [d/'original_eligible.csv', d/'generation_sources.csv',
                                            d/'validation_exclusion.csv']}}
    save_json(progress_path, metadata)
    set_seed(args.seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                             local_files_only=args.local_files_only, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    klass = Gemma3ForConditionalGeneration if args.loader == 'gemma3' else AutoModelForCausalLM
    model = klass.from_pretrained(
        args.model, revision=args.revision, local_files_only=args.local_files_only,
        torch_dtype=dtype, device_map={'': torch.cuda.current_device()},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
    model.eval()
    metadata['resolved_revision'] = getattr(model.config, '_commit_hash', None)
    metadata['gpu'] = torch.cuda.get_device_name()
    metadata['status'] = 'generating'
    save_json(progress_path, metadata)
    modes = ['false explanation', 'incorrect universal generalization', 'contradictory advice',
             'confused mechanism', 'mistaken definition', 'unsupported certainty']
    empty_batches = 0
    while any(counts[c] < targets[c] for c in (0, 1)):
        if args.limit and len(rows) >= args.limit:
            break
        slots, prompts = [], []
        scheduled = counts.copy()
        for offset in range(args.batch_size):
            available = [c for c in (0, 1) if scheduled[c] < targets[c]]
            if not available:
                break
            label = min(available, key=lambda c: (scheduled[c] / targets[c], c))
            index = attempted + offset
            source = sources.iloc[(index * 37) % len(sources)]
            context = 1 if index % 4 == 0 else 2
            mode = modes[(index // 2) % len(modes)]
            label_instruction = (
                f'Accuracy 0: include a clearly incorrect substantive claim ({mode}). '
                'The error must concern the question, not just grammar or brevity.'
                if label == 0 else
                'Accuracy 1: retain a substantive correct point from the reference but omit an '
                'explicitly requested essential component or give an incomplete explanation. '
                'A concise complete answer is Accuracy 2, NOT 1. Make the gap explicit in the rationale; '
                'do not make the whole answer false or dangerous.')
            context_instruction = (
                'Context 1: ask a natural question with two or more explicit components and '
                'answer only one of them, leaving another unanswered.' if context == 1 else
                'Context 2: address all question components directly. Accuracy can still be '
                '0 or 1: topical relevance is independent of factual correctness/completeness.')
            prompt = f'''Create ONE fictional healthcare moderation TRAINING example, including intentionally
flawed answer content for classifier research. It is not advice to a patient.
Reference training topic/question: {source.question}
Reference answer (existing dataset annotation, not a certified expert reference): {str(source.answer)[:3000]}
Write a fresh, natural question based on this topic and a genuinely new answer. Do not copy
the reference, merely negate it, or use a fixed boilerplate. Use Hindi, matching the question
language. Use 1-3 sentences in the answer, varying phrasing and length. Avoid names, contact
information, drug doses, gratuitous explicit detail and unnecessary new factual claims.
{label_instruction}
{context_instruction}
Required labels: accuracy={label}, context_relevance={context}.
For context 2 and Accuracy 1, use a correct core explanation with an important missing
qualification; do not omit a whole explicitly asked component (that is context 1).
Return ONLY one JSON object with keys question, answer, accuracy, context_relevance,
accuracy_rationale, context_rationale. Keep each English rationale to one brief sentence.
Explain the concrete error or gap, not just the requested label. Labels are integers.
Variation identifier: {index}; do not include it in the answer.'''
            prompts.append(tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                           tokenize=False, add_generation_prompt=True))
            slots.append((label, context, source.record_id, index))
            scheduled[label] += 1
        inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(model.device)
        # Persist the attempt counter before generation so resume does not replay prompts.
        attempted += len(slots)
        metadata.update(attempted=attempted, accepted_counts=counts)
        save_json(progress_path, metadata)
        set_seed(args.seed + attempted)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=0.8, top_p=0.9, use_cache=True,
                pad_token_id=tokenizer.pad_token_id)
        raws = tokenizer.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        accepted_batch = 0
        with (d / 'generation_audit.jsonl').open('a') as audit, accepted_path.open('a') as handle:
            for raw, (label, context, source_id, index) in zip(raws, slots):
                error = None
                try:
                    obj = parse_example(raw, label, context, forbidden_questions, forbidden_pairs)
                    obj.update(record_id=f'syn-model-{index:06}', source_record_id=str(source_id),
                        origin='synthetic', label_status='provisional_model_authored',
                        author=args.model, independent_label_review=False,
                        generation_method='sampled_new_question_and_answer_from_training_topic')
                    forbidden_pairs.add((norm(obj['question']), norm(obj['answer'])))
                    handle.write(json.dumps(obj, ensure_ascii=False) + '\n')
                    handle.flush()
                    rows.append(obj)
                    counts[label] += 1
                    accepted_batch += 1
                except (ValueError, TypeError, KeyError) as exc:
                    error = str(exc)
                audit.write(json.dumps({'attempt': index, 'source_record_id': str(source_id),
                    'requested_accuracy': label, 'requested_context': context,
                    'raw_output': raw, 'accepted': error is None, 'rejection_reason': error},
                    ensure_ascii=False) + '\n')
        metadata.update(accepted_counts=counts, synthetic_rows=len(rows),
                        elapsed_this_session_seconds=round(time.time()-started, 1))
        save_json(progress_path, metadata)
        print(json.dumps({'attempted': attempted, 'accepted': counts,
                          'seconds': metadata['elapsed_this_session_seconds']}), flush=True)
        empty_batches = empty_batches + 1 if not accepted_batch else 0
        if empty_batches >= 20:
            raise RuntimeError('20 batches without acceptance; inspect generation audit before resuming')
    synthetic = pd.DataFrame(rows)
    synthetic.to_csv(d / 'synthetic_examples.csv', index=False)
    complete = all(counts[c] == targets[c] for c in (0, 1))
    if complete:
        combined = pd.concat([base, synthetic], ignore_index=True)
        assert combined.record_id.is_unique
        assert len(set(zip(combined.question.map(norm), combined.answer.map(norm)))) == len(combined)
        assert combined.accuracy.value_counts().to_dict() == {0: majority, 1: majority, 2: majority}
        combined.to_csv(d / 'augmented_train.csv', index=False)
        metadata['combined_rows'] = len(combined)
        metadata['combined_class_counts'] = {k: combined[k].value_counts().sort_index().to_dict()
                                            for k in ['accuracy', 'context_relevance']}
        metadata['augmented_sha256'] = hashlib.sha256((d/'augmented_train.csv').read_bytes()).hexdigest()
    metadata.update(status='complete' if complete else 'generation_smoke_complete',
                    training_started=False, full_dataset_human_review=False)
    save_json(progress_path, metadata)
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
