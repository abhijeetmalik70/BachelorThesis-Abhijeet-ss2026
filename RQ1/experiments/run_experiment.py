"""Configurable RQ1 baseline and two-dimension QLoRA runner."""
import argparse
import hashlib
import json
import math
import re
import ast
import platform
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KEYS = ('accuracy', 'context_relevance')
RUBRIC = '''Evaluate this healthcare question and answer. Score both dimensions independently.
Accuracy: 0 = incorrect, misleading or harmful; 1 = partially correct with gaps; 2 = fully correct and safe.
Context and relevance: 0 = unrelated or misses the question; 1 = partly addresses it; 2 = directly addresses the question in its context.
Return only JSON with integer values 0, 1 or 2: {"accuracy": 1, "context_relevance": 1}.'''


def read_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg['mode'] not in ('baseline', 'train'):
        raise ValueError('mode must be baseline or train')
    if cfg['inference']['repeats'] < 1:
        raise ValueError('repeats must be positive')
    if cfg['training']['balance'] not in ('none', 'class_weighted', 'balanced_sampling'):
        raise ValueError('balance must be none, class_weighted or balanced_sampling')
    if cfg['training']['balance'] == 'balanced_sampling' and cfg['training'].get('sampling_dimension') not in KEYS:
        raise ValueError('balanced_sampling requires sampling_dimension')
    if cfg['input']['policy'] not in ('preserve_question', 'tail', 'full'):
        raise ValueError('Unknown input policy')
    if cfg['model']['loader'] not in ('gemma3', 'auto'):
        raise ValueError('model.loader must be gemma3 or auto')
    for section, names in {'input': ['max_sequence_length'],
                           'inference': ['batch_size', 'max_new_tokens'],
                           'training': ['batch_size', 'gradient_accumulation_steps', 'epochs', 'learning_rate']}.items():
        for name in names:
            if cfg[section][name] <= 0:
                raise ValueError(f'{section}.{name} must be positive')
    cap = cfg['input'].get('max_answer_tokens')
    if cap is not None and (type(cap) is not int or cap < 1):
        raise ValueError('max_answer_tokens must be null or a positive integer')
    return cfg


def load_frame(spec, training=False):
    import pandas as pd
    path = ROOT / spec['path']
    frame = pd.read_excel(path) if path.suffix == '.xlsx' else pd.read_csv(path)
    mapping = {spec['question']: 'question', spec['answer']: 'answer',
               spec['accuracy']: 'accuracy', spec['context_relevance']: 'context_relevance'}
    missing = set(mapping) - set(frame.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {missing}')
    # Explicit assignment avoids duplicate canonical columns when a CSV contains
    # both human 'accuracy' and mapped 'claude_accuracy'.
    frame = frame.assign(**{target: frame[source] for source, target in mapping.items()})
    if training:
        frame = frame.dropna(subset=list(KEYS))
    if frame[list(KEYS)].isna().any().any() or not all(frame[k].isin([0, 1, 2]).all() for k in KEYS):
        raise ValueError(f'{path}: labels must be 0, 1, 2')
    for k in KEYS:
        frame[k] = frame[k].astype(int)
    for k in ('question', 'answer'):
        if frame[k].isna().any() or frame[k].astype(str).str.strip().eq('').any():
            raise ValueError(f'{path}: empty {k}')
        frame[k] = frame[k].astype(str)
    frame['source_row'] = frame.index
    frame['record_id'] = frame[spec['id']].astype(str) if spec.get('id') else frame.index.astype(str)
    return frame, {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'rows': len(frame), 'label_columns': {k: spec[k] for k in KEYS}}


def prepare_data(cfg):
    val, vm = load_frame(cfg['data']['validation'])
    if len(val) != 100:
        raise ValueError(f'Expected fixed 100-row validation, found {len(val)}')
    if cfg.get('diagnostic', {}).get('evaluate_training_examples'):
        if cfg['mode'] != 'train':
            raise ValueError('Overfit diagnostic requires train mode')
        train, tm = load_frame(cfg['data']['train'], training=True)
        excluded_ids = {str(i) for i in cfg['data']['train'].get('exclude_record_ids', [])}
        if excluded_ids:
            excluded = train[train.record_id.isin(excluded_ids)]
            if set(excluded.record_id) != excluded_ids:
                raise ValueError('Configured excluded record IDs are missing')
            tm['excluded_records'] = excluded[['record_id', 'source_row', *KEYS]].to_dict('records')
            train = train[~train.record_id.isin(excluded_ids)].copy()
        dimension = cfg['diagnostic']['balance_dimension']
        per_class = cfg['diagnostic']['examples_per_class']
        if dimension not in KEYS or type(per_class) is not int or per_class < 1:
            raise ValueError('Invalid diagnostic class selection')
        # Keep fixed validation examples out of this deliberate training-set evaluation.
        pairs = set(zip(val.question.str.strip(), val.answer.str.strip()))
        eligible = train.loc[[pair not in pairs for pair in
                              zip(train.question.str.strip(), train.answer.str.strip())]]
        eligible = eligible.drop_duplicates(subset=['question', 'answer'])
        import pandas as pd
        groups = []
        for label in (0, 1, 2):
            group = eligible[eligible[dimension] == label]
            if len(group) < per_class:
                raise ValueError(f'Class {label}: need {per_class} unique examples, found {len(group)}')
            groups.append(group.sample(n=per_class, random_state=cfg['data']['subset_seed']))
        train = pd.concat(groups).sample(frac=1, random_state=cfg['data']['subset_seed']).copy()
        if len(train) != cfg['training']['train_size']:
            raise ValueError('Diagnostic train_size must equal three times examples_per_class')
        tm.update(selected_rows=len(train), class_counts={
            k: train[k].value_counts().sort_index().to_dict() for k in KEYS})
        return train, train.copy(), {
            'train': tm,
            'validation': dict(tm, evaluation_scope='training_examples_only'),
            'diagnostic': cfg['diagnostic'],
            'excluded_validation': dict(vm, used_for='overlap exclusion only; not evaluated'),
            'interpretation': 'Deliberate overfit diagnostic; scores do not measure generalization'}
    train = None
    manifest = {'validation': vm}
    if cfg['mode'] == 'train':
        train, tm = load_frame(cfg['data']['train'], training=True)
        excluded_ids = {str(i) for i in cfg['data']['train'].get('exclude_record_ids', [])}
        if excluded_ids:
            excluded = train[train.record_id.isin(excluded_ids)]
            if set(excluded.record_id) != excluded_ids:
                raise ValueError('Configured excluded record IDs are missing')
            tm['excluded_records'] = excluded[['record_id', 'source_row', *KEYS]].to_dict('records')
            train = train[~train.record_id.isin(excluded_ids)].copy()
        n = cfg['training']['train_size']
        if not isinstance(n, int) or not 1 <= n <= len(train):
            raise ValueError(f'train_size must be 1..{len(train)}')
        # Fixed shuffle seed creates nested subsets independent of training seed.
        train = train.sample(frac=1, random_state=cfg['data']['subset_seed']).iloc[:n].copy()
        pairs = lambda df: set(zip(df.question.str.strip(), df.answer.str.strip()))
        if pairs(train) & pairs(val):
            raise ValueError('Exact training/validation pair overlap; resolve split before running')
        tm['selected_rows'] = n
        tm['shared_questions'] = len(set(train.question.str.strip()) & set(val.question.str.strip()))
        tm['class_counts'] = {k: train[k].value_counts().sort_index().to_dict() for k in KEYS}
        manifest['train'] = tm
    return train, val, manifest


def balanced_indices(labels, seed, retain_all=False):
    """One balanced presentation list, cycling minority rows.

    Every class differs in presentations by at most one. A class is shuffled
    without replacement before cycling, so repeats do not create new examples.
    retain_all expands every class to the majority count; otherwise keeps the
    presentation count equal to the pool size.
    """
    rng = random.Random(seed)
    groups = {c: [i for i, label in enumerate(labels) if label == c] for c in (0, 1, 2)}
    if any(not group for group in groups.values()):
        raise ValueError('Balanced sampling requires all three classes in the selected pool')
    selected = []
    for c, group in groups.items():
        needed = max(map(len, groups.values())) if retain_all else len(labels)//3 + (c < len(labels)%3)
        draws = []
        while len(draws) < needed:
            cycle = group.copy()
            rng.shuffle(cycle)
            draws.extend(cycle[:needed-len(draws)])
        selected.extend(draws)
    rng.shuffle(selected)
    return selected


def parse_output(raw, strict=False):
    """Accept one unambiguous label object; wrappers never relax label validation."""
    invalid = {k: -1 for k in KEYS}
    aliases = {'accuracy': 'accuracy', 'context_relevance': 'context_relevance',
               'context_and_relevance': 'context_relevance',
               'context & relevance': 'context_relevance'}
    def normalize(obj):
        if not isinstance(obj, dict):
            return None
        normalized = {}
        for key, value in obj.items():
            canonical = key if strict else aliases.get(str(key).strip().lower())
            if canonical not in KEYS or canonical in normalized:
                return None
            normalized[canonical] = value
        if set(normalized) != set(KEYS):
            return None
        if not all(type(normalized[k]) is int and normalized[k] in (0, 1, 2) for k in KEYS):
            return None
        return normalized
    def unique_pairs(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError('Duplicate key')
            obj[key] = value
        return obj
    if not isinstance(raw, str):
        return invalid
    text = raw.strip()
    try:
        return normalize(json.loads(text, object_pairs_hook=unique_pairs)) or invalid
    except (ValueError, TypeError):
        if strict:
            return invalid
    # Object-shaped candidates only: never guess labels from explanatory numbers.
    candidates = re.findall(r'\{[^{}]*\}', text)
    if len(candidates) != 1 or '[' in text or ']' in text:
        return invalid
    candidate = candidates[0]
    try:
        obj = json.loads(candidate, object_pairs_hook=unique_pairs)
    except (ValueError, TypeError):
        try:
            node = ast.parse(candidate, mode='eval').body
            if not isinstance(node, ast.Dict):
                return invalid
            keys = [ast.literal_eval(k) for k in node.keys]
            if len(keys) != len(set(keys)):
                return invalid
            obj = ast.literal_eval(node)
        except (ValueError, TypeError, SyntaxError):
            return invalid
    return normalize(obj) or invalid


def metrics(records):
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    result = {}
    for k in KEYS:
        truth = [r['true_' + k] for r in records]
        pred = [r['pred_' + k] for r in records]
        p, r, f, support = precision_recall_fscore_support(truth, pred, labels=[0, 1, 2], zero_division=0)
        result[k] = {'macro_f1': float(f.mean()), 'exact_agreement': float(accuracy_score(truth, pred)),
                     'prediction_counts': {str(c): pred.count(c) for c in (0, 1, 2, -1)},
                     'precision': p.tolist(), 'recall': r.tolist(), 'f1': f.tolist(), 'support': support.tolist(),
                     'confusion_labels': [0, 1, 2, -1],
                     'confusion_matrix': confusion_matrix(truth, pred, labels=[0, 1, 2, -1]).tolist()}
    result['json_failure_rate'] = sum(not r['valid'] for r in records) / len(records)
    result['strict_format_failure_rate'] = sum(not r.get('strict_valid', r['valid']) for r in records) / len(records)
    result['selection_score'] = sum(result[k]['macro_f1'] for k in KEYS) / 2
    return result


def execute(cfg, train, val, out):
    import torch
    from transformers import (AutoTokenizer, AutoModelForCausalLM, Gemma3ForConditionalGeneration,
                              BitsAndBytesConfig, Trainer, TrainingArguments, set_seed)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    set_seed(cfg['seed'])
    if not torch.cuda.is_available():
        raise RuntimeError('This runner requires a CUDA GPU; use --check-data for local checks.')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_cfg = cfg['model']
    tokenizer = AutoTokenizer.from_pretrained(model_cfg['id'], revision=model_cfg.get('revision'))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    klass = Gemma3ForConditionalGeneration if model_cfg['loader'] == 'gemma3' else AutoModelForCausalLM
    kwargs = dict(revision=model_cfg.get('revision'), torch_dtype=dtype, device_map={'': torch.cuda.current_device()})
    if model_cfg['four_bit']:
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                          bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = klass.from_pretrained(model_cfg['id'], **kwargs)
    (out / 'model_metadata.json').write_text(json.dumps({
        'model_id': model_cfg['id'], 'requested_revision': model_cfg.get('revision'),
        'resolved_revision': getattr(model.config, '_commit_hash', None),
        'model_class': type(model).__name__, 'gpu': torch.cuda.get_device_name(),
        'gpu_total_bytes': torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory,
        'compute_dtype': str(dtype)}, indent=2))
    model.config.use_cache = False
    settings = cfg['input']
    rubric = cfg.get('prompt', RUBRIC)

    def encode(row):
        target = json.dumps({k: int(row[k]) for k in KEYS})
        target_ids = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
        answer_ids = tokenizer.encode(row['answer'], add_special_tokens=False)
        if settings.get('max_answer_tokens') is not None and settings['policy'] != 'full':
            answer_ids = answer_ids[:settings['max_answer_tokens']]
        def render(ids):
            answer = tokenizer.decode(ids, skip_special_tokens=True)
            text = rubric + '\nQuestion:\n' + row['question'] + '\nAnswer:\n' + answer
            return tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True, add_generation_prompt=True, return_dict=False)
        ids = render(answer_ids)
        budget = settings['max_sequence_length'] - len(target_ids)
        if budget <= 0:
            raise ValueError('Sequence budget cannot fit JSON target')
        if settings['policy'] == 'tail':
            ids = ids[-budget:]
        elif settings['policy'] == 'preserve_question':
            while len(ids) > budget and answer_ids:
                answer_ids = answer_ids[:max(0, len(answer_ids) - max(1, len(ids) - budget))]
                ids = render(answer_ids)
            if len(ids) > budget:
                raise ValueError(f'Row {row["record_id"]}: rubric/question/target exceed token budget; increase sequence length')
        if not ids or budget <= 0:
            raise ValueError('Invalid token budget')
        return ids, target_ids

    # Fail before expensive training if any selected input cannot fit.
    val_encoded = [encode(row) for _, row in val.iterrows()]
    train_encoded = [encode(row) for _, row in train.iterrows()] if train is not None else []
    (out / 'token_lengths.json').write_text(json.dumps({'validation': [len(x[0]) for x in val_encoded],
                                                      'train': [len(x[0])+len(x[1]) for x in train_encoded]}, indent=2))

    def evaluate(stage):
        model.eval()
        tokenizer.padding_side = 'left'
        all_records, summaries = [], []
        for repeat in range(cfg['inference']['repeats']):
            set_seed(cfg['seed'] + repeat)
            records = []
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            bs = cfg['inference']['batch_size']
            for start in range(0, len(val), bs):
                chunk = val_encoded[start:start+bs]
                batch = tokenizer.pad({'input_ids': [x[0] for x in chunk]}, return_tensors='pt').to(model.device)
                generation = dict(max_new_tokens=cfg['inference']['max_new_tokens'],
                                  do_sample=cfg['inference']['do_sample'], pad_token_id=tokenizer.pad_token_id)
                if generation['do_sample']:
                    generation['temperature'] = cfg['inference']['temperature']
                torch.cuda.synchronize()
                t = time.perf_counter()
                with torch.inference_mode():
                    output = model.generate(**batch, **generation)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t
                for offset, tokens in enumerate(output):
                    row = val.iloc[start+offset]
                    raw = tokenizer.decode(tokens[batch.input_ids.shape[1]:], skip_special_tokens=True)
                    parsed = parse_output(raw)
                    record = {'stage': stage, 'repeat': repeat, 'record_id': row.record_id,
                              'source_row': int(row.source_row), 'raw_output': raw,
                              'valid': all(parsed[k] != -1 for k in KEYS),
                              'strict_valid': all(v != -1 for v in parse_output(raw, strict=True).values()),
                              'batch_seconds_per_example': elapsed/len(chunk)}
                    record.update({'true_'+k: int(row[k]) for k in KEYS})
                    record.update({'pred_'+k: parsed[k] for k in KEYS})
                    records.append(record)
            summary = metrics(records)
            summary.update(repeat=repeat, seconds=time.perf_counter()-started,
                           peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
            summaries.append(summary)
            all_records.extend(records)
        with (out / f'{stage}_predictions.jsonl').open('w') as handle:
            for r in all_records:
                handle.write(json.dumps(r, ensure_ascii=False)+'\n')
        disagreement = sum(len({tuple(r['pred_'+k] for k in KEYS) for r in all_records if r['source_row']==int(row.source_row)}) > 1
                           for _, row in val.iterrows()) / len(val)
        result = {'runs': summaries, 'disagreement_rate': disagreement}
        (out / f'{stage}_metrics.json').write_text(json.dumps(result, indent=2))
        return result

    results = {'baseline': evaluate('baseline')}
    if train is not None:
        tokenizer.padding_side = 'right'
        if model_cfg['four_bit']:
            model = prepare_model_for_kbit_training(model)
        lora = cfg['lora']
        model = get_peft_model(model, LoraConfig(r=lora['rank'], lora_alpha=lora['alpha'],
                               lora_dropout=lora['dropout'], target_modules=lora['target_modules'], task_type='CAUSAL_LM'))
        weights = {}
        for k in KEYS:
            counts = train[k].value_counts()
            weights[k] = {c: len(train)/(3*counts[c]) if c in counts else 0 for c in (0, 1, 2)}
        samples = []
        for (_, row), (ids, target) in zip(train.iterrows(), train_encoded):
            # Mean of dimension-specific inverse-frequency weights, applied per example.
            weight = sum(weights[k][int(row[k])] for k in KEYS)/2 if cfg['training']['balance']=='class_weighted' else 1.0
            samples.append({'input_ids': ids+target, 'labels': [-100]*len(ids)+target, 'sample_weight': weight})

        sampling_metadata = None
        if cfg['training']['balance'] == 'balanced_sampling':
            dimension = cfg['training']['sampling_dimension']
            retain_all = cfg['training'].get('sampling_retain_all', False)
            indices = balanced_indices(train[dimension].tolist(), cfg['seed'], retain_all=retain_all)
            sampled = train.iloc[indices].copy()
            sampled.insert(0, 'presentation_index', range(len(sampled)))
            sampled[['presentation_index', 'record_id', 'source_row', *KEYS]].to_csv(
                out/'sampling_presentations.csv', index=False)
            sampling_metadata = {
                'dimension': dimension, 'seed': cfg['seed'],
                'retain_all_pool_rows': retain_all,
                'method': 'fixed balanced presentation list; shuffled class cycles; repeated each epoch',
                'pool_unique_rows': len(train),
                'presented_unique_rows': len(set(indices)),
                'presentations_per_epoch': len(indices),
                'class_counts_per_epoch': {k: sampled[k].value_counts().sort_index().to_dict() for k in KEYS},
                'epochs': cfg['training']['epochs'],
                'loss_weight': 1.0}
            (out/'sampling.json').write_text(json.dumps(sampling_metadata, indent=2))
            samples = [samples[i] for i in indices]

        def collate(rows):
            size = max(len(r['input_ids']) for r in rows)
            return {'input_ids': torch.tensor([r['input_ids']+[tokenizer.pad_token_id]*(size-len(r['input_ids'])) for r in rows]),
                    'attention_mask': torch.tensor([[1]*len(r['input_ids'])+[0]*(size-len(r['input_ids'])) for r in rows]),
                    'labels': torch.tensor([r['labels']+[-100]*(size-len(r['labels'])) for r in rows]),
                    'sample_weight': torch.tensor([r['sample_weight'] for r in rows])}

        class WeightedTrainer(Trainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Custom loss is a microbatch mean; Trainer handles accumulation scaling.
                self.model_accepts_loss_kwargs = False

            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                labels = inputs.pop('labels')
                weight = inputs.pop('sample_weight')
                outputs = model(**inputs)
                logits = outputs.logits[:, :-1, :].contiguous()
                shifted = labels[:, 1:].contiguous()
                token_loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), shifted.reshape(-1),
                                                               reduction='none', ignore_index=-100).view_as(shifted)
                per_example = token_loss.sum(1)/shifted.ne(-100).sum(1).clamp_min(1)
                loss = (per_example*weight).mean()
                return (loss, outputs) if return_outputs else loss

        ts = cfg['training']
        args = TrainingArguments(output_dir=str(out/'checkpoints'), num_train_epochs=ts['epochs'],
                  per_device_train_batch_size=ts['batch_size'], gradient_accumulation_steps=ts['gradient_accumulation_steps'],
                  learning_rate=ts['learning_rate'], warmup_steps=math.ceil(math.ceil(math.ceil(len(samples)/ts['batch_size'])/ts['gradient_accumulation_steps'])*ts['epochs']*ts['warmup_ratio']), weight_decay=ts['weight_decay'],
                  max_grad_norm=ts['max_grad_norm'], optim=ts['optimizer'], bf16=dtype==torch.bfloat16,
                  fp16=dtype==torch.float16, gradient_checkpointing=True, remove_unused_columns=False,
                  save_strategy='epoch', save_total_limit=1, logging_steps=1, report_to='none', seed=cfg['seed'])
        trainer = WeightedTrainer(model=model, args=args, train_dataset=samples, data_collator=collate)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        trainer.train()
        results['training'] = {'seconds': time.perf_counter()-started,
                              'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
                              'effective_batch_size': ts['batch_size']*ts['gradient_accumulation_steps'],
                              'checkpoint_rule': 'final epoch', 'class_weights': weights,
                              'sampling': sampling_metadata}
        trainer.save_model(str(out/'adapter'))
        tokenizer.save_pretrained(out/'adapter')
        (out/'training_log.json').write_text(json.dumps(trainer.state.log_history, indent=2))
        model.config.use_cache = True
        results['adapted'] = evaluate('adapted')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--check-data', action='store_true', help='Validate data/config without model downloads or writing results')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    cfg = read_config(args.config)
    train, val, manifest = prepare_data(cfg)
    if args.check_data:
        print(json.dumps(manifest, indent=2))
        return
    result_group = 'diagnostic_results' if cfg.get('diagnostic') else 'results'
    out = args.output_dir or ROOT/'RQ1/experiments'/result_group/cfg['name']
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory is nonempty; use a new --output-dir')
    out.mkdir(parents=True, exist_ok=True)
    (out/'config.json').write_text(json.dumps(cfg, indent=2))
    (out/'data_manifest.json').write_text(json.dumps(manifest, indent=2))
    val[['record_id','source_row']].to_csv(out/'validation_ids.csv', index=False)
    if train is not None:
        train[['record_id','source_row']].to_csv(out/'training_ids.csv', index=False)
    import importlib.metadata
    versions = {}
    for name in ('torch','transformers','peft','accelerate','bitsandbytes','pandas','scikit-learn'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    (out/'environment.json').write_text(json.dumps({'python': platform.python_version(), 'packages': versions}, indent=2))
    try:
        results = execute(cfg, train, val, out)
        (out/'results.json').write_text(json.dumps(results, indent=2))
        (out/'status.json').write_text(json.dumps({'status':'complete'}))
    except Exception as exc:
        (out/'status.json').write_text(json.dumps({'status':'failed','error':str(exc)}))
        raise

if __name__ == '__main__':
    main()
