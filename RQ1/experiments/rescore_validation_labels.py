"""Score saved predictions against Claude validation labels without retraining."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from run_experiment import KEYS, load_frame, metrics


def rescore(run_dir, out):
    cfg = json.loads((run_dir/'config.json').read_text())
    manifest = json.loads((run_dir/'data_manifest.json').read_text())
    if cfg.get('diagnostic'):
        raise ValueError('Training-set diagnostics are not validation predictions')
    if json.loads((run_dir/'status.json').read_text())['status'] != 'complete':
        raise ValueError('Source run is not complete')
    spec = dict(cfg['data']['validation'], accuracy='claude_accuracy',
                context_relevance='claude_context_relevance')
    frame, vm = load_frame(spec)
    if vm['sha256'] != manifest['validation']['sha256']:
        raise ValueError('Validation CSV differs from the source run')
    if len(frame) != 100:
        raise ValueError('Expected the fixed 100-row validation set')
    lookup = frame.set_index('source_row')
    original = json.loads((run_dir/'results.json').read_text())
    result = copy.deepcopy(original)
    saved = {}
    for stage in ('baseline', 'adapted'):
        if stage not in original:
            continue
        path = run_dir/f'{stage}_predictions.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        summaries = []
        for previous in original[stage]['runs']:
            repeat = previous['repeat']
            batch = [r for r in rows if r['repeat'] == repeat]
            if len(batch) != len(frame) or {r['source_row'] for r in batch} != set(lookup.index):
                raise ValueError(f'{stage} repeat {repeat}: incomplete or duplicate validation rows')
            for row in batch:
                label = lookup.loc[row['source_row']]
                if str(row['record_id']) != str(label.record_id):
                    raise ValueError('Record ID does not match validation source row')
                for key in KEYS:
                    row['true_'+key] = int(label[key])
            summaries.append({**previous, **metrics(batch)})
        result[stage]['runs'] = summaries
        saved[stage] = rows
    cfg['name'] += '_claude_validation_rescored'
    cfg['data']['validation'] = spec
    manifest['validation'] = vm
    manifest['rescoring'] = {
        'source_run': str(run_dir.resolve()),
        'method': 'Replace reference labels only; saved predictions and parsing unchanged',
        'label_source': 'Claude', 'no_training_or_inference_performed': True,
        'source_prediction_sha256': {stage: hashlib.sha256(
            (run_dir/f'{stage}_predictions.jsonl').read_bytes()).hexdigest() for stage in saved}}
    if out.exists():
        raise ValueError('Output already exists; choose a new directory')
    out.mkdir(parents=True)
    for name, value in [('config.json', cfg), ('data_manifest.json', manifest),
                        ('results.json', result), ('status.json', {'status': 'complete'})]:
        (out/name).write_text(json.dumps(value, indent=2)+'\n')
    for stage, rows in saved.items():
        (out/f'{stage}_predictions.jsonl').write_text(''.join(
            json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
        (out/f'{stage}_metrics.json').write_text(json.dumps(result[stage], indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    result = rescore(args.run_dir, args.output_dir)
    for stage in ('baseline', 'adapted'):
        if stage in result:
            m = result[stage]['runs'][0]
            print(stage, {key: m[key]['macro_f1'] for key in KEYS})


if __name__ == '__main__':
    main()
