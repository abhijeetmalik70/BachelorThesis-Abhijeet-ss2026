"""Reparse existing generations, preserving all original results."""
import argparse
import json
from pathlib import Path
from run_experiment import KEYS, parse_output, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.run_dir/'rescored_v1'
    if out.exists():
        raise ValueError('Rescored output already exists')
    out.mkdir()
    summaries = {}
    for stage in ('baseline', 'adapted'):
        rows = [json.loads(line) for line in (args.run_dir/f'{stage}_predictions.jsonl').read_text().splitlines()]
        for row in rows:
            pred = parse_output(row['raw_output'])
            row.update({'pred_'+k: pred[k] for k in KEYS})
            row['valid'] = all(v != -1 for v in pred.values())
            row['strict_valid'] = all(v != -1 for v in parse_output(row['raw_output'], strict=True).values())
        with (out/f'{stage}_predictions.jsonl').open('w') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False)+'\n')
        summaries[stage] = [{'repeat': repeat, **metrics([r for r in rows if r['repeat']==repeat])}
                            for repeat in sorted({r['repeat'] for r in rows})]
    (out/'metrics.json').write_text(json.dumps(summaries, indent=2))
    for stage, runs in summaries.items():
        m=runs[0]
        print(stage, 'Accuracy macro-F1:', m['accuracy']['macro_f1'], 'Context macro-F1:',
              m['context_relevance']['macro_f1'], 'parse failures:', m['json_failure_rate'],
              'strict format failures:', m['strict_format_failure_rate'])

if __name__ == '__main__':
    main()
