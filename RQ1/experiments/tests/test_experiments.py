import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

FOLDER=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('runner',FOLDER/'run_experiment.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

class ExperimentChecks(unittest.TestCase):
    def test_balanced_sampling(self):
        from collections import Counter
        labels = [0]*11 + [1]*20 + [2]*469
        indices = runner.balanced_indices(labels, 42)
        self.assertEqual(len(indices), 500)
        self.assertEqual(Counter(labels[i] for i in indices), {0:167, 1:167, 2:166})
        self.assertTrue(set(range(31)).issubset(indices))
        self.assertEqual(indices, runner.balanced_indices(labels, 42))
        self.assertNotEqual(indices, runner.balanced_indices(labels, 43))
        full = runner.balanced_indices(labels, 42, retain_all=True)
        self.assertEqual(Counter(labels[i] for i in full), {0:469,1:469,2:469})
        self.assertEqual(set(full),set(range(len(labels))))
        with self.assertRaises(ValueError):
            runner.balanced_indices([1, 2], 42)

    def test_strict_json(self):
        for raw in ['{"accuracy": true, "context_relevance": 2}',
                    '{"accuracy": 1}', '{"accuracy": 3, "context_relevance": 2}',
                    '{"accuracy": 1, "context_relevance": 2, "extra": 0}']:
            self.assertEqual(runner.parse_output(raw),dict.fromkeys(runner.KEYS,-1))
        self.assertEqual(runner.parse_output('{"accuracy": 0, "context_relevance": 2}'),{'accuracy':0,'context_relevance':2})

    def test_wrapped_output(self):
        for raw in ['```json\n{"accuracy": 0, "context_relevance": 2}\n```',
                    'Result: {"accuracy": 0, "context_relevance": 2}',
                    "{'accuracy': 0, 'context_and_relevance': 2}"]:
            self.assertEqual(runner.parse_output(raw), {'accuracy': 0, 'context_relevance': 2})
            self.assertEqual(runner.parse_output(raw, strict=True), dict.fromkeys(runner.KEYS, -1))
        for raw in ['{"accuracy":0,"accuracy":2,"context_relevance":2}',
                    '{"accuracy":0,"context_relevance":2} {"accuracy":2,"context_relevance":2}',
                    '[{"accuracy":0,"context_relevance":2}]']:
            self.assertEqual(runner.parse_output(raw), dict.fromkeys(runner.KEYS,-1))

    def test_invalid_output_is_not_a_fourth_scored_class(self):
        records=[]
        for truth,pred in [(0,-1),(1,1),(2,2)]:
            records.append({'true_'+k:truth for k in runner.KEYS}|{'pred_'+k:pred for k in runner.KEYS}|{'valid':pred!=-1})
        m=runner.metrics(records)
        self.assertAlmostEqual(m['accuracy']['macro_f1'],2/3)
        self.assertAlmostEqual(m['json_failure_rate'],1/3)
        self.assertEqual(m['accuracy']['confusion_matrix'][0][3],1)

    def test_all_configs_and_nested_subsets(self):
        subsets={}
        for path in FOLDER.joinpath('configs').glob('*.json'):
            cfg=runner.read_config(path)
            train,val,manifest=runner.prepare_data(cfg)
            if cfg.get('diagnostic'):
                self.assertEqual(len(val),30)
                self.assertEqual(train.accuracy.value_counts().to_dict(),{0:10,1:10,2:10})
                self.assertEqual(list(train.source_row),list(val.source_row))
                again, _, _ = runner.prepare_data(cfg)
                self.assertEqual(list(train.source_row),list(again.source_row))
                human, _ = runner.load_frame(cfg['data']['validation'])
                pairs = lambda df: set(zip(df.question.str.strip(),df.answer.str.strip()))
                self.assertFalse(pairs(train) & pairs(human))
            else:
                self.assertEqual(len(val),100)
                self.assertEqual(manifest['validation']['label_columns']['accuracy'],'claude_accuracy')
                self.assertEqual(val.accuracy.value_counts().to_dict(),{2:96,1:3,0:1})
                self.assertEqual(val.context_relevance.value_counts().to_dict(),{2:96,1:4})
            if train is not None:
                self.assertEqual(len(train),cfg['training']['train_size'])
            if path.stem.startswith('gemma_train_'):
                subsets[len(train)]=set(train.source_row)
        sizes=sorted(subsets)
        for a,b in zip(sizes,sizes[1:]):self.assertTrue(subsets[a]<=subsets[b])

    def test_comparison_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir()
            cfg=runner.read_config(FOLDER/'configs/gemma_train_500.json')
            records=[{'true_'+k:v for k in runner.KEYS}|{'pred_'+k:v for k in runner.KEYS}|{'valid':True} for v in (0,1,2)]
            m=runner.metrics(records)|{'repeat':0,'seconds':1,'peak_gpu_allocated_bytes':123}
            (run/'config.json').write_text(json.dumps(cfg))
            (run/'data_manifest.json').write_text(json.dumps({'validation':{'sha256':'test'}}))
            (run/'status.json').write_text('{"status":"complete"}')
            (run/'results.json').write_text(json.dumps({'adapted':{'runs':[m],'disagreement_rate':0}}))
            subprocess.run([sys.executable,str(FOLDER/'compare_results.py'),'--results-dir',str(root)],check=True,capture_output=True)
            for name in ['all_runs.csv','summary.csv','macro_f1_comparison.png']:
                self.assertTrue((root/'comparison'/name).exists())

if __name__=='__main__':unittest.main()
