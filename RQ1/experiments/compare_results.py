"""Compare completed RQ1 runs; optionally test one baseline/adapted comparison."""
import argparse
import json
from pathlib import Path


def paired_statistics(path, repeat, samples, seed):
    import numpy as np
    from sklearn.metrics import f1_score
    from scipy.stats import binomtest
    def load(stage):
        rows = [json.loads(line) for line in (path/f'{stage}_predictions.jsonl').read_text().splitlines()]
        rows = [r for r in rows if r['repeat']==repeat]
        return {r['source_row']:r for r in rows}
    a,b = load('baseline'),load('adapted')
    if not a or set(a)!=set(b):
        raise ValueError('Missing or unpaired predictions for requested repeat')
    ids=sorted(a)
    rng=np.random.default_rng(seed)
    result={}
    for key in ('accuracy','context_relevance'):
        y=np.array([a[i]['true_'+key] for i in ids]); other=np.array([b[i]['true_'+key] for i in ids])
        if not np.array_equal(y,other):raise ValueError('Reference labels differ')
        p=np.array([a[i]['pred_'+key] for i in ids]);q=np.array([b[i]['pred_'+key] for i in ids])
        def score(truth,pred):return f1_score(truth,pred,labels=[0,1,2],average='macro',zero_division=0)
        delta=score(y,q)-score(y,p)
        boot=[]; extreme=0
        for _ in range(samples):
            idx=rng.integers(0,len(y),len(y))
            boot.append(score(y[idx],q[idx])-score(y[idx],p[idx]))
            swap=rng.random(len(y))<0.5
            diff=score(y,np.where(swap,p,q))-score(y,np.where(swap,q,p))
            extreme+=abs(diff)>=abs(delta)-1e-12
        only_base=int(((p==y)&(q!=y)).sum());only_adapt=int(((q==y)&(p!=y)).sum())
        result[key]={'delta_macro_f1':float(delta),'bootstrap_95_ci':np.quantile(boot,[.025,.975]).tolist(),
                     'randomization_p':(1+extreme)/(samples+1),
                     'mcnemar_p':float(binomtest(only_adapt,only_base+only_adapt,.5).pvalue) if only_base+only_adapt else 1.0,
                     'only_baseline_correct':only_base,'only_adapted_correct':only_adapt}
    endpoints=[(k,e) for k in result for e in ('randomization_p','mcnemar_p')]
    ordered=sorted(endpoints,key=lambda pair:result[pair[0]][pair[1]])
    running=0
    for rank,(k,e) in enumerate(ordered):
        running=max(running,min(1,result[k][e]*(len(ordered)-rank)))
        result[k][e+'_holm']=running
    return {'evaluation':'validation development comparison; not final-test evidence',
            'assumption':'independent validation rows; do not use if repeated questions/users require grouping',
            'repeat':repeat,'samples':samples,'seed':seed,'dimensions':result}


def main():
    import pandas as pd
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=Path(__file__).resolve().parent/'results')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--paired-run',type=Path,help='Optional completed run containing baseline and adapted predictions')
    parser.add_argument('--assume-independent',action='store_true',help='Required for row-wise statistical tests; verify independence first')
    parser.add_argument('--repeat',type=int,default=0)
    parser.add_argument('--samples',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if args.samples<1:parser.error('--samples must be positive')
    out=args.output_dir or args.results_dir/'comparison'
    if out.exists() and any(out.iterdir()):raise ValueError('Comparison output is nonempty; choose a new directory')
    rows=[]
    for path in sorted(args.results_dir.rglob('results.json')):
        status=path.parent/'status.json'
        if not status.exists() or json.loads(status.read_text()).get('status')!='complete':continue
        cfg=json.loads((path.parent/'config.json').read_text())
        manifest=json.loads((path.parent/'data_manifest.json').read_text())
        result=json.loads(path.read_text())
        for stage in ('baseline','adapted'):
            if stage not in result:continue
            for m in result[stage]['runs']:
                rows.append({'run':str(path.parent),'name':cfg['name'],'model':cfg['model']['id'],'stage':stage,
                             'repeat':m['repeat'],'train_size':cfg['training']['train_size'] if stage=='adapted' else 0,
                             'sequence_length':cfg['input']['max_sequence_length'],'input_policy':cfg['input']['policy'],
                             'balance':cfg['training']['balance'],'seed':cfg['seed'],
                             'validation_sha256':manifest['validation']['sha256'],
                             'validation_label_columns':json.dumps(manifest['validation'].get('label_columns', cfg['data']['validation']), sort_keys=True),
                             'accuracy_macro_f1':m['accuracy']['macro_f1'],
                             'context_macro_f1':m['context_relevance']['macro_f1'],
                             'selection_score':m['selection_score'],'json_failure_rate':m['json_failure_rate'],
                             'inference_seconds':m['seconds'],'peak_gpu_bytes':m['peak_gpu_allocated_bytes'],
                             'disagreement_rate':result[stage]['disagreement_rate']})
    if not rows:raise ValueError('No completed runs found')
    frame=pd.DataFrame(rows)
    if frame.validation_sha256.nunique()!=1:raise ValueError('Runs use different validation files; compare separately')
    if frame.validation_label_columns.nunique()!=1:raise ValueError('Runs use different validation label columns; compare separately')
    out.mkdir(parents=True,exist_ok=True)
    frame.to_csv(out/'all_runs.csv',index=False)
    measures=['accuracy_macro_f1','context_macro_f1','selection_score','json_failure_rate','inference_seconds','peak_gpu_bytes','disagreement_rate']
    table=frame.groupby(['name','model','stage','train_size','sequence_length','input_policy','balance'],dropna=False)[measures].agg(['mean','std'])
    table.to_csv(out/'summary.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5))
    grouped=frame.groupby(['name','stage'])[['accuracy_macro_f1','context_macro_f1']].mean()
    grouped.plot.bar(ax=ax);ax.set_ylabel('Macro-F1');ax.set_ylim(0,1);fig.tight_layout();fig.savefig(out/'macro_f1_comparison.png');plt.close(fig)
    learning=frame[frame.name.str.match(r'^gemma_train_\d+$') & (frame.stage=='adapted')]
    if not learning.empty:
        curve=learning.groupby(['sequence_length','input_policy','balance','seed','train_size'])[['accuracy_macro_f1','context_macro_f1']].mean()
        for index,group in curve.groupby(level=[0,1,2,3]):
            fig,ax=plt.subplots();group.droplevel([0,1,2,3]).plot(marker='o',ax=ax)
            ax.set_ylabel('Validation macro-F1');ax.set_ylim(0,1);fig.tight_layout()
            fig.savefig(out/f'learning_curve_{index[0]}_{index[1]}_{index[2]}_{index[3]}.png');plt.close(fig)
    if args.paired_run:
        if not args.assume_independent:raise ValueError('Statistical tests require explicit --assume-independent after checking row independence')
        if json.loads((args.paired_run/'status.json').read_text()).get('status')!='complete':raise ValueError('Paired run is incomplete')
        stats=paired_statistics(args.paired_run,args.repeat,args.samples,args.seed)
        (out/'paired_statistics.json').write_text(json.dumps(stats,indent=2))
    print(f'Saved comparison to {out}')

if __name__=='__main__':main()
