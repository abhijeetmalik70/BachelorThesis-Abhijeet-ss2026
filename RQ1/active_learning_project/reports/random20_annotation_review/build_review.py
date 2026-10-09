"""Reproduce the random sample and compare it with documented reviewer decisions."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import pandas as pd
from openpyxl.styles import Alignment

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parents[1] / 'data/trainset/Combined_QnA_with_Labels_annotated_1100.xlsx'
SEED = 20260925
df = pd.read_excel(SOURCE, keep_default_na=False)
population = df.loc[df[['Accuracy', 'Context & Relevance']].ne('').all(axis=1)]
assert len(population) == 1100
sample = population.sample(n=20, replace=False, random_state=SEED)
decisions = json.loads((OUT/'reviewer_decisions.json').read_text())
assert sample['Row ID'].tolist() == [r['id'] for r in decisions], 'Source sample changed; review required'

def compare(label, options):
    if label not in options:
        return 'Disputed'
    return 'Agrees' if len(options) == 1 else 'Uncertain'

rows = []
for order, ((index, r), review) in enumerate(zip(sample.iterrows(), decisions), 1):
    acc = compare(int(r.Accuracy), review['accuracy'])
    ctx = compare(int(r['Context & Relevance']), review['context'])
    overall = 'Disputed' if 'Disputed' in [acc, ctx] else 'Uncertain' if 'Uncertain' in [acc, ctx] else 'Agrees'
    rows.append({
        'Sample order': order, 'Row ID': int(r['Row ID']), 'Source Excel row': int(index + 2),
        'Source Dataset': r['Source Dataset'], 'Question': r['Sanitized Question'],
        'Answer Transcription': r['Answer Transcription'],
        'Claude accuracy': int(r.Accuracy), 'Claude clarity/context': int(r['Context & Relevance']),
        'Reviewer accuracy options': ' / '.join(map(str, review['accuracy'])),
        'Reviewer clarity/context options': ' / '.join(map(str, review['context'])),
        'Accuracy result': acc, 'Clarity/context result': ctx, 'Pair result': overall,
        'Reason': review['reason'], 'Supporting sources': '\n'.join(review['sources']),
        'Expert accuracy': '', 'Expert clarity/context': '', 'Expert rationale': '',
    })
output = pd.DataFrame(rows)
summary = {
    'seed': SEED, 'method': 'pandas.DataFrame.sample(n=20, replace=False, random_state=20260925)',
    'population_size': len(population), 'sample_size': len(output),
    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'pandas_version': pd.__version__, 'sample_row_ids': output['Row ID'].tolist(),
    'source_distribution': dict(Counter(output['Source Dataset'])),
    'pair_results': dict(Counter(output['Pair result'])),
    'accuracy_results': dict(Counter(output['Accuracy result'])),
    'context_results': dict(Counter(output['Clarity/context result'])),
}
(OUT/'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
output.to_csv(OUT/'random20_review.csv', index=False)
with pd.ExcelWriter(OUT/'random20_review.xlsx', engine='openpyxl') as writer:
    output.to_excel(writer, sheet_name='20 reviewed samples', index=False)
    ws = writer.sheets['20 reviewed samples']
    ws.freeze_panes = 'G2'
    ws.auto_filter.ref = ws.dimensions
    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = 65 if col[0].value in ['Answer Transcription', 'Reason'] else 35
        for cell in col:
            cell.alignment = Alignment(vertical='top', wrap_text=True)
    for i in range(2, 22):
        ws.row_dimensions[i].height = 100

lines = [
    '# Random review of 20 annotations', '',
    '**Result: 4 pairs agree on both labels, 9 pairs have at least one disputed label, and 7 pairs remain uncertain.**', '',
    'These are reviewer assessments, not expert-certified ground truth. Do not describe this as proof that only 20% of all 1,100 annotations are correct. The sample is small, seven cases are unresolved, and this is an AI review rather than independent human validation.', '',
    '## Method', '',
    f'- Uniform random sample without replacement from the 1,100 rows with both labels; seed `{SEED}`. No resampling or substitution.',
    '- Reviewed the complete original Hindi `Answer Transcription` against the question and the supplied moderator rubric. Additional claims in an answer count toward accuracy even when they concern a different question.',
    '- Decisions were recorded before printing the current sample label comparison. This was not fully blinded: the earlier audit had already exposed some rows and labels, including 1288 and 226.',
    '- The rubric was available from its text extracted during the previous audit; the original PDF was no longer at its initial Desktop path during this follow-up.',
    '- The rubric asks for an aligned expert reference. None was established for these training rows, and the earlier audit found mismatches in `Relevant answer`. Consequently, this is a provisional factual/safety and rubric-consistency review, not certified reference-answer agreement.',
    '- Used primary healthcare guidance for material medical concerns. Ordinary Hindi loanwords alone did not trigger a downgrade. Tone was not scored. Missing an ASHA/PHC mention alone was not penalized.',
    '- All 20 sampled rows are SRHR; this sample does not directly test Mental Wellness annotation quality.',
    '- An options range such as 0 / 1 means the exact replacement needs adjudication, but an existing 2 is disputed under either option. If the existing score is one of multiple plausible options, the result is Uncertain. Disputed means the reviewer challenges the score, not an expert-confirmed error.', '',
    '## Counts', '',
    '| Assessment | Agrees | Disputed | Uncertain |',
    '|---|---:|---:|---:|',
]
for title, col in [('Accuracy', 'Accuracy result'), ('Clarity/context', 'Clarity/context result'), ('Both labels together', 'Pair result')]:
    c = Counter(output[col])
    lines.append(f"| {title} | {c['Agrees']} | {c['Disputed']} | {c['Uncertain']} |")
lines += ['', 'Agreement is not the same as an answer being good: row 226 agrees because both the answer and its zero labels correctly indicate failure to answer the question.', '',
          '## Every sampled row', '', '| Row ID | Claude A / C | Reviewer A | Reviewer C | Result |', '|---|---|---|---|---|']
for r in rows:
    lines.append(f"| {r['Row ID']} | {r['Claude accuracy']} / {r['Claude clarity/context']} | {r['Reviewer accuracy options']} | {r['Reviewer clarity/context options']} | {r['Pair result']} |")
lines += ['', 'A = accuracy; C = clarity/context. NA is explicitly allowed by the expert rubric, although the present model uses only numeric classes.', '', '## Reasons and evidence', '']
for r in rows:
    links = ' '.join(f'[Source {i}]({url})' for i, url in enumerate(r['Supporting sources'].splitlines(), 1))
    lines += [f"**Row {r['Row ID']} — {r['Pair result']}.** {r['Reason']} {links}".rstrip(), '']
lines += ['## Files', '',
          '`random20_review.xlsx` and `random20_review.csv` contain the full questions, answers, original labels, proposed score options, reasons, and blank expert-adjudication columns. `summary.json` records counts, selected IDs, seed, and source hash. `reviewer_decisions.json` contains the documented judgements; `build_review.py` reproduces the sample and comparison. The script packages these judgements; it does not independently re-evaluate answer semantics.', '',
          'Original training labels were not changed. An expert should adjudicate the disputed and uncertain examples before any correction is applied.', '']
(OUT/'README.md').write_text('\n'.join(lines))
print(json.dumps(summary, indent=2))
