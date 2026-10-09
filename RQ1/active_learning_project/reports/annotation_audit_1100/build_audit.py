"""Read-only audit of source workbooks; writes review artifacts beside this script.

Run with: python3 RQ1/active_learning_project/reports/annotation_audit_1100/build_audit.py
Requires pandas, openpyxl, scikit-learn. Review suggestions are not expert gold labels.
"""
from pathlib import Path
import hashlib
import json
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

OUT = Path(__file__).resolve().parent
DATA = OUT.parents[1] / 'data'
SOURCE = DATA / 'trainset/Combined_QnA_with_Labels_annotated_1100.xlsx'
df = pd.read_excel(SOURCE, keep_default_na=False)
labels = ['Accuracy', 'Context & Relevance']
keys = ['Sanitized Question', 'Answer Transcription']
train = df.loc[df[labels].ne('').all(axis=1)].copy()
train.insert(0, 'Excel row', train.index + 2)
conversion = df[['Row ID'] + labels].copy()
int_path = DATA / 'trainset/1100withInt.xlsx'
other = pd.read_excel(int_path, keep_default_na=False) if int_path.exists() else None
conversion['Accuracy changed'] = False
if other is not None:
    assert df['Row ID'].equals(other['Row ID'])
    assert df[keys].equals(other[keys])
    conversion['Accuracy in 1100withInt'] = other['Accuracy']
    conversion['Accuracy changed'] = df.Accuracy.ne(other.Accuracy)

# Explicit, targeted reviewer observations; not a representative random sample.
notes = {
  104: ('Clarity/context', 'Review 0; likely 1–2', 'Hindi answer directly discusses the egg/pregnancy question. An unrelated Relevant answer discusses sexual touch. Resolve answer/reference mapping before judging accuracy.'),
  172: ('Accuracy and clarity/context', 'Review both 0 labels', 'Hindi answer discusses menopause, with a core 45–55 age statement supported by WHO. Full reference-based accuracy remains unverified. Relevant answer is about a different sexual-health topic.'),
  194: ('Clarity/context', 'Review 0; likely 1–2', 'Hindi answer addresses absent menstruation and referral. Relevant answer instead discusses pre-ejaculate/anal sex. Do not interpret this flag as clinical certification.'),
  1038: ('Missing content / NA', 'Exclude missing answer; apply agreed NA rule', 'Answer Transcription is the numeric placeholder 0. The rubric provides NA for no factual content / insufficient content. Training currently treats this as a substantive class-0 answer.'),
  1243: ('Clarity/context', 'Review 0; likely 1–2', 'Hindi answer directly addresses the stated concern and is understandable. Whether its reassurance warrants an accuracy downgrade is a separate clinical review.'),
  1288: ('Clarity/context', 'Review 0; likely 1–2', 'Hindi answer explains the requested breast-development topic in largely everyday language. A 0 for clarity/context needs justification.'),
  1374: ('Clarity/context / language selection', 'Review 0; likely 1–2', 'Actual Answer Transcription is Hindi and describes school support/referral. The English translation is a different, more technical surface form; assess the actual model input.'),
  1384: ('Conflicting duplicate', 'Adjudicate together with 1385', 'Exactly the same question and Answer Transcription as 1385, but accuracy is 1 here and 0 there. Identical training inputs cannot justify different targets.'),
  1385: ('Conflicting duplicate / column selection', 'Adjudicate together with 1384', 'Actual Hindi answer is complete and identical to 1384. Only the English translation says See answer - 26J. Wrong-column annotation is a possible explanation, not established provenance.'),
  1505: ('Clarity/context / language selection', 'Proposed clarity/context 0; review accuracy separately', 'Actual Answer Transcription is a long English paragraph; its translation column is Hindi. Clarity/context 2 conflicts with the supplied rural-Hindi rubric. The English answer substantively addresses adolescent development, making accuracy 0 questionable too.'),
  1585: ('Clarity/context', 'Proposed clarity/context 2', 'Actual Hindi answer is short, understandable, and answers who adolescents are. It offers no inaccessible intervention. No visible justification for clarity/context 0. The 10–19 age statement matches WHO.'),
  1637: ('Clarity/context / language selection', 'Review 0; consider 1', 'Actual answer is Hindi with difficult medical vocabulary, not the fully English translation. The rubric allows 1 for some difficult words; expert adjudication is needed on how much jargon warrants 0.'),
}
review = train[train['Row ID'].isin(notes)].copy()
for n, col in enumerate(['Review category', 'Provisional suggestion', 'Reason']):
    review[col] = review['Row ID'].map(lambda rid: notes[rid][n])
review['Expert accuracy'] = ''
review['Expert clarity/context'] = ''
review['Expert rationale'] = ''

duplicate = train.duplicated(keys, keep=False)
placeholder = train['Answer Transcription'].astype(str).str.strip().isin(['0', ''])
english_only = (train['Answer Transcription'].astype(str).str.contains('[A-Za-z]')
                & ~train['Answer Transcription'].astype(str).str.contains('[\u0900-\u097f]'))
queue = train.copy()
queue['Conflicting duplicate candidate'] = duplicate
queue['Placeholder answer'] = placeholder
queue['English-only answer with context 2'] = english_only & train['Context & Relevance'].eq(2)
queue['Minority label: review priority, not proof of error'] = train[labels].ne(2).any(axis=1)
queue['Targeted reviewer note'] = train['Row ID'].map(lambda rid: notes.get(rid, ('', '', ''))[2])
queue['Semantically certified'] = False

test = pd.read_excel(DATA / 'testset/final_test.xlsx', keep_default_na=False)
test.insert(0, 'Test Excel row', test.index + 2)
overlap = test.merge(train[['Row ID'] + keys], on=keys, suffixes=('_test', '_train'))
test_matched = test.merge(train[keys].drop_duplicates(), on=keys)
val_path = DATA / 'varification set/anjali_dataset_with_claude_labels.xlsx'
csv_path = DATA / 'varification set/anjali_dataset_labeled.csv'
val = pd.read_excel(val_path) if val_path.exists() else None
csv = pd.read_csv(csv_path) if csv_path.exists() else None
aligned = None
if val is not None and csv is not None:
    assert val.id.is_unique and csv.id.is_unique
    aligned = csv.merge(val, on='id', suffixes=('_csv', '_xlsx'), validate='one_to_one')
    assert (aligned.model_answer == aligned['Answer Transcription']).all()
    assert all((aligned[c + '_csv'] == aligned[c + '_xlsx']).all()
               for c in ['accuracy', 'context_and_relevance'])
metrics = []
disagreements = []
for name, frame, pairs in [
    ('anjali_dataset_with_claude_labels.xlsx', val,
     [('accuracy', 'Accuracy (0-2)'), ('context_and_relevance', 'claude_relevance')]),
    ('anjali_dataset_labeled.csv', csv,
     [('accuracy', 'claude_accuracy'), ('context_and_relevance', 'claude_context_relevance')]),
]:
    if frame is None:
        continue
    for human, claude in pairs:
        a, b = frame[human], frame[claude]
        metrics.append(dict(file=name, dimension=human, n=len(frame),
                            exact_agreement=float((a == b).mean()),
                            cohen_kappa=float(cohen_kappa_score(a, b)),
                            macro_f1=float(f1_score(a, b, average='macro')),
                            confusion_matrix_human_rows_claude_columns_012=confusion_matrix(a, b, labels=[0, 1, 2]).tolist()))
        for _, r in frame[a.ne(b)].iterrows():
            disagreements.append(dict(file=name, dimension=human, id=r.id,
                                      human_label=r[human], claude_label=r[claude]))

summary = {
    'scope': 'All-row structural checks and 12 targeted semantic examples, not exhaustive expert reannotation.',
    'rows_total': len(df), 'rows_labeled': len(train), 'rows_unlabeled': len(df)-len(train),
    'label_counts': {c: {str(k): int(v) for k, v in train[c].value_counts().items()} for c in labels},
    'source_counts': train['Source Dataset'].value_counts().to_dict(),
    'int_file_present_at_generation': other is not None,
    'int_file_accuracy_changes_in_labeled_rows': int(conversion.loc[train.index, 'Accuracy changed'].sum()) if other is not None else None,
    'int_file_blank_accuracy_replaced_by_zero': int((df.Accuracy.eq('') & other.Accuracy.eq(0)).sum()) if other is not None else None,
    'test_exact_pair_overlap': len(test_matched), 'test_size': len(test),
    'validation_exact_pair_overlap': len(val.merge(train[keys].drop_duplicates(), on=keys)) if val is not None else None,
    'validation_question_overlap': len(val.merge(train[['Sanitized Question']].drop_duplicates(), on='Sanitized Question')) if val is not None else None,
    'placeholder_row_ids': train.loc[placeholder, 'Row ID'].tolist(),
    'duplicate_row_ids': train.loc[duplicate, 'Row ID'].tolist(),
    'validation_version_accuracy_differences': int(aligned.claude_accuracy.ne(aligned['Accuracy (0-2)']).sum()) if aligned is not None else None,
    'validation_version_context_differences': int(aligned.claude_context_relevance.ne(aligned.claude_relevance).sum()) if aligned is not None else None,
    'agreement_metrics': metrics,
}
inputs = [SOURCE, DATA/'trainset/1100withInt.xlsx', DATA/'testset/final_test.xlsx',
          DATA/'varification set/anjali_dataset_with_claude_labels.xlsx',
          DATA/'varification set/anjali_dataset_labeled.csv']
summary['input_sha256'] = {str(p.relative_to(DATA)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs if p.exists()}
(OUT/'audit_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
with pd.ExcelWriter(OUT/'annotation_review.xlsx', engine='openpyxl') as writer:
    for name, frame in [('Targeted review', review), ('All labeled rows', queue),
                        ('Test overlap', overlap), ('Expert disagreements', pd.DataFrame(disagreements)),
                        ('Int conversion changes', conversion[conversion['Accuracy changed']])]:
        frame.to_excel(writer, sheet_name=name, index=False)
        ws = writer.sheets[name]
        ws.freeze_panes = 'A2'
        ws.auto_filter.ref = ws.dimensions
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = min(65, max(14, len(str(col[0].value))+2))
print(json.dumps({k: v for k, v in summary.items() if k not in ['input_sha256', 'agreement_metrics']}, indent=2))
