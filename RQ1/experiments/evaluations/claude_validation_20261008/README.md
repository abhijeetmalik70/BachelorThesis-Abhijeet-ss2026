# Claude validation evaluation — 8 October 2026

Re-evaluated saved generations from the completed 500-example runs against `claude_accuracy` and `claude_context_relevance`. No new training or inference occurred. Predictions and parsing were preserved; only reference labels changed. Original human-scored files remain unchanged. Each source prediction file hash is recorded in the new data manifests. All 300 predictions per stage per source run were checked to be unchanged except for reference labels, with source-row and record-ID alignment verified.

| Model/run | Accuracy macro-F1 | Context & Relevance macro-F1 | Accuracy agreement | Context agreement |
|---|---:|---:|---:|---:|
| Base model | 0.2857 | 0.3176 | 75% | 91% |
| Unweighted natural sampling | 0.3265 | 0.3265 | 96% | 96% |
| Class-weighted natural sampling | 0.3265 | 0.3265 | 96% | 96% |
| Balanced Accuracy sampling | 0.3212 | 0.3265 | 93% | 96% |

Claude supports are Accuracy 1/3/96 and Context & Relevance 0/4/96 for classes 0/1/2. Macro-F1 averages the predefined three classes; absent class 0 in Context & Relevance contributes zero under the existing convention. Class-0 recall cannot be assessed for that dimension.

Both natural-sampling adapted runs predict only class 2. Balanced sampling predicts Accuracy class 1 on three examples whose Claude Accuracy label is 2, so it loses three agreements. It correctly identifies none of Claude's three Accuracy class-1 examples or four Context class-1 examples. On the human reference, those same three predictions were correct class-1 detections; the annotation source changes their interpretation. No JSON failures; all three greedy repetitions agree.

Higher agreement here is agreement with the teacher's overwhelmingly class-2 labels. The models still fail to detect the available minority cases. These scores do not establish improvement in human agreement. Comparing human- and Claude-scored runs together is refused by the comparison script.

`comparison/` contains aggregate tables and plots; each run directory contains re-labeled raw prediction records, metrics and provenance. No next training experiment was launched. If a new training experiment is selected, a longer balanced-sampling run remains a possible next test, scored against the now-frozen Claude columns.
