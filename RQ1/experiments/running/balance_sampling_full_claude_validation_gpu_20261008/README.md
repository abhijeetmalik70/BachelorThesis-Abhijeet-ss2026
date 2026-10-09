# Full-pool balanced Accuracy oversampling — 8 October 2026

Launched on idisc-09, RTX A4000 16 GB. Results pending. Remote process PID: 773754.

Retains all 1,650 eligible rows after excluding the contradictory duplicate pair with record IDs 1384 and 1385. The source workbook is unchanged. Unweighted loss; no synthetic data. Each Accuracy class is presented 1,558 times per epoch, giving 4,674 presentations. Context & Relevance presentations are 1,661/841/2,172; that dimension is not independently balanced. Uses 2 epochs, effective batch 24, learning rate 2e-4, seed 42, preserve-question preprocessing with sequence budget 512, and a fresh Gemma 3 4B adapter. The fixed presentation list is shuffled each epoch.

Unlike the previous 500-row pool runs, every eligible example is presented each epoch and minority rows repeat to the majority count. Training comprises 9,348 presentations and 390 optimizer steps, so both pool coverage and training exposure increase. It is not a comparison holding update count fixed.

Evaluates on the same 100 validation pairs using claude_accuracy and claude_context_relevance. No exact training/validation pair overlap, but shared question texts remain. Completed human-scored runs are preserved.

Remote root: `/home/abma00001/rq1_balance_sampling_full_20261008`. Log: `experiment.log`. Expected result folder: `RQ1/experiments/results/balance_sampling_full_claude_validation/`. The background job survives SSH disconnect. Training is estimated at 75–85 minutes plus evaluation based on previous runs; this is not a completion guarantee.

The presentation CSV here is the locally planned schedule. Verify it against the runtime sampling_presentations.csv once available. No completed results have been copied back yet.
