# Topic Recall Ablation Experiment

This folder contains a sidecar experiment for thesis section 4.8.4.

The runner is intentionally isolated from the production event-discovery entry
point. It reads `parser_newsdata`, reuses pure in-memory candidate filtering,
embedding, clustering, and event-building functions, and writes only a new
report directory under `outputs/reports/`.

It does not call `run_event_discovery()` or persist anything to MySQL.

Example:

```bash
python tests/topic_recall_ablation/run_topic_recall_ablation.py --timeline-mode none
```

For a full final-node count with the same in-memory LLM decision logic, use:

```bash
python tests/topic_recall_ablation/run_topic_recall_ablation.py --timeline-mode standard
```
