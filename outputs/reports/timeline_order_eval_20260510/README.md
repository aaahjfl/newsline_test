# Timeline Ordering Evaluation Materials

This directory contains the manual annotation materials for the section 6.3.1
end-to-end timeline ordering experiment.

## Files

- `manifest.md`: fixed input files, hashes, random seed, sampling rules and formulas.
- `sampled_nodes.csv`: sampled timeline nodes with titles, times, confidence, risk flags and evidence URLs.
- `pair_annotation.csv`: event-pair annotation sheet. Fill `human_label`, `judgment_basis` and optional `notes`.
- `pair_annotation_reviewed.csv`: superseded same-anchor sanity-check annotation sheet; do not use as independent manual evaluation.
- `metrics_summary_same_anchor_superseded.csv`: archived same-anchor sanity-check summary.
- `node_reference_independent.csv`: independent node-level reference-date review table.
- `pair_annotation_independent_review.csv`: final pair labels derived from the independent node review.
- `metrics_summary.csv`: final independent-review metric summary for thesis table 6-2.
- `pair_annotation_time_anchor_draft.csv`: machine-assisted draft labels generated only from resolved time anchors; review is required before treating them as manual labels.
- `metrics_summary_time_anchor_draft.csv`: draft metric preview computed from the time-anchor draft labels.

## Valid Human Labels

- `left_before`
- `right_before`
- `same_time`
- `uncertain`

Only `left_before` and `right_before` pairs are effective for Kendall's tau and
ordering Accuracy.
