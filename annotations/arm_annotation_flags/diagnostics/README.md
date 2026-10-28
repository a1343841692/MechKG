# Existing Follow-up Score Diagnostic

These files are copied unchanged from the analysis dated October 12, 2026. They are included to preserve the later result alongside the independently generated flags.

The primary matched comparison B did not establish lower out-of-fold MechKG scores for the flagged labels: P(control > flagged) was 0.478 (0.418–0.540) for total logits and 0.455 (0.410–0.507) for summed graph evidence. These 95% intervals include 0.5.

`flagged_label_rows.csv` has one row for each of the 137 labels; `all_oof_rows.csv` includes all 10,112 validated positives. Validation labels use their held-out fold. Test logits/contributions are averaged over five models. Model and validation-fold indices in these original diagnostic files are zero-based; the main release table uses one-based `validation_fold` and an empty value for test pairs.

`flagged_label_scores.csv` contains all planned comparisons and components. `PROTOCOL.zh-CN.md` and `summary.zh-CN.md` preserve the complete original protocol, results and limitations; other CSVs record matching coverage, per-label comparisons, residual imbalance and threshold summaries. `sources.json` and `validation.json` are the original provenance and validation records, not evidence of a new analysis performed during packaging.
