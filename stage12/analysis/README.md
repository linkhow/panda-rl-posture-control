# Stage12 analysis entry point

`report_numbers.json` supplies the numeric values for both report editions. `original_analysis.json` and `supplementary_analysis.json` keep the opened original experiment and known-family supplementary evaluation separate. The continuous tables use jointly complete successes with a separate count for nonmissing clearance; the failure tables retain every failed episode. Per-category and per-template tables preserve the generating groups.

The computation is post hoc descriptive analysis, with the supplementary analysis rule saved before those results were analyzed. The saved `protocol.json` specifies a paired template-cluster bootstrap with 5,000 replicates and PCG64 seed 12042026. It does not assume that repeated policy seeds create additional independent tasks, and its percentile ranges are sensitivity summaries rather than significance claims.

From the repository or full package, using the documented installed Python:

```bash
python stage12/analyze.py --output outputs/analysis_local
```

The ordinary clone contains the reference original episode CSV and published summaries needed for this command. Once the supplementary evaluation CSV is present, the command also analyzes it. To read the complete original traces from the full experiment package and recompute the 520 common-success smoothness values and projection leakage diagnostics:

```bash
python stage12/analyze.py \
  --raw-original artifacts/stage10/evaluation \
  --output outputs/analysis_with_original_traces
```

If raw evidence is extracted elsewhere, supply its relative or absolute location to `--raw-original`. The analysis never simulates, trains, tunes, changes models, reads feasibility-witness action files, or changes frozen historical files. It hashes its inputs and writes output to the chosen analysis directory. Saved trace diagnostics include the SHA256 of each of the 520 input state CSVs.

Useful files:

- `*_methods.csv`: every method's success counts and failure types.
- `*_strata.csv`: success/failure by category and template group.
- `*_continuous_common_primary.csv`, `*_strata_continuous.csv`: all-primary common-complete summaries, units and denominators.
- `*_paired_changes.csv`: exact gained/lost scenario IDs and their groups/categories.
- `*_paired_success.csv`, `*_paired_continuous.csv`: paired counts/differences and template-cluster ranges.
- `*_failures.csv`: full retained failure rows, including failure time and null endpoints.
- `original_pooled_compute_clipping.*`: original all-observed provider/core/outer timing samples and raw actor clipping counts; includes failure prefixes and must not be confused with complete-motion comparisons.
- `original_trace_diagnostics.*`: independently recomputed smoothness and directly observed approximate-projection leakage.
- `technical_interpretation_en.md`, `technical_interpretation_zh.md`: matching interpretation and engineering reflection without unverified student authorship.
- `analysis_validation.json`: agreement with the original paired IDs, numeric metrics, labels and common-success set; bilingual interpretation table agreement.
- `analysis_provenance.json`: exact source hashes and NumPy version.

Figure PNG/PDF assets share numeric values and method IDs across the two report languages; captions are translated in the report source. Charts label command smoothness as the RMS derivative of the **7D executed command**, not physical acceleration. Actor clipping is counted before legal Box clipping; subsequent posture-u clipping and motor speed limiting are separate quantities.
