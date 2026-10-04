# Stage12 supplemental APF and held-out experiment

The executed comparison contains100 new scenes in19 accepted template groups: tracking93/100, original fixedAPF99/100, selectedAPF100/100, and the three frozen PPO best models98/100,98/100,99/100. This supplemental set therefore does not support PPO success-rate superiority over the selectedAPF. Every13 failed episode is retained. The9 validation candidates all scored58/58; the predeclared RMS-error tie-break selected influence distance0.06m and repulsion strength2rad²/(m·s). This finite-grid selection is not a claim of globally optimal APF.

The actual scientific execution budget was522 validation episodes,101 offline witness attempts,100 verified replays, and600 comparison episodes (1323 episodes,0 new training interactions). Phase wall measurements were424.689s,204.068s and512.630s, respectively, each below its predeclared1800s cap. Of288 generated candidates,136 were geometry-rejected,100 accepted and52 unsearched after reaching the target. The accepted mix wasfar41/near49/tight10, and99 witnesses used the original fixedAPF; these distribution and screening limits matter when interpreting the results.

This independently versioned extension imports the frozen numerical core in `me5418/`; it does not edit the original controller, models, physics, success rules, test131, or historical hashes. `provenance/release_integrity.json` is checked at every phase. The original commands in `tools/` retain their original meanings.

Run from any working directory using the repository's CPU environment (the examples assume `cd` into the repository):

```bash
.venv/bin/python -B stage12/experiment.py plan
.venv/bin/python -B stage12/experiment.py tune
.venv/bin/python -B stage12/experiment.py freeze
.venv/bin/python -B stage12/experiment.py generate
.venv/bin/python -B stage12/experiment.py evaluate
.venv/bin/python -B stage12/experiment.py audit
```

The published protocol and results already exist. Commands refuse to overwrite protocol, freeze, or raw phase directories. Prepare an independent runnable copy first:

```bash
.venv/bin/python -B stage12/prepare_reproduction.py --destination ../me5418-stage12-repeat
cd ../me5418-stage12-repeat
bash delivery/stage12/install_cpu.sh /path/to/python3.11
```

The helper refuses an existing destination, copies the required code/models/data, verifies all49 historical frozen files, and retains the published Stage12 artifacts in `replication_reference/stage12/`. It clears only generated Stage12 files in the new copy; the original repository is unchanged. It excludes virtual environments, caches, Git metadata and raw outputs. Then execute the six commands above in the new copy. Preserve the supplied evidence and do not rewrite any original provenance hash. Existing public Stage12 files can instead be audited using the complete experiment package. Raw evidence lives under `outputs/stage12/` and is distributed with that package; a normal clone includes the protocol, parameter-only100-scene manifest, compact results, witness checksum index and analysis.

The finite APF grid contains nine candidates: influence distance0.06,0.10,0.14m crossed with repulsion strength2,4,8rad²/(m·s). All other APF fields stay fixed. The522 episodes use only the existing validation58, never the opened test131. Ranking is complete-success count, then mean RMS position error among complete successes, then the declared grid order. The successful subset used for that tie-break is recorded; failed prefixes are never omitted from success counts or failure logs. An incomplete grid cannot select or freeze parameters.

After freezing, generation samples288 candidates from24 new template groups, using seed2026100412 and the original fixed initial state,4s quintic3–8cm line and one6cm-radius static sphere. Category assignment precedes controller outcomes. Exact task-geometry keys and near-duplicate tolerances exclude overlaps with all696 old formal candidates (including accepted340), both20-candidate pilot pools and19 current Stage04 records (18 corrected near-sphere candidates plus one far scene). `data/reserved_geometry.json` contains these geometry parameters and original file checksums, without old test outcomes or action traces.

Acceptance uses the first100 candidates with a complete actual-motor witness followed by a matching clipped48Hz-action replay. The predeclared search order is original fixedAPF, gentleAPF and earlyAPF; it never uses the selected tunedAPF or PPO. This finite APF-family witness screen can favour APF-family-solvable scenes. An unsuccessful search does not prove infeasibility. Every generated candidate remains in the full outcome record, including geometry rejection, failed attempts and candidates left unsearched after the100-scene target. The parameter-only policy manifest excludes witness metadata and actions. Offline witness replay is a separate explicit path.

The six evaluated methods are tracking, original fixedAPF, validation-selected APF and the three original frozen PPO best models. All use the same ordered100 new scenarios and original actuator permission, physical clock, safety and success checks. Evaluation never retrains or reselects PPO, never substitutes a successful retry for a task failure, and retains every600 episode. The new set is supplemental generalization within a previously developed task family; it is not an independent, development-unaffected research test.

See `configs/protocol.json`, `configs/tuned_apf_freeze.json`, `results/tuning_summary.json`, `results/generation_summary.json`, `results/execution_complete.json` and `results/integrity_audit.json` for actual budgets, selected values, outcomes, timings and verification. `results/episodes_new600.csv` uses the same per-episode metric schema as the original1048-episode table. Original and supplemental results are analyzed separately. Continuous comparisons require common complete-success scenes and show their denominators. Template-group resampling addresses within-group dependence; repeated training seeds do not create additional independent scenes.

The provider timer covers the48Hz APF processing or PPO encoding/inference/mapping and excludes cached collision queries, primary tracking and physics. The outer240Hz loop includes observation construction, provider when updated, primary tracking, motor execution, physics, distance queries and safety checks; it excludes scene loading,240-step settling, rendering and export. These CPU timings are measurements for this machine, not a guarantee of real-time execution.

Additional read-only auditors:

```bash
.venv/bin/python -B stage12/audit_geometry.py
# The full experiment package is required for the raw audit:
.venv/bin/python -B stage12/audit_raw.py
```

The geometry audit independently fingerprints numerical task parameters and checks all340 accepted/755 reserved old scenes, new duplicates and the shared execution dictionary. The raw audit checks6000 episode-file hashes,600 scenario fingerprints, complete duration/state-count/terminal-label consistency and maximum error from570589 saved states, including all13 failure prefixes. These are integrity/statistic audits; they do not claim a second complete physics replay.

Historical reservation metadata erratum: the frozen protocol and `data/reserved_geometry.json` say17 development records, but the array actually contains all19 current Stage04 records (18 corrected near-sphere constructions plus one far scene), totaling696+20+20+19=755. This is a literal count error, not an accepted-scene count, and frozen hashes remain unchanged. [The append-only erratum](results/reserved_scope_erratum.json) verifies all19 current geometries and records a post-execution overlap check against the original Stage04 initial pool (18 earlier erroneous near-sphere constructions plus the already-reserved far scene). The [parameter-only historical addendum](data/historical_initial_geometry_addendum.json) adds18 previously unlisted unique geometries; the new100 have zero exact or near overlaps with these19 initial records. This extra audit is post hoc and does not retroactively expand the pre-generation reservation scope. No experiment, model or training is rerun.
