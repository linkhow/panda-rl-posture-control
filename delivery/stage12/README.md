# Stage12 complete course submission / 完整课程代码包

This is the current portable course submission. The historical Stage11 package
is a lightweight demonstration snapshot; it does not supply the full training
entry. Stage12 includes training, simple validation, model loading, original
evaluation, new scenario generation, APF selection and comparison, configuration,
all original parameter splits, all six frozen models (three best are primary),
formal reports and result indices.

本目录为当前推荐的完整课程提交入口。历史 Stage11 轻量包继续保留，适合演示；
完整训练、验证和本轮 APF / 新场景实验使用本 Stage12 代码包。

## Installation and practical entries / 安装与入口

Use Python 3.11. The CPU lock records exact versions. The installation script
creates a new `.venv` inside a fresh clone or extracted code package, installs
`torch==2.13.0+cpu` from the official PyTorch CPU index, installs the remaining
locked dependencies, then runs `pip check`.

```bash
bash delivery/stage12/install_cpu.sh python3.11
bash delivery/stage12/run_full.sh quick_check --output my_quick_check
bash delivery/stage12/run_full.sh validate --seed 550901 --n 3 --output my_validation
bash delivery/stage12/run_full.sh regression --output outputs/my_regression.json
```

`quick_check` loads all six models and checks five preselected motor-executed
success/failure examples against saved states and summaries. Every output name
must be new; entries reject overwrite. To use an independently created venv,
set `PYTHON_BIN` to its Python executable. There is no dependency on an old
project folder, a build-machine package path, CUDA, ROS, witness policy inputs,
or `VecNormalize` state.

```bash
# Technical smoke only: 2048 interactions; excluded from formal results.
bash delivery/stage12/run_full.sh train --smoke --steps 2048 --output smoke_2048
# Full frozen training contract: 1,024,000 interactions for the selected seed.
bash delivery/stage12/run_full.sh train --seed 550901 --output training_550901
# All three seeds use 3,072,000 interactions, with fixed all-58 validation selection.
bash delivery/stage12/run_full.sh train --all-seeds --output training_all
bash delivery/stage12/run_full.sh evaluate --help
bash delivery/stage12/run_full.sh apf --help
bash delivery/stage12/run_full.sh generate --help
bash delivery/stage12/run_full.sh audit_geometry --output outputs/my_geometry_audit.json
bash delivery/stage12/run_full.sh analyze_new --output outputs/my_new_analysis
```

The published Stage12 protocol, freeze and raw tuning/generation/evaluation
directories refuse overwrite. Preserve compact published results as reference.
The frozen `apf audit` phase regenerates its integrity report and appends audit
metadata: use it in a disposable copy. For reference-preserving checks, use
`audit_geometry` and `audit_raw --output <new-file>` instead.
For a full rerun, make a new independent copy. The helper keeps the
published protocol/results in `replication_reference/stage12/` and creates clean
new experiment destinations while retaining the frozen original numerical core:

```bash
python3 stage12/prepare_reproduction.py --destination ../my-stage12-rerun
cd ../my-stage12-rerun
bash delivery/stage12/install_cpu.sh python3.11
for phase in plan tune freeze generate evaluate audit; do
  bash delivery/stage12/run_full.sh apf "$phase"
done
```

The destination must be new and outside the source repository. This does not
remove or rewrite any published historical hash or experiment artifact.

The formal training command retains the original budget, four independent
workers, 58-scene selection protocol and no test access. A checkpoint reload
is supported; exact Bullet trajectory continuation after interruption is not
promised. Reproduction can depend on CPU/OS/library numerical behavior; inspect
saved comparison outputs when moving to a different machine.

正式训练保持冻结预算、四 worker、58 个 validation 选模及 test 隔离。
2048 步 smoke 仅验证采样、更新、保存与重载，不追加本轮科学实验训练结果。
新环境验证使用独立复制目录和独立 venv，未改变原工作环境；相同 Linux 主机的
结果不能等同于跨 OS 或跨硬件验证。

## Package contents and evidence / 包内容与证据

`me5418/`, `configs/`, `datasets/`, `models/`, `tools/`, `scripts/`, `tests/`
provide runtime code and inputs. `provenance/release_integrity.json` continues to
check the unchanged historical release. `results/reference/episodes_all1048.csv`
and companion indices preserve the original opened-test experiment. `stage12/`
is an independent extension, with its own frozen APF/new-scene protocol and
results. `docs/reports/stage12/` supplies editable bilingual report sources and
PDFs. `delivery/stage12/validation/` records the actual new-environment checks.

The ordinary clone and complete code package support the same runtime entries.
Both can independently audit geometry and shared execution contracts without
downloading raw evidence. `audit_raw` and `apf audit` require the evidence asset:

```bash
bash delivery/stage12/run_full.sh audit_raw --output outputs/my_raw_audit.json
bash delivery/stage12/run_full.sh audit_geometry --output outputs/my_geometry_check.json
```

The separate experiment-evidence asset adds every APF validation trial, every
accepted/rejected generated candidate, real-motor feasibility witness and replay,
and all new comparison episodes including failures. Witnesses are forensic
evidence and do not enter policy observations. A separate historical evidence
asset retains the original raw 1048 episodes.

普通 clone 或完整代码包均能运行训练、验证、原实验评价和新增实验入口。
原始状态、动作日志体积较大，通过 Release 的证据资产提供；不需要它们也能
加载模型、跑代表案例和重新运行协议。证据包保留全部失败，见证不进入策略输入。

```bash
# Build only after final experiments and audits complete; no environment/cache included.
python3 tools/package_stage12.py --destination ../me5418-release-assets --include-evidence
cd ../me5418-release-assets
sha256sum -c SHA256SUMS
tar -xzf me5418-stage12-complete.tar.gz
tar -xzf me5418-stage12-experiment-evidence.tar.gz
cd me5418-stage12-complete
bash delivery/stage12/install_cpu.sh python3.11
bash delivery/stage12/run_full.sh quick_check --output extracted_quick_check
```

Every asset has SHA-256 and a relative-path, per-file manifest. Keep generated
archives, installed environments and large raw evidence outside normal Git.
The top-level README and `stage12/results/` give final experiment counts and
Release download instructions; the new experiment is a supplementary evaluation
within the known task family, with development exposure explicitly acknowledged.
