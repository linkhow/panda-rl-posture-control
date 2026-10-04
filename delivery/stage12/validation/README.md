# Actual new-environment and final-source verification

The final requested checks passed in a separate source directory with the newly
installed CPU virtual environment. `final_source_verification.json` is the
current entry point to the evidence; the original runtime output files are kept
in that independent directory. Public evidence copies replace machine paths
with labels and keep numerical values unchanged.

## Exact environment and installation

Python **3.11.16**, Linux x86_64. The principal installed versions are
PyBullet **3.2.7**, Gymnasium **1.3.0**, Stable-Baselines3 **2.9.0**,
PyTorch **2.13.0+cpu**, NumPy **2.4.6**, Matplotlib **3.11.2** and
TensorBoard **2.21.0**. The complete installed package list and actual dependency
module locations are in `environment_fingerprint.json`.

An existing Python 3.11.16 interpreter and its standard library supplied the
bootstrap. A new venv was created with system site-packages disabled. All checked
scientific modules import from the new venv. The original work environment was
not modified. These are same-host checks, not a different machine or operating
system validation.

The actual installation used the following commands, with paths represented by
labels in public logs. Torch was freshly installed from the official CPU index.
The locked packages were installed, then reinstalled from official PyPI with
the exact same versions to remove dependence on the build machine's mirror.

```bash
<python3.11> -m venv <new-venv>
<new-venv>/bin/python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch==2.13.0+cpu'
<new-venv>/bin/python -m pip install --no-cache-dir -r <independent-copy>/requirements.cpu.lock.txt
<new-venv>/bin/python -m pip install --no-cache-dir --force-reinstall --no-deps --index-url https://pypi.org/simple -r <independent-copy>/requirements.cpu.lock.txt
<new-venv>/bin/python -m pip check
```

`cpu_torch_install.log`, `official_pypi_install.log` and `pip_check.log` retain
the actual installation outputs. For a normal fresh clone/package installation,
use `bash delivery/stage12/install_cpu.sh python3.11`: it directly selects the
official PyTorch and PyPI sources.

## Final source copy and checks

The latest publication source was copied to a new directory with `.git`, venvs,
runtime outputs and caches excluded. `final_source_copy_record.json` records
222 copied files and their SHA-256 values. After the checks, documentation and
uncalled analysis/audit additions were refreshed; all executed runtime inputs were
unchanged. `validation_runtime_inputs_SHA256.json` identifies the final sources
and inputs, including all **49** unchanged historical frozen files.

```bash
PYTHON_BIN=<new-venv>/bin/python bash <independent-final-source>/delivery/stage12/run_full.sh quick_check --output final_quick_check
PYTHON_BIN=<new-venv>/bin/python bash <independent-final-source>/delivery/stage12/run_full.sh regression --output <independent-final-source>/outputs/final_risk_regression.json
PYTHON_BIN=<new-venv>/bin/python bash <independent-final-source>/delivery/stage12/run_full.sh audit_geometry --output <independent-final-source>/outputs/final_geometry_audit.json
```

- `final_quick_check.json`: all six models load; five physically executed
  representatives include three successes and two expected failures. Every
  compared state agrees with the original record; maximum numeric difference
  is **0**.
- `final_risk_regression.json`: all **nine** existing risk checks pass, with no
  skipped checks. They cover action scale/clip/hold, invalid action rejection,
  termination versus truncation, latched failure, real obstacle queries and
  limit gates, information boundaries, six model loads, original versus Stage12
  fixed-APF state equality, the 100-scene parameter manifest and safe independent
  reproduction copying. The self-collision classification probe uses an injected
  monitored flag; the external collision probe uses real PyBullet geometry.
- `final_geometry_audit.json`: all **100** new scenes have zero exact or near
  duplicates and zero overlap with **340** old accepted / **755** old reserved
  geometries. Initial joints/fingers and the execution contract, excluding the
  intended obstacle placement, match the original.

The matching `.log` files preserve console output. Earlier `quick_check.json`,
`risk_regression.json` and representative state checksums record the first
independent copy verification; the files prefixed `final_` identify the latest
requested recheck.

## Earlier technical smoke and limits of evidence

`training_smoke/` and `training_smoke.log` record **2048** policy interactions
using smoke seed **559901**. One rollout produced **10** PPO optimizer epochs,
changed policy parameters and saved/loaded optimizer state; deterministic action
reload difference was **0**. The run used no test episodes. It is an engineering
entry-point check and contributes no scientific training result or additional
formal seed. `validation_smoke.json` records three validation scenes, all three
successful, as a simple sanity check rather than a selection or test result.

The final requested recheck did not repeat the full 600-episode scientific
comparison, full-budget training, PDF builds, GUI, hardware operation, or strict
real-time validation. Those scopes must not be inferred from these checks.

All public JSON/log strings are normalized consistently. Labels denote source,
copy, bootstrap, venv or unknown machine paths; public download URLs remain
intact. Original output data, numerical values, scientific results and historical
hashes are not edited by this normalization. Verify the published evidence files
from this directory with:

```bash
sha256sum -c SHA256SUMS
```

本目录记录实际新环境与最终独立源码核对。当前入口为
`final_source_verification.json`：六模型加载、五代表案例（3成功、2预期失败）
比较状态最大差0；九项实际风险检查全部通过；100新场景对340旧接受及755旧候选
几何无重复或重叠。原环境未修改，但复用同一主机的Python解释器/标准库，不属于
跨机器验证。2048步smoke只验证训练入口，未计入科学结果。公开日志仅规范化本机
路径，原始输出与数字保留；哈希清单验证所有公开证据副本。
