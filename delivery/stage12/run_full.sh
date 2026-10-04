#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
if [[ ! -x "$python_bin" || $# -lt 1 ]]; then
  echo "Install with delivery/stage12/install_cpu.sh, then run: bash delivery/stage12/run_full.sh <quick_check|train|validate|evaluate|demo|analyze|generate|apf|regression|analyze_new|audit_geometry|audit_raw> ..." >&2
  exit 2
fi
entry="$1"; shift
case "$entry" in
  quick_check|train|evaluate|demo|analyze|check_results) script="$repo_root/tools/$entry.py" ;;
  validate) script="$repo_root/tools/validate.py" ;;
  generate) script="$repo_root/stage12/experiment.py"; set -- generate "$@" ;;
  apf) script="$repo_root/stage12/experiment.py" ;;
  analyze_new) script="$repo_root/stage12/analyze.py" ;;
  audit_geometry|audit_raw) script="$repo_root/stage12/$entry.py" ;;
  regression) script="$repo_root/tests/risk_regression.py" ;;
  *) echo "Unknown entry: $entry" >&2; exit 2 ;;
esac
exec env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$python_bin" -B "$script" "$@"
