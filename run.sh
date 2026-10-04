#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 1 ]]; then
  echo "Usage: bash run.sh <quick_check|check_results|demo|evaluate|train|analyze> [arguments]" >&2
  exit 2
fi
entry="$1"; shift
case "$entry" in quick_check|check_results|demo|evaluate|train|analyze) ;; *) echo "Unknown entry: $entry" >&2; exit 2;; esac
python_bin="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "Create .venv first, or set PYTHON_BIN to an installed Python 3.11 interpreter." >&2
  exit 2
fi
exec env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$python_bin" -B "$repo_root/tools/$entry.py" "$@"
