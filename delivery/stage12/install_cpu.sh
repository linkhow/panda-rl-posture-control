#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
python_seed="${1:-python3.11}"
environment_path="${2:-$repo_root/.venv}"
if [[ -e "$environment_path" ]]; then
  echo "Environment path already exists; choose a new directory." >&2
  exit 2
fi
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$python_seed" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$python_seed" -m venv "$environment_path"
python_bin="$environment_path/bin/python"
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$python_bin" -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch==2.13.0+cpu'
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$python_bin" -m pip install --no-cache-dir --index-url https://pypi.org/simple -r "$repo_root/requirements.cpu.lock.txt"
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$python_bin" -m pip check
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$python_bin" -m pip freeze
