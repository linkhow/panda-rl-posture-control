#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_seed="${1:-python3.11}"
if [[ -e "$repo_root/.venv" ]]; then echo ".venv already exists; inspect it or use a fresh clone." >&2; exit 2; fi
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$python_seed" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$python_seed" -m venv "$repo_root/.venv"
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$repo_root/.venv/bin/python" -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch==2.13.0+cpu'
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$repo_root/.venv/bin/python" -m pip install --no-cache-dir -r "$repo_root/requirements.cpu.lock.txt"
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$repo_root/.venv/bin/python" -m pip check
