#!/usr/bin/env python3
"""Build portable course code and raw-evidence Release assets with SHA-256.

The complete code asset runs independently. The experiment-evidence asset is
optional for execution, and extracts beside the code under outputs/stage12/.
No environment, git metadata, caches, chat material, or credentials are packed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ("me5418", "configs", "datasets", "models", "provenance", "references", "tools", "scripts", "results", "docs", "media", "scenes", "environment", "stage12", "tests", "delivery/stage12")
TOP_FILES = ("README.md", "run.sh", "requirements.cpu.lock.txt", "models_index.json", ".gitignore", ".gitattributes")
EXCLUDED_PARTS = {".git", ".venv", "__pycache__", ".cache", ".matplotlib_cache", "dist"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".tmp"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def permitted(path, base):
    relative = path.relative_to(base)
    if path.is_symlink():
        raise ValueError("Symlink refused: " + str(relative))
    return path.is_file() and not EXCLUDED_PARTS.intersection(relative.parts) and path.suffix not in EXCLUDED_SUFFIXES and not path.name.startswith(".env")


def collect_code():
    files = {ROOT / name for name in TOP_FILES if (ROOT / name).is_file()}
    for directory in DIRECTORIES:
        base = ROOT / directory
        if base.exists():
            files.update(p for p in base.rglob("*") if permitted(p, ROOT))
    # The package itself must contain working complete entry points and frozen
    # task inputs, not just demo executables or a requirements file.
    required = ["tools/train.py", "tools/validate.py", "tools/evaluate.py", "tools/demo.py", "stage12/experiment.py", "delivery/stage12/run_full.sh", "provenance/release_integrity.json", "datasets/me5418-scenes-v1/manifest.json"]
    required += [f"models/best_seed{seed}.zip" for seed in (550901, 551901, 552901)]
    missing = [name for name in required if ROOT / name not in files]
    if missing:
        raise FileNotFoundError("Complete-course package prerequisites missing: " + repr(missing))
    frozen = json.loads((ROOT / "provenance/release_integrity.json").read_text())
    changed = [name for name, expected in frozen["files_SHA256"].items() if sha(ROOT / name) != expected]
    if changed:
        raise ValueError("Historical frozen files changed: " + repr(changed))
    return sorted(files), sha(ROOT / "provenance/release_integrity.json")


def make_asset(files, destination, prefix, kind, frozen_sha, extra):
    if destination.exists():
        raise FileExistsError(destination)
    records = {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in files}
    manifest = {"format": "me5418-stage12-package-v1", "kind": kind, "files": records, "file_count": len(records), "uncompressed_bytes": sum(r["bytes"] for r in records.values()), "historical_release_integrity_sha256": frozen_sha, "relative_paths_only": True, **extra}
    encoded = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()
    import io
    with tarfile.open(destination, "w:gz", compresslevel=6) as tar:
        for path in files:
            info = tar.gettarinfo(str(path), arcname=f"{prefix}/{path.relative_to(ROOT)}")
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as f:
                tar.addfile(info, f)
        info = tarfile.TarInfo(f"{prefix}/PACKAGE_MANIFEST_{kind.upper()}.json")
        info.size = len(encoded)
        info.mode = 0o644
        info.mtime = 0
        tar.addfile(info, io.BytesIO(encoded))
    manifest_path = destination.with_suffix("").with_suffix(".manifest.json")
    manifest_path.write_bytes(encoded)
    return {"asset": destination.name, "sha256": sha(destination), "bytes": destination.stat().st_size, "manifest": manifest_path.name, "manifest_sha256": sha(manifest_path), "kind": kind, "file_count": len(records), "uncompressed_bytes": manifest["uncompressed_bytes"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path, required=True, help="New directory outside the source checkout")
    p.add_argument("--include-evidence", action="store_true", help="Add separate raw stage12 asset including every failure and witness replay")
    args = p.parse_args()
    target = args.destination.resolve()
    if target == ROOT or ROOT in target.parents:
        p.error("Destination must be outside the source checkout to prevent recursive packaging")
    target.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    files, frozen_sha = collect_code()
    prefix = "me5418-stage12-complete"
    assets = [make_asset(files, target / "me5418-stage12-complete.tar.gz", prefix, "code", frozen_sha, {"training_entry": "tools/train.py", "recommended_entry": "delivery/stage12/run_full.sh", "historical_demo_separate": "Original delivery/stage11 is lightweight and lacks complete training entry"})]
    if args.include_evidence:
        base = ROOT / "outputs/stage12"
        if not base.is_dir():
            raise FileNotFoundError("Raw stage12 evidence is absent")
        evidence = sorted(p for p in base.rglob("*") if permitted(p, ROOT))
        for required in ["stage12/results/execution_complete.json", "stage12/results/integrity_audit.json"]:
            if not (ROOT / required).is_file():
                raise FileNotFoundError("Finish experiment verification before packaging: " + required)
        # Includes candidates, unsuccessful witnesses, tuned-grid failures and
        # failed comparisons. There is no success-only payload filter.
        assets.append(make_asset(evidence, target / "me5418-stage12-experiment-evidence.tar.gz", prefix, "evidence", frozen_sha, {"policy_input": False, "extraction": "Extract beside complete package to add outputs/stage12", "all_successes_and_failures": True}))
    index = {"format": "me5418-stage12-release-assets-v1", "assets": assets, "historical_release_integrity_sha256": frozen_sha, "build_wall_s": time.perf_counter() - started, "ordinary_clone_can": ["load all six models", "run preselected success/failure examples", "train from scratch and run validation", "repeat original 131x8 evaluation", "run new APF and heldout protocol in a prepared independent copy", "independently audit geometry and shared execution contracts", "recompute cluster analysis from compact full episode rows", "build portable complete package"], "complete_code_package_can": ["same runtime operations as clone without git metadata", "read bilingual reports and compact experiment indices"], "evidence_asset_adds": ["every validation-grid episode raw state/action log", "all heldout acceptance/rejection and executed witness/replay evidence", "all six-method heldout raw state/action logs", "independent raw600 checksum, terminal failure and statistic audit"], "historical_1048_raw_asset": "Separate historical evidence asset maintained by release publisher; original compact1048 result rows are already in code package"}
    (target / "release_assets_index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    checks = [f"{sha(path)}  {path.name}" for path in sorted(target.iterdir()) if path.is_file()]
    (target / "SHA256SUMS").write_text("\n".join(checks) + "\n")
    print(json.dumps(index, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
