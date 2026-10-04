"""Portable release boundary, frozen-file checks, and shared CLI helpers."""
from pathlib import Path
import csv
import datetime
import hashlib
import importlib.metadata
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (550901, 551901, 552901)
EXPECTED_SUCCESSES = {
    "tracking": 106, "APF": 123,
    "PPO_best_550901": 129, "PPO_best_551901": 127, "PPO_best_552901": 131,
    "PPO_last_550901": 125, "PPO_last_551901": 129, "PPO_last_552901": 125,
}
_AUDIT = None


def contained(path, base):
    path, base = Path(path).resolve(), Path(base).resolve()
    return path == base or base in path.parents


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def is_original_archive(path):
    # The guard travels with a clone and contains no machine-specific absolute path.
    return "ME5418_Group44" in Path(path).parts and not contained(path, ROOT)


class AccessAudit:
    def __init__(self, allow_test=False):
        self.allow_test = allow_test
        self.project_reads = set()
        self.dataset_reads = set()
        self.denied = []

        def hook(event, arguments):
            if event != "open" or not isinstance(arguments[0], (str, bytes, os.PathLike)):
                return
            path = Path(os.fsdecode(arguments[0])).resolve()
            witness = "private_witnesses" in path.parts
            forbidden_test = not self.allow_test and path.name == "test.json" and path.parent.name == "splits" and "me5418-scenes-v1" in path.parts
            if witness or is_original_archive(path) or forbidden_test:
                self.denied.append({"path": str(path), "reason": "witness" if witness else "test_split" if forbidden_test else "original_archive"})
                raise PermissionError("Release data boundary rejected " + str(path))
            if contained(path, ROOT):
                self.project_reads.add(str(path.relative_to(ROOT)))
                if "datasets" in path.parts:
                    self.dataset_reads.add(str(path.relative_to(ROOT)))

        sys.addaudithook(hook)

    def record(self):
        return {
            "allow_test_parameters_for_evaluation_only": self.allow_test,
            "opened_project_paths": sorted(self.project_reads),
            "opened_dataset_paths": sorted(self.dataset_reads),
            "denied_probe_or_reads": self.denied,
            "original_archive_reads": 0, "private_witness_reads": 0,
            "manifest_scope": "Parameter-only public manifest; no offline witness action input",
        }


def bootstrap(allow_test=False):
    global _AUDIT
    if _AUDIT is not None:
        if _AUDIT.allow_test != allow_test:
            raise RuntimeError("Cannot change data permission inside a running process")
        return _AUDIT
    # Imports from the release root are explicit, independent of the invocation cwd.
    clean = []
    for entry in sys.path:
        if not entry:
            continue
        path = Path(entry).resolve()
        if is_original_archive(path) or "/opt/ros" in str(path):
            continue
        if path != ROOT:
            clean.append(entry)
    sys.path[:] = [str(ROOT)] + clean
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib_cache"))
    _AUDIT = AccessAudit(allow_test)
    module_paths()
    return _AUDIT


def module_paths():
    paths = {
        name: str(Path(module.__file__).resolve())
        for name, module in sys.modules.items()
        if (name == "me5418" or name.startswith("me5418.")) and getattr(module, "__file__", None)
    }
    if not all(contained(path, ROOT / "me5418") for path in paths.values()):
        raise RuntimeError("Imported me5418 code outside this release: " + repr(paths))
    return paths


def reserve_output(tag):
    output = Path(tag)
    if not output.is_absolute():
        output = ROOT / "outputs" / output
    output = output.resolve()
    if not contained(output, ROOT / "outputs") or output == ROOT / "outputs":
        raise ValueError("Output must be a new directory inside this repository's outputs/")
    output.mkdir(parents=True, exist_ok=False)
    return output


def verify_release(check_environment=True):
    manifest = read(ROOT / "provenance/release_integrity.json")
    failures = []
    excluded = []
    for relative, expected in manifest["files_SHA256"].items():
        if _AUDIT is not None and not _AUDIT.allow_test and relative == "datasets/me5418-scenes-v1/splits/test.json":
            # Training validates its train/validation inputs without opening the test split.
            excluded.append(relative)
            continue
        path = ROOT / relative
        if not contained(path, ROOT) or not path.is_file() or sha(path) != expected:
            failures.append(relative)
    if failures:
        raise RuntimeError("Missing or changed frozen release files: " + repr(failures))
    report = {"hashes_match": True, "checked_files": len(manifest["files_SHA256"]) - len(excluded), "excluded_by_training_test_boundary": excluded}
    if check_environment:
        recorded = manifest["environment"]
        actual_python = sys.version.split()[0]
        if actual_python.split(".")[:2] != recorded["python"].split(".")[:2]:
            raise RuntimeError("Recorded Python major/minor required: " + recorded["python"])
        actual_packages = {}
        package_versions = recorded["packages"] if "packages" in recorded else {name: recorded[name] for name in ("pybullet", "gymnasium", "stable-baselines3", "torch", "numpy", "matplotlib", "tensorboard")}
        for name, expected in package_versions.items():
            actual = importlib.metadata.version(name)
            actual_packages[name] = actual
            # CPU and CUDA builds of the same PyTorch release share the model format.
            equal = actual.split("+")[0] == expected.split("+")[0] if name == "torch" else actual == expected
            if not equal:
                raise RuntimeError(f"Recorded package version required: {name}={expected}; installed {actual}")
        import pybullet_data
        urdf = Path(pybullet_data.getDataPath()) / "franka_panda/panda.urdf"
        if sha(urdf) != recorded["builtin_panda_urdf_sha256"]:
            raise RuntimeError("Installed Panda URDF differs from frozen asset")
        report["environment"] = {
            "python": actual_python, "recorded_python": recorded["python"],
            "python_patch_matches": actual_python == recorded["python"],
            "packages": actual_packages, "device": "cpu", "panda_urdf_matches": True,
            "torch_build_may_differ": "Same base version; outputs must still pass numeric comparisons",
        }
    return report


def methods():
    entries = read(ROOT / "models_index.json")["models"]
    expected_keys = {(seed, kind) for seed in SEEDS for kind in ("best", "last")}
    if len(entries) != 6 or {(e["seed"], e["kind"]) for e in entries} != expected_keys:
        raise ValueError("Exactly six frozen best/last model entries are required")
    result = [
        {"name": "tracking", "kind": "tracking", "role": "primary", "seed": None, "checkpoint": None, "model_SHA256": None},
        {"name": "APF", "kind": "apf", "role": "primary", "seed": None, "checkpoint": None, "model_SHA256": None},
    ]
    for kind in ("best", "last"):
        for seed in SEEDS:
            entry = next(e for e in entries if e["seed"] == seed and e["kind"] == kind)
            if entry["role"] != ("primary" if kind == "best" else "precommitted_secondary"):
                raise ValueError("Frozen model role changed")
            result.append({
                "name": f"PPO_{kind}_{seed}", "kind": "policy", "role": entry["role"],
                "seed": seed, "checkpoint": kind, "model_SHA256": entry.get("original_archive_SHA256", entry["SHA256"]),
                "release_model_SHA256": entry["SHA256"],
                "model_path": entry["package_model_path"], "config_path": entry["package_configuration_path"],
                "config_SHA256": entry["configuration_SHA256"], "selected_step": entry["selected_step"],
            })
    return result


def load_model(method):
    from stable_baselines3 import PPO
    # Accept either a models_index entry or an evaluation-method entry.
    model_path = method.get("package_model_path", method.get("model_path"))
    config_path = method.get("package_configuration_path", method.get("config_path"))
    model_hash = method.get("SHA256", method.get("release_model_SHA256", method.get("model_SHA256")))
    config_hash = method.get("configuration_SHA256", method.get("config_SHA256"))
    if sha(ROOT / model_path) != model_hash or sha(ROOT / config_path) != config_hash:
        raise RuntimeError("Model/configuration checksum changed")
    config = read(ROOT / config_path)
    model = PPO.load(ROOT / model_path, device="cpu")
    if model.observation_space.shape != (91,) or model.action_space.shape != (7,):
        raise ValueError("Frozen observation/action shapes are 91/7")
    if config["observation_version"] != "geometry91-v1" or config["reward_version"] != "bounded-v1":
        raise ValueError("Frozen configuration versions changed")
    return model, config


def snapshot_checks():
    rows = list(csv.DictReader((ROOT / "results/reference/episodes_all1048.csv").open()))
    counts, order = {}, {}
    for row in rows:
        name = row["method"]
        counts[name] = counts.get(name, 0) + (row["success"] == "True")
        order.setdefault(name, []).append(row["scene_id"])
    split_ids = read(ROOT / "datasets/me5418-scenes-v1/splits/test.json")["scene_ids"]
    split_order = {split: read(ROOT / f"datasets/me5418-scenes-v1/splits/{split}.json")["scene_ids"] for split in ("train", "validation", "test")}
    public = read(ROOT / "datasets/me5418-scenes-v1/manifest.json")["scenes"]
    method_index = {method["name"]: method for method in methods()}
    checks = {
        "all1048_rows": len(rows) == 1048,
        "counts_match": counts == EXPECTED_SUCCESSES,
        "all_eight_methods": set(order) == set(EXPECTED_SUCCESSES),
        "same_fixed131_order": len(split_ids) == len(set(split_ids)) == 131 and all(ids == split_ids for ids in order.values()),
        "fixed151_58_131_split_counts": {split: len(ids) for split, ids in split_order.items()} == {"train": 151, "validation": 58, "test": 131},
        "disjoint340_scenes": len({identifier for ids in split_order.values() for identifier in ids}) == 340 and len(public) == 340 and all(set(ids) == {scene["id"] for scene in public if scene["split"] == split} for split, ids in split_order.items()),
        "model_roles_and_archive_hashes_match": all(row["role"] == method_index[row["method"]]["role"] and row["model_SHA256"] == (method_index[row["method"]]["model_SHA256"] or "") for row in rows if row["method"] in method_index),
    }
    return {"pass": all(checks.values()), "checks": checks, "official_successes_snapshot": counts, "physics_episodes": 0, "training_interactions": 0}
