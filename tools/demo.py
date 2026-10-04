#!/usr/bin/env python
"""Reproduce the five preselected success/failure pairs and compare all state columns."""
import argparse
from common import ROOT, bootstrap, load_model, methods, module_paths, read, reserve_output, sha, verify_release, write_json
from comparison import compare_states, compare_summary


def inspect_models(output, audit):
    import torch
    torch.set_num_threads(1)
    entries = read(ROOT / "models_index.json")["models"]
    loaded = []
    for entry in entries:
        model, config = load_model(entry)
        loaded.append({
            "seed": entry["seed"], "kind": entry["kind"], "role": entry["role"],
            "loaded": True, "model_SHA256": entry["SHA256"],
            "original_archive_SHA256": entry.get("original_archive_SHA256", entry["SHA256"]),
            "observation_shape": list(model.observation_space.shape), "action_shape": list(model.action_space.shape),
            "observation_version": config["observation_version"], "reward_version": config["reward_version"],
            "normalization": "Frozen manual scales; no VecNormalize dependency",
        })
    probes = {}
    for name, path in {
        "original_archive": ROOT.parent / "ME5418_Group44/me5418/scene.py",
        "private_witness": ROOT / "datasets/me5418-scenes-v1/private_witnesses/guard_probe.json",
    }.items():
        try:
            path.read_bytes()
            probes[name] = False
        except PermissionError:
            probes[name] = True
    result = {
        "pass": len(loaded) == 6 and all(probes.values()), "models": loaded,
        "intentional_guard_probes_denied": probes, "module_paths": module_paths(), "access": audit.record(),
        "physics_episodes": 0, "training_interactions": 0,
    }
    write_json(output / "models_inspection.json", result)
    if not result["pass"]:
        raise RuntimeError("Six-model/independence inspection failed")
    return result


def run_case(case_name, method_name, output, audit):
    import torch
    from me5418.final_evaluation import logged_run
    from me5418.scenarios import canonical_hash
    torch.set_num_threads(1)
    cases = read(ROOT / "scenes/representatives.json")
    case = cases[case_name]
    if method_name not in case["allowed_methods"]:
        raise ValueError("Method is not one of the preselected five representative pairs")
    method = next(method for method in methods() if method["name"] == method_name)
    scenario = case["parameters"]
    # Representatives must be exactly the public fixed-test parameters, never an adapted task.
    manifest = read(ROOT / "datasets/me5418-scenes-v1/manifest.json")
    public = next(scene for scene in manifest["scenes"] if scene["id"] == scenario["id"])
    from me5418.arm_env import PARAMETER_KEYS
    if canonical_hash({key: public[key] for key in PARAMETER_KEYS}) != canonical_hash(scenario):
        raise RuntimeError("Representative differs from fixed dataset parameters")
    model, config = load_model(method) if method["kind"] == "policy" else (None, None)
    episode = output / "episode"
    result = logged_run(scenario, method, episode, model, config)
    reference = ROOT / case["references"][method_name]["directory"]
    state_comparison = compare_states(reference / "states.csv", episode / "states.csv")
    # The frozen exporter converts NumPy arrays to JSON lists; compare those saved values.
    summary_comparison = compare_summary(reference / "summary.json", episode / "summary.json")
    passed = state_comparison["pass"] and summary_comparison["pass"] and result["connection_closed"] and result["joint_resets_during_motion"] == 0
    report = {
        "pass": passed, "case": case_name, "method": method_name, "scene_id": scenario["id"],
        "success": result["success"], "first_failure_time_s": result["first_failure_time_s"],
        "state_comparison": state_comparison, "summary_comparison": summary_comparison,
        "reference": str(reference.relative_to(ROOT)), "reference_states_SHA256": sha(reference / "states.csv"),
        "module_paths": module_paths(), "access": audit.record(),
        "physics_episodes": 1, "training_interactions": 0,
        "purpose": "Reproduction of preselected existing success/failure examples, not a new formal score",
    }
    write_json(output / "comparison.json", report)
    print({"case": case_name, "method": method_name, "success": result["success"], "first_failure_time_s": result["first_failure_time_s"], "pass": passed, "maximum_state_difference": state_comparison["maximum_numeric_difference"]}, flush=True)
    if not passed:
        raise RuntimeError("Representative differs; preserve comparison.json for investigation")
    return report


def run_suite(output, audit):
    reports = []
    for case_name, case in read(ROOT / "scenes/representatives.json").items():
        for method_name in case["allowed_methods"]:
            target = output / case_name / method_name
            target.mkdir(parents=True, exist_ok=False)
            reports.append(run_case(case_name, method_name, target, audit))
    if len(reports) != 5:
        raise RuntimeError("Exactly five frozen representative pairs required")
    result = {
        "pass": all(report["pass"] for report in reports), "physics_episodes": 5, "training_interactions": 0,
        "cases": [{key: report[key] for key in ("case", "method", "scene_id", "success", "first_failure_time_s", "pass")} for report in reports],
        "maximum_state_difference": max(report["state_comparison"]["maximum_numeric_difference"] for report in reports),
        "module_paths": module_paths(), "access": audit.record(),
    }
    write_json(output / "demonstrations_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--inspect-models", action="store_true")
    choice.add_argument("--suite", action="store_true", help="All original three-scene/five-method pairs")
    choice.add_argument("--case", choices=("learning_success", "apf_success_learning_failure", "best_last_difference"))
    parser.add_argument("--method", choices=tuple(methods()[index]["name"] for index in range(8)))
    parser.add_argument("--output", required=True, help="New name under outputs/; refuses overwrite")
    args = parser.parse_args()
    if bool(args.case) != bool(args.method):
        parser.error("A single --case requires exactly one --method")
    audit = bootstrap(True)
    verification = verify_release()
    output = reserve_output(args.output)
    write_json(output / "preflight.json", verification)
    if args.inspect_models:
        result = inspect_models(output, audit)
    elif args.suite:
        result = run_suite(output, audit)
    else:
        result = run_case(args.case, args.method, output, audit)
    print({"pass": result["pass"], "physics_episodes": result["physics_episodes"], "output": str(output)}, flush=True)


if __name__ == "__main__":
    main()
