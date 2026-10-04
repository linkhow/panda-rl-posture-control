#!/usr/bin/env python
"""Run eight frozen methods on the same ordered 131 test scenes; never train/select."""
import argparse
import concurrent.futures
import csv
import multiprocessing
from pathlib import Path
import shutil
import time
from common import ROOT, append_json, bootstrap, load_model, methods, module_paths, read, reserve_output, sha, snapshot_checks, utc, verify_release, write_json
from comparison import compare_episode_csv


def episode_row(method, scenario, summary):
    row = {
        "method": method["name"], "role": method["role"], "seed": method["seed"],
        "checkpoint": method["checkpoint"], "model_SHA256": method["model_SHA256"],
        "scene_id": scenario["id"], "geometry": scenario["preassigned_geometric_category"],
        "template_group": scenario["template_group"], "success": summary["success"],
        **{key: summary[key] for key in ("failure_category", "failure_reason", "failure_phase", "first_failure_time_s")},
        "completed_fraction": summary["actual_duration_s"] / summary["requested_duration_s"],
    }
    row.update({key: summary[key] for key in (
        "max_error_m", "rmse_m", "endpoint_error_m", "minimum_external_distance_m", "minimum_self_distance_m",
        "external_censored_states", "self_censored_states", "command_smoothness_rms_rad_s2", "max_actual_arm_velocity_rad_s",
        "speed_saturated_fraction", "soft_limited_fraction", "raw_policy_component_clip_fraction", "raw_policy_decisions_with_clip_fraction",
        "u_component_clip_fraction", "u_decisions_with_clip_fraction", "mean_u_abs_rad_s", "joint_resets_during_motion",
    )})
    row.update(summary["distance_and_collision_flags"])
    row.update(summary["state_violation_counts"])
    row.update({f"{key}_mean_s": summary["timing"][field]["mean_s"] for key, field in (("provider", "provider_48Hz"), ("core", "core_physical_step"), ("outer", "outer_physical_loop"))})
    row.update({key: summary["timing"][key] for key in ("scene_load_wall_s", "settle_wall_s", "setup_including_settle_wall_s")})
    row["whole_episode_wall_s"] = summary["wall_s_including_setup_and_io"]
    return row


def evaluate_method(method, output_string, scene_ids, parameter_hashes):
    audit = bootstrap(True)
    verification = verify_release()
    import torch
    from me5418.final_evaluation import EvaluationAccessAudit, load_evaluation_pool, logged_run
    from me5418.scenarios import canonical_hash
    torch.set_num_threads(1)
    pool = load_evaluation_pool("test", EvaluationAccessAudit(True))
    if list(pool) != scene_ids or any(canonical_hash(scene) != parameter_hashes[key] for key, scene in pool.items()):
        raise RuntimeError("Fixed test parameters/order changed")
    output = Path(output_string) / "evaluation" / method["name"]
    output.mkdir(parents=True, exist_ok=False)
    model, config = load_model(method) if method["kind"] == "policy" else (None, None)
    started, rows = time.perf_counter(), []
    for identifier, scenario in pool.items():
        if shutil.disk_usage(ROOT).free < 2 * 1024**3:
            raise RuntimeError("Insufficient disk reserve: free space is below 2 GiB")
        target = output / identifier
        try:
            result = logged_run(scenario, method, target, model, config)
            if not result["connection_closed"] or result["joint_resets_during_motion"] != 0:
                raise RuntimeError("Connection/reset invariants failed")
            write_json(target / "COMPLETE.json", {
                "method": method["name"], "scene_id": identifier,
                "parameter_SHA256": parameter_hashes[identifier], "model_SHA256": method["model_SHA256"],
                "release_model_SHA256": method.get("release_model_SHA256", method["model_SHA256"]),
                "task_success": result["success"], "completed_UTC": utc(),
                "files_SHA256": {path.name: sha(path) for path in target.iterdir() if path.is_file()},
            })
            rows.append(episode_row(method, scenario, result))
            append_json(output / "progress.jsonl", {"UTC": utc(), "method": method["name"], "completed": len(rows), "planned": 131, "scene_id": identifier, "task_success": result["success"], "failure_category": result["failure_category"]})
            if len(rows) % 20 == 0:
                print(f'{method["name"]}: {len(rows)}/131 completed', flush=True)
        except BaseException as error:
            write_json(output / "infrastructure_failure.json", {"UTC": utc(), "method": method["name"], "scene_id": identifier, "exception": repr(error), "completed_episodes": len(rows), "task_failure_retried": False})
            raise
    report = {
        "method": method, "n": len(rows), "successes": sum(row["success"] for row in rows),
        "wall_s": time.perf_counter() - started, "environment_verification": verification,
        "module_paths": module_paths(), "access": audit.record(),
        "training_interactions": 0, "task_failure_retried": 0,
    }
    write_json(output / "method_summary.json", report)
    return rows, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New name under outputs/; refuses existing directories")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=1, help="Independent DIRECT processes; parallel wall timings are not formal performance evidence")
    args = parser.parse_args()
    audit = bootstrap(True)
    verification = verify_release()
    snapshot = snapshot_checks()
    if not snapshot["pass"]:
        raise RuntimeError("Reference snapshot checks failed")
    import torch
    from me5418.final_evaluation import EvaluationAccessAudit, load_evaluation_pool
    from me5418.scenarios import canonical_hash
    torch.set_num_threads(1)
    pool = load_evaluation_pool("test", EvaluationAccessAudit(True))
    if len(pool) != 131:
        raise RuntimeError("Fixed evaluation requires all 131 test scenes")
    output = reserve_output(args.output)
    selected = methods()
    hashes = {key: canonical_hash(scene) for key, scene in pool.items()}
    plan = {
        "created_UTC": utc(), "purpose": "Release reproduction of already-used fixed test set, not a new unseen evaluation",
        "test_IDs": list(pool), "scene_parameter_SHA256": hashes,
        "methods": selected, "n_scenes": 131, "n_methods": 8, "planned_episodes": 1048,
        "environment_verification": verification, "workers": args.workers, "torch_threads_per_worker": 1, "device": "cpu",
        "no_training_tuning_reselection": True, "normal_task_failure_not_retried": True,
        "timing": "Parallel runs change CPU contention. Wall clocks are reported, excluded from deterministic reference comparison and any real-time claim.",
        "snapshot_SHA256": sha(ROOT / "results/reference/episodes_all1048.csv"),
        "integrity_manifest_SHA256": sha(ROOT / "provenance/release_integrity.json"),
    }
    write_json(output / "evaluation_plan.json", plan)
    started, result_by_name, method_reports = time.perf_counter(), {}, []
    try:
        if args.workers == 1:
            for method in selected:
                rows, report = evaluate_method(method, str(output), list(pool), hashes)
                result_by_name[method["name"]] = rows
                method_reports.append(report)
                print(f'{method["name"]}: {report["successes"]}/131 successful', flush=True)
        else:
            context = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
                futures = {executor.submit(evaluate_method, method, str(output), list(pool), hashes): method for method in selected}
                for future in concurrent.futures.as_completed(futures):
                    method = futures[future]
                    rows, report = future.result()
                    result_by_name[method["name"]] = rows
                    method_reports.append(report)
                    print(f'{method["name"]}: {report["successes"]}/131 successful', flush=True)
        rows = [row for method in selected for row in result_by_name[method["name"]]]
        with (ROOT / "results/reference/episodes_all1048.csv").open(newline="") as handle:
            fieldnames = csv.DictReader(handle).fieldnames
        with (output / "episodes_all1048.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        comparison = compare_episode_csv(ROOT / "results/reference/episodes_all1048.csv", output / "episodes_all1048.csv")
        write_json(output / "comparison.json", comparison)
        report = {
            "complete": len(rows) == 1048, "comparison_pass": comparison["pass"], "physics_episodes": len(rows),
            "successes": {method["name"]: sum(row["success"] for row in result_by_name[method["name"]]) for method in selected},
            "wall_s": time.perf_counter() - started, "workers": args.workers,
            "method_reports": method_reports, "parent_module_paths": module_paths(), "parent_access": audit.record(),
            "training_interactions": 0, "task_failure_retried": 0,
            "state_comparison_scope": "All 1048 deterministic snapshot columns compared; five representatives separately compare every saved state column. Full historical1048 traces stay in the original archive.",
        }
        write_json(output / "execution_complete.json", report)
        print({"complete": report["complete"], "comparison_pass": comparison["pass"], "successes": report["successes"], "output": str(output)}, flush=True)
        if not comparison["pass"]:
            raise RuntimeError("Reproduction differs from reference; preserve outputs/comparison.json for investigation")
    except BaseException as error:
        write_json(output / "run_failure.json", {"UTC": utc(), "exception": repr(error), "completed_methods": sorted(result_by_name), "output_preserved": True})
        raise


if __name__ == "__main__":
    main()
