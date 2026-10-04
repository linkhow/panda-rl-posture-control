"""Compare deterministic physics and policy values, excluding wall-clock timings."""
import csv
import math
from pathlib import Path
from common import read

TIMING_COLUMNS = {"provider_mean_s", "core_mean_s", "outer_mean_s", "scene_load_wall_s", "settle_wall_s", "setup_including_settle_wall_s", "whole_episode_wall_s"}
TEXT_COLUMNS = {"method", "role", "seed", "checkpoint", "model_SHA256", "scene_id", "geometry", "template_group", "success", "failure_category", "failure_reason", "failure_phase"}
INTEGER_COLUMNS = {"external_censored_states", "self_censored_states", "joint_resets_during_motion", "obstacle_contact_band", "obstacle_negative_penetration", "obstacle_buffer", "self_contact_band", "self_negative_penetration", "self_buffer", "obstacle_negative_distance", "self_negative_distance", "hard_position_violation", "actual_speed_violation", "tracking_error_violation", "non_finite_state"}
IGNORE_SUMMARY = {"decision_mean_s", "decision_max_s", "loop_mean_s", "loop_max_s", "timing", "trace_csv", "actions_json", "wall_s_including_setup_and_io", "maximum_prediction_check_difference"}


def csv_rows(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def numeric_delta(a, b):
    if a in (None, "") or b in (None, ""):
        return (a in (None, "") and b in (None, "")), None
    x, y = float(a), float(b)
    if not math.isfinite(x) or not math.isfinite(y):
        return x == y, None
    return True, abs(x - y)


def compare_episode_csv(reference, actual):
    fields, old = csv_rows(reference)
    new_fields, new = csv_rows(actual)
    keys = lambda rows: [(row["method"], row["scene_id"]) for row in rows]
    mismatches = []
    if fields != new_fields:
        mismatches.append({"kind": "columns"})
    if keys(old) != keys(new):
        mismatches.append({"kind": "fixed_method_scene_order"})
    maximum = {key: 0.0 for key in fields if key not in TIMING_COLUMNS | TEXT_COLUMNS}
    checked = 0
    for previous, current in zip(old, new):
        for key in fields:
            if key in TIMING_COLUMNS:
                continue
            a, b = previous[key], current.get(key)
            checked += 1
            if key in TEXT_COLUMNS:
                good, delta = a == b, None
            else:
                good, delta = numeric_delta(a, b)
                if delta is not None:
                    maximum[key] = max(maximum[key], delta)
                    tolerance = 0.0 if key in INTEGER_COLUMNS else 1e-12 if key == "first_failure_time_s" else 1e-7
                    good = good and delta <= tolerance
            if not good:
                mismatches.append({"method": previous["method"], "scene_id": previous["scene_id"], "field": key, "reference": a, "actual": b, "difference": delta})
    return {
        "pass": not mismatches and len(old) == len(new) == 1048,
        "reference_rows": len(old), "actual_rows": len(new),
        "checked_values": checked, "maximum_differences_by_field": maximum,
        "mismatches": mismatches, "ignored_wall_clock_columns": sorted(TIMING_COLUMNS),
        "tolerances": {"discrete_fields": 0, "first_failure_time_s": 1e-12, "other_numeric_physics_and_policy_fields": 1e-7},
        "scope": "All 1048 episode snapshot physics/policy columns; full original per-step trajectories are held in the archive",
    }


def compare_states(reference, actual):
    fields, old = csv_rows(reference)
    new_fields, new = csv_rows(actual)
    mismatches, maximum = [], {key: 0.0 for key in fields}
    if fields != new_fields:
        mismatches.append({"kind": "columns"})
    if len(old) != len(new):
        mismatches.append({"kind": "state_count", "reference": len(old), "actual": len(new)})
    for index, (previous, current) in enumerate(zip(old, new)):
        for key in fields:
            a, b = previous[key], current.get(key)
            try:
                good, delta = numeric_delta(a, b)
            except (ValueError, TypeError):
                good, delta = a == b, None
            if delta is not None:
                maximum[key] = max(maximum[key], delta)
                good = good and delta <= (1e-12 if key == "time_s" else 1e-7)
            if not good:
                # Bounded example output; the total mismatch count remains exact.
                mismatches.append({"row": index, "field": key, "reference": a, "actual": b, "difference": delta})
    return {"pass": not mismatches, "state_counts": [len(old), len(new)], "compared_columns": fields, "maximum_differences_by_field": maximum, "maximum_numeric_difference": max(maximum.values(), default=0.0), "mismatch_count": len(mismatches), "mismatch_examples": mismatches[:100], "tolerances": {"time_s": 1e-12, "all_other_numeric_columns": 1e-7, "strings_and_missing_values": "exact"}}


def compare_summary(reference, actual):
    old = read(reference) if isinstance(reference, (str, Path)) else reference
    new = read(actual) if isinstance(actual, (str, Path)) else actual
    mismatches, numeric_differences = [], {}

    def walk(a, b, path):
        if isinstance(a, dict):
            if not isinstance(b, dict):
                mismatches.append(path)
                return
            for key, value in a.items():
                if key not in IGNORE_SUMMARY:
                    if key not in b:
                        mismatches.append(path + "." + key + ":missing")
                    else:
                        walk(value, b[key], path + "." + key)
        elif isinstance(a, list):
            if not isinstance(b, list) or len(a) != len(b):
                mismatches.append(path + ":length")
            else:
                for index, (x, y) in enumerate(zip(a, b)):
                    walk(x, y, f"{path}[{index}]")
        elif isinstance(a, float):
            good, delta = numeric_delta(a, b)
            numeric_differences[path] = delta
            good = good and (delta is None or delta <= (1e-12 if path.endswith("first_failure_time_s") else 1e-7))
            if not good:
                mismatches.append(path)
        elif a != b:
            mismatches.append(path)

    walk(old, new, "summary")
    return {"pass": not mismatches, "mismatches": mismatches, "numeric_differences": numeric_differences, "ignored_fields": sorted(IGNORE_SUMMARY)}
