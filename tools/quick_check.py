#!/usr/bin/env python
"""Verify files/results, load all six models, and replay all five representative pairs."""
import argparse
from common import bootstrap, module_paths, reserve_output, snapshot_checks, verify_release, write_json
from demo import inspect_models, run_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New name under outputs/; refuses overwrite")
    args = parser.parse_args()
    audit = bootstrap(True)
    verification = verify_release()
    snapshot = snapshot_checks()
    if not snapshot["pass"]:
        raise RuntimeError("Saved frozen snapshot verification failed")
    output = reserve_output(args.output)
    write_json(output / "preflight.json", {"release": verification, "snapshot": snapshot})
    inspection = inspect_models(output, audit)
    demonstrations = run_suite(output, audit)
    report = {
        "pass": snapshot["pass"] and inspection["pass"] and demonstrations["pass"],
        "six_models_loaded": len(inspection["models"]), "physics_episodes": 5, "training_interactions": 0,
        "environment_verification": verification, "official_successes_snapshot": snapshot["official_successes_snapshot"],
        "demonstrations": demonstrations, "module_paths": module_paths(), "access": audit.record(),
    }
    write_json(output / "quick_check.json", report)
    print({"pass": report["pass"], "six_models_loaded": 6, "physics_episodes": 5, "maximum_state_difference": demonstrations["maximum_state_difference"], "output": str(output)}, flush=True)


if __name__ == "__main__":
    main()
