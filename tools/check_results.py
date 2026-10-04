#!/usr/bin/env python
"""Stdlib-only frozen snapshot/hash verification; no physics, training or writes."""
import argparse
import json
from common import bootstrap, snapshot_checks, verify_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    audit = bootstrap(True)
    verification = verify_release(check_environment=False)
    snapshot = snapshot_checks()
    report = {**snapshot, "release_verification": verification, "access": audit.record(), "scope": "Saved frozen CSV and release file hashes only; physics reproduction uses quick_check/evaluate"}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not snapshot["pass"]:
        raise RuntimeError("Saved result snapshot consistency failed")


if __name__ == "__main__":
    main()
