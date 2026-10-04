#!/usr/bin/env python3
"""Simple deterministic validation smoke, using validation parameters only.

This is a sanity check, not model selection or a formal scientific result.
"""
import argparse
from common import ROOT, bootstrap, read, reserve_output, verify_release, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=550901, choices=(550901, 551901, 552901))
    p.add_argument("--n", type=int, default=3, help="First n validation IDs in frozen order (1..58)")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if not 1 <= args.n <= 58:
        p.error("n must be within 1..58")
    audit = bootstrap(False)
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from me5418.arm_env import json_safe
    from me5418.diagnostic_env import DiagnosticPandaEnv
    torch.set_num_threads(1)
    verification = verify_release()
    out = reserve_output(args.output)
    model = PPO.load(ROOT / f"models/best_seed{args.seed}.zip", device="cpu")
    env = DiagnosticPandaEnv("validation", ROOT / "configs/stage09/C.json")
    rows = []
    try:
        for identifier in list(env.pool)[:args.n]:
            obs, _ = env.reset(seed=541812, options={"scene_id": identifier})
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            rows.append(env.last_summary)
    finally:
        env.close()
    report = {"completed": len(rows) == args.n, "split": "validation", "n": len(rows), "successes": sum(r["success"] for r in rows), "seed": args.seed, "purpose": "Simple smoke; not checkpoint selection or official performance", "episodes": rows, "release_verification": verification, "data_access": audit.record()}
    write_json(out / "validation_smoke.json", json_safe(report))
    print({k: report[k] for k in ("completed", "split", "n", "successes", "seed")})


if __name__ == "__main__":
    main()
