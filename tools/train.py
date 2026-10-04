#!/usr/bin/env python3
"""Fresh CPU PPO training with the frozen task and all-58 selection rule.

Run as a script from a release checkout. Historical stage artifacts are not
dependencies. A smoke run verifies sampling, PPO updates and model round-trip;
its models and scores are explicitly excluded from the official results.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from common import bootstrap, module_paths, read, reserve_output, sha, verify_release

# Spawned workers import this module too: every process installs its own guard
# before importing the project's environment and reading scenario parameters.
AUDIT = bootstrap(allow_test=False)

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from me5418.arm_env import json_safe, load_pool
from me5418.diagnostic_env import DiagnosticPandaEnv
from me5418.sampling import BalancedPandaEnv, CATEGORIES, validate_pools
from me5418.tracking import write_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage09/C.json"
SEEDS = (550901, 551901, 552901)
SMOKE_SEED = 559901
FORMAL_STEPS = 1_024_000
N_ENVS = 4
ROLLOUT_STEPS = 2048
EVALUATION_INTERVAL = 102_400
RECOVERY_INTERVAL = 51_200
SELECTION_RULES = [
    "maximum complete successes on fixed 58 validation scenes",
    "maximum mean discounted bounded-v1 return on exact success tie",
    "earliest step on exact return tie",
]
torch.set_num_threads(1)


def now():
    return datetime.now(timezone.utc).isoformat()


def line_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(json_safe(value), allow_nan=False) + "\n")


def global_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
    }


def same_global(a, b):
    return (
        a["python"] == b["python"]
        and a["numpy"][0] == b["numpy"][0]
        and np.array_equal(a["numpy"][1], b["numpy"][1])
        and a["numpy"][2:] == b["numpy"][2:]
        and torch.equal(a["torch"], b["torch"])
    )


@contextmanager
def isolated_global_rng():
    state = global_state()
    try:
        yield
    finally:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if not same_global(state, global_state()):
            raise RuntimeError("Global RNG restoration failed")


def worker_rng(vec):
    return [copy.deepcopy(g.bit_generator.state) for g in vec.get_attr("np_random")]


def rng_key(value):
    return json.dumps(json_safe(value), sort_keys=True, allow_nan=False)


def checkpoint_diagnostics(model):
    log_std = model.policy.log_std.detach().cpu().numpy().copy()
    return {
        "log_std": log_std,
        "std": np.exp(log_std),
        "parameters": sum(x.numel() for x in model.policy.parameters()),
    }


def evaluate_validation(model, ids, config_path, output, seed):
    """Exact Stage7 deterministic policy means and bounded-v1 returns.

    Each scene resets a separate validation client; the learner and its four
    continuously advancing workers stay alive and retain their RNG state.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    env = DiagnosticPandaEnv("validation", config_path, output / "samples")
    episodes = []
    begin = time.perf_counter()
    try:
        if ids != list(env.pool):
            raise ValueError("Validation must use every fixed ID in original manifest order")
        for identifier in ids:
            obs, _ = env.reset(seed=seed, options={"scene_id": identifier})
            while True:
                with torch.no_grad():
                    raw = model.policy.get_distribution(
                        torch.as_tensor(obs[None], device=model.device)
                    ).distribution.mean.cpu().numpy()[0]
                action = np.clip(raw, -1, 1)
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            episodes.append(copy.deepcopy(env.last_summary))
        category_results = {}
        for category in CATEGORIES:
            rows = [e for e in episodes if env.pool[e["scene_id"]]["preassigned_geometric_category"] == category]
            category_results[category] = {
                "n": len(rows),
                "successes": sum(e["success"] for e in rows),
                "failure_categories": {
                    kind: sum(e["failure_category"] == kind for e in rows)
                    for kind in sorted({e["failure_category"] for e in rows if not e["success"]})
                },
                "mean_abs_action": float(np.mean([e["mean_abs_action"] for e in rows])) if rows else None,
            }
    finally:
        env.close()
    result = {
        "ids": ids,
        "n": len(ids),
        "successes": sum(e["success"] for e in episodes),
        "success_rate": sum(e["success"] for e in episodes) / len(ids),
        "category_results": category_results,
        "mean_return": float(np.mean([e["return"] for e in episodes])),
        "mean_discounted_return": float(np.mean([e["discounted_return"] for e in episodes])),
        "mean_abs_action": float(np.mean([e["mean_abs_action"] for e in episodes])),
        "wall_s": time.perf_counter() - begin,
        "episodes": episodes,
        "deterministic": True,
        "split": "validation",
        "scope": "all58 validation for precommitted selection; zero test episodes",
    }
    if not np.isfinite(result["mean_discounted_return"]):
        raise FloatingPointError("Nonfinite validation selection return")
    write_json(output / "summary.json", result)
    return result


def make_env(rank, config_path, output):
    def init():
        torch.set_num_threads(1)
        worker_audit = bootstrap(allow_test=False)
        folder = Path(output) / f"env{rank}"
        env = BalancedPandaEnv("train", config_path, folder, access_audit=worker_audit)
        return Monitor(env, str(folder / "monitor.csv"))
    return init


def rss_of_process_tree():
    """Linux live RSS of the learner and descendants, checked each rollout."""
    if not Path("/proc/self/stat").exists():
        return None
    parents, sizes = {}, {}
    for folder in Path("/proc").iterdir():
        if not folder.name.isdigit():
            continue
        try:
            raw = (folder / "stat").read_text()
            fields = raw[raw.rfind(")") + 2:].split()
            pid = int(folder.name)
            parents[pid] = int(fields[1])
            sizes[pid] = int(fields[21]) * os.sysconf("SC_PAGE_SIZE")
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError):
            continue
    pids, previous = {os.getpid()}, set()
    while pids != previous:
        previous = pids.copy()
        pids |= {pid for pid, parent in parents.items() if parent in pids}
    return sum(sizes.get(pid, 0) for pid in pids)


def assert_frozen_settings(config):
    if config["budget"] != {
        "max_steps": FORMAL_STEPS,
        "evaluation_interval_steps": EVALUATION_INTERVAL,
        "recovery_interval_steps": RECOVERY_INTERVAL,
        "n_envs": N_ENVS,
        "global_wall_cap_s": 32400,
    }:
        raise ValueError("Frozen training budget changed")
    if config["ppo"]["n_steps"] != 512 or config["ppo"]["device"] != "cpu":
        raise ValueError("Frozen CPU rollout configuration changed")
    if config["threads_per_process"] != 1 or config["validation_selection"]["n"] != 58:
        raise ValueError("Frozen worker/validation configuration changed")


def train_one(seed, output, smoke_steps, command_begin):
    smoke = smoke_steps is not None
    total = smoke_steps if smoke else FORMAL_STEPS
    eligible = [total] if smoke else list(range(EVALUATION_INTERVAL, total + 1, EVALUATION_INTERVAL))
    recovery = total if smoke else RECOVERY_INTERVAL
    config = read(CONFIG)
    assert_frozen_settings(config)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    train_pool, validation_pool = load_pool("train"), load_pool("validation")
    if len(train_pool) != 151 or len(validation_pool) != 58 or set(train_pool) & set(validation_pool):
        raise ValueError("Frozen 151/58 train-validation split contract failed")
    category_ids = {c: [i for i in train_pool if train_pool[i]["preassigned_geometric_category"] == c] for c in CATEGORIES}
    pool_contract = validate_pools(train_pool, category_ids, config["sampling"]["weights"])
    if pool_contract["counts"] != dict(zip(CATEGORIES, [72, 64, 15])):
        raise ValueError("Frozen train category counts changed")
    validation_ids = list(validation_pool)
    write_json(output / "config.json", config)
    write_json(output / "pool_contract.json", pool_contract)
    write_json(output / "actual_execution.json", {
        "seed": seed, "worker_seeds": list(range(seed, seed + N_ENVS)),
        "target_steps": total, "from_scratch": True,
        "formal_protocol_reproduction": not smoke,
        "is_official_saved_result": False,
        "smoke_only": smoke,
        "model_role_note": "smoke models are not official best/last results" if smoke else "new training results; do not replace saved original results",
        "one_continuous_learn": True, "no_VecNormalize": True,
        "eligible_steps": eligible, "validation_ids": validation_ids,
        "validation_order": "original manifest order used by frozen load_pool",
        "selection_rules": SELECTION_RULES,
        "configuration_sha256": sha(CONFIG),
        "source_module_paths": module_paths(),
        "training_entry_sha256": sha(Path(__file__)),
        "test_episodes": 0,
    })
    begin = time.perf_counter()
    vec = model = None
    history, actions, evaluated, saved = [], [], set(), set()
    best = None
    validation_wall = checkpoint_wall = 0.0
    peak_rss = 0
    initial_parameters = None

    def protect():
        nonlocal peak_rss
        if time.perf_counter() - command_begin >= config["budget"]["global_wall_cap_s"]:
            raise TimeoutError("Fresh command wall cap (9h) reached")
        if shutil.disk_usage(output).free < 2 * 1024 ** 3:
            raise RuntimeError("Free disk below 2 GiB")
        rss = rss_of_process_tree()
        if rss is not None:
            peak_rss = max(peak_rss, rss)
            if rss > 12e9:
                raise MemoryError("Aggregate live RSS over 12 GB")
        if Path("/proc/meminfo").exists():
            available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
            if available < 512 * 1024 ** 2:
                raise MemoryError("Available memory below 512 MiB")

    def save_checkpoint(step, label=None):
        nonlocal checkpoint_wall
        started = time.perf_counter()
        name = label or f"step_{step}"
        target = output / "checkpoints" / f"{name}.zip"
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        model.save(target)
        state = global_state()
        state["worker_private_rng"] = worker_rng(vec) if vec is not None else None
        torch.save(state, target.with_suffix(".rng.pt"))
        line_json(output / "checkpoint_index.jsonl", {
            "step": step, "name": name, "sha256": sha(target), "UTC": now(),
            "optimizer_in_SB3_zip": True, "global_and_worker_RNG_saved": state["worker_private_rng"] is not None,
            "complete_Bullet_episode_state_saved": False,
            "eligible_for_best": label is None and step in eligible,
            "after_PPO_update": step > 0, "smoke_only": smoke,
        })
        saved.add(name)
        checkpoint_wall += time.perf_counter() - started

    def validate(step):
        nonlocal best, validation_wall
        if step in evaluated:
            raise RuntimeError("Duplicate validation milestone")
        protect()
        started = time.perf_counter()
        before_global, before_worker = global_state(), worker_rng(vec)
        with isolated_global_rng():
            result = evaluate_validation(model, validation_ids, CONFIG, output / "validation" / f"step_{step}", seed)
        after_worker = worker_rng(vec)
        preserved = same_global(before_global, global_state()) and rng_key(before_worker) == rng_key(after_worker)
        if not preserved:
            raise RuntimeError("Validation changed learner/worker RNG")
        validation_wall += time.perf_counter() - started
        line_json(output / "validation_rng_audit.jsonl", {
            "step": step, "parent_global_preserved": True,
            "worker_rng_before": before_worker, "worker_rng_after": after_worker,
            "worker_RNG_unchanged": True, "worker_rebuilds": 0, "worker_reseeds": 0,
            "fixed58_ID_order": result["ids"] == validation_ids,
        })
        record = {
            "step": step, **{k: v for k, v in result.items() if k != "episodes"},
            **checkpoint_diagnostics(model), "eligible_for_best": step in eligible,
            "evaluation_after_update": step > 0, "smoke_only": smoke,
        }
        history.append(record)
        evaluated.add(step)
        write_json(output / "validation/history.json", history)
        selected = False
        if step in eligible:
            key = (result["successes"], result["mean_discounted_return"])
            # Strict > is deliberate: exact ties retain the earlier checkpoint.
            if best is None or key > best:
                best = key
                shutil.copyfile(output / f"checkpoints/step_{step}.zip", output / "checkpoints/best.zip")
                write_json(output / "checkpoints/best_selection.json", {
                    "step": step, "key": key, "rules": SELECTION_RULES,
                    "model_sha256": sha(output / "checkpoints/best.zip"),
                    "validation_summary": f"validation/step_{step}/summary.json",
                    "smoke_only": smoke,
                })
                selected = True
            line_json(output / "selection_process.jsonl", {
                "step": step, "successes": result["successes"],
                "mean_discounted_return": result["mean_discounted_return"],
                "replaced_best": selected, "best_so_far": read(output / "checkpoints/best_selection.json"),
                "smoke_only": smoke,
            })
        print(json.dumps({"seed": seed, "step": step, "validation58_successes": result["successes"], "selected_best": selected, "smoke_only": smoke}), flush=True)

    class Observe(BaseCallback):
        def finite(self):
            if not all(torch.isfinite(x).all() for x in model.policy.parameters()):
                raise FloatingPointError("Nonfinite policy")
            for optimizer_state in model.policy.optimizer.state.values():
                if any(isinstance(x, torch.Tensor) and not torch.isfinite(x).all() for x in optimizer_state.values()):
                    raise FloatingPointError("Nonfinite optimizer")

        def _on_step(self):
            raw = np.asarray(self.locals["actions"])
            actions.append(raw.copy())
            if not all(np.isfinite(x).all() for x in [raw, self.locals["new_obs"], self.locals["rewards"]]):
                raise FloatingPointError("Nonfinite samples")
            for index, info in enumerate(self.locals["infos"]):
                if "episode_summary" in info:
                    episode = info["episode_summary"]
                    line_json(output / "training_progress.jsonl", {
                        "step": self.num_timesteps, "env_index": index,
                        "scene_id": episode["scene_id"],
                        "category": train_pool[episode["scene_id"]]["preassigned_geometric_category"],
                        **{key: episode[key] for key in ["success", "failure_category", "first_failure_time_s", "actual_duration_s", "policy_steps", "return", "discounted_return", "mean_abs_action", "reward_terms"]},
                    })
            return True

        def _on_rollout_end(self):
            raw = np.concatenate(actions).reshape(-1, 7)
            actions.clear()
            line_json(output / "action_distribution.jsonl", {
                "step": self.num_timesteps, "raw_mean_per_joint": raw.mean(0),
                "raw_std_per_joint": raw.std(0), "raw_components": raw.size,
                "raw_component_clip_fraction": float(np.mean(abs(raw) > 1)),
                "mean_abs_raw": float(np.mean(abs(raw))),
                "mean_abs_clipped": float(np.mean(abs(np.clip(raw, -1, 1)))),
                **checkpoint_diagnostics(model),
            })
            if not all(np.isfinite(x).all() for x in [model.rollout_buffer.observations, model.rollout_buffer.actions, model.rollout_buffer.rewards]):
                raise FloatingPointError("Nonfinite rollout")

        def milestone(self, final=False):
            self.finite()
            protect()
            step = self.num_timesteps
            for key, value in model.logger.name_to_value.items():
                if key.startswith("train/") and isinstance(value, (int, float, np.number)) and not np.isfinite(value):
                    raise FloatingPointError("Nonfinite PPO diagnostic: " + key)
            if step:
                line_json(output / "ppo_update_diagnostics.jsonl", {
                    "step": step, "rollouts": step // ROLLOUT_STEPS,
                    "optimizer_epoch_updates": model._n_updates,
                    "metrics": {k: v for k, v in model.logger.name_to_value.items() if k.startswith("train/")},
                    "policy_optimizer_finite": True, "final": final,
                })
            if step and step % recovery == 0 and f"step_{step}" not in saved:
                save_checkpoint(step)
            if step in eligible and step not in evaluated:
                validate(step)
            if step:
                write_json(output / "live_status.json", {
                    "step": step, "optimizer_epoch_updates": model._n_updates,
                    "total_wall_s": time.perf_counter() - begin,
                    "validation_wall_s": validation_wall,
                    "last_validation58_successes": history[-1]["successes"],
                    "best_step": read(output / "checkpoints/best_selection.json")["step"] if best is not None else None,
                    "smoke_only": smoke,
                })

        def _on_rollout_start(self):
            # SB3 invokes this after the preceding rollout's PPO update.
            self.milestone()

        def _on_training_end(self):
            # Final PPO update has finished; there is no next rollout_start.
            self.milestone(final=True)

    try:
        protect()
        vec = SubprocVecEnv([make_env(rank, CONFIG, output / "training") for rank in range(N_ENVS)], start_method="spawn")
        model = PPO("MlpPolicy", vec, seed=seed, tensorboard_log=str(output / "tensorboard"), verbose=0, **config["ppo"])
        initial = checkpoint_diagnostics(model)
        if not np.array_equal(initial["std"], np.ones(7)):
            raise ValueError("Frozen initial Gaussian std must be one")
        initial_parameters = [parameter.detach().clone() for parameter in model.policy.parameters()]
        # Same frozen startup: one seeded reset, then one continuous learn.
        model._last_obs = vec.reset()
        model._last_episode_starts = np.ones((N_ENVS,), dtype=bool)
        save_checkpoint(0, "untrained")
        validate(0)
        learn_begin = time.perf_counter()
        initial_validation_wall = validation_wall
        initial_checkpoint_wall = checkpoint_wall
        model.learn(total_timesteps=total, reset_num_timesteps=False, callback=Observe(), tb_log_name=f"C_seed{seed}", log_interval=1)
        learning_session_wall = time.perf_counter() - learn_begin
        if model.num_timesteps != total or len(history) != len(eligible) + 1:
            raise RuntimeError("Training did not finish the exact requested budget/validation schedule")
        learning_checkpoint_wall = checkpoint_wall - initial_checkpoint_wall
        save_checkpoint(total, "last")
        parameter_difference = max(float(torch.max(torch.abs(before - after.detach())).cpu()) for before, after in zip(initial_parameters, model.policy.parameters()))
        if model._n_updates <= 0 or parameter_difference <= 0:
            raise RuntimeError("Training sampled but did not update policy parameters")
        observations = np.asarray(model._last_obs).copy()
        before = model.predict(observations, deterministic=True)[0]
        with isolated_global_rng():
            loaded = PPO.load(output / "checkpoints/last.zip", device="cpu")
        after = loaded.predict(observations, deterministic=True)[0]
        reload_difference = float(np.max(abs(before - after)))
        if reload_difference != 0 or not loaded.policy.optimizer.state:
            raise RuntimeError("Saved policy/optimizer round-trip failed")
        np.save(output / "reload_observations.npy", observations)
        write_json(output / "reload_check.json", {
            "actual_observations_shape": list(observations.shape),
            "maximum_deterministic_action_difference": reload_difference,
            "pass": True, "optimizer_states_loaded": len(loaded.policy.optimizer.state),
            "maximum_parameter_change_since_initialization": parameter_difference,
            "optimizer_epoch_updates": model._n_updates,
        })
        best_selection = read(output / "checkpoints/best_selection.json")
        write_json(output / "checkpoints/final_lock.json", {
            "best": best_selection, "last_sha256": sha(output / "checkpoints/last.zip"),
            "actual_steps": total, "eligible_evaluation_steps": eligible,
            "selection_complete": True, "smoke_only": smoke,
            "all58_precommitted_selection_not_test": True,
        })
        vec.close()
        vec = None
        learning_wall = learning_session_wall - (validation_wall - initial_validation_wall) - learning_checkpoint_wall
        result = {
            "seed": seed, "formal_protocol_reproduction": not smoke,
            "is_official_saved_result": False, "smoke_only": smoke,
            "complete": True, "actual_steps": total,
            "stop_reason": "planned_steps_completed", "continuous_session": True,
            "worker_restarts": 0, "worker_seed_range": list(range(seed, seed + N_ENVS)),
            "rollouts": total // ROLLOUT_STEPS, "optimizer_epoch_updates": model._n_updates,
            "wall_s": time.perf_counter() - begin, "validation_wall_s": validation_wall,
            "learning_wall_s_estimate": learning_wall,
            "learning_steps_per_s_estimate": total / learning_wall if learning_wall > 0 else None,
            "peak_aggregate_RSS_bytes_at_rollout_checks": peak_rss,
            "initial_network": initial, "last_network": checkpoint_diagnostics(model),
            "best_step": best_selection["step"],
            "best58": next(row["successes"] for row in history if row["step"] == best_selection["step"]),
            "last58": history[-1]["successes"], "history": history,
            "validation_rng_checks_passed": len(history),
            "reload_action_diff": reload_difference,
            "maximum_parameter_change_since_initialization": parameter_difference,
            "test_episodes": 0, "data_access": AUDIT.record(),
            "configuration_sha256": sha(CONFIG),
            "resume_limitation": "Checkpoint stores optimizer and RNG but not complete Bullet episode state; exact same-trajectory resume is not provided",
        }
        write_json(output / "run.json", result)
        print(json.dumps({"finished_seed": seed, "steps": total, "smoke_only": smoke, "optimizer_updates": model._n_updates, "save_reload_action_diff": reload_difference, "wall_s": result["wall_s"]}), flush=True)
        return result
    except BaseException as exc:
        if model is not None:
            try:
                save_checkpoint(model.num_timesteps, "interrupted")
            except Exception as saving_error:
                line_json(output / "save_faults.jsonl", {"error": repr(saving_error)})
        write_json(output / "interruption.json", {
            "error": repr(exc), "actual_steps": model.num_timesteps if model is not None else 0,
            "wall_s": time.perf_counter() - begin, "complete": False,
            "smoke_only": smoke, "test_episodes": 0,
        })
        raise
    finally:
        if vec is not None:
            vec.close()


def main():
    parser = argparse.ArgumentParser(description="按冻结配置从头训练；默认 CPU、四 worker，每个正式种子 1,024,000 步。")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seed", type=int, choices=SEEDS, help="单个正式种子（默认 550901）")
    group.add_argument("--all-seeds", action="store_true", help="按 550901、551901、552901 顺序训练")
    group.add_argument("--smoke", action="store_true", help="只检查采样、更新和模型保存；不产生正式结果")
    parser.add_argument("--steps", type=int, help="仅用于 smoke，必须为 2048 的正整数倍；默认 2048")
    parser.add_argument("--output", required=True, help="outputs 下的新目录名；已存在则拒绝覆盖")
    args = parser.parse_args()
    if args.steps is not None and not args.smoke:
        parser.error("--steps 只能用于 --smoke；正式预算固定为每种子 1,024,000 步")
    smoke_steps = (args.steps if args.steps is not None else ROLLOUT_STEPS) if args.smoke else None
    if smoke_steps is not None and (smoke_steps < ROLLOUT_STEPS or smoke_steps % ROLLOUT_STEPS):
        parser.error("smoke 步数必须为 2048 的正整数倍")
    command_begin = time.perf_counter()
    verified = verify_release(check_environment=True)
    output = reserve_output(args.output)
    write_json(output / "release_verification.json", verified)
    seeds = SEEDS if args.all_seeds else (SMOKE_SEED if args.smoke else (args.seed or SEEDS[0]),)
    runs = []
    for seed in seeds:
        runs.append(train_one(seed, output / f"seed{seed}", smoke_steps, command_begin))
    write_json(output / "training_summary.json", {
        "seeds": list(seeds), "smoke_only": args.smoke,
        "complete": all(run["complete"] for run in runs),
        "total_interaction_steps": sum(run["actual_steps"] for run in runs),
        "wall_s": time.perf_counter() - command_begin,
        "all_run_paths": [f"seed{seed}/run.json" for seed in seeds],
        "saved_original_results_replaced": False,
    })


if __name__ == "__main__":
    main()
