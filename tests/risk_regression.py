#!/usr/bin/env python3
"""Risk-focused checks against the unchanged motor-based task implementation.

No test reward or synthetic fault is included in scientific performance scores.
Run with the locked CPU environment; no pytest dependency is required.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
import unittest
import tempfile
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from common import bootstrap, module_paths, read, sha, verify_release, write_json
AUDIT = bootstrap(False)

import numpy as np
import torch
from stable_baselines3 import PPO
from me5418.arm_env import PandaPostureEnv, json_safe
from me5418.diagnostic_env import DiagnosticPandaEnv, encode_observation
from me5418.evaluation import classify_sample
from me5418.scene import PandaScene
from me5418.tracking import PositionTracker, state

torch.set_num_threads(1)
EVIDENCE = {}


class TaskRisks(unittest.TestCase):
    def env(self, **options):
        env = DiagnosticPandaEnv("validation", ROOT / "configs/stage09/C.json", trace=True)
        self.addCleanup(env.close)
        # A preassigned far scene gives a clear baseline for injected faults.
        identifier = next(i for i, s in env.pool.items() if s["preassigned_geometric_category"] == "far")
        obs, info = env.reset(seed=541812, options={"scene_id": identifier, **options})
        return env, obs, info

    def test_action_scale_clip_hold_and_finite_guard(self):
        env, _, _ = self.env()
        invalid = [np.full(7, np.nan), np.zeros(6)]
        for action in invalid:
            with self.assertRaises(ValueError):
                env.step(action)
            self.assertEqual(env.episode.k, 0, "Invalid action must not advance motors")
        action = np.array([-2., -.5, 0., .25, .5, 1., 2.])
        expected = np.array([-.2, -.1, 0., .05, .1, .2, .2])
        _, _, terminated, truncated, info = env.step(action)
        np.testing.assert_allclose(env.episode.schedule.u, expected, rtol=0, atol=1e-15)
        np.testing.assert_allclose(env.episode.schedule.raw, .2 * action, rtol=0, atol=1e-15)
        self.assertEqual(info["executed_substeps"], 5)
        self.assertFalse(terminated or truncated)
        applied = [row for row in env.episode.rows if row["command_applied"]]
        self.assertEqual(len(applied), 5)
        for row in applied:
            np.testing.assert_allclose([row[f"u{i}"] for i in range(7)], expected, rtol=0, atol=1e-15)
            self.assertEqual(row["held_from_step"], 0)
            self.assertLessEqual(max(abs(row[f"command{i}"]) for i in range(7)), 1 + 1e-12)
        EVIDENCE["action"] = {"tested_normalized_action": action.tolist(), "executed_preprojection_rad_s": expected.tolist(), "actual_motor_substeps": len(applied), "invalid_action_motor_steps": 0}

    def test_truncation_has_no_terminal_success_bonus(self):
        env, _, _ = self.env(external_limit_steps=3)
        _, reward, terminated, truncated, info = env.step(np.zeros(7))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertFalse(info["success"])
        self.assertEqual(info["executed_substeps"], 3)
        self.assertEqual(info["reward_terms"]["terminal"], 0)
        self.assertFalse(env.last_summary["success"])
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(7))
        EVIDENCE["truncation"] = {"physical_steps": 3, "terminated": terminated, "truncated": truncated, "terminal_reward": info["reward_terms"]["terminal"], "reward": reward}

    def test_failure_latches_before_more_motor_motion(self):
        outcomes = []
        for kind, category in [("tracking_error", "tracking_error"), ("hard_position", "constraint_violation"), ("external_contact", "external_collision_band"), ("non_finite", "non_finite")]:
            env, _, _ = self.env(synthetic_fault={"step": 3, "kind": kind})
            obs, _, terminated, truncated, info = env.step(np.zeros(7))
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["executed_substeps"], 3)
            self.assertEqual(info["episode_summary"]["failure_category"], category)
            self.assertEqual(info["reward_terms"]["terminal"], -20)
            self.assertFalse(env.last_summary["success"])
            self.assertTrue(np.isfinite(obs).all())
            summary = copy.deepcopy(env.last_summary)
            self.assertEqual(env.episode.rows[-1]["command_applied"], 0)
            with self.assertRaises(RuntimeError):
                env.step(np.zeros(7))
            self.assertEqual(json_safe(env.last_summary), json_safe(summary), "Failure record must stay latched")
            outcomes.append({"fault": kind, "category": category, "failure_time_s": summary["first_failure_time_s"], "motor_substeps": 3, "terminal_reward": -20})
        EVIDENCE["injected_failures"] = outcomes

    def test_real_collision_query_and_limit_gates(self):
        env, _, _ = self.env()
        e = env.episode
        # Real PyBullet geometry, not a fabricated distance or contact flag.
        e.scene.move_obstacle(e.scene.tool_pose()["position"])
        collision = e.scene.detect()
        q, v = state(e.scene)
        reason, category, _ = classify_sample(e.scene, q, v, e.scene.tool_pose(), collision, e.ex["validation"])
        self.assertTrue(collision["obstacle"]["contact_or_penetration"])
        self.assertEqual(category, "external_collision_band")
        tracker = e.tracker
        near = tracker.upper - tracker.config["soft_margin_rad"] + .001
        outward, scale, limited = tracker.limit_velocity(near, np.full(7, 10.))
        self.assertTrue(limited)
        self.assertGreater(scale, 0)
        self.assertLess(scale, 1)
        np.testing.assert_equal(outward, np.zeros(7))
        inward, _, _ = tracker.limit_velocity(near, -np.ones(7))
        self.assertTrue(np.all(inward < 0))
        self.assertTrue(np.all(abs(inward) <= tracker.speed_limits + 1e-12))
        # Collision checks cover both monitored obstacle and self pairs.
        self_collision = copy.deepcopy(collision)
        self_collision["obstacle"]["contact_or_penetration"] = False
        self_collision["self"]["contact_or_penetration"] = True
        _, self_category, _ = classify_sample(e.scene, q, v, e.scene.tool_pose(), self_collision, e.ex["validation"])
        self.assertEqual(self_category, "self_collision_band")
        EVIDENCE["collision_limits"] = {"real_obstacle_distance_m": collision["obstacle"]["minimum_signed_distance"], "real_obstacle_failure_reason": reason, "self_flag_classification_only": self_category, "outward_soft_limit_command_rad_s": outward.tolist(), "uniform_scale_for_10_rad_s": scale}

    def test_split_and_witness_information_boundaries(self):
        env, obs, _ = self.env()
        manifest = read(ROOT / "datasets/me5418-scenes-v1/manifest.json")
        test_id = next(scene["id"] for scene in manifest["scenes"] if scene["split"] == "test")
        with self.assertRaisesRegex(ValueError, "enforced split"):
            env.reset(options={"scene_id": test_id})
        with self.assertRaises(ValueError):
            PandaPostureEnv("test")
        env, obs, _ = self.env()
        context = env.episode.observation_context()
        enriched = copy.deepcopy(context)
        enriched.update({"witness_actions": [[.2] * 7] * 192, "discovery_success": True, "scene_id": "probe", "template_group": "probe", "split": "test"})
        np.testing.assert_array_equal(obs, encode_observation(enriched, env.config))
        rejected = []
        paths = {"opened_test_parameters": ROOT / "datasets/me5418-scenes-v1/splits/test.json", "private_witness": ROOT / "private_witnesses/probe.json", "old_project": ROOT.parent / "ME5418_Group44/probe.json"}
        for name, path in paths.items():
            with self.assertRaises(PermissionError):
                path.read_bytes()
            rejected.append(name)
        EVIDENCE["isolation"] = {"denied_probes": rejected, "extra_privileged_metadata_changes_observation": False, "training_supports_test_reset": False}

    def test_all_models_load_and_deterministic_action_bounds(self):
        env, obs, _ = self.env()
        loaded = []
        for entry in read(ROOT / "models_index.json")["models"]:
            path = ROOT / entry["model_path"]
            self.assertEqual(sha(path), entry["SHA256"])
            model = PPO.load(path, device="cpu")
            self.assertEqual(model.observation_space.shape, (91,))
            self.assertEqual(model.action_space.shape, (7,))
            a, _ = model.predict(obs, deterministic=True)
            b, _ = model.predict(obs, deterministic=True)
            np.testing.assert_array_equal(a, b)
            self.assertTrue(np.isfinite(a).all())
            self.assertTrue(np.all(abs(a) <= 1))
            loaded.append({"seed": entry["seed"], "kind": entry["kind"], "sha256": entry["SHA256"], "shape": list(a.shape), "repeat_action_difference": float(np.max(abs(a - b)))})
        self.assertEqual(len(loaded), 6)
        EVIDENCE["models"] = loaded

    def test_stage12_extension_keeps_original_motor_execution(self):
        from stage12.experiment import new_logged_run
        from me5418.final_evaluation import logged_run
        from comparison import compare_states
        env, _, _ = self.env()
        scenario = next(s for s in env.pool.values() if s["preassigned_geometric_category"] == "near_link")
        base = {"name": "APF", "kind": "apf", "role": "regression", "seed": None, "checkpoint": None, "model_SHA256": None}
        extension = {**base, "apf": read(ROOT / "configs/stage04_posture_apf.json")["apf"]}
        output_base = ROOT / "outputs"
        output_base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="stage12_fixed_apf_regression_", dir=output_base) as temporary:
            target = Path(temporary)
            original = logged_run(scenario, base, target / "original")
            added = new_logged_run(scenario, extension, target / "extension")
            comparison = compare_states(target / "original/states.csv", target / "extension/states.csv")
            self.assertTrue(comparison["pass"])
            self.assertEqual(comparison["maximum_numeric_difference"], 0)
            self.assertEqual(original["success"], added["success"])
            self.assertEqual(original["failure_category"], added["failure_category"])
            self.assertEqual(original["joint_resets_during_motion"], 0)
            self.assertEqual(added["joint_resets_during_motion"], 0)
            EVIDENCE["stage12_wrapper"] = {"scene_id": scenario["id"], "success": added["success"], "failure_category": added["failure_category"], "full_state_maximum_numeric_difference": comparison["maximum_numeric_difference"], "joint_resets_during_motion": 0}

    def test_stage12_protocol_and_parameter_isolation(self):
        from me5418.arm_env import PARAMETER_KEYS
        protocol_path = ROOT / "stage12/configs/protocol.json"
        protocol = read(protocol_path)
        self.assertEqual(sha(protocol_path), (ROOT / "stage12/configs/protocol.sha256").read_text().split()[0])
        self.assertEqual(sha(ROOT / "stage12/experiment.py"), protocol["executed_source_SHA256"]["stage12/experiment.py"])
        self.assertEqual(sha(ROOT / "stage12/data/reserved_geometry.json"), protocol["reserved_geometry_SHA256"])
        self.assertEqual(protocol["validation"]["ordered_ids"], read(ROOT / "datasets/me5418-scenes-v1/splits/validation.json")["scene_ids"])
        self.assertEqual(len(protocol["validation"]["finite_grid"]), 9)
        self.assertEqual(protocol["validation"]["maximum_episodes"], 9 * 58)
        self.assertEqual(protocol["evaluation"]["training_interactions"], 0)
        self.assertEqual(protocol["evaluation"]["methods"], ["tracking", "APF_fixed", "APF_tuned", "PPO_best_550901", "PPO_best_551901", "PPO_best_552901"])
        manifest_path = ROOT / "stage12/data/manifest.json"
        self.assertTrue(manifest_path.is_file(), "Final heldout manifest is required for the complete release check")
        scenes = read(manifest_path)["scenes"]
        self.assertEqual(len(scenes), 100)
        for scene in scenes:
            self.assertEqual(set(scene), set(PARAMETER_KEYS))
            self.assertEqual(scene["split"], "supplemental_heldout")
            self.assertEqual(scene["execution"]["posture_cap_rad_s"], .2)
            self.assertEqual(scene["execution"]["posture_period_steps"], 5)
            self.assertEqual(scene["execution"]["duration_s"], 4)
            self.assertEqual(scene["execution"]["validation"]["max_tracking_error_m"], .01)
        EVIDENCE["stage12_protocol"] = {"source_sha256": protocol["executed_source_SHA256"]["stage12/experiment.py"], "protocol_sha256": sha(protocol_path), "validation_candidates": 9, "validation_episodes": 522, "n_scenes": len(scenes), "manifest_exact_parameter_keys": True, "new_evaluation_training_steps": 0}

    def test_reproduction_copy_preserves_history_and_refuses_overwrite(self):
        helper = ROOT / "stage12/prepare_reproduction.py"
        original_protocol_sha = sha(ROOT / "stage12/configs/protocol.json")
        command = [sys.executable, "-B", str(helper), "--destination"]
        with tempfile.TemporaryDirectory(prefix="me5418_reproduction_guard_") as temporary:
            target = Path(temporary) / "independent_copy"
            completed = subprocess.run([*command, str(target)], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            record = read(target / "preparation_record.json")
            self.assertTrue(record["source_numeric_files_all_match"])
            self.assertEqual(record["checked_frozen_files"], len(read(ROOT / "provenance/release_integrity.json")["files_SHA256"]))
            self.assertEqual(sha(target / "provenance/release_integrity.json"), sha(ROOT / "provenance/release_integrity.json"))
            self.assertEqual(sha(target / "stage12/experiment.py"), sha(ROOT / "stage12/experiment.py"))
            self.assertEqual(sha(target / "stage12/data/reserved_geometry.json"), sha(ROOT / "stage12/data/reserved_geometry.json"))
            self.assertEqual(sha(target / "replication_reference/stage12/configs/protocol.json"), original_protocol_sha)
            self.assertFalse((target / "stage12/configs/protocol.json").exists())
            self.assertFalse((target / "stage12/data/manifest.json").exists())
            self.assertFalse((target / ".venv").exists())
            self.assertEqual(list((target / "outputs").iterdir()), [])
            for entry in ("tools/train.py", "tools/validate.py", "tests/risk_regression.py", "delivery/stage12/run_full.sh", "delivery/stage12/install_cpu.sh"):
                self.assertTrue((target / entry).is_file(), entry)
            refused = subprocess.run([*command, str(target)], capture_output=True, text=True)
            self.assertNotEqual(refused.returncode, 0)
            refused_inside = subprocess.run([*command, str(ROOT / "outputs/refused_internal_copy")], capture_output=True, text=True)
            self.assertNotEqual(refused_inside.returncode, 0)
            self.assertFalse((ROOT / "outputs/refused_internal_copy").exists())
        self.assertEqual(sha(ROOT / "stage12/configs/protocol.json"), original_protocol_sha)
        EVIDENCE["reproduction_copy"] = {"checked_frozen_files": record["checked_frozen_files"], "source_protocol_unchanged": True, "published_protocol_archived_in_new_copy": True, "existing_destination_refused": True, "inside_source_destination_refused": True, "full_entries_copied": True, "no_environment_or_raw_output_copied": True}


class EvidenceResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = []
    def addSuccess(self, test):
        super().addSuccess(test)
        self.rows.append({"test": test.id(), "passed": True})
    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.rows.append({"test": test.id(), "passed": False, "kind": "assertion", "detail": self._exc_info_to_string(err, test)})
    def addError(self, test, err):
        super().addError(test, err)
        self.rows.append({"test": test.id(), "passed": False, "kind": "runtime", "detail": self._exc_info_to_string(err, test)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    began = time.perf_counter()
    verification = verify_release()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TaskRisks)
    result = unittest.TextTestRunner(verbosity=2, resultclass=EvidenceResult).run(suite)
    write_json(args.output, {"pass": result.wasSuccessful(), "tests_run": result.testsRun, "checks": result.rows, "evidence": EVIDENCE, "frozen_release": verification, "module_paths": module_paths(), "data_access": AUDIT.record(), "wall_s": time.perf_counter() - began, "synthetic_faults_excluded_from_performance": True, "actual_regression_scope": "Locked original implementation, direct motor substeps, real obstacle geometry, injected safety faults and information boundary probes"})
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
