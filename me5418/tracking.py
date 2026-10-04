"""Position-only Jacobian control and actual-motor episode execution.

Official API references and the restricted local-point convention are recorded
in docs/stage03_api_sources.md. Stage 2 scene/tool/collision code is reused.
"""
from __future__ import annotations

import csv
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pybullet as p

from .scene import PandaScene, ROOT
from .trajectory import QuinticLine

DEFAULT_CONFIG = ROOT / "configs/stage03_tracking.json"


def load_config(path=DEFAULT_CONFIG):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    # Invalid state must be recorded as a failure, never lose all evidence
    # because strict JSON forbids NaN/Inf. CSV keeps the raw invalid value.
    def finite_json(x):
        if isinstance(x, np.ndarray):
            return finite_json(x.tolist())
        if isinstance(x, np.generic):
            return finite_json(x.item())
        if isinstance(x, dict):
            return {k: finite_json(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [finite_json(v) for v in x]
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(finite_json(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


@contextmanager
def forbid_joint_resets():
    """Runtime guard: init/FD may reset, physical hold/tracking may not."""
    original = p.resetJointState
    def forbidden(*args, **kwargs):
        raise RuntimeError("resetJointState is forbidden during physical motion")
    p.resetJointState = forbidden
    try:
        yield
    finally:
        p.resetJointState = original


class ToolJacobian:
    def __init__(self, scene):
        self.scene = scene
        self.movable = sorted((j for j in scene.joints if j["movable_position_order"] is not None),
                              key=lambda j: j["movable_position_order"])
        self.indices = [j["joint_index"] for j in self.movable]
        self.arm_columns = [self.indices.index(i) for i in scene.arm]
        inertial = p.getDynamicsInfo(scene.robot, scene.tool_link, physicsClientId=scene.cid)
        # Restrict this implementation to the verified Stage 2 zero-point tool.
        # This prevents silently applying ambiguous localPosition conventions to
        # a future tool having a nonzero inertial offset or rotation.
        if not (np.allclose(inertial[3], [0, 0, 0], atol=1e-12)
                and np.allclose(inertial[4], [0, 0, 0, 1], atol=1e-12)
                and np.allclose(scene.config["tool"]["local_position"], [0, 0, 0], atol=1e-12)
                and np.allclose(scene.config["base_orientation_xyzw"], [0, 0, 0, 1], atol=1e-12)):
            raise ValueError("Jacobian requires the verified zero-offset tool and identity base rotation")
        self.metadata = {
            "tool": scene.config["tool"], "tool_link_index": scene.tool_link,
            "localPosition": [0.0, 0.0, 0.0], "local_inertial_position": list(inertial[3]),
            "local_inertial_orientation_xyzw": list(inertial[4]),
            "input_order": [{"slot": k, "joint_name": j["joint_name"], "joint_index": j["joint_index"],
                             "raw_q_index": j["q_index"]} for k, j in enumerate(self.movable)],
            "arm_columns": self.arm_columns, "output_shape": [3, 7],
            "frame": "world (verified identity base rotation)", "units": "m/rad for seven arm columns",
            "raw_q_index_is_not_array_index": True}

    def matrix(self):
        q = [s[0] for s in p.getJointStates(self.scene.robot, self.indices, physicsClientId=self.scene.cid)]
        linear, _ = p.calculateJacobian(self.scene.robot, self.scene.tool_link, [0, 0, 0],
                                        q, [0.0] * len(q), [0.0] * len(q), physicsClientId=self.scene.cid)
        full = np.asarray(linear, dtype=float)
        if full.shape != (3, len(q)) or len(q) != 9:
            raise ValueError(f"Unexpected full Jacobian shape {full.shape}, DOFs={len(q)}")
        return full[:, self.arm_columns]


class PositionTracker:
    def __init__(self, scene, config):
        self.scene, self.config = scene, config
        self.jacobian = ToolJacobian(scene)
        self.lower = np.array([scene.joints[i]["lower_limit"] for i in scene.arm])
        self.upper = np.array([scene.joints[i]["upper_limit"] for i in scene.arm])
        self.urdf_speed = np.array([scene.joints[i]["urdf_max_velocity"] for i in scene.arm])
        self.forces = [scene.joints[i]["urdf_max_force"] for i in scene.arm]
        self.speed_limits = np.minimum(self.urdf_speed, config["command_speed_cap_rad_s"])

    def limit_velocity(self, q, raw):
        ratio = float(np.max(np.abs(raw) / self.speed_limits))
        scale = 1.0 / max(1.0, ratio)
        scaled = raw * scale
        # One-step command cannot continue outward past the soft boundary.
        # If already outside soft range, only inward or zero commands are allowed.
        dt = self.scene.config["time_step"]
        margin = self.config["soft_margin_rad"]
        vmax = np.maximum(0.0, (self.upper - margin - q) / dt)
        vmin = np.minimum(0.0, (self.lower + margin - q) / dt)
        command = np.clip(scaled, vmin, vmax)
        return command, scale, bool(np.any(np.abs(command - scaled) > 1e-12))

    def compute(self, position, desired_position, desired_velocity, q, posture_u=None, posture_cap=0.2):
        j = self.jacobian.matrix()
        singular_values = np.linalg.svd(j, compute_uv=False)
        error = np.asarray(desired_position) - position
        task_velocity = np.asarray(desired_velocity) + self.config["kp_per_s"] * error
        damping = self.config["damping_m_per_rad"]
        a = j @ j.T + damping**2 * np.eye(3)
        # Keep Stage 3's operation order for the primary task and u=0 regression.
        track = j.T @ np.linalg.solve(a, task_velocity)
        damped_inverse = j.T @ np.linalg.solve(a, np.eye(3))
        u = np.zeros(7) if posture_u is None else np.asarray(posture_u, dtype=float)
        if u.shape != (7,) or not np.isfinite(u).all() or not np.isfinite(posture_cap) or posture_cap <= 0:
            raise ValueError("posture_u must be seven finite pre-projection velocities (rad/s)")
        limited_u = np.clip(u, -posture_cap, posture_cap)
        secondary = (np.eye(7) - damped_inverse @ j) @ limited_u
        raw = track + secondary
        if not np.isfinite([*raw, *singular_values, *task_velocity]).all():
            raise FloatingPointError("Non-finite control output")
        command, scale, soft_limited = self.limit_velocity(q, raw)
        return {"raw": raw, "command": command, "singular_values": singular_values,
                "task_velocity": task_velocity, "scale": scale, "soft_limited": soft_limited,
                "track": track, "u_input": u, "u_limited": limited_u, "secondary": secondary,
                "secondary_leakage": j @ secondary,
                "limit_task_distortion": j @ (command - raw),
                "posture_clipped": bool(np.any(np.abs(u - limited_u) > 1e-12))}

    def apply(self, command):
        p.setJointMotorControlArray(self.scene.robot, self.scene.arm, p.VELOCITY_CONTROL,
                                    targetVelocities=command.tolist(), forces=self.forces,
                                    velocityGains=[self.config["velocity_gain"]] * 7, physicsClientId=self.scene.cid)


def state(scene):
    states = p.getJointStates(scene.robot, scene.arm + scene.fingers, physicsClientId=scene.cid)
    return np.array([s[0] for s in states]), np.array([s[1] for s in states])


def safety_reason(scene, q, velocity, pose, collision, validation):
    if not np.isfinite([*q, *velocity, *pose["position"], *pose["orientation_xyzw"]]).all():
        return "non_finite_state"
    for k, i in enumerate(scene.arm + scene.fingers):
        j = scene.joints[i]
        tol = validation["hard_limit_tolerance"]
        if q[k] < j["lower_limit"] - tol or q[k] > j["upper_limit"] + tol:
            return f"hard_position_limit:{j['joint_name']}"
        if abs(velocity[k]) > j["urdf_max_velocity"] + validation["actual_speed_urdf_tolerance_rad_s"]:
            return f"actual_urdf_speed_limit:{j['joint_name']}"
    for kind in ["self", "obstacle"]:
        if collision[kind]["contact_or_penetration"]:
            return f"related_collision:{kind}"
    return None


def run_episode(config, displacement, output, name, gui=False):
    """Fresh scene per episode; settle first, then exactly N timed physical steps."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with PandaScene(ROOT / config["scene_config"], gui=gui) as scene:
        scene.move_obstacle(config["obstacle_override_position"])
        tracker = PositionTracker(scene, config["controller"])
        dt = scene.config["time_step"]
        nsteps = round(config["duration_s"] / dt)
        if abs(nsteps * dt - config["duration_s"]) > 1e-12:
            raise ValueError("Duration must be an integer number of physical steps")
        initial_q = scene.joint_positions()
        initial_tool_position = np.array(scene.tool_pose()["position"])
        settle_rows, rows = [], []
        failure = None
        with forbid_joint_resets():
            for k in range(config["settle_steps"]):
                collision = scene.step()
                q, velocity = state(scene)
                pose = scene.tool_pose()
                failure = safety_reason(scene, q, velocity, pose, collision, config["validation"])
                settle_rows.append({"time_s": (k + 1) * dt, **{f"q{i+1}": float(v) for i, v in enumerate(q)},
                                    "self_min_distance_m": collision["self"]["minimum_signed_distance"],
                                    "failure": failure or ""})
                if failure:
                    break
            settled_q, _ = state(scene)
            start_pose = scene.tool_pose()
            start = np.array(start_pose["position"])
            # A failed non-finite settling state cannot define a trajectory.
            # Keep valid planned endpoints solely to finish the failure record.
            start_valid = bool(np.isfinite(start).all())
            if not start_valid:
                failure = failure or "non_finite_settled_tool_position"
                start = initial_tool_position
            line = QuinticLine(start, start + np.asarray(displacement), config["duration_s"])
            if gui:
                cam = scene.config["camera"]
                p.resetDebugVisualizerCamera(cam["distance"], cam["yaw"], cam["pitch"], cam["target"], physicsClientId=scene.cid)
                p.addUserDebugLine(line.start, line.goal, [1, 0.3, 0], 4, physicsClientId=scene.cid)
            collision = scene.detect()
            for k in range(nsteps + 1) if not failure else []:
                t = k * dt  # Reference time never depends on actual tool progress.
                q, velocity = state(scene)
                pose = scene.tool_pose()
                xyz = np.array(pose["position"])
                pd, vd, _ = line.sample(t)
                failure = safety_reason(scene, q, velocity, pose, collision, config["validation"])
                error = float(np.linalg.norm(pd - xyz))
                if error > config["validation"]["max_tracking_error_m"]:
                    failure = failure or "tracking_error_exceeded_0.01_m"
                control = None
                if failure is None:
                    try:
                        control = tracker.compute(xyz, pd, vd, q[:7])
                    except (FloatingPointError, np.linalg.LinAlgError) as exc:
                        failure = f"control_error:{type(exc).__name__}:{exc}"
                row = {"time_s": t, "error_m": error, "command_applied": int(k < nsteps and failure is None),
                       "self_min_distance_m": collision["self"]["minimum_signed_distance"],
                       "obstacle_min_distance_m": collision["obstacle"]["minimum_signed_distance"],
                       "obstacle_distance_status": collision["obstacle"]["distance_status"],
                       "self_buffer_warning": int(collision["self"]["within_safety_buffer"]),
                       "self_collision": int(collision["self"]["contact_or_penetration"]),
                       "obstacle_collision": int(collision["obstacle"]["contact_or_penetration"]),
                       "failure": failure or ""}
                for prefix, values in [("ref", pd), ("actual", xyz), ("ref_velocity", vd)]:
                    row.update({f"{prefix}_{axis}": float(v) for axis, v in zip("xyz", values)})
                for prefix, values in [("q", q), ("actual_velocity", velocity)]:
                    row.update({f"{prefix}{i+1}": float(v) for i, v in enumerate(values)})
                row.update({f"tool_quat_{a}": float(v) for a, v in zip("xyzw", pose["orientation_xyzw"])})
                for prefix in ["raw", "command"]:
                    row.update({f"{prefix}{i+1}": float(control[prefix][i]) if control else None for i in range(7)})
                row.update({f"singular_value{i+1}": float(control["singular_values"][i]) if control else None for i in range(3)})
                row["speed_scale"] = control["scale"] if control else None
                row["soft_limited"] = int(control["soft_limited"]) if control else 0
                rows.append(row)
                if failure or k == nsteps:
                    break
                tracker.apply(control["command"])
                collision = scene.step()  # Physical response, refresh, and all collision queries every step.
                if gui:
                    time.sleep(dt)
            tracker.apply(np.zeros(7))  # No post-T settling is used to improve endpoint metrics.
        for suffix, records in [("steps", rows), ("settle", settle_rows)]:
            if records:
                with (output / f"{name}_{suffix}.csv").open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(records[0]))
                    writer.writeheader()
                    writer.writerows(records)
        finished = bool(rows and len(rows) == nsteps + 1 and failure is None)
        positions = np.array([[r[f"q{i+1}"] for i in range(9)] for r in rows]) if rows else np.empty((0, 9))
        velocities = np.array([[r[f"actual_velocity{i+1}"] for i in range(9)] for r in rows]) if rows else np.empty((0, 9))
        applied = [r for r in rows if r["command_applied"]]
        errors = np.array([r["error_m"] for r in rows])
        distance = [r["self_min_distance_m"] for r in rows if r["self_min_distance_m"] is not None]
        all_joints = [scene.joints[i] for i in scene.arm + scene.fingers]
        hard_margins = np.minimum(positions - [j["lower_limit"] for j in all_joints],
                                  [j["upper_limit"] for j in all_joints] - positions)
        end_pose = scene.tool_pose()
        dot = abs(float(np.dot(start_pose["orientation_xyzw"], end_pose["orientation_xyzw"])))
        summary = {
            "name": name, "success": finished, "complete_duration": finished,
            "failure_reason": failure, "failure_time_s": rows[-1]["time_s"] if failure and rows else None,
            "settle_failure_time_s": settle_rows[-1]["time_s"] if failure and not rows else None,
            "requested_duration_s": line.duration, "actual_duration_s": rows[-1]["time_s"] if rows else 0,
            "physical_tracking_steps": max(0, len(rows) - 1), "logged_states": len(rows),
            "time_step_s": dt, "start_world_m": line.start.tolist(), "goal_world_m": line.goal.tolist(),
            "start_method": "tool_pose() after 240 actual POSITION_CONTROL settle steps",
            "measured_start_valid": start_valid,
            "displacement_m": list(displacement), "settled_q_arm_rad": settled_q[:7].tolist(),
            "settle_max_arm_drift_rad": float(max(np.max(np.abs([r[f'q{i+1}'] - initial_q[i] for i in range(7)])) for r in settle_rows)),
            "max_error_m": float(errors.max()) if len(errors) else None,
            "rmse_m": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
            "endpoint_error_m": float(errors[-1]) if finished else None,
            "last_logged_error_m": float(errors[-1]) if len(errors) else None,
            "max_abs_actual_arm_velocity_rad_s": np.max(np.abs(velocities[:, :7]), axis=0).tolist() if len(rows) else None,
            "max_abs_raw_command_rad_s": [max(abs(r[f"raw{i+1}"]) for r in applied) for i in range(7)] if applied else None,
            "max_abs_limited_command_rad_s": [max(abs(r[f"command{i+1}"]) for r in applied) for i in range(7)] if applied else None,
            "speed_saturated_step_fraction": sum(r["speed_scale"] < 1 - 1e-12 for r in applied) / len(applied) if applied else None,
            "soft_limited_step_fraction": sum(r["soft_limited"] for r in applied) / len(applied) if applied else None,
            "min_arm_hard_limit_margin_rad": float(np.min(hard_margins[:, :7])) if len(rows) else None,
            "max_arm_change_from_settled_rad": float(np.max(np.abs(positions[:, :7] - settled_q[:7]))) if len(rows) else None,
            "max_finger_error_from_0_02_m": float(np.max(np.abs(positions[:, 7:] - 0.02))) if len(rows) else None,
            "self_collision_state_count": sum(r["self_collision"] for r in rows),
            "obstacle_collision_state_count": sum(r["obstacle_collision"] for r in rows),
            "self_buffer_warning_state_count": sum(r["self_buffer_warning"] for r in rows),
            "minimum_queried_self_distance_m": min(distance) if distance else None,
            "obstacle_distance": {"status": "beyond_query_range_censored" if rows and all(r["obstacle_distance_status"] == "beyond_query_range_censored" for r in rows) else "see_step_log",
                                  "range_m": scene.config["collision"]["query_range"], "minimum_measured_m": min([r["obstacle_min_distance_m"] for r in rows if r["obstacle_min_distance_m"] is not None], default=None)},
            "min_singular_value_m_per_rad": min([r["singular_value3"] for r in rows if r["singular_value3"] is not None], default=None),
            "orientation_change_rad_uncontrolled": float(2 * np.arccos(np.clip(dot, 0, 1))),
            "reset_guard_active": True, "joint_resets_during_motion": 0,
            "initialization_reset_count": scene.reset_count, "obstacle_position_m": config["obstacle_override_position"],
            "motor": {"mode": "VELOCITY_CONTROL", "velocity_gain": config["controller"]["velocity_gain"], "forces_Nm": tracker.forces,
                      "finger_mode": "POSITION_CONTROL", "finger_parameters": scene.config["finger_motor"]},
            "joint_limits": all_joints, "jacobian_mapping": tracker.jacobian.metadata,
            "tool_final_pose": end_pose, "csv_semantics": "row t_k is state/reference at t_k; raw/command calculated at t_k and command_applied=1 means sent for [t_k,t_k+dt). Final row command is diagnostic only.",
            "steps_csv": str(output / f"{name}_steps.csv"), "settle_csv": str(output / f"{name}_settle.csv")}
        if name in ["pilot_x", "x_positive"] and failure is None:
            summary["final_render"] = scene.render(output / f"{name}_final.png")
    summary["connection_closed"] = not bool(p.isConnected(scene.cid))
    write_json(output / f"{name}_summary.json", summary)
    return summary
