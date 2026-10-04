"""Paired Stage 4 episodes, reusing Stage 3 tracker, motors and safety rules."""
from pathlib import Path
import csv
import json
import time
import numpy as np
import pybullet as p
from .scene import PandaScene, ROOT
from .trajectory import QuinticLine
from .tracking import PositionTracker, load_config, state, safety_reason, forbid_joint_resets, write_json
from .posture import ArtificialPotentialPosture, LinkPointJacobian


def read_trace(path):
    """Respect CSV quoting: point metadata contains commas inside JSON cells."""
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Empty trajectory CSV")
    columns = {}
    for key in rows[0]:
        values = [r[key] for r in rows]
        try:
            columns[key] = np.array([float(v) if v else np.nan for v in values])
        except ValueError:
            columns[key] = np.array(values, dtype=object)
    result = np.empty(len(rows), dtype=[(k, v.dtype) for k, v in columns.items()])
    for key, values in columns.items():
        result[key] = values
    return result


def run_posture_episode(config, case, method, output, name, fixed_u=None, replay_csv=None, gui=False, render=False):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / config["tracking_config"])
    replay = None
    if replay_csv:
        replay = read_trace(replay_csv)
    rows, settle = [], []
    failure, failure_phase = None, None
    with PandaScene(ROOT / cfg["scene_config"], gui=gui) as scene:
        scene.move_obstacle(case["sphere_position_m"])
        if abs(case["sphere_radius_m"] - scene.config["obstacle"]["radius"]) > 1e-12:
            raise ValueError("This baseline keeps the Stage 2 sphere radius")
        tracker = PositionTracker(scene, cfg["controller"])
        apf = ArtificialPotentialPosture(scene, config["apf"])
        points = LinkPointJacobian(scene)
        dt = scene.config["time_step"]
        nsteps = round(cfg["duration_s"] / dt)
        initial_q, _ = state(scene)
        collision = scene.detect()
        pose = scene.tool_pose()
        failure = safety_reason(scene, *state(scene), pose, collision, cfg["validation"])
        if failure:
            failure_phase = "initialization"
        with forbid_joint_resets():
            for k in range(cfg["settle_steps"]) if not failure else []:
                collision = scene.step()
                q, v = state(scene)
                pose = scene.tool_pose()
                failure = safety_reason(scene, q, v, pose, collision, cfg["validation"])
                settle.append({"time_s": (k+1)*dt, **{f"q{i+1}": float(x) for i,x in enumerate(q)},
                               "external_distance_m": collision["obstacle"]["minimum_signed_distance"],
                               "self_distance_m": collision["self"]["minimum_signed_distance"], "failure": failure or ""})
                if failure:
                    failure_phase = "settle"
                    break
            settled_q, _ = state(scene)
            measured_start = np.asarray(scene.tool_pose()["position"])
            reference_start = np.asarray(case.get("reference_start_world_m", measured_start))
            start_mismatch = float(np.linalg.norm(reference_start - measured_start))
            if start_mismatch > config["validation"]["regression_position_tolerance_m"] and not failure:
                failure, failure_phase = "reference_start_mismatch", "settle"
            line = QuinticLine(reference_start, reference_start + case["displacement_m"], cfg["duration_s"])
            if replay is not None and len(replay) != nsteps + 1:
                raise ValueError("Command replay requires a complete source episode")
            if gui:
                camera = scene.config["camera"]
                p.resetDebugVisualizerCamera(camera["distance"], camera["yaw"], camera["pitch"], camera["target"], physicsClientId=scene.cid)
                p.addUserDebugLine(line.start, line.goal, [0, .2, 1], 3, physicsClientId=scene.cid)
            if render and not failure:
                scene.render(out / f"{name}_initial.png")
            for k in range(nsteps + 1) if not failure else []:
                t = k * dt
                q, v = state(scene)
                pose = scene.tool_pose()
                position = np.asarray(pose["position"])
                pd, vd, _ = line.sample(t)
                failure = safety_reason(scene, q, v, pose, collision, cfg["validation"])
                error = float(np.linalg.norm(pd - position))
                if error > cfg["validation"]["max_tracking_error_m"]:
                    failure = failure or "tracking_error_exceeded_0.01_m"
                provider = {"u": np.zeros(7), "u_obstacle": np.zeros(7), "u_joint": np.zeros(7), "active_points": [], "immobile_base_in_influence": False}
                if not failure:
                    if method == "apf":
                        provider = apf.compute(q[:7], collision["obstacle"])
                    elif method == "fixed":
                        provider["u"] = np.asarray(fixed_u, dtype=float)
                control = None
                if not failure:
                    try:
                        control = tracker.compute(position, pd, vd, q[:7], provider["u"], config["posture_cap_rad_s"])
                    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
                        failure = f"control_error:{type(exc).__name__}:{exc}"
                if control is not None and replay is not None:
                    replay_command = np.array([replay[f"command{i+1}"][k] for i in range(7)])
                    if not np.isfinite(replay_command).all() or np.any(np.abs(replay_command) > tracker.speed_limits+1e-12):
                        failure = "invalid_replay_command"
                    control["command"] = replay_command
                    # Do not run the policy or re-limit the saved executed commands.
                    # Independent safety checks still apply to every actual state.
                row = {"time_s": t, "error_m": error, "command_applied": int(k < nsteps and not failure),
                       "external_distance_m": collision["obstacle"]["minimum_signed_distance"],
                       "external_distance_status": collision["obstacle"]["distance_status"],
                       "self_distance_m": collision["self"]["minimum_signed_distance"],
                       "external_collision": int(collision["obstacle"]["contact_or_penetration"]),
                       "self_collision": int(collision["self"]["contact_or_penetration"]),
                       "self_buffer": int(collision["self"]["within_safety_buffer"]),
                       "active_apf_links": len(provider["active_points"]),
                       "apf_points_json": json.dumps(provider["active_points"], separators=(",", ":")),
                       "failure": failure or ""}
                for prefix, values in [("ref",pd),("actual",position),("ref_velocity",vd)]:
                    row.update({f"{prefix}_{a}":float(x) for a,x in zip("xyz",values)})
                for prefix, values in [("q",q),("actual_velocity",v),("u_obstacle",provider["u_obstacle"]),("u_joint",provider["u_joint"])]:
                    row.update({f"{prefix}{i+1}":float(x) for i,x in enumerate(values)})
                for prefix in ["track","u_input","u_limited","secondary","raw","command"]:
                    row.update({f"{prefix}{i+1}":float(control[prefix][i]) if control is not None else None for i in range(7)})
                for prefix in ["secondary_leakage", "limit_task_distortion"]:
                    row.update({f"{prefix}_{a}":float(x) if control is not None else None for a,x in zip("xyz",control[prefix] if control is not None else [0]*3)})
                row.update({f"singular_value{i+1}":float(control["singular_values"][i]) if control is not None else None for i in range(3)})
                row["speed_scale"] = control["scale"] if control is not None else None
                row["soft_limited"] = int(control["soft_limited"]) if control is not None else 0
                row["posture_clipped"] = int(control["posture_clipped"]) if control is not None else 0
                for link in [3,4,8]:
                    origin,_ = points.link_frame(link)
                    row.update({f"link{link}_{a}":float(x) for a,x in zip("xyz",origin)})
                nearest = collision["obstacle"]["nearest"]
                row["nearest_external_link"] = nearest["link_a"] if nearest else None
                row["nearest_external_point_json"] = json.dumps(nearest, separators=(",", ":")) if nearest else ""
                rows.append(row)
                if failure or k == nsteps:
                    failure_phase = "tracking" if failure else None
                    break
                tracker.apply(control["command"])
                collision = scene.step()
                if gui:
                    time.sleep(dt)
            tracker.apply(np.zeros(7))
        if render:
            scene.render(out / f"{name}_final.png")
        all_joints = [scene.joints[i] for i in scene.arm + scene.fingers]
        complete = bool(len(rows) == nsteps+1 and not failure)
        applied = [r for r in rows if r["command_applied"]]
        errors = np.array([r["error_m"] for r in rows])
        distance = [r["external_distance_m"] for r in rows if r["external_distance_m"] is not None]
        self_distance = [r["self_distance_m"] for r in rows if r["self_distance_m"] is not None]
        command_array = np.array([[r[f"command{i+1}"] for i in range(7)] for r in applied])
        smoothness = float(np.sqrt(np.mean(np.sum((np.diff(command_array, axis=0)/dt)**2, axis=1)))) if len(applied)>1 else None
        q_array = np.array([[r[f"q{i+1}"] for i in range(9)] for r in rows]) if rows else np.empty((0,9))
        margin = np.minimum(q_array[:,:7] - tracker.lower, tracker.upper - q_array[:,:7])
        norms = lambda prefix: [float(np.linalg.norm([r[f"{prefix}_{a}"] for a in "xyz"])) for r in applied]
        summary = {"name":name,"case":case,"method":method,"success":complete,"complete_duration":complete,
                   "failure_reason":failure,"failure_phase":failure_phase,
                   "failure_time_s":rows[-1]["time_s"] if failure and rows else settle[-1]["time_s"] if failure and settle else 0 if failure else None,
                   "actual_duration_s":rows[-1]["time_s"] if rows else 0,"requested_duration_s":line.duration,
                   "tracking_physical_steps":max(0,len(rows)-1),"logged_states":len(rows),"settle_steps_completed":len(settle),
                   "reference_start_world_m":line.start.tolist(),"reference_goal_world_m":line.goal.tolist(),
                   "measured_start_world_m":measured_start.tolist(),"start_mismatch_m":start_mismatch,
                   "settled_q_arm_rad":settled_q[:7].tolist(),"max_error_m":float(errors.max()) if len(rows) else None,
                   "rmse_m":float(np.sqrt(np.mean(errors**2))) if len(rows) else None,
                   "endpoint_error_m":float(errors[-1]) if complete else None,
                   "minimum_external_distance_m":min(distance) if distance else None,
                   "external_distance_censored_states":sum(r["external_distance_m"] is None for r in rows),
                   "external_query_range_m":scene.config["collision"]["query_range"],
                   "minimum_self_distance_m":min(self_distance) if self_distance else None,
                   "external_collision_states":sum(r["external_collision"] for r in rows),
                   "self_collision_states":sum(r["self_collision"] for r in rows),
                   "self_buffer_states":sum(r["self_buffer"] for r in rows),
                   "speed_saturated_fraction":sum(r["speed_scale"] < 1-1e-12 for r in applied)/len(applied) if applied else None,
                   "soft_limited_fraction":sum(r["soft_limited"] for r in applied)/len(applied) if applied else None,
                   "posture_clipped_fraction":sum(r["posture_clipped"] for r in applied)/len(applied) if applied else None,
                   "max_actual_arm_velocity_rad_s":max([abs(r[f"actual_velocity{i+1}"]) for r in rows for i in range(7)],default=None),
                   "max_executed_command_rad_s":float(np.max(np.abs(command_array))) if len(applied) else None,
                   "max_secondary_leakage_m_s":max(norms("secondary_leakage"),default=None),
                   "max_limit_task_distortion_m_s":max(norms("limit_task_distortion"),default=None),
                   "min_singular_value_m_per_rad":min([r["singular_value3"] for r in applied],default=None),
                   "command_smoothness_rms_rad_s2":smoothness,
                   "smoothness_definition":"sqrt(mean(||(command[k+1]-command[k])/dt||_2^2)); applied commands only; partial runs not compared as full performance",
                   "max_arm_change_rad":float(np.max(np.abs(q_array[:,:7]-settled_q[:7]))) if rows else None,
                   "min_arm_hard_limit_margin_rad":float(np.min(margin)) if rows else None,
                   "max_finger_error_m":float(np.max(np.abs(q_array[:,7:]-.02))) if rows else None,
                   "joint_resets_during_motion":0,"reset_guard_active":True,
                   "joint_limits":all_joints,"motor":cfg["controller"],"posture_cap_rad_s":config["posture_cap_rad_s"],
                   "apf_config":config["apf"],"apf_active_states":sum(r["active_apf_links"]>0 for r in rows),
                   "steps_csv":str(out/f"{name}_steps.csv"),"settle_csv":str(out/f"{name}_settle.csv"),
                   "replay_source_csv":str(replay_csv) if replay_csv else None,
                   "replay_control_fields_note":"u/track/projection diagnostics are hypothetical zero-u controller; command is the saved executed APF command" if replay_csv else None}
        for suffix,records in [("steps",rows),("settle",settle)]:
            if records:
                with (out/f"{name}_{suffix}.csv").open("w",newline="") as f:
                    writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    summary["connection_closed"]=not bool(p.isConnected(scene.cid))
    write_json(out/f"{name}_summary.json",summary)
    return summary
