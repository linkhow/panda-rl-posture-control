"""Stage 5: shared 240 Hz execution and 48 Hz posture action interface.

Shared physical stepping also serves the Stage 6 Gymnasium wrapper. Policy loaders receive scenario parameters
only; witness sequences are a separate, explicit replay argument.
"""
from pathlib import Path
import csv
import copy
import json
import time
import numpy as np
import pybullet as p
from .scene import PandaScene, ROOT
from .trajectory import QuinticLine
from .tracking import PositionTracker, state, safety_reason, forbid_joint_resets, write_json
from .posture import ArtificialPotentialPosture


def classify_sample(scene, q, velocity, pose, collision, validation, error=None):
    reason = safety_reason(scene, q, velocity, pose, collision, validation)
    if reason is None and error is not None:
        if not np.isfinite(error):
            reason = "non_finite_error"
        elif error > validation["max_tracking_error_m"]:
            reason = "tracking_error_exceeded_0.01_m"
    category = None
    if reason:
        if reason.startswith("non_finite"):
            category = "non_finite"
        elif reason.startswith("related_collision:self"):
            category = "self_collision_band"
        elif reason.startswith("related_collision:obstacle"):
            category = "external_collision_band"
        elif reason.startswith("tracking_error"):
            category = "tracking_error"
        else:
            category = "constraint_violation"
    flags = {f"{kind}_{flag}": bool(collision[kind][field])
             for kind in ["obstacle", "self"]
             for flag, field in [("contact_band", "contact_or_penetration"),
                                 ("negative_penetration", "penetration"),
                                 ("buffer", "within_safety_buffer")]}
    # Negative penetration uses the inherited < -0.1 mm rule. Also retain any
    # d<0 sample separately; a tiny negative value is not hidden by tolerance.
    for kind in ["obstacle", "self"]:
        pts = collision[kind]["closest_points"] + collision[kind]["contact_points"]
        flags[f"{kind}_negative_distance"] = any(x["signed_distance"] < 0 for x in pts)
    return reason, category, flags


class PostureSchedule:
    """At k=0,5,...,955 decide once; hold clipped u for five intervals."""
    def __init__(self, scene, execution, spec, replay=None, decision_function=None):
        self.period = execution["posture_period_steps"]
        self.cap = execution["posture_cap_rad_s"]
        self.spec, self.replay = spec, replay
        self.decision_function=decision_function
        self.apf = ArtificialPotentialPosture(scene, spec.get("apf", execution["apf"]))
        self.raw, self.u = np.zeros(7), np.zeros(7)
        self.last_step, self.clipped = -1, False
        self.decisions, self.times = [], []
        if replay is not None:
            expected = list(range(0, round(execution["duration_s"] / scene.config["time_step"]), self.period))
            if [x["physical_step"] for x in replay] != expected:
                raise ValueError("Witness update indices do not match the 48 Hz protocol")
            if any(np.asarray(x["u_rad_s"]).shape != (7,) or
                   not np.isfinite(x["u_rad_s"]).all() or
                   np.max(np.abs(x["u_rad_s"])) > self.cap + 1e-12 for x in replay):
                raise ValueError("Invalid witness posture actions")

    def value(self, k, t, q, collision, observation=None):
        update = k % self.period == 0
        if update:
            begin = time.perf_counter()
            kind = self.spec["kind"]
            if self.replay is not None:
                raw = np.asarray(self.replay[k // self.period]["u_rad_s"], dtype=float)
            elif self.decision_function is not None:
                # Future policy adapter returns only seven preprojection rad/s.
                # It receives no witness index, sequence, or discovery result.
                raw=np.asarray(self.decision_function(observation),dtype=float)
            elif kind == "tracking":
                raw = np.zeros(7)
            elif kind == "apf":
                raw = self.apf.compute(q[:7], collision["obstacle"])["u"]
            elif kind == "envelope":
                # Fixed offline candidate; not tailored to any test failure.
                u = np.clip(t / self.spec["duration_s"], 0, 1)
                raw = np.asarray(self.spec["vector_rad_s"]) * np.sin(np.pi * u)**2
            else:
                raise ValueError(f"Unknown posture provider {kind}")
            if raw.shape != (7,) or not np.isfinite(raw).all():
                raise FloatingPointError("non_finite_posture_action")
            self.raw = raw.copy()
            self.u = np.clip(raw, -self.cap, self.cap)
            self.clipped = bool(np.any(np.abs(raw-self.u) > 1e-12))
            self.last_step = k
            elapsed = time.perf_counter() - begin
            self.times.append(elapsed)
            self.decisions.append({"physical_step": k, "time_s": t,
                                   "u_raw_rad_s": raw.tolist(), "u_rad_s": self.u.tolist(),
                                   "clipped": self.clipped, "decision_wall_s": elapsed})
        return self.u, update


def _csv(path, rows):
    if rows:
        with Path(path).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


class EpisodeStepper:
    """Shared motors/clock/safety for evaluator and Gym, version stage06-v1.

    A sample is checked after every physical step, including final/failed state.
    resetJointState is permitted only inside scene construction, never motion.
    """
    def __init__(self, scenario, provider=None, replay=None, decision_function=None,
                 synthetic_fault=None):
        from .scenarios import validate_scene
        validate_scene(scenario)
        self.scenario, self.ex = scenario, scenario["execution"]
        self.provider = provider or {"kind":"tracking"}
        self.replay, self.decision_function = replay, decision_function
        self.synthetic_fault = synthetic_fault
        self.scene = PandaScene(self.ex["scene_config"])
        self.tracker = PositionTracker(self.scene, self.ex["controller"])
        self.schedule = PostureSchedule(self.scene,self.ex,self.provider,replay,decision_function)
        self.dt = self.scene.config["time_step"]
        self.nsteps = round(self.ex["duration_s"]/self.dt)
        self.k=0; self.rows=[]; self.settle=[]; self.loops=[]
        self.failure=None; self.category=None; self.phase=None; self.failure_time=None
        self.truncated=False; self.previous_command=np.zeros(7); self.safety_checks=0
        try:
            q,v=state(self.scene); pose=self.scene.tool_pose(); self.collision=self.scene.detect()
            reason,category,flags=classify_sample(self.scene,q,v,pose,self.collision,self.ex["validation"])
            self.safety_checks+=1
            self.initial_record={"q":q,"velocity":v,"pose":pose,"collision":self.collision,"failure":reason,"flags":flags}
            if reason: raise RuntimeError("accepted_scene_initialization_failed:"+reason)
            with forbid_joint_resets():
                for k in range(self.ex["settle_steps"]):
                    self.collision=self.scene.step(); q,v=state(self.scene); pose=self.scene.tool_pose()
                    reason,category,flags=classify_sample(self.scene,q,v,pose,self.collision,self.ex["validation"])
                    self.safety_checks+=1
                    self.settle.append({"physical_step":k+1,"time_s":(k+1)*self.dt,"failure":reason or "",
                        **{f"q{i}":float(x) for i,x in enumerate(q)},**{f"velocity{i}":float(x) for i,x in enumerate(v)},
                        "external_distance_m":self.collision["obstacle"]["minimum_signed_distance"],
                        "self_distance_m":self.collision["self"]["minimum_signed_distance"],**flags})
                    if reason: raise RuntimeError("accepted_scene_settle_failed:"+reason)
            self.settled_q,self.settled_vel=state(self.scene)
            self.measured_start=np.asarray(self.scene.tool_pose()["position"])
            start=np.asarray(scenario["trajectory"]["start_world_m"])
            self.mismatch=float(np.linalg.norm(start-self.measured_start))
            if self.mismatch>self.ex["reference_start_tolerance_m"]:
                raise RuntimeError("accepted_scene_reference_start_mismatch")
            self.line=QuinticLine(start,scenario["trajectory"]["goal_world_m"],self.ex["duration_s"])
            self.inspect()
        except BaseException:
            self.close(); raise

    @property
    def done(self): return bool(self.failure or self.truncated or self.k>=self.nsteps)

    def inspect(self):
        self.q,self.vel=state(self.scene); self.pose=self.scene.tool_pose()
        self.actual=np.asarray(self.pose["position"]); self.t=self.k*self.dt
        self.pd,self.vd,_=self.line.sample(self.t)
        self.error=float(np.linalg.norm(self.pd-self.actual))
        if self.synthetic_fault and self.k==self.synthetic_fault["step"]:
            kind=self.synthetic_fault["kind"]
            if kind=="non_finite": self.q[0]=np.nan
            elif kind=="hard_position": self.q[0]=self.tracker.upper[0]+self.synthetic_fault.get('excess_rad',.01)
            elif kind=="tracking_error": self.error=.02
            elif kind=="external_contact": self.collision["obstacle"]["contact_or_penetration"]=True
            else: raise ValueError(kind)
        self.failure,self.category,self.flags=classify_sample(self.scene,self.q,self.vel,self.pose,self.collision,self.ex["validation"],self.error)
        self.safety_checks+=1
        if self.failure: self.phase="tracking"; self.failure_time=self.t

    def observation_context(self):
        return {"time_s":self.t,"time_fraction":self.t/self.line.duration,
            "q_rad_or_m":self.q.copy(),"actual_joint_velocity":self.vel.copy(),
            "tool_position_world_m":self.actual.copy(),"reference_position_world_m":self.pd.copy(),
            "reference_velocity_world_m_s":self.vd.copy(),"goal_world_m":self.line.goal.copy(),
            "start_world_m":self.line.start.copy(),"duration_s":self.line.duration,
            "previous_command_rad_s":self.previous_command.copy(),
            "sphere":copy.deepcopy(self.scenario["sphere"]),
            "obstacle_distance_query":copy.deepcopy(self.collision["obstacle"])}

    def record(self,control=None,updated=False):
        k,t,q,vel,actual,pd,vd,error=self.k,self.t,self.q,self.vel,self.actual,self.pd,self.vd,self.error
        collision,flags,failure,category,schedule=self.collision,self.flags,self.failure,self.category,self.schedule
        row = {"physical_step": k, "time_s": t, "posture_updated": int(updated),
               "held_from_step": schedule.last_step, "error_m": error,
               "command_applied": int(control is not None and not failure),
               "failure": failure or "", "failure_category": category or "",
               "external_distance_m": collision["obstacle"]["minimum_signed_distance"],
               "external_censored": int(collision["obstacle"]["minimum_signed_distance"] is None),
               "self_distance_m": collision["self"]["minimum_signed_distance"],
               "self_censored": int(collision["self"]["minimum_signed_distance"] is None),
               **{key:int(value) for key,value in flags.items()},
               "u_clipped": int(schedule.clipped),
               "speed_scale": control["scale"] if control else None,
               "soft_limited": int(control["soft_limited"]) if control else 0}
        for prefix, values in [("actual",actual),("ref",pd),("ref_v",vd)]:
            row.update({f"{prefix}_{axis}":float(x) for axis,x in zip("xyz",values)})
        for prefix, values in [("q",q),("velocity",vel),("u_raw",schedule.raw),("u",schedule.u)]:
            row.update({f"{prefix}{i}":float(x) for i,x in enumerate(values)})
        for prefix in ["track","secondary","raw","command"]:
            row.update({f"{prefix}{i}":float(control[prefix][i]) if control else None for i in range(7)})
        row["secondary_leakage_m_s"] = float(np.linalg.norm(control["secondary_leakage"])) if control else None
        row["min_singular_value"] = float(control["singular_values"][-1]) if control else None
        self.rows.append(row)
        return row

    def advance(self,u,updated=False,raw=None):
        if self.done: raise RuntimeError("Cannot step a completed episode")
        begin=time.perf_counter()
        u=np.asarray(u,dtype=float)
        if u.shape!=(7,) or not np.isfinite(u).all(): raise ValueError("seven finite posture velocities required")
        self.schedule.raw=np.asarray(raw if raw is not None else u).copy()
        self.schedule.u=np.clip(u,-self.ex["posture_cap_rad_s"],self.ex["posture_cap_rad_s"])
        self.schedule.clipped=bool(np.any(abs(self.schedule.raw-self.schedule.u)>1e-12))
        if updated: self.schedule.last_step=self.k
        try:
            control=self.tracker.compute(self.actual,self.pd,self.vd,self.q[:7],self.schedule.u,self.ex["posture_cap_rad_s"])
        except (FloatingPointError,ValueError,np.linalg.LinAlgError) as exc:
            self.failure=f"control_error:{type(exc).__name__}:{exc}"; self.category="non_finite" if isinstance(exc,FloatingPointError) else "runtime_exception"
            self.phase="tracking";self.failure_time=self.t;self.record();return None
        self.record(control,updated)
        before=self.previous_command.copy()
        with forbid_joint_resets():
            self.tracker.apply(control["command"]);self.collision=self.scene.step()
        self.previous_command=control["command"].copy();self.k+=1;self.inspect()
        self.loops.append(time.perf_counter()-begin)
        if self.done: self.record(); self.stop()
        return {"control":control,"previous_command":before,"dt":self.dt}

    def stop(self): self.tracker.apply(np.zeros(7))
    def close(self): self.scene.close()

    def summary(self,output=None):
        scenario,provider,ex,scene,tracker,schedule=self.scenario,self.provider,self.ex,self.scene,self.tracker,self.schedule
        rows,settle,loops,nsteps,dt=self.rows,self.settle,self.loops,self.nsteps,self.dt
        failure,category,phase,failure_time=self.failure,self.category,self.phase,self.failure_time
        measured_start,mismatch,settled_q,settled_vel=self.measured_start,self.mismatch,self.settled_q,self.settled_vel
        flags=self.flags;replay_actions=self.replay;synthetic_fault=self.synthetic_fault;decision_function=self.decision_function
        complete = bool(not failure and self.k==nsteps and not self.truncated)
        applied = [r for r in rows if r["command_applied"]]
        errors = np.array([r["error_m"] for r in rows])
        cmds = np.array([[r[f"command{i}"] for i in range(7)] for r in applied])
        qa = np.array([[r[f"q{i}"] for i in range(7)] for r in rows])
        margin = np.minimum(qa-tracker.lower,tracker.upper-qa) if rows else []
        ext = [r["external_distance_m"] for r in rows if r["external_distance_m"] is not None]
        sd = [r["self_distance_m"] for r in rows if r["self_distance_m"] is not None]
        mean = lambda x: float(np.mean(x)) if len(x) else None
        maximum = lambda x: float(np.max(x)) if len(x) else None
        summary = {"scene_id":scenario["id"],"split":scenario["split"],"provider":provider,
                   "success":complete,"complete_duration":complete,"failure_reason":failure,
                   "failure_category":category,"failure_phase":phase,"first_failure_time_s":failure_time,
                   "actual_duration_s":rows[-1]["time_s"] if rows else 0.0,
                   "requested_duration_s":ex["duration_s"],"tracking_physical_steps":max(0,len(rows)-1),
                   "logged_states":len(rows),"settle_steps_completed":len(settle),
                   "measured_start_world_m":measured_start,"reference_start_mismatch_m":mismatch,
                   "settled_q_arm_rad":settled_q[:7],"settled_velocity":settled_vel,
                   "max_error_m":maximum(errors),"rmse_m":float(np.sqrt(np.mean(errors**2))) if len(rows) else None,
                   "endpoint_error_m":float(errors[-1]) if complete else None,
                   "minimum_external_distance_m":min(ext) if ext else None,
                   "minimum_self_distance_m":min(sd) if sd else None,
                   "external_query_range_m":scene.config["collision"]["query_range"],
                   "external_censored_states":sum(r["external_censored"] for r in rows),
                   "self_censored_states":sum(r["self_censored"] for r in rows),
                   "posture_decisions":len(schedule.decisions),
                   "posture_clipped_fraction":mean([r["u_clipped"] for r in applied]),
                   "posture_clipped_decision_fraction":mean([d["clipped"] for d in schedule.decisions]),
                   "speed_saturated_fraction":mean([r["speed_scale"]<1-1e-12 for r in applied]),
                   "soft_limited_fraction":mean([r["soft_limited"] for r in applied]),
                   "max_actual_arm_velocity_rad_s":maximum([abs(r[f"velocity{i}"]) for r in rows for i in range(7)]),
                   "max_actual_finger_velocity_m_s":maximum([abs(r[f"velocity{i}"]) for r in rows for i in [7,8]]),
                   "min_arm_hard_limit_margin_rad":float(np.min(margin)) if rows else None,
                   "command_smoothness_rms_rad_s2":float(np.sqrt(np.mean(np.sum((np.diff(cmds,axis=0)/dt)**2,axis=1)))) if len(cmds)>1 else None,
                   "smoothness_definition":"RMS 7D norm of adjacent executed command difference/dt; tracking applied commands only; excludes mode switch and terminal stop; partial episodes are not ranked against complete episodes",
                   "decision_mean_s":mean(schedule.times),"decision_max_s":maximum(schedule.times),
                   "decision_timing_scope":"48 Hz provider only: APF point processing and point Jacobians included; cached 240 Hz collision-distance query excluded; main task and physics excluded",
                   "loop_mean_s":mean(loops),"loop_max_s":maximum(loops),
                   "loop_timing_scope":"state/reference/safety/provider/tracker/log-row construction/motor/physics/collision queries; excludes load, settle, rendering, CSV/JSON IO",
                   "joint_resets_during_motion":0,"reset_guard_active":True,
                   "safety_checked_states_including_initial_settle":1+len(settle)+len(rows),
                   "distance_and_collision_flags":{key:sum(int(r[key]) for r in rows) for key in flags},
                   "evidence_type":"48 Hz u replay with current-state 240 Hz recomputation" if replay_actions is not None else "48 Hz provider with 240 Hz main task",
                   "synthetic_fault":synthetic_fault,"excluded_from_performance":synthetic_fault is not None,
                   "external_callback_used":decision_function is not None,
                   "trace_csv":str((output/"states.csv").relative_to(ROOT)) if output else None,
                   "actions_json":str((output/"posture_actions.json").relative_to(ROOT)) if output else None}
        summary["safety_checked_states_including_initial_settle"]=self.safety_checks
        summary["truncated"]=self.truncated
        return summary


def run_scene(scenario,provider,output,replay_actions=None,render=False,synthetic_fault=None,decision_function=None):
    """Whole-episode evidence wrapper around the same stepper used by Gym."""
    begin=time.perf_counter();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/"summary.json").exists(): raise FileExistsError(output)
    write_json(output/"executed_scene_config.json",scenario["execution"]["scene_config"])
    write_json(output/"executed_scenario.json",scenario);write_json(output/"provider.json",provider)
    episode=EpisodeStepper(scenario,provider,replay_actions,decision_function,synthetic_fault)
    try:
        if render:episode.scene.render(output/"initial.png")
        while not episode.done:
            u,updated=episode.schedule.value(episode.k,episode.t,episode.q,episode.collision,episode.observation_context())
            episode.advance(u,updated,episode.schedule.raw)
        if render:episode.scene.render(output/"final.png")
        summary=episode.summary(output)
        write_json(output/"initial_state.json",episode.initial_record)
        _csv(output/"states.csv",episode.rows);_csv(output/"settle.csv",episode.settle)
        write_json(output/"posture_actions.json",episode.schedule.decisions)
    finally:episode.close()
    summary["connection_closed"]=not bool(p.isConnected(episode.scene.cid))
    summary["wall_s_including_setup_and_io"]=time.perf_counter()-begin
    write_json(output/"summary.json",summary);return summary


def load_parameters(manifest_path, split):
    """Parameters-only interface for future environments; no witness lookup."""
    if split not in ["train","validation","test"]:
        raise ValueError("Explicit split required")
    manifest=json.loads(Path(manifest_path).read_text())
    return [x for x in manifest["scenes"] if x["split"]==split]


def paired_summary(pairs):
    """Continuous comparisons only for BOTH complete, with population stated."""
    eligible=[x for x in pairs if x["tracking"]["success"] and x["apf"]["success"]]
    metrics=["max_error_m","rmse_m","endpoint_error_m","command_smoothness_rms_rad_s2"]
    return {"scope":"development/validation pipeline demonstration; not final test performance",
            "paired_count":len(pairs),"both_complete_count":len(eligible),
            "success_counts":{m:sum(x[m]["success"] for x in pairs) for m in ["tracking","apf"]},
            "conditional_continuous_means":{m:{key:float(np.mean([x[m][key] for x in eligible])) if eligible else None for key in metrics} for m in ["tracking","apf"]},
            "partial_metrics_not_used_for_continuous_ranking":True}
