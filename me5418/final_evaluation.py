"""Stage10 logging-only evaluator; frozen numeric core remains unchanged.
Constructor copied from frozen EpisodeStepper to timestamp load/settle explicitly.
Normal motion delegates to inherited advance and real motors. No witness inputs.
"""
from pathlib import Path
import csv,copy,json,time,hashlib,sys,os
import numpy as np
import pybullet as p
from .scene import PandaScene,ROOT
from .evaluation import EpisodeStepper,PostureSchedule,classify_sample,_csv
from .tracking import PositionTracker,state,forbid_joint_resets,write_json
from .trajectory import QuinticLine
from .diagnostic_env import encode_observation
from .arm_env import PARAMETER_KEYS

class LoggedEpisode(EpisodeStepper):
    def __init__(self, scenario, provider=None, replay=None, decision_function=None,
                 synthetic_fault=None):
        setup_begin=time.perf_counter()
        from .scenarios import validate_scene
        validate_scene(scenario)
        self.scenario, self.ex = scenario, scenario["execution"]
        self.provider = provider or {"kind":"tracking"}
        self.replay, self.decision_function = replay, decision_function
        self.synthetic_fault = synthetic_fault
        load_begin=time.perf_counter()
        self.scene = PandaScene(self.ex["scene_config"])
        self.scene_load_wall_s=time.perf_counter()-load_begin
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
            settle_begin=time.perf_counter()
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
            self.settle_wall_s=time.perf_counter()-settle_begin
            self.settled_q,self.settled_vel=state(self.scene)
            self.measured_start=np.asarray(self.scene.tool_pose()["position"])
            start=np.asarray(scenario["trajectory"]["start_world_m"])
            self.mismatch=float(np.linalg.norm(start-self.measured_start))
            if self.mismatch>self.ex["reference_start_tolerance_m"]:
                raise RuntimeError("accepted_scene_reference_start_mismatch")
            self.line=QuinticLine(start,scenario["trajectory"]["goal_world_m"],self.ex["duration_s"])
            self.inspect()
            self.setup_wall_s=time.perf_counter()-setup_begin
        except BaseException:
            self.close(); raise

    def record(self,control=None,updated=False):
        row=super().record(control,updated)
        joints=[self.scene.joints[i] for i in self.scene.arm+self.scene.fingers]
        tol=self.ex['validation']['hard_limit_tolerance'];vtol=self.ex['validation']['actual_speed_urdf_tolerance_rad_s']
        row['hard_position_violation']=int(any(q<j['lower_limit']-tol or q>j['upper_limit']+tol for q,j in zip(self.q,joints)))
        row['actual_speed_violation']=int(any(abs(v)>j['urdf_max_velocity']+vtol for v,j in zip(self.vel,joints)))
        row['tracking_error_violation']=int(self.error>self.ex['validation']['max_tracking_error_m'])
        row['non_finite_state']=int(not np.isfinite([*self.q,*self.vel,*self.actual,self.error]).all())
        for kind,prefix in [('obstacle','obstacle'),('self','self')]:
            pts=self.collision[kind]['closest_points']+self.collision[kind]['contact_points']
            nearest=min(pts,key=lambda x:x['signed_distance']) if pts else None
            row[prefix+'_nearest_link_a']=nearest['link_name_a'] if nearest else ''
            row[prefix+'_nearest_link_b']=nearest['link_name_b'] if nearest else ''
        return row

class EvaluationAccessAudit:
    """Test permission only in new evaluation commands; legacy train guard unchanged."""
    def __init__(self,allow_test=False):
        self.allow_test=allow_test;self.paths=set();self.denied=[]
        def hook(event,args):
            if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
            name=str(Path(os.fsdecode(args[0])).resolve())
            if 'datasets/me5418-scenes-v1' not in name:return
            if 'private_witnesses' in name or ('/splits/test.json' in name and not self.allow_test):
                self.denied.append(name);raise PermissionError('Stage10 evaluation data boundary: '+name)
            self.paths.add(name)
        sys.addaudithook(hook)
    def record(self):return {'allow_test_parameters_for_evaluation_only':self.allow_test,'opened_dataset_paths':sorted(self.paths),'denied_probe_or_reads':self.denied,'private_witness_reads':0,'manifest_scope':'Full public manifest parsed including acceptance metadata; only PARAMETER_KEYS copied to selected split; no witness file and no metadata/ID feature to policy'}

def load_evaluation_pool(split,audit):
    if split not in ('validation','test') or (split=='test' and not audit.allow_test):raise PermissionError('Explicit evaluator test permission required')
    dataset=ROOT/'datasets/me5418-scenes-v1';ids=json.loads((dataset/f'splits/{split}.json').read_text())['scene_ids']
    manifest=json.loads((dataset/'manifest.json').read_text());selected={x['id']:{k:copy.deepcopy(x[k]) for k in PARAMETER_KEYS} for x in manifest['scenes'] if x['split']==split}
    if len(ids)!=len(set(ids)) or set(ids)!=set(selected):raise ValueError('Split/order/manifest mismatch')
    return {i:selected[i] for i in ids}

class FrozenPolicyAdapter:
    """One deterministic SB3 _predict; same float32 clip then float64 .2 map.

    Installed ActorCriticPolicy.predict implementation is archived in checks.
    predict equivalence check is validation-only and off in all test timings.
    """
    def __init__(self,model,config,check_predictions=False):
        self.model,self.config,self.check_predictions=model,config,check_predictions
        self.last=None;self.prediction_differences=[]
        assert not model.policy.squash_output and model.action_space.shape==(7,)
    def __call__(self,context):
        import torch
        obs=encode_observation(context,self.config)
        if obs.shape!=(91,) or not np.isfinite(obs).all():raise ValueError('Frozen91 observation required')
        self.model.policy.set_training_mode(False)
        tensor,_=self.model.policy.obs_to_tensor(obs)
        with torch.no_grad():raw=self.model.policy._predict(tensor,deterministic=True).cpu().numpy().reshape(7)
        legal=np.clip(raw,self.model.action_space.low,self.model.action_space.high)
        if self.check_predictions:
            expected=self.model.predict(obs,deterministic=True)[0]
            difference=float(np.max(abs(expected-legal)));self.prediction_differences.append(difference)
            if difference!=0:raise RuntimeError('Adapter differs from sealed model.predict')
        self.last={'raw':raw.copy(),'legal':legal.copy()}
        return .2*legal.astype(float)

def stats(values):
    if not values:return {'n':0,'mean_s':None,'p95_s':None,'p99_s':None,'max_s':None,'sum_s':0.}
    a=np.asarray(values);return {'n':len(a),'mean_s':float(a.mean()),'p95_s':float(np.percentile(a,95)),'p99_s':float(np.percentile(a,99)),'max_s':float(a.max()),'sum_s':float(a.sum())}

def logged_run(scenario,method,output,model=None,config=None,check_predictions=False):
    """One real physical episode. Only export at end, no rendering in timings."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False);begin=time.perf_counter()
    adapter=FrozenPolicyAdapter(model,config,check_predictions) if model is not None else None
    provider={'kind':method['kind']}
    e=LoggedEpisode(scenario,provider,decision_function=adapter);timings=[];actions=[]
    try:
        while not e.done:
            k,t=e.k,e.t;outer_begin=time.perf_counter()
            context=e.observation_context()
            u,updated=e.schedule.value(k,t,e.q,e.collision,context)
            tr=e.advance(u,updated,e.schedule.raw)
            outer_elapsed=time.perf_counter()-outer_begin
            core=e.loops[-1] if tr else None
            decision=e.schedule.times[-1] if updated else None
            timings.append({'physical_step':k,'time_s':t,'provider_updated':int(updated),'provider_wall_s':decision,'core_advance_wall_s':core,'outer_physical_loop_wall_s':outer_elapsed})
            if updated:
                raw_u=e.schedule.raw;actual_u=e.schedule.u
                row={'physical_step':k,'time_s':t,'decision_wall_s':decision,'u_component_clip_fraction':float(np.mean(abs(raw_u-actual_u)>1e-12)),'u_any_clip':int(np.any(abs(raw_u-actual_u)>1e-12)),'raw_policy_component_clip_fraction':None,'raw_policy_any_clip':None}
                if adapter:
                    raw,legal=adapter.last['raw'],adapter.last['legal'];row['raw_policy_component_clip_fraction']=float(np.mean(raw!=legal));row['raw_policy_any_clip']=int(np.any(raw!=legal))
                for j in range(7):
                    row[f'raw_policy{j}']=float(raw[j]) if adapter else None;row[f'legal_policy{j}']=float(legal[j]) if adapter else None
                    row[f'raw_u{j}']=float(raw_u[j]);row[f'u{j}']=float(actual_u[j]);row[f'first_executed_command{j}']=float(tr['control']['command'][j]) if tr else None
                actions.append(row)
        summary=e.summary(output)
        summary['loop_timing_scope']='Inherited advance timer only; excludes outer observation/provider; use outer_physical_loop timing'
        summary['timing']={'provider_48Hz':stats(e.schedule.times),'core_physical_step':stats(e.loops),'outer_physical_loop':stats([x['outer_physical_loop_wall_s'] for x in timings]),'scene_load_wall_s':e.scene_load_wall_s,'settle_wall_s':e.settle_wall_s,'setup_including_settle_wall_s':e.setup_wall_s,'motion_wall_s':time.perf_counter()-begin-e.setup_wall_s,'definitions':{'provider':'PostureSchedule.value timer around actual raw provider/validation/clip; PPO includes encoding, one deterministic inference, mapping; APF includes current cached point processing/Jacobians; excludes context construction/core','core':'Frozen advance timer around tracker, extended log row, motors, physical step, queries and inspect','outer':'One 1/240s physical loop: begin BEFORE observation_context and schedule.value, end AFTER advance. Includes context/provider when updated/main control/log row/motor/physics/collision/safety. Excludes setup/settle/render/export and packaging of timing/action CSV rows after timer. Not a five-step Gym decision','setup':'Constructor total: scene load, initial check, 240-step settle including state/query/safety/log rows, start/line checks','real_time_claim':False}}
        summary['raw_policy_component_clip_fraction']=float(np.mean([x['raw_policy_component_clip_fraction'] for x in actions])) if adapter else None
        summary['raw_policy_decisions_with_clip_fraction']=float(np.mean([x['raw_policy_any_clip'] for x in actions])) if adapter else None
        summary['u_component_clip_fraction']=float(np.mean([x['u_component_clip_fraction'] for x in actions]))
        summary['u_decisions_with_clip_fraction']=float(np.mean([x['u_any_clip'] for x in actions]))
        summary['raw_policy_mean_abs']=float(np.mean([abs(x[f'raw_policy{j}']) for x in actions for j in range(7)])) if adapter else None
        summary['legal_policy_mean_abs']=float(np.mean([abs(x[f'legal_policy{j}']) for x in actions for j in range(7)])) if adapter else None
        summary['mean_u_abs_rad_s']=float(np.mean([abs(x[f'u{j}']) for x in actions for j in range(7)]))
        summary['maximum_prediction_check_difference']=max(adapter.prediction_differences,default=0.) if adapter and check_predictions else None
        summary['state_violation_counts']={k:sum(r[k] for r in e.rows) for k in ['hard_position_violation','actual_speed_violation','tracking_error_violation','non_finite_state']}
        summary['final_nearest_links']={k:e.rows[-1][k] for k in ['obstacle_nearest_link_a','obstacle_nearest_link_b','self_nearest_link_a','self_nearest_link_b']}
        write_json(output/'initial_state.json',e.initial_record);write_json(output/'terminal_collision.json',e.collision)
        write_json(output/'executed_scenario.json',scenario);write_json(output/'method.json',method)
        _csv(output/'states.csv',e.rows);_csv(output/'settle.csv',e.settle);_csv(output/'physical_timing.csv',timings);_csv(output/'actions48.csv',actions)
        write_json(output/'posture_actions.json',e.schedule.decisions)
    finally:e.close()
    summary['connection_closed']=not bool(p.isConnected(e.scene.cid));summary['wall_s_including_setup_and_io']=time.perf_counter()-begin
    write_json(output/'summary.json',summary)
    return summary
