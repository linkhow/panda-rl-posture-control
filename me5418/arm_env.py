"""Gymnasium 48 Hz posture interface to the shared 240 Hz EpisodeStepper.

Only selected train/validation parameters enter this module. No witness loader.
Manual normalization and bounded-v1 rewards are fixed before pilot learning.
"""
from pathlib import Path
import copy,json,time,os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .scene import ROOT
from .evaluation import EpisodeStepper,_csv
from .tracking import write_json
from .trajectory import QuinticLine

CONFIG=ROOT/'configs/stage06_ppo.json'
PARAMETER_KEYS=['id','dataset_version','schema_version','split','template_group','generator_version','seed','candidate_index','initial_arm_q_rad','finger_positions_m','sphere','trajectory','execution','preassigned_geometric_category']


def load_pool(split):
    if split not in ('train','validation'):raise ValueError('Stage 6 permits train/validation only')
    manifest=json.loads((ROOT/'datasets/me5418-scenes-v1/manifest.json').read_text())
    ids=json.loads((ROOT/f'datasets/me5418-scenes-v1/splits/{split}.json').read_text())['scene_ids']
    # Metadata is stripped here; IDs/group/split are bookkeeping, never features.
    pool={x['id']:{k:copy.deepcopy(x[k]) for k in PARAMETER_KEYS} for x in manifest['scenes'] if x['split']==split}
    if set(ids)!=set(pool):raise ValueError('Split/manifest mismatch')
    return pool


def json_safe(value):
    if isinstance(value,np.ndarray):return json_safe(value.tolist())
    if isinstance(value,np.generic):return json_safe(value.item())
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def geometry_observation(context,scales):
    q=context['q_rad_or_m'][:7];v=context['actual_joint_velocity'][:7]
    actual=context['tool_position_world_m'];t=context['time_s']
    line=QuinticLine(context['start_world_m'],context['goal_world_m'],context['duration_s'])
    values=[q/scales['q_rad'],v/scales['velocity_rad_s'],
        (context['reference_position_world_m']-actual)/scales['error_m'],
        context['reference_velocity_world_m_s']/scales['reference_velocity_m_s'],[context['time_fraction']]]
    values += [(line.sample(t+future)[0]-actual)/scales['future_relative_m'] for future in (.2,.5,1.)]
    values += [(np.asarray(context['sphere']['position_world_m'])-actual)/scales['sphere_relative_m'],
        [context['sphere']['radius_m']/scales['radius_m']],context['previous_command_rad_s']/scales['previous_command_rad_s']]
    obs=np.concatenate(values).astype(np.float32)
    # Synthetic nonfinite termination must still return a finite terminal obs;
    # physical invalid state remains in failure diagnostics, never reaches motors.
    return obs


OBS_LAYOUT=[('arm_q',0,7,'rad','q_rad'),('arm_actual_velocity',7,14,'rad/s','velocity_rad_s'),('position_error',14,17,'m','error_m'),('reference_velocity',17,20,'m/s','reference_velocity_m_s'),('time_fraction',20,21,'1',None),('future_relative_0.2',21,24,'m','future_relative_m'),('future_relative_0.5',24,27,'m','future_relative_m'),('future_relative_1.0',27,30,'m','future_relative_m'),('sphere_relative',30,33,'m','sphere_relative_m'),('sphere_radius',33,34,'m','radius_m'),('previous_executed_command',34,41,'rad/s','previous_command_rad_s')]


def dense_reward(episode,transition,u,config):
    r=config['reward'];dt=transition['dt'];c=transition['control']
    d=episode.collision['obstacle']['minimum_signed_distance']
    proximity=0. if d is None else np.clip(1-max(d,0)/r['proximity_range_m'],0,1)**2
    error=np.clip(episode.error/episode.ex['validation']['max_tracking_error_m'],0,1)**2
    if not np.isfinite(error):error=1.
    jump=np.clip((c['command']-transition['previous_command'])/r['command_jump_scale_rad_s'],-1,1)
    return {'progress':r['progress_per_s']*dt,'error':-r['error_per_s']*error*dt,
        'proximity':-r['proximity_per_s']*proximity*dt,
        'action':-r['action_per_s']*float(np.mean((u/.2)**2))*dt,
        'smoothness':-r['smoothness_per_s']*float(np.mean(jump**2))*dt,'terminal':0.}


class PandaPostureEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self,split='train',config_path=CONFIG,log_dir=None,trace=False):
        self.config=json.loads(Path(config_path).read_text());self.split=split
        self.pool=load_pool(split);self.ids=list(self.pool);self.log_dir=Path(log_dir) if log_dir else None
        self.trace=trace;self.episode=None;self.reset_count=0;self.episode_returns=[];self.reward_rows=[]
        self.action_space=spaces.Box(-1.,1.,(7,),np.float32)
        self.observation_space=spaces.Box(-2.,2.,(41,),np.float32)
        self.last_obs=None;self.last_summary=None;self.closed_ids=[]
        self.external_limit=None;self.decision_count=0;self.return_total=0.;self.discounted_return=0.;self.terms={};self.action_abs=[];self.clips=0

    def _obs(self):
        obs=geometry_observation(self.episode.observation_context(),self.config['observation_scales'])
        if not np.isfinite(obs).all():
            if not self.episode.failure:raise FloatingPointError('nonfinite observation')
            obs=self.last_obs.copy()
        self.terminal_observation_sanitized=not self.observation_space.contains(obs) or not np.isfinite(geometry_observation(self.episode.observation_context(),self.config['observation_scales'])).all()
        if not self.observation_space.contains(obs):
            if not self.episode.failure:raise ValueError('Observation outside declared bounds')
            obs=np.clip(obs,self.observation_space.low,self.observation_space.high).astype(np.float32)
        self.last_obs=obs;return obs.copy()

    def _info(self):
        e=self.episode
        return {'scene_id':e.scenario['id'],'split':self.split,'sampling_seed':self.sampling_seed,
            'time_s':e.t,'time_fraction':e.t/e.line.duration,'physical_steps':e.k,
            'failure_reason':e.failure,'failure_category':e.category,'success':bool(e.k==e.nsteps and not e.failure and not e.truncated)}

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);self.close();options=options or {}
        if set(options)-{'scene_id','synthetic_fault','external_limit_steps'}:raise ValueError('Unknown reset option')
        scene_id=options.get('scene_id') or self.ids[int(self.np_random.integers(len(self.ids)))]
        if scene_id not in self.pool:raise ValueError('scene not in enforced split')
        self.sampling_seed=int(seed) if seed is not None else None
        self.reset_count+=1;self.external_limit=options.get('external_limit_steps')
        begin=time.perf_counter()
        try:self.episode=EpisodeStepper(self.pool[scene_id],{'kind':'gym_action'},synthetic_fault=options.get('synthetic_fault'))
        except BaseException as exc:
            if self.log_dir:write_json(self.log_dir/f'reset_failure_{self.reset_count}.json',{'scene_id':scene_id,'exception':repr(exc)})
            raise
        self.last_obs=None;self.last_summary=None;self.reward_rows=[]
        self.decision_count=0;self.return_total=0.;self.discounted_return=0.;self.terms={k:0. for k in ['progress','error','proximity','action','smoothness','terminal']};self.action_abs=[];self.clips=0
        if self.log_dir:
            self.log_dir.mkdir(parents=True,exist_ok=True)
            with (self.log_dir/'samples.jsonl').open('a') as f:f.write(json.dumps({'scene_id':scene_id,'split':self.split,'seed':self.sampling_seed,'reset_index':self.reset_count,'process_id':os.getpid(),'physics_client_id':self.episode.scene.cid,'rng_state':self.np_random.bit_generator.state,'reset_wall_s':time.perf_counter()-begin})+'\n')
        return self._obs(),self._info()

    def step(self,action):
        e=self.episode
        if e is None or e.done:raise RuntimeError('reset required before step')
        action=np.asarray(action,dtype=float)
        if action.shape!=(7,) or not np.isfinite(action).all():raise ValueError('Action must be seven finite numbers')
        clipped=np.clip(action,-1,1);self.clips+=int(np.any(action!=clipped));u=.2*clipped
        e.schedule.last_step=e.k;e.schedule.decisions.append({'physical_step':e.k,'time_s':e.t,'u_raw_rad_s':(.2*action).tolist(),'u_rad_s':u.tolist(),'clipped':bool(np.any(action!=clipped)),'decision_wall_s':0.})
        sums={k:0. for k in self.terms};performed=0
        for sub in range(5):
            tr=e.advance(u,updated=(sub==0),raw=.2*action)
            if tr:
                performed+=1
                for k,v in dense_reward(e,tr,u,self.config).items():sums[k]+=v
            if self.external_limit is not None and e.k>=self.external_limit and not e.done:
                e.truncated=True;e.record();e.stop()
            if e.done:break
        terminated=bool(e.failure or e.k>=e.nsteps);truncated=bool(e.truncated)
        if terminated:sums['terminal']=self.config['reward']['failure'] if e.failure else self.config['reward']['success']
        reward=float(sum(sums.values()));self.return_total+=reward
        self.discounted_return+=self.config['ppo']['gamma']**self.decision_count*reward
        self.decision_count+=1;self.action_abs.append(float(np.mean(abs(clipped))))
        for k,v in sums.items():self.terms[k]+=v
        info=self._info();info.update({'reward_terms':sums,'executed_substeps':performed,'action_clipped':bool(np.any(action!=clipped))})
        self.reward_rows.append({'policy_step':self.decision_count,'time_s':e.t,'reward':reward,**sums,'terminated':terminated,'truncated':truncated,'performed':performed})
        obs=self._obs();info['terminal_observation_sanitized']=self.terminal_observation_sanitized
        if terminated or truncated:
            summary=e.summary();summary.update({'policy_steps':self.decision_count,'return':self.return_total,'discounted_return':self.discounted_return,'reward_terms':self.terms.copy(),'mean_abs_action':float(np.mean(self.action_abs)),'action_clip_fraction':self.clips/self.decision_count,'environment_version':self.config['environment_version'],'observation_version':self.config['observation_version'],'reward_version':self.config['reward_version']})
            e.close();self.closed_ids.append(e.scene.cid);summary['connection_closed']=True
            self.last_summary=summary;info['episode_summary']=copy.deepcopy(summary);self.episode_returns.append(summary)
            if self.log_dir:
                with (self.log_dir/'episodes.jsonl').open('a') as f:f.write(json.dumps(json_safe(summary),allow_nan=False)+'\n')
        return obs,reward,terminated,truncated,info

    def close(self):
        if self.episode is not None:self.episode.close();self.episode=None

    def save_trace(self,path):
        if not self.trace:raise ValueError('trace flag required')
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        _csv(path/'states.csv',self.episode.rows);_csv(path/'rewards.csv',self.reward_rows)
        write_json(path/'summary.json',self.last_summary)
        write_json(path/'actions.json',self.episode.schedule.decisions)
