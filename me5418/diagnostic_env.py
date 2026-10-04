"""Stage7 representation/sampling variants; same Stage6 reward and stepping."""
from pathlib import Path
import json,os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .arm_env import PandaPostureEnv,geometry_observation

LINK_ORDER=[f'panda_link{i}' for i in range(1,8)]+['panda_hand','panda_leftfinger','panda_rightfinger']


def link_features(query,scale=.1):
    """One minimum signed surface distance per named movable collision link.

    Uses the already computed 240Hz collision cache. Far encoding 3,0,0,0,0
    means censored beyond0.3m, not an exact measured0.3m distance.
    """
    nearest={}
    for point in query['closest_points']:
        name=point['link_name_a']
        if name in LINK_ORDER and (name not in nearest or point['signed_distance']<nearest[name]['signed_distance']):nearest[name]=point
    values=[]
    for name in LINK_ORDER:
        point=nearest.get(name)
        if point is None:values.extend([query['query_range']/scale,0.,0.,0.,0.]);continue
        normal=np.asarray(point['normal_on_b_towards_a_world'],dtype=float)
        if not np.isfinite(normal).all() or np.linalg.norm(normal)<1e-12:raise FloatingPointError('Invalid cached surface normal')
        normal=normal/np.linalg.norm(normal)
        values.extend([point['signed_distance']/scale,*normal,1.])
    return np.asarray(values,dtype=np.float32)


def encode_observation(context,config):
    base=geometry_observation(context,config['observation_scales'])
    if config['observation_version']=='geometry41-v1':return base
    if config['observation_version']!='geometry91-v1':raise ValueError('Unknown observation version')
    return np.concatenate([base,link_features(context['obstacle_distance_query'],config['link_feature_scale_m'])]).astype(np.float32)


class DiagnosticPandaEnv(PandaPostureEnv):
    def __init__(self,split='train',config_path=None,log_dir=None,trace=False):
        super().__init__(split,config_path,log_dir,trace)
        self.category_ids={cat:[i for i in self.ids if self.pool[i]['preassigned_geometric_category']==cat] for cat in ['far','near_link','tight_layout']}
        if self.config['observation_version']=='geometry91-v1':
            low=[-2.]*41;high=[2.]*41
            for _ in LINK_ORDER:low.extend([-3.,-1.,-1.,-1.,0.]);high.extend([3.,1.,1.,1.,1.])
            self.observation_space=spaces.Box(np.asarray(low,np.float32),np.asarray(high,np.float32),dtype=np.float32)

    def _obs(self):
        obs=encode_observation(self.episode.observation_context(),self.config)
        sanitized=not np.isfinite(obs).all() or not self.observation_space.contains(obs)
        if sanitized:
            if not self.episode.failure:raise ValueError('Invalid normal-state observation')
            obs=self.last_obs.copy() if not np.isfinite(obs).all() else np.clip(obs,self.observation_space.low,self.observation_space.high).astype(np.float32)
        self.terminal_observation_sanitized=sanitized;self.last_obs=obs
        return obs.copy()

    def reset(self,*,seed=None,options=None):
        options=dict(options or {});entry_seed=seed
        sampling=self.config['sampling']
        if self.split=='train' and sampling['kind']=='category_weighted' and not options.get('scene_id'):
            gym.Env.reset(self,seed=seed)
            cats=list(sampling['weights']);prob=np.asarray(list(sampling['weights'].values()))
            cat=str(self.np_random.choice(cats,p=prob));options['scene_id']=self.category_ids[cat][int(self.np_random.integers(len(self.category_ids[cat])))]
            seed=None # Do not reseed after selection; preserve continuously advancing RNG.
        obs,info=super().reset(seed=seed,options=options)
        if self.log_dir:
            with (self.log_dir/'sampling_design.jsonl').open('a') as f:f.write(json.dumps({'scene_id':info['scene_id'],'category':self.pool[info['scene_id']]['preassigned_geometric_category'],'split':self.split,'entrypoint_seed':entry_seed,'sampling':sampling,'rng_state_after_selection':self.np_random.bit_generator.state,'process_id':os.getpid()})+'\n')
        return obs,info
