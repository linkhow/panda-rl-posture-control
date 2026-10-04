"""Stage8 geometry-only sampling; physics, observations and reward inherited.

The draw algorithm is exactly the existing Stage7 C algorithm, exposed as a
pure function so its distribution can be checked without simulating episodes.
There is no adaptive weighting, answer-label input, curriculum or global RNG.
"""
import copy
import json
import os
import numpy as np
import gymnasium as gym
from .diagnostic_env import DiagnosticPandaEnv
from .tracking import write_json

CATEGORIES = ('far', 'near_link', 'tight_layout')


def validate_pools(pool, category_ids, weights):
    if set(weights) != set(CATEGORIES):
        raise ValueError('Exactly the three preassigned geometry categories are required')
    probabilities = np.asarray(list(weights.values()), dtype=float)
    if not np.isfinite(probabilities).all() or np.any(probabilities <= 0) or abs(probabilities.sum()-1) > 1e-12:
        raise ValueError('Positive category probabilities must sum to one')
    flat = [i for c in CATEGORIES for i in category_ids[c]]
    if len(flat) != len(set(flat)) or set(flat) != set(pool):
        raise ValueError('Category pools duplicate or omit allowed IDs')
    for category in CATEGORIES:
        if not category_ids[category]:
            raise ValueError('Empty category pool')
        for identifier in category_ids[category]:
            s = pool[identifier]
            if s['split'] != 'train' or s['preassigned_geometric_category'] != category:
                raise ValueError('Pool must contain train parameters with matching geometry category')
    return {'counts': {c: len(category_ids[c]) for c in CATEGORIES},
            'total': len(flat), 'probability_sum': float(probabilities.sum()),
            'weights': dict(weights), 'category_pool_ids': copy.deepcopy(category_ids),
            'per_id_probability': {i: weights[c]/len(category_ids[c]) for c in CATEGORIES for i in category_ids[c]},
            'all_train': True, 'unique_and_complete': True,
            'replacement': True, 'selection_inputs': 'preassigned geometric category and ID only'}


def draw_category_id(rng, category_ids, weights):
    """Independent draws with replacement, using the caller's private RNG."""
    cats = list(weights)
    cat = str(rng.choice(cats, p=np.asarray(list(weights.values()), dtype=float)))
    return category_ids[cat][int(rng.integers(len(category_ids[cat])))], cat


class BalancedPandaEnv(DiagnosticPandaEnv):
    """Same 91D task as B; change only train reset selection and audit logging."""
    def __init__(self, *args, access_audit=None, **kwargs):
        self.access_audit = access_audit
        super().__init__(*args, **kwargs)
        if self.split == 'train':
            if self.config['sampling']['kind'] != 'category_weighted':
                raise ValueError('Stage8 train environment requires category_weighted')
            self.pool_contract = validate_pools(self.pool, self.category_ids, self.config['sampling']['weights'])
            if self.log_dir:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                write_json(self.log_dir/'pool_contract.json', self.pool_contract)

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        gym.Env.reset(self, seed=seed)
        before = copy.deepcopy(self.np_random.bit_generator.state)
        weighted = self.split == 'train' and not options.get('scene_id')
        if weighted:
            identifier, category = draw_category_id(self.np_random, self.category_ids, self.config['sampling']['weights'])
            options['scene_id'] = identifier
        # Fixed ID bypasses the legacy weighted selector. Do not reseed after a
        # weighted draw: worker RNG advances continuously across all episodes.
        obs, info = super().reset(seed=None if weighted else seed, options=options)
        if self.log_dir:
            record = {'reset_index': self.reset_count, 'scene_id': info['scene_id'],
                      'category': self.pool[info['scene_id']]['preassigned_geometric_category'],
                      'split': self.split, 'entrypoint_seed': seed, 'weighted_draw': weighted,
                      'rng_state_before_selection': before,
                      'rng_state_after_selection': copy.deepcopy(self.np_random.bit_generator.state),
                      'process_id': os.getpid()}
            with (self.log_dir/'sampling_audit.jsonl').open('a') as f:
                f.write(json.dumps(record, allow_nan=False)+'\n')
            if self.access_audit:
                write_json(self.log_dir/'access_guard.json', self.access_audit.record())
        return obs, info

    def close(self):
        super().close()
        if getattr(self, 'log_dir', None) and self.access_audit:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            write_json(self.log_dir/'access_guard.json', self.access_audit.record())
