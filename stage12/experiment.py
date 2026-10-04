#!/usr/bin/env python3
"""Stage12: predeclared validation APF search and supplemental held-out experiment.

Frozen Stage09/10 numerical modules are imported, hash-checked and never edited.
Witness search is a separate offline command; no policy execution reads witnesses.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'stage12'
RAW = ROOT / 'outputs/stage12'
sys.path.insert(0, str(ROOT / 'tools'))
from common import bootstrap, verify_release, read, sha, utc, write_json, append_json, methods, load_model, module_paths


def checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def safe_phase(name):
    audit = bootstrap(False)
    verification = verify_release()
    protocol = read(STAGE / 'configs/protocol.json')
    if sha(Path(__file__)) != protocol['executed_source_SHA256']['stage12/experiment.py']:
        raise RuntimeError('Stage12 source changed after plan; create a separately versioned protocol')
    if sha(STAGE / 'data/reserved_geometry.json') != protocol['reserved_geometry_SHA256']:
        raise RuntimeError('Reserved geometry changed after plan')
    RAW.mkdir(parents=True, exist_ok=True)
    append_json(RAW / 'phase_audit.jsonl', {'UTC': utc(), 'phase': name, 'protocol_SHA256': sha(STAGE / 'configs/protocol.json'), 'frozen_verification': verification})
    return protocol, audit, verification


def plan():
    """Write the finite search, selection, budget and new geometry BEFORE physics."""
    if (STAGE / 'configs/protocol.json').exists():
        raise FileExistsError('A saved protocol cannot be overwritten')
    audit = bootstrap(False); verification = verify_release()
    base = read(ROOT / 'configs/stage04_posture_apf.json')['apf']
    grid = []
    for d in (.06, .10, .14):
        for strength in (2., 4., 8.):
            grid.append({'id': f'APF_grid_{len(grid):02d}', 'parameters': {**base, 'influence_distance_m': d, 'repulsion_max_rad2_per_m_s': strength}})
    pool_ids = read(ROOT / 'datasets/me5418-scenes-v1/splits/validation.json')['scene_ids']
    protocol = {
        'version': 'stage12-apf-heldout-v1', 'created_UTC': utc(),
        'purpose': 'Supplemental generalization within the already known fixed-q0 single-sphere short-line task family; development-informed, not independent untouched research.',
        'original_experiment': 'Stage09 frozen models and Stage10 opened test131 remain unchanged; no old test episode can enter APF selection.',
        'task_contract': {'physical_Hz': 240, 'posture_Hz': 48, 'u_cap_rad_s': .2, 'duration_s': 4., 'length_m': [.03, .08], 'fixed_q0': True, 'single_sphere_radius_m': .06, 'tracking_success_threshold_m': .01, 'same_tracker_projection_motors_collision_limits_as_original': True},
        'validation': {'split': 'validation', 'ordered_ids': pool_ids, 'n': 58, 'finite_grid': grid,
            'selection_rule': 'Descending complete-success count, then ascending mean RMS tracking error among complete successes, then ascending predeclared grid index. Failure prefixes are retained but excluded from the tie-break metric. No test scores or PPO scores used.',
            'maximum_episodes': 9 * 58, 'wall_budget_s': 1800, 'workers': 1, 'task_failure_retries': 0, 'infrastructure_failure_rule': 'Preserve evidence and abort; do not rank an incomplete grid.'},
        'heldout': {'dataset_version': 'me5418-supplement-v1', 'seed': 2026100412, 'template_groups': 24, 'variants_per_group': 12, 'candidate_pool': 288, 'accepted_target': 100,
            'acceptance_order': 'Candidates are generated once in variant-major/group-minor order before any witness search. Accept the first100 whose predeclared ordered provider finds a complete actual-motor trajectory and whose saved clipped48Hz action replay passes. Stop witness search at100; keep all remaining candidates as not_run_target_met.',
            'no_tuned_APF_or_PPO_in_witness_search': True, 'search_order': ['original_fixed_APF', 'predeclared_gentle_APF', 'predeclared_early_APF'],
            'witness_providers': [{'name': 'original_fixed_APF', 'kind': 'apf', 'apf': base}, {'name': 'predeclared_gentle_APF', 'kind': 'apf', 'apf': {**base, 'influence_distance_m': .10, 'repulsion_max_rad2_per_m_s': 2.}}, {'name': 'predeclared_early_APF', 'kind': 'apf', 'apf': {**base, 'influence_distance_m': .15, 'repulsion_max_rad2_per_m_s': 4.}}],
            'maximum_provider_attempts_per_candidate': 3, 'maximum_attempt_episodes': 864, 'maximum_replay_episodes': 288, 'wall_budget_s': 1800,
            'replay_tolerance_m_or_rad': 1e-7, 'witness_bias': 'Finite sequential APF-family witness search can favour APF-family-solvable scenes; no witness does not prove infeasibility. Witness data never enter policy input.',
            'category_cycle': ['far','near_link','tight_layout','near_link','tight_layout'],
            'sampling': {'direction_cone_noise_sd': .25, 'surface_cone_noise_sd': .3, 'length_range_m': [.03,.08], 'gap_ranges_m': {'near_link': [.012,.05], 'tight_layout': [-.006,.008]}, 'tight_surface_frame': 'linearized_goal_configuration', 'target_link_indices': [3,4], 'far_cube_m': [1.5,2.5]},
            'geometry': {'minimum_initial_gap_m': .002, 'minimum_base_gap_m': .01, 'minimum_tool_segment_gap_m': .03},
            'deduplication': {'q_rad': .01, 'start_m': .005, 'goal_m': .005, 'sphere_m': .005, 'radius_m': .001, 'duration_s': .05},
            'reserved_scope': 'All696 original formal candidates (includes accepted340), both20-candidate pilot pools, and17 development candidates; exact geometric keys and componentwise near-duplicate tolerances checked, including rejected candidates.'},
        'evaluation': {'methods': ['tracking','APF_fixed','APF_tuned','PPO_best_550901','PPO_best_551901','PPO_best_552901'], 'maximum_episodes': 600, 'wall_budget_s': 1800, 'workers': 1, 'task_failure_retries': 0, 'training_interactions': 0,
            'primary': 'Complete4s success per scene, per seed; paired gains/losses. Same100 scenarios and initial motors/physics/success rules for all six methods.',
            'continuous': 'Pairwise common complete-success subsets with explicit n. Censored external clearance is not treated as exact0.3m.',
            'uncertainty': 'Resample accepted template groups together for paired cluster bootstrap; seeds reuse the same100 scenarios and are not independent task replicates.',
            'timing': 'CPU sequential single-thread per run; logged provider48Hz excludes cached collision queries and primary/physics; outer240Hz loop includes context/provider/primary/motors/physics/safety, excludes load/settle/render/export. No real-time guarantee.'},
        'frozen_release_verification': verification, 'executed_source_SHA256': {'stage12/experiment.py': sha(Path(__file__))}, 'reserved_geometry_SHA256': sha(STAGE / 'data/reserved_geometry.json'),
        'models': [m for m in methods() if m['checkpoint'] == 'best'], 'original_integrity_manifest_SHA256': sha(ROOT / 'provenance/release_integrity.json'),
        'data_boundary': audit.record(),
    }
    write_json(STAGE / 'configs/protocol.json', protocol)
    (STAGE / 'configs/protocol.sha256').write_text(sha(STAGE / 'configs/protocol.json') + '  protocol.json\n')
    print({'plan_saved': True, 'protocol_SHA256': sha(STAGE / 'configs/protocol.json'), 'validation_episodes': 522, 'heldout_target': 100}, flush=True)


def new_logged_run(scenario, method, output, model=None, config=None):
    """Stage12 method-parameter extension of the frozen logging-only wrapper."""
    import numpy as np
    import pybullet as p
    from me5418.final_evaluation import LoggedEpisode, FrozenPolicyAdapter, stats
    from me5418.evaluation import _csv
    from me5418.tracking import write_json as physical_json
    output=Path(output); output.mkdir(parents=True,exist_ok=False); begin=time.perf_counter()
    adapter=FrozenPolicyAdapter(model,config) if model is not None else None
    provider={'kind':method['kind']}
    if 'apf' in method: provider['apf']=method['apf']
    e=LoggedEpisode(scenario,provider,decision_function=adapter); timings=[];actions=[]
    try:
        while not e.done:
            k,t=e.k,e.t; outer_begin=time.perf_counter()
            u,updated=e.schedule.value(k,t,e.q,e.collision,e.observation_context())
            tr=e.advance(u,updated,e.schedule.raw); elapsed=time.perf_counter()-outer_begin
            decision=e.schedule.times[-1] if updated else None
            timings.append({'physical_step':k,'time_s':t,'provider_updated':int(updated),'provider_wall_s':decision,'core_advance_wall_s':e.loops[-1] if tr else None,'outer_physical_loop_wall_s':elapsed})
            if updated:
                raw_u,actual_u=e.schedule.raw,e.schedule.u
                row={'physical_step':k,'time_s':t,'decision_wall_s':decision,'u_component_clip_fraction':float(np.mean(abs(raw_u-actual_u)>1e-12)),'u_any_clip':int(np.any(abs(raw_u-actual_u)>1e-12)),'raw_policy_component_clip_fraction':None,'raw_policy_any_clip':None}
                raw=legal=None
                if adapter:
                    raw,legal=adapter.last['raw'],adapter.last['legal'];row['raw_policy_component_clip_fraction']=float(np.mean(raw!=legal));row['raw_policy_any_clip']=int(np.any(raw!=legal))
                for j in range(7):
                    row[f'raw_policy{j}']=float(raw[j]) if adapter else None;row[f'legal_policy{j}']=float(legal[j]) if adapter else None
                    row[f'raw_u{j}']=float(raw_u[j]);row[f'u{j}']=float(actual_u[j]);row[f'first_executed_command{j}']=float(tr['control']['command'][j]) if tr else None
                actions.append(row)
        summary=e.summary(output)
        summary['timing']={'provider_48Hz':stats(e.schedule.times),'core_physical_step':stats(e.loops),'outer_physical_loop':stats([r['outer_physical_loop_wall_s'] for r in timings]),'scene_load_wall_s':e.scene_load_wall_s,'settle_wall_s':e.settle_wall_s,'setup_including_settle_wall_s':e.setup_wall_s,'motion_wall_s':time.perf_counter()-begin-e.setup_wall_s,'definitions':read(STAGE/'configs/protocol.json')['evaluation']['timing']}
        summary['raw_policy_component_clip_fraction']=float(np.mean([r['raw_policy_component_clip_fraction'] for r in actions])) if adapter else None
        summary['raw_policy_decisions_with_clip_fraction']=float(np.mean([r['raw_policy_any_clip'] for r in actions])) if adapter else None
        summary['u_component_clip_fraction']=float(np.mean([r['u_component_clip_fraction'] for r in actions]))
        summary['u_decisions_with_clip_fraction']=float(np.mean([r['u_any_clip'] for r in actions]))
        summary['mean_u_abs_rad_s']=float(np.mean([abs(r[f'u{j}']) for r in actions for j in range(7)]))
        summary['state_violation_counts']={k:sum(r[k] for r in e.rows) for k in ('hard_position_violation','actual_speed_violation','tracking_error_violation','non_finite_state')}
        summary['final_nearest_links']={k:e.rows[-1][k] for k in ('obstacle_nearest_link_a','obstacle_nearest_link_b','self_nearest_link_a','self_nearest_link_b')}
        physical_json(output/'initial_state.json',e.initial_record);physical_json(output/'terminal_collision.json',e.collision)
        physical_json(output/'executed_scenario.json',scenario);physical_json(output/'method.json',method)
        _csv(output/'states.csv',e.rows);_csv(output/'settle.csv',e.settle);_csv(output/'physical_timing.csv',timings);_csv(output/'actions48.csv',actions)
        physical_json(output/'posture_actions.json',e.schedule.decisions)
    finally: e.close()
    summary['connection_closed']=not bool(p.isConnected(e.scene.cid));summary['wall_s_including_setup_and_io']=time.perf_counter()-begin
    physical_json(output/'summary.json',summary)
    write_json(output/'COMPLETE.json',{'scene_id':scenario['id'],'method':method['name'],'parameter_SHA256':checksum(scenario),'task_success':summary['success'],'completed_UTC':utc(),'files_SHA256':{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


def tune():
    protocol,audit,verification=safe_phase('tune')
    from me5418.arm_env import load_pool
    from evaluate import episode_row
    pool=load_pool('validation');ids=protocol['validation']['ordered_ids']
    if set(pool)!=set(ids) or len(ids)!=58:raise ValueError('Frozen validation pool mismatch')
    output=RAW/'tuning';output.mkdir(exist_ok=False)
    begin=time.perf_counter();rows=[];reports=[]
    for idx,candidate in enumerate(protocol['validation']['finite_grid']):
        method={'name':candidate['id'],'kind':'apf','apf':candidate['parameters'],'role':'validation_selection','seed':None,'checkpoint':None,'model_SHA256':None}
        scores=[]
        for identifier in ids:
            if time.perf_counter()-begin>protocol['validation']['wall_budget_s']:raise TimeoutError('Finite grid budget exceeded; no APF may be selected')
            scenario=pool[identifier];result=new_logged_run(scenario,method,output/candidate['id']/identifier)
            scores.append(result);rows.append(episode_row(method,scenario,result))
        successful=[s for s in scores if s['success']]
        report={'id':candidate['id'],'grid_index':idx,'parameters':candidate['parameters'],'n':len(scores),'successes':len(successful),'mean_complete_success_rmse_m':sum(s['rmse_m'] for s in successful)/len(successful) if successful else 1e99,'failure_categories':dict(Counter(s['failure_category'] for s in scores if not s['success']))}
        reports.append(report);write_json(output/candidate['id']/'candidate_summary.json',report)
        table(STAGE/'results/validation_tuning.csv',rows)
        print({'candidate':candidate['id'],'successes':len(successful),'n':58,'elapsed_s':round(time.perf_counter()-begin,1)},flush=True)
    selected=min(reports,key=lambda x:(-x['successes'],x['mean_complete_success_rmse_m'],x['grid_index']))
    summary={'complete':len(rows)==522,'selection_split':'validation58_only','reports':reports,'selected':selected,'selection_rule':protocol['validation']['selection_rule'],'protocol_SHA256':sha(STAGE/'configs/protocol.json'),'validation_csv_SHA256':sha(STAGE/'results/validation_tuning.csv'),'wall_s':time.perf_counter()-begin,'training_interactions':0,'task_failure_retries':0,'access':audit.record(),'frozen_verification':verification}
    write_json(STAGE/'results/tuning_summary.json',summary)
    print({'complete_grid':summary['complete'],'selected':selected,'wall_s':summary['wall_s']},flush=True)


def freeze():
    protocol,audit,verification=safe_phase('freeze')
    target=STAGE/'configs/tuned_apf_freeze.json'
    if target.exists():raise FileExistsError('Frozen tuned parameters cannot be overwritten')
    if (STAGE/'data/manifest.json').exists() or (RAW/'generation').exists():raise RuntimeError('APF freeze must precede new heldout generation')
    result=read(STAGE/'results/tuning_summary.json')
    if not result['complete'] or len(result['reports'])!=9 or any(r['n']!=58 for r in result['reports']):raise ValueError('A complete finite grid is mandatory')
    if result['protocol_SHA256']!=sha(STAGE/'configs/protocol.json') or result['validation_csv_SHA256']!=sha(STAGE/'results/validation_tuning.csv'):raise ValueError('Tuning evidence changed')
    selected=min(result['reports'],key=lambda x:(-x['successes'],x['mean_complete_success_rmse_m'],x['grid_index']))
    value={'version':protocol['version'],'frozen_UTC':utc(),'selected_candidate':selected,'parameters':selected['parameters'],'protocol_SHA256':sha(STAGE/'configs/protocol.json'),'tuning_summary_SHA256':sha(STAGE/'results/tuning_summary.json'),'validation_csv_SHA256':sha(STAGE/'results/validation_tuning.csv'),'source_SHA256':protocol['executed_source_SHA256'],'no_old_test_used_for_selection':True,'no_new_scenes_generated_at_freeze':True,'models_unchanged':protocol['models'],'frozen_verification':verification,'access':audit.record()}
    write_json(target,value);(STAGE/'configs/tuned_apf_freeze.sha256').write_text(sha(target)+'  tuned_apf_freeze.json\n')
    print({'frozen':True,'selected_candidate':selected['id'],'parameters':selected['parameters'],'freeze_SHA256':sha(target)},flush=True)


def geometric_key(s):
    return {k:s[k] for k in ('initial_arm_q_rad','finger_positions_m','sphere','trajectory')}


def verify_freeze(protocol):
    value=read(STAGE/'configs/tuned_apf_freeze.json')
    if value['protocol_SHA256']!=sha(STAGE/'configs/protocol.json') or value['tuning_summary_SHA256']!=sha(STAGE/'results/tuning_summary.json'):raise ValueError('Freeze input changed')
    expected=(STAGE/'configs/tuned_apf_freeze.sha256').read_text().split()[0]
    if expected!=sha(STAGE/'configs/tuned_apf_freeze.json'):raise ValueError('Tuned freeze changed')
    return value


def sample_candidates(protocol):
    import numpy as np
    import pybullet as p
    from me5418.scenarios import execution_template, measured_start, make_scene, near_duplicate
    from me5418.scene import PandaScene
    from me5418.tracking import state,safety_reason,ToolJacobian
    from me5418.posture import LinkPointJacobian
    h=protocol['heldout'];sampling=h['sampling'];ex=execution_template({});start,metadata=measured_start(ex)
    ex['model_metadata']={k:metadata[k] for k in ('load_flags','joints','self_pair_policy','monitored_pair_count','obstacle_monitored_link_indices','tool')}
    rng=np.random.default_rng(h['seed']);reserved=read(STAGE/'data/reserved_geometry.json')['scenes'];templates=[]
    with PandaScene() as scene:
        points=LinkPointJacobian(scene)
        for group in range(h['template_groups']):
            direction=rng.normal(size=3);direction/=np.linalg.norm(direction);axis=rng.normal(size=3);axis/=np.linalg.norm(axis)
            templates.append({'group':f'supplement-g{group:03d}','direction':direction,'axis':axis,'category':h['category_cycle'][group%5],'link':int(rng.choice(sampling['target_link_indices']))})
        records=[];previous=[];hashes={}
        for variant in range(h['variants_per_group']):
            for tpl in templates:
                direction=tpl['direction']+rng.normal(0,sampling['direction_cone_noise_sd'],3);direction/=np.linalg.norm(direction)
                delta=direction*rng.uniform(*sampling['length_range_m'])
                if tpl['category']=='far':sphere=rng.uniform(*sampling['far_cube_m'],3);construction={'type':'far cube'}
                else:
                    proxy=None
                    if tpl['category']=='tight_layout':
                        scene.reset();j=ToolJacobian(scene).matrix();dq=j.T@np.linalg.solve(j@j.T+.02**2*np.eye(3),delta);qproxy=np.asarray(ex['scene_config']['initial_arm_positions'])+dq;scene.reset(qproxy)
                        proxy={'type':'static damped linearized goal construction, not feasible motion','q_rad':qproxy.tolist()}
                    axis=tpl['axis']+rng.normal(0,sampling['surface_cone_noise_sd'],3);axis/=np.linalg.norm(axis);origin,_=points.link_frame(tpl['link'])
                    scene.move_obstacle((origin+.4*axis).tolist());raw=p.getClosestPoints(scene.robot,scene.obstacle,2,linkIndexA=tpl['link'],physicsClientId=scene.cid);probe=min(raw,key=lambda x:x[8]);outward=-np.asarray(probe[7]);outward/=np.linalg.norm(outward)
                    gap=rng.uniform(*sampling['gap_ranges_m'][tpl['category']]);sphere=np.asarray(probe[5])+(.06+gap)*outward
                    construction={'type':'real collision surface plus outward normal*(radius+gap)','target_link':scene.links[tpl['link']],'surface_point_m':list(probe[5]),'outward_normal':outward.tolist(),'sampled_gap_m':gap,'proxy':proxy};scene.reset()
                idx=len(records);s=make_scene(f"{h['dataset_version']}-{idx:04d}",'supplemental_heldout',tpl['group'],h['seed'],idx,ex,start,delta,sphere,tpl['category'],h['dataset_version']);s['generator_version']='stage12-surface-cone-v1'
                scene.move_obstacle(sphere.tolist());collision=scene.detect();reason=safety_reason(scene,*state(scene),scene.tool_pose(),collision,ex['validation']);gap=collision['obstacle']['minimum_signed_distance']
                base=p.getClosestPoints(scene.robot,scene.obstacle,4,linkIndexA=-1,physicsClientId=scene.cid);base_gap=min(x[8] for x in base) if base else None
                segment=np.asarray(delta);v=np.clip(np.dot(np.asarray(sphere)-start,segment)/np.dot(segment,segment),0,1);tool_gap=float(np.linalg.norm(np.asarray(sphere)-(np.asarray(start)+v*segment))-.06)
                rejects=[]
                if reason:rejects.append('initialization_failure:'+reason)
                if gap is not None and gap<h['geometry']['minimum_initial_gap_m']:rejects.append('initial_gap_below_2mm')
                if base_gap is not None and base_gap<h['geometry']['minimum_base_gap_m']:rejects.append('immobile_base_gap_below_10mm')
                if tool_gap<h['geometry']['minimum_tool_segment_gap_m']:rejects.append('sphere_blocks_tool_segment_buffer')
                key=checksum(geometric_key(s))
                if key in hashes:rejects.append('exact_new_duplicate:'+hashes[key])
                dup=next((a for a in previous if near_duplicate(s,a,h['deduplication'])),None)
                if dup:rejects.append('near_new_duplicate:'+dup['id'])
                dup=next((a for a in reserved if near_duplicate(s,a,h['deduplication'])),None)
                if dup:rejects.append('old_reserved_overlap:'+dup['id'])
                hashes[key]=s['id'];previous.append(s)
                records.append({'parameters':s,'geometric_SHA256':key,'parameter_SHA256':checksum(s),'construction':construction,'geometry':{'initial_min_gap_m':gap,'base_gap_m':base_gap,'tool_segment_gap_m':tool_gap,'initial_self_gap_m':collision['self']['minimum_signed_distance']},'status':'geometry_rejected' if rejects else 'pending_search','rejection_reasons':rejects,'attempts':[]})
    return {'version':h['dataset_version'],'protocol_SHA256':sha(STAGE/'configs/protocol.json'),'tuned_freeze_SHA256':sha(STAGE/'configs/tuned_apf_freeze.json'),'created_UTC':utc(),'measured_reference_start_world_m':start,'candidates':records,'count':len(records),'all_geometry_before_witness_search':True}


def compare_motor_replay(a,b):
    import numpy as np
    fields=['time_s','physical_step','error_m','actual_x','actual_y','actual_z','external_distance_m','self_distance_m']+[f'{prefix}{i}' for prefix in ('q','velocity','command','u') for i in range(7)]
    rows=[]
    for path in (a,b):
        with path.open(newline='') as f:rows.append(list(csv.DictReader(f)))
    differences={};missing=[]
    for field in fields:
        maximum=0.
        for x,y in zip(*rows):
            vx,vy=x[field],y[field]
            if not vx or not vy:
                if vx!=vy:missing.append(field)
            else:maximum=max(maximum,abs(float(vx)-float(vy)))
        differences[field]=maximum
    maximum=max(differences.values(),default=0.)
    return {'pass':len(rows[0])==len(rows[1]) and not missing and maximum<=1e-7,'state_counts':[len(x) for x in rows],'maximum_numeric_difference':maximum,'maximum_differences_by_field':differences,'missing_value_mismatches':missing,'scope':'Every actual motor-executed tool/state/velocity/command/clipped-u/error/distance sample; excludes preclip-provider diagnostic raw-u, flags derived from those states and wall clocks','tolerance_m_or_rad':1e-7}


def generate():
    protocol,audit,verification=safe_phase('generate');verify_freeze(protocol)
    from me5418.evaluation import run_scene
    from me5418.scenarios import near_duplicate
    out=RAW/'generation';out.mkdir(exist_ok=False);started=time.perf_counter();h=protocol['heldout'];data=sample_candidates(protocol)
    write_json(out/'candidates.json',data);accepted=[];index=[]
    for item in data['candidates']:
        if item['status']!='pending_search':continue
        if len(accepted)>=h['accepted_target']:item['status']='not_run_target_met';continue
        if time.perf_counter()-started>=h['wall_budget_s']:item['status']='budget_not_run';continue
        scenario=item['parameters'];found=False
        for attempt,provider in enumerate(h['witness_providers'],1):
            target=out/'witnesses'/scenario['id']/f"attempt{attempt}_{provider['name']}"
            try:summary=run_scene(scenario,provider,target)
            except Exception as exc:
                write_json(target/'exception.json',{'exception':repr(exc),'infrastructure_failure':True});item['attempts'].append({'provider':provider['name'],'success':False,'failure_category':'runtime_exception','exception':repr(exc)});item['status']='runtime_exception';break
            item['attempts'].append({'provider':provider['name'],'success':summary['success'],'failure_category':summary['failure_category'],'failure_reason':summary['failure_reason'],'summary_path':str((target/'summary.json').relative_to(ROOT)),'summary_SHA256':sha(target/'summary.json')})
            if not summary['success']:continue
            actions=read(target/'posture_actions.json');replay_target=out/'witnesses'/scenario['id']/f'replay{attempt}'
            replay=run_scene(scenario,{'kind':'tracking','name':'offline_saved_u_replay'},replay_target,replay_actions=actions)
            comparison=compare_motor_replay(target/'states.csv',replay_target/'states.csv')
            evidence={'scene_id':scenario['id'],'provider':provider['name'],'attempt_count':attempt,'replay_pass':bool(replay['success'] and comparison['pass']),'source_directory':str(target.relative_to(ROOT)),'replay_directory':str(replay_target.relative_to(ROOT)),'source_files_SHA256':{p.name:sha(p) for p in sorted(target.iterdir()) if p.is_file()},'replay_files_SHA256':{p.name:sha(p) for p in sorted(replay_target.iterdir()) if p.is_file()},'motor_replay_comparison':comparison,'policy_input_excluded':True}
            write_json(replay_target/'replay_comparison.json',evidence)
            if evidence['replay_pass']:
                item['status']='accepted';item['witness_index']=len(index);accepted.append(scenario);index.append(evidence);found=True;break
            item['attempts'][-1]['replay_failed']=True
        if not found and item['status']=='pending_search':item['status']='search_no_witness_not_proven_infeasible'
        write_json(out/'candidate_outcomes.json',data);write_json(out/'witness_index.json',index)
        if len(accepted)%10==0:print({'accepted':len(accepted),'searched_scene':scenario['id'],'elapsed_s':round(time.perf_counter()-started,1)},flush=True)
    write_json(out/'candidate_outcomes.json',data);write_json(out/'witness_index.json',index)
    if len(accepted)!=h['accepted_target']:raise RuntimeError(f'Only{len(accepted)} accepted; predeclared pool/budget exhausted; do not change the current protocol.')
    write_json(STAGE/'data/manifest.json',{'dataset_version':h['dataset_version'],'protocol_SHA256':data['protocol_SHA256'],'tuned_freeze_SHA256':data['tuned_freeze_SHA256'],'parameter_only':True,'witness_actions_excluded':True,'witness_metadata_excluded':True,'scenes':accepted})
    write_json(STAGE/'data/heldout.json',{'split':'supplemental_heldout','scene_ids':[s['id'] for s in accepted],'n':len(accepted)})
    summary={'complete':True,'candidate_count':len(data['candidates']),'accepted_count':len(accepted),'accepted_template_groups':len({s['template_group'] for s in accepted}),'outcome_counts':dict(Counter(i['status'] for i in data['candidates'])),'generated_categories':dict(Counter(i['parameters']['preassigned_geometric_category'] for i in data['candidates'])),'accepted_categories':dict(Counter(s['preassigned_geometric_category'] for s in accepted)),'witness_provider_counts':dict(Counter(i['provider'] for i in index)),'provider_attempt_episodes':sum(len(i['attempts']) for i in data['candidates']),'replay_episodes':len(index),'all_replay_pass':all(i['replay_pass'] for i in index),'maximum_motor_replay_difference':max(i['motor_replay_comparison']['maximum_numeric_difference'] for i in index),'wall_s':time.perf_counter()-started,'manifest_SHA256':sha(STAGE/'data/manifest.json'),'outcomes_SHA256':sha(out/'candidate_outcomes.json'),'witness_index_SHA256':sha(out/'witness_index.json'),'protocol_SHA256':data['protocol_SHA256'],'tuned_freeze_SHA256':data['tuned_freeze_SHA256'],'selection_bias':h['witness_bias'],'access':audit.record(),'frozen_verification':verification}
    write_json(STAGE/'results/generation_summary.json',summary);write_json(STAGE/'results/witness_index.json',index)
    # Compact public ledger includes every accepted/rejected/unsearched candidate and outcome.
    table(STAGE/'results/candidate_ledger.csv',[{'scene_id':i['parameters']['id'],'template_group':i['parameters']['template_group'],'geometry':i['parameters']['preassigned_geometric_category'],'status':i['status'],'rejection_reasons':';'.join(i['rejection_reasons']),'attempts':len(i['attempts']),'attempt_successes':sum(a['success'] for a in i['attempts']),'parameter_SHA256':i['parameter_SHA256']} for i in data['candidates']])
    print(summary,flush=True)


def evaluate():
    protocol,audit,verification=safe_phase('evaluate');freeze=verify_freeze(protocol)
    import torch
    from evaluate import episode_row
    torch.set_num_threads(1)
    manifest=read(STAGE/'data/manifest.json');ids=read(STAGE/'data/heldout.json')['scene_ids'];pool={s['id']:s for s in manifest['scenes']}
    if len(ids)!=100 or list(pool)!=ids:raise ValueError('New heldout pool/order changed')
    generation=read(STAGE/'results/generation_summary.json')
    if generation['manifest_SHA256']!=sha(STAGE/'data/manifest.json'):raise ValueError('Frozen new manifest changed')
    selected=[{'name':'tracking','kind':'tracking','role':'supplemental_primary','seed':None,'checkpoint':None,'model_SHA256':None},{'name':'APF_fixed','kind':'apf','apf':read(ROOT/'configs/stage04_posture_apf.json')['apf'],'role':'supplemental_primary','seed':None,'checkpoint':None,'model_SHA256':None},{'name':'APF_tuned','kind':'apf','apf':freeze['parameters'],'role':'supplemental_primary','seed':None,'checkpoint':None,'model_SHA256':None}]+protocol['models']
    out=RAW/'evaluation';out.mkdir(exist_ok=False);started=time.perf_counter();rows=[];method_reports=[]
    write_json(out/'evaluation_plan.json',{'saved_UTC':utc(),'methods':selected,'ordered_IDs':ids,'manifest_SHA256':sha(STAGE/'data/manifest.json'),'protocol_SHA256':sha(STAGE/'configs/protocol.json'),'tuned_freeze_SHA256':sha(STAGE/'configs/tuned_apf_freeze.json'),'source_SHA256':sha(Path(__file__)),'task_failures_retried':0,'workers':1})
    for method in selected:
        model,config=load_model(method) if method['kind']=='policy' else (None,None);current=[];method_begin=time.perf_counter()
        for identifier in ids:
            if time.perf_counter()-started>protocol['evaluation']['wall_budget_s']:raise TimeoutError('Predeclared evaluation wall budget exceeded')
            scenario=pool[identifier];result=new_logged_run(scenario,method,out/method['name']/identifier,model,config)
            if not result['connection_closed'] or result['joint_resets_during_motion']!=0:raise RuntimeError('Execution invariants failed')
            row=episode_row(method,scenario,result);current.append(row);rows.append(row)
            append_json(out/'progress.jsonl',{'UTC':utc(),'method':method['name'],'scene_id':identifier,'success':result['success'],'completed':len(current),'planned':100})
            if len(current)%25==0:print({'method':method['name'],'completed':len(current),'successes':sum(r['success'] for r in current),'elapsed_s':round(time.perf_counter()-started,1)},flush=True)
        report={'method':method['name'],'n':len(current),'successes':sum(r['success'] for r in current),'failure_categories':dict(Counter(r['failure_category'] for r in current if not r['success'])),'wall_s':time.perf_counter()-method_begin};method_reports.append(report)
        write_json(out/method['name']/'method_summary.json',report);table(STAGE/'results/episodes_new600.csv',rows)
    summary={'complete':len(rows)==600,'physics_episodes':len(rows),'n_independent_scenes':100,'accepted_template_groups':generation['accepted_template_groups'],'method_reports':method_reports,'successes':{r['method']:r['successes'] for r in method_reports},'wall_s':time.perf_counter()-started,'training_interactions':0,'task_failures_retried':0,'manifest_SHA256':sha(STAGE/'data/manifest.json'),'episodes_csv_SHA256':sha(STAGE/'results/episodes_new600.csv'),'protocol_SHA256':sha(STAGE/'configs/protocol.json'),'tuned_freeze_SHA256':sha(STAGE/'configs/tuned_apf_freeze.json'),'modules':module_paths(),'access':audit.record(),'frozen_verification':verification}
    write_json(STAGE/'results/execution_complete.json',summary);print(summary,flush=True)


def audit_run():
    protocol,audit,verification=safe_phase('audit');freeze=verify_freeze(protocol)
    from me5418.scenarios import near_duplicate
    manifest=read(STAGE/'data/manifest.json');scenes=manifest['scenes'];reserved=read(STAGE/'data/reserved_geometry.json')['scenes'];index=read(STAGE/'results/witness_index.json');overlaps=[];duplicates=[];exact=[];hashes={}
    for i,scene in enumerate(scenes):
        key=checksum(geometric_key(scene))
        if key in hashes:exact.append([scene['id'],hashes[key]])
        hashes[key]=scene['id']
        for other in scenes[:i]:
            if near_duplicate(scene,other,protocol['heldout']['deduplication']):duplicates.append([scene['id'],other['id']])
        for old in reserved:
            if near_duplicate(scene,old,protocol['heldout']['deduplication']):overlaps.append([scene['id'],old['id']])
    witnessed={x['scene_id']:x for x in index};evidence_failures=[];checked=0
    for scene in scenes:
        w=witnessed.get(scene['id'])
        if not w or not w['replay_pass']:evidence_failures.append(scene['id']);continue
        for folder,key in (('source_directory','source_files_SHA256'),('replay_directory','replay_files_SHA256')):
            for name,expected in w[key].items():
                path=ROOT/w[folder]/name;checked+=1
                if not path.exists() or sha(path)!=expected:evidence_failures.append(str(path))
    with (STAGE/'results/episodes_new600.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    method_names=protocol['evaluation']['methods'];expected=[(m,s['id']) for m in method_names for s in scenes];actual=[(r['method'],r['scene_id']) for r in rows]
    leaks=[s['id'] for s in scenes if any(k in s for k in ('witness_index','replay_index','feasibility_status','u_sequence','posture_actions','witness_actions'))]
    complete=read(STAGE/'results/execution_complete.json');generation=read(STAGE/'results/generation_summary.json')
    integrity={'pass':not(overlaps or duplicates or exact or leaks or evidence_failures) and expected==actual and len(scenes)==100 and complete['complete'] and generation['all_replay_pass'],'n_scenes':len(scenes),'n_reserved_geometry':len(reserved),'n_rows':len(rows),'template_groups':len({s['template_group'] for s in scenes}),'exact_duplicates':exact,'near_duplicates_new':duplicates,'near_overlaps_old':overlaps,'policy_manifest_metadata_or_action_leaks':leaks,'witness_evidence_failures':evidence_failures,'witness_file_hashes_checked':checked,'method_scene_order_match':expected==actual,'frozen_release_verification':verification,'stage12_protocol_SHA256':sha(STAGE/'configs/protocol.json'),'stage12_freeze_SHA256':sha(STAGE/'configs/tuned_apf_freeze.json'),'outputs_preserve_task_failures':True,'training_interactions':0,'access':audit.record()}
    write_json(STAGE/'results/integrity_audit.json',integrity)
    print(integrity,flush=True)
    if not integrity['pass']:raise RuntimeError('Stage12 audit failed')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('plan','tune','freeze','generate','evaluate','audit'));args=parser.parse_args()
    globals()[{'audit':'audit_run'}.get(args.command,args.command)]()

if __name__=='__main__':main()
