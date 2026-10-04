#!/usr/bin/env python3
"""Independently audit numerical task geometry and shared execution contracts."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text())


def fingerprint(scene):
    tr=scene['trajectory'];sphere=scene['sphere']
    value={'q0':scene['initial_arm_q_rad'],'fingers':scene['finger_positions_m'],'start':tr['start_world_m'],'goal':tr['goal_world_m'],'T':tr['duration_s'],'sphere_position':sphere['position_world_m'],'radius':sphere['radius_m']}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def close(a,b,tolerance):return max(abs(x-y) for x,y in zip(a,b))<=tolerance


def near(a,b,tol):
    return (close(a['initial_arm_q_rad'],b['initial_arm_q_rad'],tol['q_rad']) and close(a['trajectory']['start_world_m'],b['trajectory']['start_world_m'],tol['start_m']) and close(a['trajectory']['goal_world_m'],b['trajectory']['goal_world_m'],tol['goal_m']) and close(a['sphere']['position_world_m'],b['sphere']['position_world_m'],tol['sphere_m']) and abs(a['sphere']['radius_m']-b['sphere']['radius_m'])<=tol['radius_m'] and abs(a['trajectory']['duration_s']-b['trajectory']['duration_s'])<=tol['duration_s'])


def execution_contract(scene):
    value=json.loads(json.dumps(scene['execution']));value['scene_config'].pop('obstacle');return value


def audit():
    new=read(ROOT/'stage12/data/manifest.json')['scenes'];old=read(ROOT/'datasets/me5418-scenes-v1/manifest.json')['scenes'];reserved=read(ROOT/'stage12/data/reserved_geometry.json')['scenes'];tol=read(ROOT/'stage12/configs/protocol.json')['heldout']['deduplication']
    oldkeys={fingerprint(s):s['id'] for s in old};reservedkeys={fingerprint(s):s['id'] for s in reserved};newkeys=[fingerprint(s) for s in new]
    report={'pass':True,'n_new':len(new),'n_old_accepted':len(old),'n_old_reserved':len(reserved),'fingerprint_definition':'SHA256 canonical JSON of numerical q0,fingers,measured start,goal,duration,sphere center/radius; excludes IDs,split,template,controller outcomes and witness','exact_overlap_old_accepted':[[s['id'],oldkeys[fingerprint(s)]] for s in new if fingerprint(s) in oldkeys],'exact_overlap_all_old_reserved':[[s['id'],reservedkeys[fingerprint(s)]] for s in new if fingerprint(s) in reservedkeys],'exact_duplicate_new_count':len(newkeys)-len(set(newkeys)),'near_overlap_old_accepted':[[a['id'],b['id']] for a in new for b in old if near(a,b,tol)],'near_overlap_all_old_reserved':[[a['id'],b['id']] for a in new for b in reserved if near(a,b,tol)],'near_duplicate_new':[[a['id'],b['id']] for i,a in enumerate(new) for b in new[:i] if near(a,b,tol)],'same_execution_contract_excluding_obstacle':all(execution_contract(s)==execution_contract(old[0]) for s in new),'same_fixed_q0_and_fingers':all(s['initial_arm_q_rad']==old[0]['initial_arm_q_rad'] and s['finger_positions_m']==old[0]['finger_positions_m'] for s in new),'path_length_range_m':[min(sum(v*v for v in s['trajectory']['displacement_m'])**.5 for s in new),max(sum(v*v for v in s['trajectory']['displacement_m'])**.5 for s in new)],'source_manifest_SHA256':hashlib.sha256((ROOT/'stage12/data/manifest.json').read_bytes()).hexdigest()}
    report['pass']=not any(report[k] for k in ('exact_overlap_old_accepted','exact_overlap_all_old_reserved','exact_duplicate_new_count','near_overlap_old_accepted','near_overlap_all_old_reserved','near_duplicate_new')) and report['same_execution_contract_excluding_obstacle'] and report['same_fixed_q0_and_fingers']
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',help='Optional new JSON file; refuses overwriting evidence');args=parser.parse_args();result=audit()
    if args.output:
        target=Path(args.output).resolve()
        if target.exists():parser.error('Output already exists; preserve the recorded audit')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result['pass']:raise RuntimeError('Geometry or execution-contract audit failed')


if __name__=='__main__':main()
