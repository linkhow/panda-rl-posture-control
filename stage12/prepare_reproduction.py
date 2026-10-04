#!/usr/bin/env python3
"""Prepare a fresh runnable copy without overwriting published experiment evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
DIRECTORIES=('configs','datasets','delivery','docs','environment','me5418','media','models','provenance','references','results','scenes','scripts','stage12','tests','tools')
FILES=('README.md','run.sh','models_index.json','requirements.cpu.lock.txt')


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):value.update(block)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',required=True,help='A new directory outside the source repository; refuses any existing destination')
    args=parser.parse_args();target=Path(args.destination).expanduser().resolve()
    if target.exists():parser.error('Destination already exists; choose a new directory')
    if target==ROOT or ROOT in target.parents:parser.error('Use an independent destination outside the source repository')
    target.mkdir(parents=True)
    for name in FILES:
        if (ROOT/name).is_file():shutil.copy2(ROOT/name,target/name)
    for name in DIRECTORIES:
        if (ROOT/name).is_dir():shutil.copytree(ROOT/name,target/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.DS_Store','dist','cache','.venv'))
    reference=target/'replication_reference/stage12';reference.parent.mkdir(parents=True)
    shutil.copytree(target/'stage12',reference)
    generated=('configs/protocol.json','configs/protocol.sha256','configs/tuned_apf_freeze.json','configs/tuned_apf_freeze.sha256','data/manifest.json','data/heldout.json')
    removed=[]
    for name in generated:
        path=target/'stage12'/name
        if path.exists():path.unlink();removed.append('stage12/'+name)
    for name in ('results','analysis'):
        path=target/'stage12'/name
        if path.exists():shutil.rmtree(path);removed.append('stage12/'+name+'/')
    analysis_protocol=reference/'analysis/protocol.json'
    if analysis_protocol.is_file():
        (target/'stage12/analysis').mkdir(exist_ok=True)
        shutil.copy2(analysis_protocol,target/'stage12/analysis/protocol.json')
    (target/'stage12/results').mkdir()
    (target/'outputs').mkdir()
    source_integrity=json.loads((ROOT/'provenance/release_integrity.json').read_text())
    mismatches=[]
    for name,expected in source_integrity['files_SHA256'].items():
        path=target/name
        if not path.is_file() or digest(path)!=expected:mismatches.append(name)
    report={'source_numeric_files_all_match':not mismatches,'checked_frozen_files':len(source_integrity['files_SHA256']),'mismatches':mismatches,'reference':'replication_reference/stage12/','cleared_generated_paths_only_in_new_copy':removed,'not_copied':['.git','outputs','virtualenv','cache','credentials'],'original_source_changed':False,'command_after_install':'python -B stage12/experiment.py plan; then tune, freeze, generate, evaluate, audit','source_experiment_SHA256':digest(ROOT/'stage12/experiment.py'),'statistical_analysis_protocol_preserved':analysis_protocol.is_file(),'statistical_analysis_protocol_SHA256':digest(analysis_protocol) if analysis_protocol.is_file() else None}
    (target/'preparation_record.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'destination':str(target),**report},ensure_ascii=False,indent=2))
    if mismatches:raise RuntimeError('Fresh copy failed frozen file checks; preserve it for inspection')


if __name__=='__main__':main()
