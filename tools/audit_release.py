#!/usr/bin/env python3
"""Inspect the actual Git index before publishing this prepared repository."""
from pathlib import Path
import argparse,base64,hashlib,json,pickletools,re,subprocess,zipfile

ROOT=Path(__file__).resolve().parents[1]
ROOT_FILES={'.gitignore','README.md','run.sh','models_index.json','requirements.cpu.lock.txt'}
PREFIXES=('me5418/','scenes/','configs/','datasets/me5418-scenes-v1/','docs/','environment/','media/','models/','provenance/','references/','results/reference/','scripts/','tools/')
PERSONAL=re.compile(r'(interview|resume|defense|exercise_feedback|hands_on_exercises|learning_route|learning_log|contribution_checklist|request_stage|已粘贴)',re.I)
MACHINE=re.compile(rb'(?:/home|/tmp|/mnt)/[a-zA-Z0-9_.-]+/')
TOKEN=re.compile(rb'(?:github_pat_[a-zA-Z0-9_]{20,}|gh[pousr]_[a-zA-Z0-9]{20,})')
TEXT_EXT={'.py','.sh','.md','.json','.txt','.csv'}
MEMBERS={'data','pytorch_variables.pth','policy.pth','policy.optimizer.pth','_stable_baselines3_version','system_info.txt'}

def inspect_bytes(data,name,issues):
    if MACHINE.search(data) or b'file:' + b'/' * 3 in data:issues.append({'file':name,'reason':'machine_absolute_path'})
    if TOKEN.search(data):issues.append({'file':name,'reason':'credential_pattern'})

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,help='Audit JSON outside the staged publication files, such as outputs/preupload.json')
    a=p.parse_args();names=subprocess.check_output(['git','diff','--cached','--name-only','--diff-filter=ACMR','-z'],cwd=ROOT).decode().split('\0');names=[n for n in names if n]
    if not names:raise RuntimeError('Stage the intended publication files first')
    issues=[];model_checks=[];files={}
    for n in names:
        path=ROOT/n
        if n not in ROOT_FILES and not n.startswith(PREFIXES):issues.append({'file':n,'reason':'outside_allowlist'})
        if PERSONAL.search(n):issues.append({'file':n,'reason':'personal_material_filename'})
        if path.is_symlink():issues.append({'file':n,'reason':'symlink'})
        payload=subprocess.check_output(['git','show',':'+n],cwd=ROOT)
        if payload!=path.read_bytes():issues.append({'file':n,'reason':'staged_worktree_difference'})
        files[n]=hashlib.sha256(payload).hexdigest()
        if path.suffix in TEXT_EXT:inspect_bytes(payload,n,issues)
        if path.suffix=='.zip':
            if not n.startswith('models/'):issues.append({'file':n,'reason':'only_model_zips_allowed'})
            with zipfile.ZipFile(path) as z:
                if set(z.namelist())!=MEMBERS or len(z.namelist())!=6:issues.append({'file':n,'reason':'unexpected_zip_member'})
                for member in z.namelist():inspect_bytes(z.read(member),n+':'+member,issues)
                data=json.loads(z.read('data'));serial_fields=0
                if data.get('tensorboard_log') is not None:issues.append({'file':n,'reason':'unclean_tensorboard_metadata'})
                for key,v in data.items():
                    if isinstance(v,dict) and ':serialized:' in v:
                        decoded=base64.b64decode(v[':serialized:']);serial_fields+=1;inspect_bytes(decoded,n+':'+key,issues)
                        for op,value,_ in pickletools.genops(decoded):
                            if isinstance(value,str):inspect_bytes(value.encode(),n+':'+key+':pickle',issues)
                model_checks.append({'path':n,'standard_six_members':True,'serialized_fields_checked':serial_fields,'tensorboard_log_null':data.get('tensorboard_log') is None})
    if len(model_checks)!=6:issues.append({'reason':'all_six_models_must_be_staged'})
    manifest=json.loads((ROOT/'provenance/public_file_manifest.json').read_text())
    if set(names)!=(set(manifest['files_SHA256'])|{'provenance/public_file_manifest.json'}):issues.append({'reason':'staged_manifest_file_set_mismatch'})
    for n,h in manifest['files_SHA256'].items():
        if files.get(n)!=h:issues.append({'file':n,'reason':'staged_manifest_sha_mismatch'})
    report={'pass':not issues,'staged_files':len(names),'files_SHA256':files,'six_model_zip_checks':model_checks,'issues':issues,'ignored_output_directories_in_staging':not any(n.startswith(('outputs/','.venv/')) for n in names),'personal_files_in_staging':False if not any(PERSONAL.search(n) for n in names) else True}
    out=(ROOT/a.output).resolve()
    if not out.is_relative_to(ROOT/'outputs'):raise ValueError('Keep audit output inside untracked outputs/')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:report[k] for k in ['pass','staged_files','issues']},ensure_ascii=False))
    if issues:raise SystemExit(1)

if __name__=='__main__':main()
