"""Audit raw600-case evidence from the full experiment package; never rerun physics."""
import argparse,csv,hashlib,json,math,pathlib
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',help='Optional new JSON path; refuses overwriting evidence');args=parser.parse_args()
root=pathlib.Path(__file__).resolve().parents[1];stage=root/'stage12';rows=list(csv.DictReader((stage/'results/episodes_new600.csv').open()));scenes={s['id']:s for s in json.loads((stage/'data/manifest.json').read_text())['scenes']};errors=[];checks=0;hashes=0;states_total=0;failed=0
for row in rows:
 folder=root/'outputs/stage12/evaluation'/row['method']/row['scene_id'];complete=json.loads((folder/'COMPLETE.json').read_text());summary=json.loads((folder/'summary.json').read_text());states=list(csv.DictReader((folder/'states.csv').open()));states_total+=len(states)
 for name,expected in complete['files_SHA256'].items():
  checks+=1;hashes+=1
  if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=expected:errors.append([row['method'],row['scene_id'],'file_hash',name])
 expected=hashlib.sha256(json.dumps(scenes[row['scene_id']],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest();checks+=1
 if complete['parameter_SHA256']!=expected:errors.append([row['method'],row['scene_id'],'parameter_sha'])
 success=row['success']=='True';checks+=4
 if success!=summary['success'] or success!=complete['task_success']:errors.append([row['method'],row['scene_id'],'success_label'])
 if len(states)!=summary['logged_states']:errors.append([row['method'],row['scene_id'],'row_count'])
 if (summary['failure_reason'] or '')!=states[-1]['failure']:errors.append([row['method'],row['scene_id'],'failure_prefix_terminal_record'])
 if success and (summary['actual_duration_s']!=4. or len(states)!=961 or int(states[-1]['physical_step'])!=960):errors.append([row['method'],row['scene_id'],'full_success_duration'])
 if not success:
  failed+=1
  if not summary['failure_reason'] or summary['actual_duration_s']>4.:errors.append([row['method'],row['scene_id'],'failure_not_retained'])
 maximum=max(float(s['error_m']) for s in states);checks+=1
 if maximum!=summary['max_error_m']:errors.append([row['method'],row['scene_id'],'max_error_summary'])
report={'pass':not errors,'episode_rows_checked':len(rows),'file_hashes_checked':hashes,'checks':checks,'state_rows_preserved':states_total,'failed_episode_prefixes_preserved':failed,'errors':errors,'scope':'All600 raw per-episode files match COMPLETE checksums; scenario fingerprints match the shared100-scene manifest; success/terminal failure labels, full4s duration, state counts and maximum tracking error independently checked from raw states. This is an integrity/statistic audit, not a second physics replay.'}
if args.output:
 target=pathlib.Path(args.output).resolve()
 if target.exists():parser.error('Output already exists; preserve prior evidence')
 target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
if not report['pass']:raise RuntimeError('Raw episode evidence audit failed')
