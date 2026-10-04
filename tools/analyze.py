#!/usr/bin/env python
"""Descriptive analysis of the saved fixed-scene CSV; no model selection."""
from pathlib import Path
import argparse,csv,json,collections

ROOT=Path(__file__).resolve().parents[1]
METHODS=['tracking','APF']+[f'PPO_{k}_{s}' for k in ['best','last'] for s in [550901,551901,552901]]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='results/reference/episodes_all1048.csv')
    p.add_argument('--output',required=True,help='New name under outputs/')
    a=p.parse_args();src=ROOT/a.input;dest=(ROOT/'outputs'/a.output).resolve()
    from common import bootstrap
    bootstrap(True)
    if not dest.is_relative_to(ROOT/'outputs') or dest==ROOT/'outputs':p.error('Output must be a new subdirectory of outputs/')
    with src.open() as f:rows=list(csv.DictReader(f))
    groups={m:[r for r in rows if r['method']==m] for m in METHODS}
    expected=json.loads((ROOT/'datasets/me5418-scenes-v1/splits/test.json').read_text())['scene_ids']
    if len(rows)!=1048 or any([r['scene_id'] for r in groups[m]]!=expected for m in METHODS):raise ValueError('Expected fixed131 ordered IDs for all eight methods')
    ok=lambda r:r['success'] in ('True','true','1')
    table=[{'method':m,'n':len(groups[m]),'successes':sum(ok(r) for r in groups[m]),'rate':sum(ok(r) for r in groups[m])/len(groups[m]),'failure_categories':dict(collections.Counter(r['failure_category'] for r in groups[m] if not ok(r)))} for m in METHODS]
    keyed={m:{r['scene_id']:r for r in groups[m]} for m in METHODS};pairs={}
    for seed in [550901,551901,552901]:
        b=f'PPO_best_{seed}';baseline=keyed['APF'];candidate=keyed[b]
        shared=[i for i in expected if ok(baseline[i]) and ok(candidate[i])]
        metrics={}
        for k in ['max_error_m','rmse_m','minimum_external_distance_m','command_smoothness_rms_rad_s2','max_actual_arm_velocity_rad_s']:
            finite=[i for i in shared if baseline[i][k] not in ('','None','null') and candidate[i][k] not in ('','None','null')]
            metrics[k]={'common_complete_n':len(shared),'uncensored_paired_n':len(finite),'APF_mean':sum(float(baseline[i][k]) for i in finite)/len(finite) if finite else None,'PPO_mean':sum(float(candidate[i][k]) for i in finite)/len(finite) if finite else None}
        pairs[b]={'gained':[i for i in expected if not ok(baseline[i]) and ok(candidate[i])],'lost':[i for i in expected if ok(baseline[i]) and not ok(candidate[i])],'common_complete_n':len(shared),'metrics':metrics}
    dest.mkdir(parents=True,exist_ok=False)
    category_counts={m:{c:{'n':sum(r['geometry']==c for r in groups[m]),'successes':sum(ok(r) and r['geometry']==c for r in groups[m])} for c in ['far','near_link','tight_layout']} for m in METHODS}
    failures=[r for r in rows if not ok(r)]
    from comparison import compare_episode_csv
    comparison=compare_episode_csv(ROOT/'results/reference/episodes_all1048.csv',src)
    (dest/'failures.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2)+'\n')
    (dest/'reference_comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2)+'\n')
    result={'category_counts':category_counts,'scope':'descriptive fixed131 regression set; no test reselection or significance inference','counts':table,'paired_vs_APF':pairs,'three_seed_policies_share_same_scenes':True}
    (dest/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (dest/'success_counts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['method','n','successes','rate']);w.writeheader();w.writerows({k:r[k] for k in w.fieldnames} for r in table)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4));ax.bar([r['method'] for r in table],[r['successes'] for r in table],color=['#888888','#35778f']+['#388b5b']*3+['#a79651']*3)
    ax.set_ylim(0,140);ax.set_ylabel('Complete successes / 131');ax.tick_params(axis='x',labelrotation=30)
    for j,r in enumerate(table):ax.text(j,r['successes']+2,str(r['successes']),ha='center')
    fig.tight_layout();fig.savefig(dest/'success_counts.png',dpi=160);plt.close(fig)
    print(json.dumps({r['method']:r['successes'] for r in table}))

if __name__=='__main__':main()
