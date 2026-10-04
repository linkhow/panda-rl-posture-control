from pathlib import Path
import json
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
r=Path(__file__).resolve().parents[4];d=json.loads((r/'stage12/analysis/original_analysis.json').read_text());groups=sorted(d['template_groups']);methods=d['primary_methods'];s=d['strata']
mat=[];counts=[];ns=[]
for m in methods:
 records=[next(x for x in s if x['method']==m and x['stratum_type']=='template_group' and x['stratum']==g) for g in groups]
 mat.append([x['success_rate'] for x in records]);counts.append([x['successes'] for x in records]);ns.append([x['n'] for x in records])
fig,ax=plt.subplots(figsize=(7.1,2.75));im=ax.imshow(mat,cmap='YlGnBu',vmin=0,vmax=1,aspect='auto')
ax.set_yticks(range(5),['tracking','APF','PPO550901','PPO551901','PPO552901'],fontsize=8)
ax.set_xticks(range(len(groups)),[g.replace('formal-','')+'\n'+str(n) for g,n in zip(groups,ns[0])],fontsize=7)
ax.tick_params(length=0);ax.set_xlabel('Template group / scenario denominator (n)',fontsize=8)
for i in range(5):
 for j in range(len(groups)):ax.text(j,i,str(counts[i][j]),ha='center',va='center',fontsize=8,color='white' if mat[i][j]>.65 else '#153F5A')
fig.colorbar(im,ax=ax,fraction=.025,pad=.015).ax.tick_params(labelsize=7);fig.tight_layout(pad=.35)
fig.savefig(r/'docs/reports/stage12/assets/template_heatmap.png',dpi=240);plt.close(fig)
