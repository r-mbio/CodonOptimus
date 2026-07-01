#!/usr/bin/env python3
"""
Figure 3 v2 — CPB benchmark with 4 native genes per organism (A4, 7×8 in)

Panel A: 3 rows × 4 cols
  Rows: Native (host org.) | CodonOptimus | CodonTransformer
  Cols: E.coli | B.subtilis | S.cerevisiae | K.phaffii
  Each subplot: 4 thin individual-gene lines + bold mean line + IQR shaded band
  Reference: CPB=0 (random synonymous choice)
  n=4 genes per organism, randomly sampled from top-10%-tAI, 200-300 codons

Panel B: Aggregate CPB boxplots — 4 organisms, 4 heterologous benchmark proteins

Saves to: figures/fig3_combined_v2.{png,pdf,svg}
"""

import sys, math
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / '02_foundation_pretraining'))
sys.path.insert(0, str(_REPO_ROOT / '03_finetuning'))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from scipy import stats

from train_mlm_industrial import (
    IndustrialMLM, ID_TO_PAIR, AA_MASK, ORG_TO_ID, CLS_TOKEN,
)

BASE    = Path(__file__).resolve().parent.parent
OUT_PNG = BASE / 'figures/fig3_combined_v2.png'
OUT_PDF = BASE / 'figures/fig3_combined_v2.pdf'
OUT_SVG = BASE / 'figures/fig3_combined_v2.svg'

STOP_C = {'TAA','TAG','TGA'}
WINDOW = 20   # smoothing window

GENETIC_CODE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q','AAT':'N','AAC':'N',
    'AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGA':'R','AGG':'R','AGT':'S','AGC':'S','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
    'TAA':'*','TAG':'*','TGA':'*',
}
def dna_to_aa(dna):
    aas=[]
    for i in range(0,len(dna)-2,3):
        aa=GENETIC_CODE.get(dna[i:i+3].upper(),'X')
        if aa=='*': break
        if aa!='X': aas.append(aa)
    return ''.join(aas)

# ── Organism config — 4 genes each, randomly sampled top-10%-tAI 200-300 aa ──
ORGS = [
    {'org':'ecoli',        'label':'E. coli',       'col_label':'$\\it{E.\\ coli}$',
     'gene_ids':['ecoli_b4384','ecoli_b3860','ecoli_b0055','ecoli_b3751'],
     'co_org':'ecoli', 'ct_id':52,
     'raw':['data/raw/ecoli/ecoli_train.csv','data/raw/ecoli/ecoli_val.csv']},
    {'org':'bacillus',     'label':'B. subtilis',   'col_label':'$\\it{B.\\ subtilis}$',
     'gene_ids':['bacillus_rpsB','bacillus_licT','bacillus_tasA','bacillus_lipA'],
     'co_org':'bacillus', 'ct_id':2,
     'raw':['data/raw/bacillus/bacillus_train.csv','data/raw/bacillus/bacillus_val.csv']},
    {'org':'s_cerevisiae', 'label':'S. cerevisiae', 'col_label':'$\\it{S.\\ cerevisiae}$',
     'gene_ids':['s_cerevisiae_YIL070C','s_cerevisiae_YDL082W',
                 's_cerevisiae_YML105C','s_cerevisiae_YHR088W'],
     'co_org':'s_cerevisiae', 'ct_id':120,
     'raw':['data/raw/s_cerevisiae/s_cerevisiae_train.csv',
            'data/raw/s_cerevisiae/s_cerevisiae_val.csv']},
    {'org':'pichia',       'label':'K. phaffii',    'col_label':'$\\it{K.\\ phaffii}$',
     'gene_ids':['pichia_GQ67_00707','pichia_GQ67_03452',
                 'pichia_GQ67_02802','pichia_GQ67_02027'],
     'co_org':'pichia', 'ct_id':None,
     'raw':['data/raw/pichia/pichia_train.csv','data/raw/pichia/pichia_val.csv']},
]

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA, BUILD CPB TABLES, EXTRACT GENE SEQUENCES
# ══════════════════════════════════════════════════════════════════════════════
print('Building CPB tables and extracting genes...')
for cfg in ORGS:
    dfs=[pd.read_csv(BASE/p) for p in cfg['raw'] if (BASE/p).exists()]
    if not dfs: cfg['cpb']={}; cfg['genes']=[]; continue
    df=pd.concat(dfs,ignore_index=True)
    dna_col=next(c for c in df.columns if 'dna' in c.lower())
    id_col =next(c for c in df.columns if 'sequence_id' in c.lower())

    # Build CPB table
    pair_cnt=defaultdict(int); cod_cnt=defaultdict(int); tot_pairs=0
    for dna in df[dna_col].dropna():
        dna=str(dna).upper().replace(' ',''); n=len(dna)//3
        cod=[dna[i*3:(i+1)*3] for i in range(n)]
        for c in cod: cod_cnt[c]+=1
        for i in range(n-1): pair_cnt[(cod[i],cod[i+1])]+=1; tot_pairs+=1
    total_cod=sum(cod_cnt.values())
    cpb={}
    for (c1,c2),cnt in pair_cnt.items():
        obs=cnt/tot_pairs; exp=(cod_cnt[c1]/total_cod)*(cod_cnt[c2]/total_cod)
        cpb[(c1,c2)]=math.log2(obs/exp) if (obs>0 and exp>0) else 0.0
    cfg['cpb']=cpb

    # Extract the 4 benchmark genes (for CO/CT rows)
    genes=[]
    for gid in cfg['gene_ids']:
        row=df[df[id_col]==gid]
        if row.empty:
            print(f'  WARNING: {gid} not found'); continue
        r=row.iloc[0]; dna=str(r[dna_col]).upper().replace(' ','')
        aa=dna_to_aa(dna)
        cod=[dna[i*3:(i+1)*3] for i in range(len(dna)//3)]
        m=np.mean([cpb.get((cod[i],cod[i+1]),0.0) for i in range(len(cod)-1)])
        genes.append({'id':gid,'dna':dna,'aa':aa,'native_cpb':m})
    cfg['genes']=genes
    means=[g['native_cpb'] for g in genes]
    print(f"  {cfg['org']}: {len(genes)} genes, native CPB mean={np.mean(means):+.3f} "
          f"[{np.min(means):+.3f}–{np.max(means):+.3f}]")

    # Top-10% tAI native genes for Native row — high-expression, consistently pos CPB
    N_NATIVE = 50
    tai_col = next((c for c in df.columns if 'tai' in c.lower()), None)
    cand = df[(df[dna_col].str.len() >= 450) & (df[dna_col].str.len() <= 1200)].copy()
    if tai_col and tai_col in cand.columns:
        thresh = cand[tai_col].quantile(0.90)
        hiexp = cand[cand[tai_col] >= thresh]
        if len(hiexp) >= N_NATIVE:
            cand = hiexp.sample(n=N_NATIVE, random_state=42)
        else:
            cand = hiexp
        print(f"  {cfg['org']}: top-10% tAI threshold={thresh:.3f}, n={len(cand)} hi-expr genes")
    else:
        cand = cand.sample(n=min(N_NATIVE, len(cand)), random_state=42)
        print(f"  {cfg['org']}: no tAI col, random sample n={len(cand)}")
    cfg['native_dnas'] = [str(r[dna_col]).upper().replace(' ','')
                          for _, r in cand.iterrows()]

# ══════════════════════════════════════════════════════════════════════════════
# 2. GENERATE CO SEQUENCES (ep11, greedy)
# ══════════════════════════════════════════════════════════════════════════════
print('\nGenerating CodonOptimus sequences (CSI-FT all39)...')
ckpt=torch.load(BASE/'models/industrial_mlm_csi_all39.pt',map_location='cpu',weights_only=False)
co_model=IndustrialMLM(use_grad_ckpt=False)
co_model.load_state_dict(ckpt['model']); co_model.eval()

def co_gen(aa,org):
    tok=[CLS_TOKEN]+[AA_MASK[a] for a in aa.upper() if a in AA_MASK]
    ids=torch.tensor([tok],dtype=torch.long)
    msk=torch.zeros(1,len(tok),dtype=torch.bool)
    oid=torch.tensor([ORG_TO_ID[org]],dtype=torch.long)
    with torch.no_grad(): logits=co_model(ids,msk,oid)
    return ''.join(ID_TO_PAIR[t.item()][1] for t in logits[0,1:,:61].argmax(-1))

for cfg in ORGS:
    for g in cfg['genes']:
        g['co_dna']=co_gen(g['aa'],cfg['co_org'])
        cod=[g['co_dna'][i*3:(i+1)*3] for i in range(len(g['co_dna'])//3)]
        g['co_cpb']=np.mean([cfg['cpb'].get((cod[i],cod[i+1]),0.0) for i in range(len(cod)-1)])
    vals=[g['co_cpb'] for g in cfg['genes']]
    print(f"  {cfg['org']}: CO CPB mean={np.mean(vals):+.3f} [{np.min(vals):+.3f}–{np.max(vals):+.3f}]")

# ══════════════════════════════════════════════════════════════════════════════
# 3. GENERATE CT SEQUENCES
# ══════════════════════════════════════════════════════════════════════════════
print('\nGenerating CodonTransformer sequences...')
try:
    from CodonTransformer.CodonPrediction import predict_dna_sequence,load_model,load_tokenizer
    ct_model=load_model(); ct_tok=load_tokenizer()
    for cfg in ORGS:
        if cfg['ct_id'] is None:
            for g in cfg['genes']: g['ct_dna']=None
            continue
        for g in cfg['genes']:
            r=predict_dna_sequence(protein=g['aa'],organism=cfg['ct_id'],
                                   device=torch.device('cpu'),model=ct_model,
                                   tokenizer=ct_tok,deterministic=True)
            dna=r.predicted_dna.replace(' ','').upper()
            if dna[-3:] in STOP_C: dna=dna[:-3]
            g['ct_dna']=dna
            cod=[dna[i*3:(i+1)*3] for i in range(len(dna)//3)]
            g['ct_cpb']=np.mean([cfg['cpb'].get((cod[i],cod[i+1]),0.0) for i in range(len(cod)-1)])
        vals=[g['ct_cpb'] for g in cfg['genes']]
        print(f"  {cfg['org']}: CT CPB mean={np.mean(vals):+.3f} [{np.min(vals):+.3f}–{np.max(vals):+.3f}]")
except Exception as e:
    print(f'CT error: {e}')
    for cfg in ORGS:
        for g in cfg['genes']: g['ct_dna']=None

# ══════════════════════════════════════════════════════════════════════════════
# 4. CPB PROFILE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def cpb_profile(dna,cpb,window=WINDOW):
    dna=str(dna).upper().replace(' ','')
    if len(dna)>3 and dna[-3:] in STOP_C: dna=dna[:-3]
    n=len(dna)//3; cod=[dna[i*3:(i+1)*3] for i in range(n)]
    raw=np.array([cpb.get((cod[i],cod[i+1]),0.0) for i in range(n-1)],dtype=float)
    if window>1 and len(raw)>=window:
        k=np.ones(window)/window; sm=np.convolve(raw,k,mode='same')
        e=window//2; sm[:e]=np.nan; sm[-e:]=np.nan; return sm
    return raw

def band_stats(profiles):
    """Align profiles to min length, return x, mean, p25, p75."""
    min_len=min(len(p) for p in profiles)
    arr=np.array([p[:min_len] for p in profiles])  # (n_genes, min_len)
    x=np.arange(min_len)
    mn=np.nanmean(arr,axis=0)
    p25=np.nanpercentile(arr,25,axis=0)
    p75=np.nanpercentile(arr,75,axis=0)
    return x,mn,p25,p75

# ══════════════════════════════════════════════════════════════════════════════
# 5. FIGURE
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Helvetica','Arial','DejaVu Sans'],
    'font.size':8.5,'axes.labelsize':9,'axes.titlesize':9.5,
    'xtick.labelsize':7.5,'ytick.labelsize':7.5,'axes.linewidth':0.8,
    'pdf.fonttype':42,'ps.fonttype':42,
})

TOOL_COLOR={'CodonOptimus':'#0072B2','CodonTransformer':'#888888',
            'IDT':'#E69F00','Twist':'#D55E00','Genscript':'#009E73','WT':'#BBBBBB'}
TOOLS_BOX=['WT','IDT','Genscript','Twist','CodonTransformer','CodonOptimus']
TOOL_RENAME={'WT':'Unoptimized','CodonTransformer':'CT','CodonOptimus':'CO',
             'IDT':'IDT','Genscript':'GS','Twist':'Twist'}
PROT_MARKERS={'BLG':'o','HSA':'s','PHYA':'^','XYN2':'D'}
ORG_LABEL={'ecoli':'E. coli','bacillus':'B. subtilis',
           's_cerevisiae':'S. cerevisiae','pichia':'K. phaffii'}

ROWS=[
    ('Native\n(top-10% tAI)', '#C8860A', 'native'),   # amber — high-expression native
    ('CodonOptimus',           '#0072B2', 'co'),        # dark blue
    ('CodonTransformer',       '#888888', 'ct'),        # grey — consistent with all figures
]
YLIM=(-0.42,0.50); YTICKS=[-0.4,-0.2,0.0,0.2,0.4]

fig=plt.figure(figsize=(7.0,8.2))
gs_main=gridspec.GridSpec(2,1,figure=fig,height_ratios=[2.0,1.0],
                           hspace=0.52,left=0.10,right=0.88,top=0.88,bottom=0.09)
gs_A=gridspec.GridSpecFromSubplotSpec(3,4,subplot_spec=gs_main[0],hspace=0.20,wspace=0.22)

for ri,(row_name,col,seq_key) in enumerate(ROWS):
    row_means=[]

    for ci,cfg in enumerate(ORGS):
        ax=fig.add_subplot(gs_A[ri,ci])

        # Background shading: green=good, red=harmful
        ax.axhspan(0,0.6,  color='#EEFFEE',alpha=0.25,zorder=0)
        ax.axhspan(-0.5,0, color='#FFEEEE',alpha=0.25,zorder=0)
        ax.axhline(0,color='#555555',lw=0.9,ls='--',zorder=1)

        # Get DNA sequences for this tool
        if seq_key=='native':
            dnas=cfg.get('native_dnas', [g['dna'] for g in cfg['genes']])
        elif seq_key=='co':
            dnas=[g['co_dna'] for g in cfg['genes'] if g.get('co_dna')]
        else:
            dnas=[g['ct_dna'] for g in cfg['genes'] if g.get('ct_dna')]

        if dnas:
            # Compute per-gene profiles
            profs=[cpb_profile(d,cfg['cpb']) for d in dnas]
            x,mn,p25,p75=band_stats(profs)

            # Individual gene traces (thin, low alpha)
            for p in profs:
                ax.plot(np.arange(len(p[:len(x)])),p[:len(x)],
                        color=col,lw=0.6,alpha=0.20,zorder=2)

            # IQR band + mean line
            ax.fill_between(x,p25,p75,color=col,alpha=0.25,zorder=3)
            ax.plot(x,mn,color=col,lw=1.8,alpha=0.95,zorder=4)

            # Mean CPB annotation
            gene_means=[np.nanmean(cpb_profile(d,cfg['cpb'])) for d in dnas]
            m=np.mean(gene_means); row_means.append(m)
            ax.text(0.97,0.05,f'{m:+.3f}',transform=ax.transAxes,
                    ha='right',va='bottom',fontsize=7,color=col,fontweight='bold')

        elif seq_key=='ct' and cfg['ct_id'] is None:
            ax.set_axis_off()
            h0=mlines.Line2D([],[],color='#555',lw=1.2,ls='--',label='CPB = 0 (random)')
            h1=mpatches.Patch(facecolor='#EEFFEE',alpha=0.8,edgecolor='#AACCAA',label='CPB > 0 (favourable)')
            h2=mpatches.Patch(facecolor='#FFEEEE',alpha=0.8,edgecolor='#CCAAAA',label='CPB < 0 (harmful)')
            h3=mpatches.Patch(facecolor='#888888',alpha=0.35,edgecolor='none',label='IQR across genes')
            hnat=mlines.Line2D([],[],color='#C8860A',lw=1.5,label='Native (top-10% tAI)')
            hco =mlines.Line2D([],[],color='#0072B2',lw=1.5,label='CodonOptimus')
            hct =mlines.Line2D([],[],color='#888888',lw=1.5,label='CodonTransformer')
            leg=ax.legend(handles=[hnat,hco,hct,h0,h1,h2,h3],fontsize=7.0,loc='center',
                          framealpha=0.95,edgecolor='#CCCCCC',frameon=True,
                          handlelength=1.6,handleheight=1.0,labelspacing=0.5)
            ax.text(0.5,0.07,'CT not available for $\\it{K.\\ phaffii}$',
                    transform=ax.transAxes,ha='center',va='bottom',
                    fontsize=7,color='#888',style='italic')
            continue

        ax.set_ylim(YLIM); ax.set_yticks(YTICKS)
        ax.set_yticklabels([f'{v:.1f}' if ci==0 else '' for v in YTICKS],fontsize=6.5)
        ax.set_xlim(left=0)
        if ri==0:
            n_lbl = len(cfg.get('native_dnas', cfg['genes']))
            ax.set_title(cfg['col_label']+f'\n(n={n_lbl} hi-expr genes)',fontsize=8.5,pad=5,fontweight='bold')
        ax.set_xlabel('Codon position' if ri==2 else '',fontsize=7.5)
        if ri<2: ax.set_xticklabels([])
        for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.8)

        # Panel A label on first subplot only
        if ri==0 and ci==0:
            ax.text(-0.38,1.22,'A',transform=ax.transAxes,
                    fontsize=14,fontweight='bold',ha='left',va='bottom')

    # Right-side row label
    last_ax=fig.axes[ri*4+3] if len(fig.axes)>ri*4+3 else fig.axes[-1]
    yc=last_ax.get_position().y0+last_ax.get_position().height/2
    short=row_name.replace('CodonTransformer','CT').replace('CodonOptimus','CO')
    fig.text(0.895,yc+0.014,short,va='center',ha='left',
             fontsize=7.5,fontweight='bold',color=col,linespacing=1.3)
    if row_means:
        n_label=f'  (n={len(row_means)} orgs)' if len(row_means)<4 else ''
        fig.text(0.895,yc-0.016,f'mean={np.mean(row_means):+.3f}{n_label}',
                 va='center',ha='left',fontsize=6.5,color='#444')

fig.text(0.018,
         gs_main[0].get_position(fig).y0+gs_main[0].get_position(fig).height/2,
         'CPB score  (host-organism-specific)',
         va='center',ha='center',fontsize=8,rotation=90,color='#444')

# ── Panel B: aggregate CPB boxplots — 4 heterologous proteins ────────────────
gs_B=gridspec.GridSpecFromSubplotSpec(1,4,subplot_spec=gs_main[1],wspace=0.30)
df_bench=pd.read_csv(BASE/'other_optimizers/benchmark/benchmark_results_tai.csv')
df_bench['protein']=df_bench['protein'].str.upper()

for ci,org in enumerate(['ecoli','bacillus','s_cerevisiae','pichia']):
    ax=fig.add_subplot(gs_B[ci])
    org_df=df_bench[df_bench['organism']==org].copy()
    present=[t for t in TOOLS_BOX if t in org_df['optimizer'].values]
    pal={t:TOOL_COLOR[t] for t in present}
    sns.boxplot(data=org_df[org_df['optimizer'].isin(present)][['optimizer','mean_cpb']],
                x='optimizer',y='mean_cpb',ax=ax,order=present,palette=pal,
                width=0.55,linewidth=0.8,fliersize=0,saturation=0.95,zorder=2,
                hue='optimizer',legend=False)
    ax.axhline(0,color='#555',lw=0.8,ls='--',zorder=1)
    rng=np.random.default_rng(hash(org)%2**31)
    for xi,tool in enumerate(present):
        for prot,marker in PROT_MARKERS.items():
            row=org_df[(org_df['optimizer']==tool)&(org_df['protein']==prot)]
            if row.empty: continue
            val=row['mean_cpb'].values[0]
            if pd.isna(val): continue
            ax.scatter(xi+rng.uniform(-0.15,0.15),val,color=TOOL_COLOR[tool],marker=marker,
                       s=55 if tool=='CodonOptimus' else 35,
                       alpha=1.0 if tool=='CodonOptimus' else 0.78,
                       zorder=5,edgecolors='white',linewidths=0.4)
    # Significance CO vs best competitor
    if 'CodonOptimus' in present:
        others=[t for t in present if t!='CodonOptimus']
        best=max(others,key=lambda t:org_df[org_df['optimizer']==t]['mean_cpb'].mean()
                 if len(org_df[org_df['optimizer']==t])>0 else -np.inf,default=None)
        if best:
            co_v=[org_df[(org_df['optimizer']=='CodonOptimus')&(org_df['protein']==p)]['mean_cpb'].values for p in PROT_MARKERS]
            bv  =[org_df[(org_df['optimizer']==best)&(org_df['protein']==p)]['mean_cpb'].values for p in PROT_MARKERS]
            pairs=[(a[0],b[0]) for a,b in zip(co_v,bv) if len(a) and len(b)]
            if len(pairs)>=2:
                try: _,pval=stats.ttest_rel([p[0] for p in pairs],[p[1] for p in pairs])
                except: pval=1.0
                if pval<0.05:
                    pstr='***' if pval<0.001 else '**' if pval<0.01 else '*'
                    av=org_df['mean_cpb'].dropna().values
                    ymax=np.nanmax(av); yr=ymax-np.nanmin(av); yb=ymax+yr*0.12
                    x1,x2=present.index(best),present.index('CodonOptimus')
                    ax.plot([x1,x1,x2,x2],[yb,yb+yr*0.02,yb+yr*0.02,yb],lw=0.9,color='#444',zorder=6)
                    ax.text((x1+x2)/2,yb+yr*0.03,pstr,ha='center',va='bottom',fontsize=10,fontweight='bold')
                    ax.set_ylim(top=yb+yr*0.22)
    parts=ORG_LABEL[org].split()
    ax.set_title(' '.join(f'$\\it{{{p}}}$' for p in parts),fontsize=8.5,pad=4)
    ax.set_xlabel('')
    ticks=ax.get_xticks(); ax.set_xticks(ticks)
    ax.set_xticklabels([TOOL_RENAME.get(t,t) for t in present],rotation=40,ha='right',fontsize=7)
    ax.set_ylabel('CPB — 4 heterologous proteins\n(Unoptimized = source organism codons)' if ci==0 else '',fontsize=7)
    if ci==0:
        ax.text(-0.60,1.14,'B',transform=ax.transAxes,fontsize=14,fontweight='bold',ha='left',va='bottom')
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.8)

prot_handles=[mlines.Line2D([],[],marker=m,color='#555',markersize=6,linewidth=0,label=p)
              for p,m in PROT_MARKERS.items()]
fig.legend(handles=prot_handles,loc='lower center',ncol=4,
           bbox_to_anchor=(0.50,0.005),fontsize=7.5,frameon=True,
           framealpha=0.95,edgecolor='#CCC',handlelength=1.2)

fig.suptitle(
    'CodonOptimus generates host-compatible codon pair contexts\n'
    'that commercial tools and CodonTransformer systematically fail to reproduce',
    fontsize=10,y=0.96,style='italic')

fig.savefig(OUT_PNG,dpi=300,bbox_inches='tight')
fig.savefig(OUT_PDF,bbox_inches='tight')
fig.savefig(OUT_SVG,format='svg',bbox_inches='tight')
print(f'\nSaved → {OUT_PNG}')
