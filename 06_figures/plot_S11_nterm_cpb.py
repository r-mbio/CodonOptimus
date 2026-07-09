#!/usr/bin/env python3
"""
Supplementary Figure — N-terminal Codon Pair Bias (first 30 codons)

N-terminal CPB is critical for translation initiation efficiency.
Negative N-terminal CPB reduces expression >1000× (Coleman et al. 2008 Science).

Panel A: Violin plots — 100 native highly-expressed genes per organism
         Native (host org.) | CodonOptimus | CodonTransformer
         Data: results/native_cpb_100genes.csv

Panel B: Boxplots — 4 heterologous benchmark proteins
         All tools, 4 organisms

Output: figures/figS11_nterm_cpb.{png,pdf,svg}
"""

import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy import stats

BASE    = Path(__file__).resolve().parent.parent
OUT_PNG = BASE / 'figures/figS11_nterm_cpb.png'
OUT_PDF = BASE / 'figures/figS11_nterm_cpb.pdf'
OUT_SVG = BASE / 'figures/figS11_nterm_cpb.svg'

TOOL_COLOR = {
    'CodonOptimus':     '#0072B2',
    'CodonTransformer': '#888888',
    'IDT':              '#E69F00',
    'Twist':            '#D55E00',
    'Genscript':        '#009E73',
    'WT':               '#BBBBBB',
    'Native\n(host org.)': '#AAAAAA',
}
TOOLS_BOX    = ['WT','IDT','Genscript','Twist','CodonTransformer','CodonOptimus']
TOOL_RENAME  = {'WT':'Unoptimized','CodonTransformer':'CT','CodonOptimus':'CO',
                'IDT':'IDT','Genscript':'GS','Twist':'Twist'}
PROT_MARKERS = {'BLG':'o','HSA':'s','PHYA':'^','XYN2':'D'}
ORG_LABEL    = {'ecoli':'E. coli','bacillus':'B. subtilis',
                's_cerevisiae':'S. cerevisiae','pichia':'K. phaffii'}
ORG_ORDER    = ['E. coli','B. subtilis','S. cerevisiae','K. phaffii']

VIOLIN_COLORS = {
    'Native\n(host org.)': '#AAAAAA',
    'CodonOptimus':        '#0072B2',
    'CodonTransformer':    '#888888',
}

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Helvetica','Arial','DejaVu Sans'],
    'font.size':9,'axes.labelsize':10,'axes.titlesize':10.5,
    'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.linewidth':0.9,
    'pdf.fonttype':42,'ps.fonttype':42,
})

# ── Data ──────────────────────────────────────────────────────────────────────
df_nat  = pd.read_csv(BASE/'results/OLD_RESULTS/native_cpb_100genes.csv')
df_het  = pd.read_csv(BASE/'other_optimizers/benchmark/benchmark_results_tai.csv')
df_het['protein'] = df_het['protein'].str.upper()

# ── Layout ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 8.0))
gs  = gridspec.GridSpec(2, 4, figure=fig,
                        height_ratios=[1.15, 1.0],
                        hspace=0.55, wspace=0.30,
                        left=0.07, right=0.97, top=0.91, bottom=0.11)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — violin plots, 100 native genes
# ══════════════════════════════════════════════════════════════════════════════
for ci, org_label in enumerate(ORG_ORDER):
    ax  = fig.add_subplot(gs[0, ci])
    sub = df_nat[df_nat['org_label'] == org_label].copy()
    n   = len(sub)

    # Build long-form
    rows = []
    for _, r in sub.iterrows():
        rows.append({'tool': 'Native\n(host org.)', 'cpb': r['native_nterm_cpb']})
        rows.append({'tool': 'CodonOptimus',        'cpb': r['co_nterm_cpb']})
        if not np.isnan(r['ct_nterm_cpb']):
            rows.append({'tool': 'CodonTransformer', 'cpb': r['ct_nterm_cpb']})
    df_long  = pd.DataFrame(rows).dropna(subset=['cpb'])
    tool_ord = [t for t in ['Native\n(host org.)','CodonOptimus','CodonTransformer']
                if t in df_long['tool'].unique()]
    pal = {t: VIOLIN_COLORS[t] for t in tool_ord}

    sns.violinplot(data=df_long, x='tool', y='cpb', order=tool_ord,
                   palette=pal, hue='tool', legend=False,
                   ax=ax, inner='box', cut=0.5, linewidth=0.8, saturation=0.85)

    ax.axhline(0, color='#444', lw=1.0, ls='--', zorder=5)

    # Significance
    def sig(p):
        return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'

    ymax = df_long['cpb'].max(); yrng = ymax - df_long['cpb'].min()

    # CO vs Native
    if 'CodonOptimus' in tool_ord and 'Native\n(host org.)' in tool_ord:
        co_v  = sub['co_nterm_cpb'].dropna().values
        nat_v = sub['native_nterm_cpb'].dropna().values
        n_p   = min(len(co_v), len(nat_v))
        if n_p >= 10:
            _, pv = stats.wilcoxon(co_v[:n_p], nat_v[:n_p])
            xi_co  = tool_ord.index('CodonOptimus')
            xi_nat = tool_ord.index('Native\n(host org.)')
            yb = ymax + yrng*0.10
            ax.plot([xi_nat,xi_nat,xi_co,xi_co],[yb,yb+yrng*0.02,yb+yrng*0.02,yb],
                    lw=0.9,color='#333',zorder=6)
            ax.text((xi_nat+xi_co)/2, yb+yrng*0.03, sig(pv),
                    ha='center',va='bottom',fontsize=11,fontweight='bold')
            ax.set_ylim(top=yb+yrng*0.28)

    # CO vs CT
    if 'CodonOptimus' in tool_ord and 'CodonTransformer' in tool_ord:
        co_v = sub['co_nterm_cpb'].dropna().values
        ct_v = sub['ct_nterm_cpb'].dropna().values
        n_p  = min(len(co_v),len(ct_v))
        if n_p >= 10:
            _, pv = stats.wilcoxon(co_v[:n_p], ct_v[:n_p])
            xi_co = tool_ord.index('CodonOptimus')
            xi_ct = tool_ord.index('CodonTransformer')
            ylim_top = ax.get_ylim()[1]
            yb = ylim_top * 0.88
            ax.plot([xi_ct,xi_ct,xi_co,xi_co],[yb,yb+yrng*0.02,yb+yrng*0.02,yb],
                    lw=0.9,color='#333',zorder=6)
            ax.text((xi_ct+xi_co)/2, yb+yrng*0.03, sig(pv),
                    ha='center',va='bottom',fontsize=11,fontweight='bold')

    ax.text(0.97, 0.02, f'n={n}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8, color='#555')

    parts = org_label.split()
    ax.set_title(' '.join(f'$\\it{{{p}}}$' for p in parts), fontsize=10.5, pad=5)
    ax.set_xlabel('')
    ax.set_xticklabels(
        [t.replace('\n(host org.)','').replace('CodonTransformer','CT')
          .replace('CodonOptimus','CO') for t in tool_ord],
        fontsize=8.5, rotation=25, ha='right'
    )
    ax.set_ylabel('N-terminal CPB\n(first 30 codons, native genes)' if ci==0 else '',
                  fontsize=9.5)
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.8)
    if ci == 0:
        ax.text(-0.35, 1.14, 'A', transform=ax.transAxes,
                fontsize=14, fontweight='bold', ha='left', va='bottom')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — boxplots, 4 heterologous benchmark proteins
# ══════════════════════════════════════════════════════════════════════════════
for ci, org in enumerate(['ecoli','bacillus','s_cerevisiae','pichia']):
    ax     = fig.add_subplot(gs[1, ci])
    org_df = df_het[df_het['organism']==org].copy()
    present= [t for t in TOOLS_BOX if t in org_df['optimizer'].values]
    pal    = {t: TOOL_COLOR[t] for t in present}

    sns.boxplot(
        data=org_df[org_df['optimizer'].isin(present)][['optimizer','nterm_cpb']],
        x='optimizer', y='nterm_cpb', ax=ax,
        order=present, palette=pal, hue='optimizer', legend=False,
        width=0.55, linewidth=0.8, fliersize=0, saturation=0.95, zorder=2,
    )
    ax.axhline(0, color='#444', lw=0.9, ls='--', zorder=1)

    rng = np.random.default_rng(hash(org+'nt') % 2**31)
    for xi, tool in enumerate(present):
        for prot, marker in PROT_MARKERS.items():
            row = org_df[(org_df['optimizer']==tool)&(org_df['protein']==prot)]
            if row.empty: continue
            val = row['nterm_cpb'].values[0]
            if pd.isna(val): continue
            ax.scatter(xi + rng.uniform(-0.15,0.15), val,
                       color=TOOL_COLOR[tool], marker=marker,
                       s=60 if tool=='CodonOptimus' else 38,
                       alpha=1.0 if tool=='CodonOptimus' else 0.80,
                       zorder=5, edgecolors='white', linewidths=0.4)

    # Significance CO vs best competitor
    if 'CodonOptimus' in present:
        others = [t for t in present if t!='CodonOptimus']
        best   = max(others,
                     key=lambda t: org_df[org_df['optimizer']==t]['nterm_cpb'].mean()
                     if len(org_df[org_df['optimizer']==t])>0 else -np.inf,
                     default=None)
        if best:
            co_v  = [org_df[(org_df['optimizer']=='CodonOptimus')&(org_df['protein']==p)]['nterm_cpb'].values for p in PROT_MARKERS]
            best_v= [org_df[(org_df['optimizer']==best)&(org_df['protein']==p)]['nterm_cpb'].values for p in PROT_MARKERS]
            pairs = [(a[0],b[0]) for a,b in zip(co_v,best_v) if len(a) and len(b)]
            if len(pairs)>=2:
                try: _,pv=stats.ttest_rel([p[0] for p in pairs],[p[1] for p in pairs])
                except: pv=1.0
                if pv<0.05:
                    ps='***' if pv<0.001 else '**' if pv<0.01 else '*'
                    av=org_df['nterm_cpb'].dropna().values
                    ym=np.nanmax(av); yr=ym-np.nanmin(av); yb=ym+yr*0.12
                    x1,x2=present.index(best),present.index('CodonOptimus')
                    ax.plot([x1,x1,x2,x2],[yb,yb+yr*0.02,yb+yr*0.02,yb],
                            lw=0.9,color='#444',zorder=6)
                    ax.text((x1+x2)/2,yb+yr*0.03,ps,
                            ha='center',va='bottom',fontsize=11,fontweight='bold')
                    ax.set_ylim(top=yb+yr*0.22)

    parts = ORG_LABEL[org].split()
    ax.set_title(' '.join(f'$\\it{{{p}}}$' for p in parts), fontsize=10.5, pad=5)
    ax.set_xlabel('')
    ticks=ax.get_xticks(); ax.set_xticks(ticks)
    ax.set_xticklabels([TOOL_RENAME.get(t,t) for t in present],
                       rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('N-terminal CPB\n(first 30 codons, 4 heterologous proteins)'
                  if ci==0 else '', fontsize=9.5)
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.8)
    if ci==0:
        ax.text(-0.35, 1.14, 'B', transform=ax.transAxes,
                fontsize=14, fontweight='bold', ha='left', va='bottom')

# ── Shared legends ────────────────────────────────────────────────────────────
# Panel A violin legend
h_viol = [mpatches.Patch(facecolor=VIOLIN_COLORS[t], alpha=0.85,
                          label=t.replace('\n(host org.)','').replace('\n',' '))
          for t in ['Native\n(host org.)','CodonOptimus','CodonTransformer']]
h_ref  = mlines.Line2D([],[],color='#444',lw=1.0,ls='--',label='CPB = 0 (random)')
# Panel B protein markers
h_prot = [mlines.Line2D([],[],marker=m,color='#555',markersize=7,
                         linewidth=0, label=p)
          for p,m in PROT_MARKERS.items()]

fig.legend(handles=h_viol + [h_ref] + h_prot,
           loc='lower center', ncol=8,
           bbox_to_anchor=(0.5, 0.01),
           fontsize=8.5, frameon=True, framealpha=0.95,
           edgecolor='#CCC', handlelength=1.6, columnspacing=1.0)


fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
fig.savefig(OUT_PDF, bbox_inches='tight')
fig.savefig(OUT_SVG, format='svg', bbox_inches='tight')
print(f'Saved → {OUT_PNG}')
