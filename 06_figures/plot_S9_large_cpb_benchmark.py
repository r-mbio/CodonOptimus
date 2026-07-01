#!/usr/bin/env python3
"""
Supplementary Figure S9: Large-scale CPB benchmark — 489 proteins.
Style matches Fig 3 exactly: seaborn boxplot + strip, ticks style.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from pathlib import Path

sns.set_style('ticks')
sns.set_context('paper', font_scale=1.45)

BASE = Path(__file__).resolve().parent.parent
OUT  = BASE / 'figures/figS9_large_cpb_benchmark.pdf'
OUTP = BASE / 'figures/figS9_large_cpb_benchmark.png'
OUTS = BASE / 'figures/figS9_large_cpb_benchmark.svg'

TOOL_COLORS = {
    'CodonOptimus':     '#0072B2',   # CO dark blue
    'CodonTransformer': '#888888',   # CT grey
}
TOOLS_ORDER = ['CodonTransformer', 'CodonOptimus']

ORGS = ['ecoli', 'bacillus', 's_cerevisiae']   # K.phaffii excluded — CT not supported
ORG_LABELS = {
    'ecoli':        'E. coli',
    'bacillus':     'B. subtilis',
    's_cerevisiae': 'S. cerevisiae',
}
METRICS = [
    ('cpb',       'Codon Pair Bias (CPB)',            'A'),
    ('nterm_cpb', 'N-terminal Codon Pair Bias', 'B'),
]

df = pd.read_csv(BASE / 'results/OLD_RESULTS/large_benchmark_combined.csv')

fig, axes = plt.subplots(
    len(METRICS), len(ORGS),
    figsize=(11, 9),
    gridspec_kw={'hspace': 0.65, 'wspace': 0.34},
)

for row_i, (metric, ylabel, panel_letter) in enumerate(METRICS):
    for col_i, org in enumerate(ORGS):
        ax = axes[row_i, col_i]
        org_df = df[df['organism'] == org].copy()
        present = [t for t in TOOLS_ORDER if t in org_df['tool'].values]
        pal = {t: TOOL_COLORS[t] for t in present}

        # ── Boxplot (seaborn) ─────────────────────────────────────────────────
        plot_df = org_df[org_df['tool'].isin(present)][['tool', metric]].copy()
        sns.boxplot(
            data=plot_df, x='tool', y=metric, ax=ax,
            order=present, palette=pal,
            width=0.52, linewidth=1.0, fliersize=0,
            saturation=0.90, zorder=2,
        )

        # ── Strip (individual proteins, small dots) ───────────────────────────
        rng = np.random.default_rng(hash(org + metric) % 2**31)
        for xi, tool in enumerate(present):
            vals = org_df[org_df['tool'] == tool][metric].dropna().values
            jitter = rng.uniform(-0.18, 0.18, len(vals))
            is_co = (tool == 'CodonOptimus')
            ax.scatter(
                xi + jitter, vals,
                color=TOOL_COLORS[tool],
                s=6 if is_co else 4,
                alpha=0.18 if is_co else 0.12,
                zorder=3, edgecolors='none',
            )

        # ── Significance bracket (CO vs CT) ───────────────────────────────────
        if 'CodonOptimus' in present and 'CodonTransformer' in present:
            co_vals = org_df[org_df['tool']=='CodonOptimus'][metric].dropna().values
            ct_vals = org_df[org_df['tool']=='CodonTransformer'][metric].dropna().values
            _, pval = stats.ttest_ind(co_vals, ct_vals)
            pstr = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
            all_vals = org_df[metric].dropna().values
            y_top = np.nanmax(all_vals)
            y_rng = np.nanmax(all_vals) - np.nanmin(all_vals)
            yb = y_top + y_rng * 0.12
            x1 = present.index('CodonTransformer')
            x2 = present.index('CodonOptimus')
            ax.plot([x1, x1, x2, x2],
                    [yb, yb + y_rng*0.02, yb + y_rng*0.02, yb],
                    lw=1.1, color='#444', zorder=6)
            ax.text((x1+x2)/2, yb + y_rng*0.03, pstr,
                    ha='center', va='bottom',
                    fontsize=13, color='#333', fontweight='bold')
            ax.set_ylim(top=yb + y_rng * 0.22)

        # ── Axes cleanup ──────────────────────────────────────────────────────
        ax.set_xlabel('')
        ax.set_xticklabels(
            [t.replace('CodonTransformer','CT').replace('CodonOptimus','CO')
             for t in present],
            rotation=30, ha='right', fontsize=11
        )
        ax.tick_params(axis='y', labelsize=11)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.9)

        if row_i == 0:
            parts = ORG_LABELS[org].split()
            ax.set_title(' '.join(f'$\\it{{{p}}}$' for p in parts),
                         fontsize=13, pad=8)

        if col_i == 0:
            ax.set_ylabel(ylabel, fontsize=12, labelpad=6)
            ax.text(-0.30, 1.08, panel_letter,
                    transform=ax.transAxes,
                    ha='left', va='bottom', fontsize=18, fontweight='bold')
        else:
            ax.set_ylabel('')


# ── Legend ────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(facecolor=TOOL_COLORS[t], alpha=0.90, label=t)
    for t in ['CodonOptimus', 'CodonTransformer']
]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.01),
           ncol=2, fontsize=12, frameon=False,
           handlelength=1.5, handletextpad=0.6)

fig.suptitle(
    'CodonOptimus achieves superior codon pair bias across 489 industrially relevant proteins\n'
    '(n = 489 proteins per organism per tool)',
    fontsize=11.5, y=1.01, color='#222'
)
fig.savefig(OUT,  dpi=300, bbox_inches='tight')
fig.savefig(OUTP, dpi=300, bbox_inches='tight')
fig.savefig(OUTS, format='svg', bbox_inches='tight')
print(f'Saved → {OUTP}')
