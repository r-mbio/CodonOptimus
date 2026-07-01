#!/usr/bin/env python3
"""
Supplementary Figure: Dual-head expression predictor R² per organism.
Shows how well codon usage alone predicts mRNA expression (Ribo-seq TE)
across the four main organisms.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT  = BASE / 'figures/figS6_dualhead_r2.png'
OUTP = BASE / 'figures/figS6_dualhead_r2.pdf'
OUTS = BASE / 'figures/figS6_dualhead_r2.svg'

plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':        10,
    'axes.labelsize':   11,
    'axes.titlesize':   12,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'axes.linewidth':   0.9,
    'legend.fontsize':  9,
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'pdf.fonttype':     42,
})

# ── Dual-head R² data (verified by independent reruns, 2026-06-29) ────────────
ORGS = [
    ('E. coli',       0.355, 'ep11',    'PRJNA1335396+906596', 'normal growth'),
    ('B. subtilis',   0.309, 'pretrain','GSE249448',  'WT0'),
    ('K. phaffii',    0.553, 'ep11',    'GSE159336',  'abs Ribo'),
    ('S. cerevisiae', 0.511, 'pretrain','GSE185458',  'WT'),
]

labels     = [o[0] for o in ORGS]
r2_vals    = [o[1] for o in ORGS]
backbones  = [o[2] for o in ORGS]
datasets   = [o[3] for o in ORGS]
conditions = [o[4] for o in ORGS]

C_BAR     = '#0072B2'
C_LOW     = '#56B4E9'   # ep11 backbone
bar_colors = [C_LOW if bb == 'ep11' else C_BAR for bb in backbones]

fig, ax = plt.subplots(figsize=(7, 4.5))

x = np.arange(len(labels))
bars = ax.bar(x, r2_vals, color=bar_colors, width=0.55, alpha=0.88,
              edgecolor='#333', linewidth=0.8, zorder=3)

# Value labels on bars
for xi, (bar, r2) in enumerate(zip(bars, r2_vals)):
    ax.text(xi, r2 + 0.013, f'R² = {r2:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            color='#222')

# Dataset annotations below bars
for xi, (ds, cond) in enumerate(zip(datasets, conditions)):
    ax.text(xi, -0.030, f'{ds}\n({cond})',
            ha='center', va='top', fontsize=7.5, color='#666', style='italic')

ax.set_xticks(x)
ax.set_xticklabels([f'$\\it{{{l.split()[0]}}}$\n$\\it{{{l.split()[1]}}}$'
                     for l in labels], fontsize=10)
ax.set_ylabel('Validation R²\n(Ribo-seq translation efficiency)', fontsize=11)
ax.set_ylim(-0.09, 0.82)
ax.set_xlim(-0.55, len(labels) - 0.45)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, linestyle='--', alpha=0.45, linewidth=0.8)
ax.set_axisbelow(True)

# Backbone legend
patches = [
    mpatches.Patch(color=C_BAR, label='Foundation backbone (pretrain)'),
    mpatches.Patch(color=C_LOW, label='ep11 backbone'),
]
ax.legend(handles=patches, fontsize=8.5, frameon=True, framealpha=0.9,
          edgecolor='#CCC', loc='upper left')

ax.set_title(
    'Dual-head expression predictor: codon → Ribo-seq TE\n'
    'R² measures how well synonymous codon identity alone predicts translation efficiency',
    fontsize=10.5, pad=10, color='#333'
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(OUT,  dpi=200, bbox_inches='tight')
fig.savefig(OUTP, dpi=300, bbox_inches='tight')
fig.savefig(OUTS, format='svg', bbox_inches='tight')
print(f'Saved → {OUT}')
print(f'Saved → {OUTP}')
print(f'Saved → {OUTS}')
