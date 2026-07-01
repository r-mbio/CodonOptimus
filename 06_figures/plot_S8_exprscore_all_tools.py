#!/usr/bin/env python3
"""
Supplementary Figure S6 — ExprScore for all competitor sequences, 4 organisms.
Data: results/benchmark_consistent_v4.csv (canonical, new dual-head models).

K.phaffii is the PRIMARY panel (wet-lab validated).
S.cer and E.coli: ExprScore is unreliable for heterologous proteins (models trained
on native-gene expression; out-of-distribution); included for completeness with note.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
DF   = pd.read_csv(BASE / 'results' / 'benchmark_consistent_v4.csv')
OUT  = BASE / 'figures' / 'figS8_exprscore_all_tools.png'
OUTPDF = BASE / 'figures' / 'figS8_exprscore_all_tools.pdf'

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':9,'axes.labelsize':9,'axes.titlesize':9.5,
    'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':0.8,
    'figure.facecolor':'white','axes.facecolor':'white','pdf.fonttype':42,
})

TOOL_COLORS = {
    'CO Ribo-seq':      '#0072B2',
    'CO CSI-FT':        '#56B4E9',
    'CO Foundation':    '#A8D4F0',
    'CodonTransformer': '#888888',
    'IDT':              '#E69F00',
    'Twist':            '#D55E00',
    'Genscript':        '#009E73',
}
TOOL_ORDER = ['CO Foundation','CO CSI-FT','CO Ribo-seq','CodonTransformer','IDT','Twist','Genscript']

ORG_INFO = [
    ('pichia',       'K. phaffii',    False, 'PRIMARY — wet-lab validated'),
    ('bacillus',     'B. subtilis',   False, ''),
    ('s_cerevisiae', 'S. cerevisiae', True,  '[!] Out-of-distribution for heterologous proteins'),
    ('ecoli',        'E. coli',       True,  '[!] Out-of-distribution for heterologous proteins'),
]

fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), gridspec_kw={'wspace': 0.35})

for col, (org, org_label, unreliable, note) in enumerate(ORG_INFO):
    ax = axes[col]
    sub = DF[DF['organism'] == org]

    # Build per-tool distributions
    box_data, box_labels, box_colors = [], [], []
    for tool in TOOL_ORDER:
        td = sub[sub['optimizer'] == tool]['expr_score'].values
        if len(td) > 0:
            box_data.append(td)
            box_labels.append(tool.replace('CodonTransformer','CT').replace('CO Ribo-seq','CO-RS')
                                  .replace('CO CSI-FT','CO-CSI').replace('CO Foundation','CO-PT')
                                  .replace('Genscript','GS'))
            box_colors.append(TOOL_COLORS.get(tool, '#888'))

    bp = ax.boxplot(box_data, patch_artist=True, widths=0.5,
                    medianprops=dict(color='white', linewidth=2),
                    whiskerprops=dict(linewidth=1.0, color='#333'),
                    capprops=dict(linewidth=1.0, color='#333'),
                    boxprops=dict(linewidth=1.0),
                    flierprops=dict(marker='o', markersize=3, alpha=0.5,
                                   markeredgewidth=0))
    for patch, col_c in zip(bp['boxes'], box_colors):
        patch.set_facecolor(col_c)
        patch.set_alpha(0.85)
        patch.set_edgecolor('#333333')

    # Mean annotations
    for i, (td, col_c) in enumerate(zip(box_data, box_colors)):
        ax.text(i+1, np.median(td), f'{np.mean(td):.3f}',
                ha='center', va='bottom', fontsize=6.5,
                color='white', fontweight='bold')

    ax.set_xticks(range(1, len(box_labels)+1))
    ax.set_xticklabels(box_labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('ExprScore', fontsize=9) if col == 0 else ax.set_ylabel('')
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(axis='both', length=3)

    # Title
    title_color = '#888' if unreliable else '#222'
    ax.set_title(f'$\\it{{{org_label.replace(" ","\\;")}}}$',
                 fontsize=9.5, color=title_color)
    if note:
        ax.text(0.5, -0.22, note, transform=ax.transAxes,
                ha='center', va='top', fontsize=6.5, color='#888', style='italic')

    # Panel letter
    ax.text(-0.15, 1.08, chr(65+col), transform=ax.transAxes,
            fontsize=13, fontweight='bold', ha='left', va='bottom')

    # K.phaffii: highlight CO RS vs IDT
    if org == 'pichia':
        co_vals = sub[sub['optimizer']=='CO Ribo-seq']['expr_score'].values
        idt_vals = sub[sub['optimizer']=='IDT']['expr_score'].values
        if len(co_vals) and len(idt_vals):
            ratio = np.mean(co_vals)/np.mean(idt_vals)
            ax.text(0.97, 0.03, f'CO RS / IDT = {ratio:.2f}×\nwet-lab: 4.77×',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=7.5, color='#333',
                    bbox=dict(facecolor='#f8f8f8', edgecolor='#ccc', boxstyle='round,pad=0.3'))

    # Reference line at 0
    ax.axhline(0, color='#bbb', lw=0.7, linestyle='--', alpha=0.6)

# Legend
handles = [mpatches.Patch(facecolor=TOOL_COLORS[t], edgecolor='#555', lw=0.5,
                           alpha=0.85, label=t) for t in TOOL_ORDER
           if t in TOOL_COLORS]
fig.legend(handles=handles, loc='upper center', ncol=7,
           bbox_to_anchor=(0.5, 1.03), fontsize=8.5,
           framealpha=0.95, edgecolor='#ddd', handlelength=1.2)

fig.suptitle(
    'ExprScore (dual-head Ribo-seq expression predictor) applied to all tools — '
    'K. phaffii is the primary validated panel',
    fontsize=8.5, style='italic', y=1.09, color='#444'
)

fig.savefig(OUT,    dpi=200, bbox_inches='tight')
fig.savefig(OUTPDF, dpi=200, bbox_inches='tight')
print(f'Saved → {OUT}')
