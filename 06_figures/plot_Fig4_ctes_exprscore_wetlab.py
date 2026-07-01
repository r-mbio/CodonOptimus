#!/usr/bin/env python3
"""
Figure 4 v5 — full aesthetic polish.

Row 1 (A): CTES box+strip, per organism, per-panel y-axis
Row 2: B = K.phaffii ExprScore box+strip | C = Wet-lab 4.77× | D = SDS placeholder
"""

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / '02_foundation_pretraining'))
sys.path.insert(0, str(_REPO_ROOT / '03_finetuning'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from pathlib import Path

BASE    = Path(__file__).resolve().parent.parent
BENCH   = BASE / 'other_optimizers/benchmark/benchmark_results_tai.csv'
BENCH4  = BASE / 'results/benchmark_consistent_v4.csv'
OUT_PNG = BASE / 'figures/fig4_csi_exprscore_final.png'
OUT_PDF = BASE / 'figures/fig4_csi_exprscore_final.pdf'
OUT_SVG = BASE / 'figures/fig4_csi_exprscore_final.svg'

df_tai = pd.read_csv(BENCH)
bench4 = pd.read_csv(BENCH4)
df_tai['ctes'] = df_tai['tai'] * (1 + df_tai['mean_cpb']) * df_tai['csi']

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':        13,
    'axes.labelsize':   13,
    'axes.titlesize':   13.5,
    'xtick.labelsize':  12,
    'ytick.labelsize':  12,
    'axes.linewidth':   1.2,
    'xtick.major.width': 1.1,
    'ytick.major.width': 1.1,
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'pdf.fonttype':     42,
})

# Okabe-Ito colorblind-safe palette (full saturation)
C_CO    = '#0072B2'   # dark blue — CO Ribo-seq
C_CT    = '#888888'   # grey  — CodonTransformer
C_IDT   = '#E69F00'   # orange
C_TWIST = '#D55E00'   # vermillion
C_GS    = '#009E73'   # bluish-green
C_CO_PT = '#882255'   # dark wine — CO Foundation
C_CO_CS = '#56B4E9'   # sky blue — CO CSI-FT

# ── Tool mappings ──────────────────────────────────────────────────────────────
# Panel A: CTES — CO, CT, IDT, Twist, GS
CTES_RENAME = {
    'CodonOptimus':     'CO',
    'CodonTransformer': 'CT',
    'IDT':              'IDT',
    'Twist':            'Twist',
    'Genscript':        'GS',
}
CTES_ORDER   = ['CO', 'CT', 'IDT', 'Twist', 'GS']
CTES_PALETTE = {'CO': C_CO, 'CT': C_CT, 'IDT': C_IDT, 'Twist': C_TWIST, 'GS': C_GS}

# Panel B: ExprScore K.phaffii — CO-RS, CO-CSI, CO-PT, IDT, Twist, GS
EXPR_RENAME = {
    'CO Ribo-seq':   'CO RS',
    'CO CSI-FT':     'CO CSI',
    'CO Foundation': 'CO PT',
    'IDT':           'IDT',
    'Twist':         'Twist',
    'Genscript':     'GS',
}
EXPR_ORDER   = ['CO RS', 'CO CSI', 'CO PT', 'IDT', 'Twist', 'GS']
EXPR_PALETTE = {
    'CO RS':  C_CO, 'CO CSI': C_CO_CS, 'CO PT': C_CO_PT,
    'IDT':    C_IDT, 'Twist': C_TWIST, 'GS':    C_GS,
}

ORG_CFG = [
    ('ecoli',        '$\\it{E.\\ coli}$'),
    ('bacillus',     '$\\it{B.\\ subtilis}$'),
    ('s_cerevisiae', '$\\it{S.\\ cerevisiae}$'),
    ('pichia',       '$\\it{K.\\ phaffii}$'),
]

df_tai['tool'] = df_tai['optimizer'].map(CTES_RENAME)
df_ctes = df_tai[df_tai['tool'].notna()].copy()

bench4['tool'] = bench4['optimizer'].map(EXPR_RENAME)
df_expr = bench4[bench4['tool'].notna()].copy()

# ── Layout ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs_outer = gridspec.GridSpec(
    2, 1, figure=fig,
    height_ratios=[1.1, 0.90],
    hspace=0.55,
    left=0.06, right=0.97, top=0.93, bottom=0.07,
)
gs_A  = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_outer[0], wspace=0.35)
gs_BC = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[1],
                                          wspace=0.38, width_ratios=[1.35, 0.80, 1.0])

# ── Helper: clean box+strip panel ─────────────────────────────────────────────
def make_box_panel(ax, data, x_col, y_col, order, palette,
                   ylabel='', title='', ref_tool=None, ref_color=C_CT,
                   ylim_pad=0.20):
    present = [t for t in order if t in data[x_col].values]
    sub = data[data[x_col].isin(present)].copy()

    # Compute y-limits before plotting
    ymin = sub[y_col].min()
    ymax = sub[y_col].max()
    yrange = ymax - ymin
    ax_ymin = max(0, ymin - yrange * 0.15)
    ax_ymax = ymax + yrange * ylim_pad

    sns.boxplot(
        data=sub, x=x_col, y=y_col, hue=x_col, order=present,
        palette=palette, width=0.58, linewidth=1.0,
        fliersize=0, saturation=1.0, legend=False,
        medianprops=dict(color='white', linewidth=2.5),
        whiskerprops=dict(linewidth=1.0, color='#333'),
        capprops=dict(linewidth=1.0, color='#333'),
        boxprops=dict(edgecolor='#333333', linewidth=1.0),
        ax=ax,
    )
    sns.stripplot(
        data=sub, x=x_col, y=y_col, hue=x_col, order=present,
        palette=palette, size=6, jitter=0.18, alpha=0.9,
        edgecolor='white', linewidth=0.7, zorder=5, legend=False, ax=ax,
    )

    # CO RS vs CT/ref significance bracket
    co_key = 'CO' if 'CO' in present else ('CO RS' if 'CO RS' in present else None)
    ct_key = ref_tool if ref_tool and ref_tool in present else None
    if co_key and ct_key:
        co_d = sub[sub[x_col] == co_key][y_col].dropna().values
        ct_d = sub[sub[x_col] == ct_key][y_col].dropna().values
        n = min(len(co_d), len(ct_d))
        if n >= 3:
            try:
                _, pv = stats.wilcoxon(co_d[:n], ct_d[:n])
                pstr = '***' if pv < 0.001 else ('**' if pv < 0.01 else ('*' if pv < 0.05 else ''))
            except Exception:
                pstr = ''
            if pstr:
                xi_co = present.index(co_key)
                xi_ct = present.index(ct_key)
                yb = ax_ymax * 0.93
                ax.plot([xi_co, xi_co, xi_ct, xi_ct],
                        [yb, yb + yrange*0.04, yb + yrange*0.04, yb],
                        lw=1.0, color='#333')
                ax.text((xi_co + xi_ct) / 2, yb + yrange*0.05, pstr,
                        ha='center', va='bottom', fontsize=11,
                        fontweight='bold', color='#2a7a2a')

    ax.set_ylim(ax_ymin, ax_ymax)
    ax.set_xlabel('', labelpad=2)
    ax.set_ylabel(ylabel, fontsize=13, labelpad=6)
    ax.set_title(title, fontsize=13.5, pad=8)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.4)
        sp.set_color('#333333')
    ax.tick_params(axis='x', length=0, pad=5)
    ax.tick_params(axis='y', length=3)

# ── Panel A: CTES per organism ─────────────────────────────────────────────────
for oi, (org, org_label) in enumerate(ORG_CFG):
    ax = fig.add_subplot(gs_A[oi])
    sub = df_ctes[df_ctes['organism'] == org].copy()

    make_box_panel(
        ax, sub, 'tool', 'ctes',
        order=CTES_ORDER,
        palette=CTES_PALETTE,
        ylabel='CTES = tAI × (1+CPB) × CSI' if oi == 0 else '',
        title=org_label,
        ref_tool='CT',
    )

    # K.phaffii: CT absent
    if org == 'pichia':
        ax.text(1, ax.get_ylim()[0] + 0.002, 'CT: n/a',
                ha='center', va='bottom', fontsize=8, color=C_CT,
                style='italic', transform=ax.get_xaxis_transform())

    if oi == 0:
        ax.text(-0.28, 1.08, 'A', transform=ax.transAxes,
                fontsize=17, fontweight='bold', ha='left', va='bottom')

# Shared legend top-left above panel A
legend_handles = [mpatches.Patch(color=CTES_PALETTE[t], label=t) for t in CTES_ORDER]
fig.axes[0].legend(
    handles=legend_handles, fontsize=11, frameon=False,
    loc='upper left', ncol=5, handlelength=1.2,
    bbox_to_anchor=(0.0, 1.26),
)

# ── Panel B: K.phaffii ExprScore ──────────────────────────────────────────────
ax_b = fig.add_subplot(gs_BC[0])
kp_expr = df_expr[df_expr['organism'] == 'pichia'].copy()

make_box_panel(
    ax_b, kp_expr, 'tool', 'expr_score',
    order=EXPR_ORDER,
    palette=EXPR_PALETTE,
    ylabel='ExprScore',
    title='$\\it{K.\\ phaffii}$ — expression by model variant',
    ref_tool=None,
    ylim_pad=0.22,
)
ax_b.text(-0.15, 1.08, 'B', transform=ax_b.transAxes,
          fontsize=17, fontweight='bold', ha='left', va='bottom')

# B legend
b_handles = [mpatches.Patch(color=EXPR_PALETTE[t], label=t) for t in EXPR_ORDER]
ax_b.legend(handles=b_handles, fontsize=10, frameon=False,
            loc='lower left', ncol=1, handlelength=1.0)

# ── Panel C: Wet-lab 4.77× ────────────────────────────────────────────────────
ax_c = fig.add_subplot(gs_BC[1])
means_c = [48.6, 232.0]; errs_c = [9.8, 32.9]
clones  = [[50, 60, 45, 54, 34], [250, 245, 270, 200, 195]]

ax_c.bar([0, 1], means_c, yerr=errs_c, color=[C_IDT, C_CO], width=0.54,
         capsize=7, error_kw=dict(lw=1.8, capthick=1.8),
         edgecolor='white', linewidth=0.5, zorder=3)

rng = np.random.default_rng(77)
for xi, cl, col_c in [(0, clones[0], C_IDT), (1, clones[1], C_CO)]:
    jx = rng.uniform(-0.11, 0.11, len(cl))
    ax_c.scatter(xi + jx, cl, s=34, color='white', edgecolors=col_c,
                 linewidths=1.6, zorder=5)

y_top = max(means_c) + max(errs_c)
yb    = y_top * 1.17
ax_c.plot([0, 0, 1, 1], [yb, yb * 1.04, yb * 1.04, yb], lw=1.1, color='#444')
ax_c.text(0.5, yb * 1.055, r'$p = 2.2\times10^{-6}$',
          ha='center', va='bottom', fontsize=10)
ax_c.text(0.5, means_c[0] * 0.88, '4.77×',
          ha='center', va='bottom', fontsize=22, fontweight='bold', color=C_CO)

ax_c.set_xticks([0, 1])
ax_c.set_xticklabels(['IDT', 'CO\nRibo-seq'], fontsize=13)
ax_c.set_ylabel('BLG yield (mg/L)', fontsize=13)
ax_c.set_ylim(0, y_top * 1.68)
for sp in ax_c.spines.values():
    sp.set_visible(True)
    sp.set_linewidth(1.4)
    sp.set_color('#333333')
ax_c.tick_params(axis='x', length=0, pad=5)
ax_c.tick_params(axis='y', length=3)
ax_c.set_title('Wet-lab validation\n$\\it{K.\\ phaffii}$, BLG, n=5 clones',
               fontsize=13.5, pad=8)
ax_c.text(-0.32, 1.08, 'C', transform=ax_c.transAxes,
          fontsize=17, fontweight='bold', ha='left', va='bottom')

# ── Panel D: SDS-PAGE placeholder ─────────────────────────────────────────────
ax_d = fig.add_subplot(gs_BC[2])
ax_d.set_xlim(0, 1); ax_d.set_ylim(0, 1)
ax_d.set_facecolor('#F5F5F5')
for spine in ax_d.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor('#BBBBBB')
    spine.set_linewidth(1.3)
ax_d.set_xticks([]); ax_d.set_yticks([])
ax_d.text(0.5, 0.52, 'SDS-PAGE', ha='center', va='center',
          fontsize=14, color='#BBBBBB', style='italic')
ax_d.text(0.5, 0.42, '(gel image)', ha='center', va='center',
          fontsize=12, color='#CCCCCC', style='italic')
ax_d.set_title('$\\it{K.\\ phaffii}$ X33 supernatant',
               fontsize=13.5, pad=8, style='italic')
ax_d.text(-0.12, 1.08, 'D', transform=ax_d.transAxes,
          fontsize=17, fontweight='bold', ha='left', va='bottom')

# ── Supertitle ─────────────────────────────────────────────────────────────────

fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
fig.savefig(OUT_PDF, dpi=300, bbox_inches='tight')
fig.savefig(OUT_SVG, format='svg', bbox_inches='tight')
print(f'Saved → {OUT_PNG}')
print(f'Saved → {OUT_PDF}')
print(f'Saved → {OUT_SVG}')
