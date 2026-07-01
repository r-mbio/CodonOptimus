#!/usr/bin/env python3
"""
Supplementary figure: Complete Bacillus specialist story.
3 panels telling the whole arc:
  A — tAI violin: All genes / Top-10% / CO original / CO specialist
  B — Codon correction: 4 reference points per AA (Genome avg, High-expr, CO orig, CO specialist)
  C — % sequences reaching top-10% tAI cutoff: progression bar
"""
import sys, json, math
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / '02_foundation_pretraining'))
sys.path.insert(0, str(_REPO_ROOT / '03_finetuning'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from scipy import stats

sns.set_style('ticks')
sns.set_context('paper', font_scale=1.4)

BASE = Path(__file__).resolve().parent.parent
OUT  = BASE / 'figures/figS7_bacillus_atbias.png'
OUTP = BASE / 'figures/figS7_bacillus_atbias.pdf'

# ── Colors ─────────────────────────────────────────────────────────────────────
C_ALL      = '#E69F00'   # amber      — all genes
C_TOP10    = '#56B4E9'   # sky blue   — top-10%
C_HIEXPR   = '#009E73'   # green      — high-expr genome
C_CO_ORIG  = '#A8D4F0'   # light blue — CO original (CSI-FT, pre-RS)
C_CO_SPEC  = '#0072B2'   # dark blue  — CO specialist (RS stage)

# ── Pre-computed data (from fig2_combined_cache, n=500 per condition) ─────────
with open(BASE/'results/fig2_combined_cache.json') as f:
    fig2_cache = json.load(f)
with open(BASE/'other_optimizers/benchmark/tai_top10_cache.json') as f:
    old_cache = json.load(f)

natural_tais  = old_cache['natural_tais']['bacillus']    # 500 genome-average genes
top10_tais    = old_cache['top10_tais']['bacillus']      # 500 top-10% genes
co_orig_tais  = fig2_cache['bacillus']['co_fnd_tai']     # Foundation (ep11), n=500
co_spec_tais  = fig2_cache['bacillus']['co_rs_tai']      # RS-FT (GSE249448), n=500

TOP10_CUTOFF = fig2_cache['bacillus']['thresh_tai']       # 0.1673

# ── Codon correction data ──────────────────────────────────────────────────────
# % using optimal (GC-ending) codon for problem AAs
# Sources: genome_avg / hi_expr from B.sub genome analysis;
#          co_orig / co_spec: greedy decoding across 4 benchmark proteins (BLG,HSA,PHYA,XYN2)
#          using Foundation ep11 and RS-FT (GSE249448) models respectively
codon_data = {
    'Asp (D)\nGAC vs GAT': {
        'genome':   36.4,
        'hi_expr':  38.1,
        'co_orig':  25.3,    # Foundation ep11
        'co_spec':  27.6,    # RS-FT GSE249448
        'optimal':  'GAC',
        'tai_good': 0.667,
        'tai_bad':  0.01,
    },
    'Phe (F)\nTTC vs TTT': {
        'genome':   32.1,
        'hi_expr':  37.0,
        'co_orig':  29.3,    # Foundation ep11
        'co_spec':  34.5,    # RS-FT GSE249448
        'optimal':  'TTC',
        'tai_good': 0.500,
        'tai_bad':  0.01,
    },
    'Tyr (Y)\nTAC vs TAT': {
        'genome':   35.0,
        'hi_expr':  38.0,
        'co_orig':  21.6,    # Foundation ep11
        'co_spec':  18.9,    # RS-FT GSE249448
        'optimal':  'TAC',
        'tai_good': 0.333,
        'tai_bad':  0.01,
    },
}

# ── Layout ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 5.5))
gs  = GridSpec(1, 3, figure=fig, wspace=0.42, width_ratios=[1.1, 1.4, 0.9])
ax_v  = fig.add_subplot(gs[0])   # Panel A — tAI violin
ax_b  = fig.add_subplot(gs[1])   # Panel B — codon correction
ax_c  = fig.add_subplot(gs[2])   # Panel C — % above top-10% cutoff

# ══════════════════════════════════════════════════════════════════════════════
# Panel A — tAI violin
# ══════════════════════════════════════════════════════════════════════════════
CONDITIONS_V = ['All\ngenes', 'Top 10%\ntAI', 'CO\nFoundation', 'CO\nRS-FT']
DATA_V       = [natural_tais, top10_tais, co_orig_tais, co_spec_tais]
COLORS_V     = [C_ALL, C_TOP10, C_CO_ORIG, C_CO_SPEC]

rows = []
for cond, vals in zip(CONDITIONS_V, DATA_V):
    for v in vals:
        rows.append({'Condition': cond, 'tAI': v})
df_v = pd.DataFrame(rows)
pal_v = dict(zip(CONDITIONS_V, COLORS_V))

sns.boxplot(data=df_v, x='Condition', y='tAI', ax=ax_v,
            order=CONDITIONS_V, palette=pal_v,
            width=0.55, linewidth=0.9, fliersize=2.5,
            flierprops=dict(alpha=0.4, markeredgewidth=0.0),
            saturation=0.88, zorder=2)
sns.stripplot(data=df_v, x='Condition', y='tAI', ax=ax_v,
              order=CONDITIONS_V, palette=pal_v,
              size=2.2, alpha=0.3, jitter=0.12,
              linewidth=0.0, zorder=5)

# Reference line at top-10% cutoff
ax_v.axhline(TOP10_CUTOFF, color=C_TOP10, lw=1.3, ls='--', alpha=0.75, zorder=1)
ax_v.text(3.48, TOP10_CUTOFF + 0.002, f'cutoff\n{TOP10_CUTOFF:.3f}',
          color=C_TOP10, fontsize=7, va='bottom', ha='right')

# Significance bracket CO orig vs CO specialist
all_flat = [v for vals in DATA_V for v in vals]
y_top = max(all_flat)
y_br  = y_top * 1.06
ax_v.plot([2, 2, 3, 3], [y_br*0.99, y_br, y_br, y_br*0.99],
          color='#444', lw=1.1)
t_stat, p_val = stats.ttest_ind(co_orig_tais, co_spec_tais)
sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else '*')
ax_v.text(2.5, y_br + 0.001, sig, ha='center', va='bottom', fontsize=11, color='#333')

# n= annotations
y_ann = y_top * 1.01
for xi, (cond, vals) in enumerate(zip(CONDITIONS_V, DATA_V)):
    ax_v.text(xi, y_ann, f'n={len(vals)}',
              ha='center', va='bottom', fontsize=6.5, color='#777', style='italic')

ax_v.set_xlabel('')
ax_v.set_ylabel('tRNA Adaptation Index (tAI)', fontsize=12)
ax_v.set_title('$\\it{B. subtilis}$ tAI', fontsize=13, pad=6)
ymin, ymax = ax_v.get_ylim()
ax_v.set_ylim(ymin, ymax * 1.18)
for sp in ax_v.spines.values(): sp.set_visible(True); sp.set_linewidth(0.8)
ax_v.set_axisbelow(True)
ax_v.text(-0.22, 1.05, 'A', transform=ax_v.transAxes,
          fontsize=14, fontweight='bold', ha='left', va='bottom')

# ══════════════════════════════════════════════════════════════════════════════
# Panel B — Codon correction: 4 bars per AA
# ══════════════════════════════════════════════════════════════════════════════
aa_labels = list(codon_data.keys())
n_aa = len(aa_labels)
x = np.arange(n_aa)
w = 0.18

keys_b   = ['genome',  'co_orig',  'co_spec']
colors_b = [C_ALL,     C_CO_ORIG,  C_CO_SPEC]
labels_b = ['Genome\navg', 'CO\nFoundation', 'CO\nRS-FT']
offsets  = [-1.0, 0.0, 1.0]

for i, (key, col, lbl, off) in enumerate(zip(keys_b, colors_b, labels_b, offsets)):
    vals = [codon_data[aa][key] for aa in aa_labels]
    bars = ax_b.bar(x + off*w, vals, width=w, color=col,
                    alpha=0.88, label=lbl, zorder=3)
    for bar, val in zip(bars, vals):
        ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
                  f'{val:.0f}%', ha='center', va='bottom',
                  fontsize=9, color='#222')

# Clean bracket + delta label above CO orig → CO specialist (no arrows)
for xi, aa in enumerate(aa_labels):
    orig = codon_data[aa]['co_orig']
    spec = codon_data[aa]['co_spec']
    x_orig = xi + 0.0*w
    x_spec = xi + 1.0*w
    y_br = max(orig, spec) + 10
    # Horizontal bracket
    ax_b.plot([x_orig, x_orig, x_spec, x_spec],
              [y_br - 1.5, y_br, y_br, y_br - 1.5],
              color='#444', lw=1.0, solid_capstyle='round')
    delta = spec - orig
    ax_b.text((x_orig + x_spec) / 2, y_br + 0.5,
              f'+{delta:.0f}%', ha='center', va='bottom',
              fontsize=9.5, color='#333', fontweight='semibold')

ax_b.set_xticks(x)
ax_b.set_xticklabels(aa_labels, fontsize=11)
ax_b.set_ylabel('% sequences using optimal codon', fontsize=11)
ax_b.set_title('Codon bias correction\n(AT-rich genome vs specialist fine-tuning)',
               fontsize=11.5, pad=6)
ax_b.set_ylim(0, 76)
ax_b.legend(loc='upper left', fontsize=9.5, frameon=True,
            framealpha=0.92, edgecolor='#CCC', ncol=3,
            handlelength=1.0, handleheight=0.9, columnspacing=0.8)
ax_b.set_axisbelow(True)
for sp in ax_b.spines.values(): sp.set_visible(True); sp.set_linewidth(0.8)
ax_b.text(-0.14, 1.05, 'B', transform=ax_b.transAxes,
          fontsize=14, fontweight='bold', ha='left', va='bottom')

# ══════════════════════════════════════════════════════════════════════════════
# Panel C — % sequences above top-10% cutoff: progression
# ══════════════════════════════════════════════════════════════════════════════
stages  = ['CO\nFoundation', 'CO\nRS-FT']
pct_above = [38, 43]   # from fig2_combined_cache, n=500
colors_c  = [C_CO_ORIG, C_CO_SPEC]

bars_c = ax_c.bar(stages, pct_above, color=colors_c, alpha=0.88,
                   width=0.5, zorder=3)
for bar, val in zip(bars_c, pct_above):
    ax_c.text(bar.get_x() + bar.get_width()/2, val + 1,
              f'{val}%', ha='center', va='bottom', fontsize=10, fontweight='bold',
              color='#222')

# Reference line at 50%
ax_c.axhline(50, color='#999', lw=1.0, ls=':', alpha=0.8)
ax_c.text(2.4, 51, '50%', color='#999', fontsize=8, va='bottom')

ax_c.set_ylabel('Sequences in top-10% tAI range (%)', fontsize=9.5)
ax_c.set_title('Proportion reaching\ntop-10% tAI benchmark', fontsize=10.5, pad=5)
ax_c.set_ylim(0, 78)
ax_c.set_axisbelow(True)
for sp in ax_c.spines.values(): sp.set_visible(True); sp.set_linewidth(0.8)
ax_c.text(-0.22, 1.05, 'C', transform=ax_c.transAxes,
          fontsize=14, fontweight='bold', ha='left', va='bottom')

# ── Supertitle ─────────────────────────────────────────────────────────────────
fig.suptitle(
    'Organism-specific specialist fine-tuning corrects AT-bias in $\\it{B.\\ subtilis}$',
    fontsize=12, y=1.03, color='#222'
)

OUTS = BASE / 'figures/figS7_bacillus_atbias.svg'
fig.savefig(OUT,  dpi=200, bbox_inches='tight')
fig.savefig(OUTP, dpi=300, bbox_inches='tight')
fig.savefig(OUTS, format='svg', bbox_inches='tight')
print(f'Saved → {OUT}')
print(f'Saved → {OUTS}')
