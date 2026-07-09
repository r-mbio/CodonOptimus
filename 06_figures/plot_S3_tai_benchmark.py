#!/usr/bin/env python3
"""
Re-plot figS_other_organisms_tai pages from existing cache.
Skips model loading — cache must be complete (all organisms have co + co_foundation).
Shared y-axis per page so comparisons are honest.
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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import seaborn as sns
from pathlib import Path

sns.set_style('ticks')
sns.set_context('paper', font_scale=1.5)

BASE  = Path(__file__).resolve().parent.parent
CACHE = BASE / 'results/figS_tai_other_orgs_cache.json'

plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':       10,
    'axes.labelsize':  10,
    'axes.titlesize':  10.5,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.linewidth':  0.9,
    'pdf.fonttype':    42,
})

C_ALL   = '#E69F00'
C_TOP10 = '#56B4E9'
C_FOUND = '#A8D4F0'   # CO Foundation — light blue
C_CO    = '#0072B2'   # CO CSI-FT — dark blue

CONDITIONS = ['All genes', 'Top 10%\ntAI', 'CO\nFoundation', 'CO\nCSI-FT']
COLORS_V   = [C_ALL, C_TOP10, C_FOUND, C_CO]

# ── Organism order matches original script ─────────────────────────────────────
ORGS_ORDER = [
    'yarrowia', 'kluyveromyces_marxianus', 'kluyveromyces', 'ogataea',
    'rhodotorula', 'scheffersomyces', 'ashbya',
    'trichoderma', 'aspergillus_niger', 'aspergillus_oryzae',
    'aspergillus_terreus', 'neurospora', 'penicillium', 'fusarium', 'rhizopus',
    'pseudomonas_putida', 'corynebacterium', 'lactobacillus', 'lactococcus',
    'streptomyces', 'gluconobacter', 'cupriavidus', 'brevibacillus',
    'bacillus_amyloliquefaciens', 'clostridium',
    'nannochloropsis', 'cho', 'human',
]

# ── Load cache ─────────────────────────────────────────────────────────────────
print(f'Loading cache from {CACHE}')
with open(CACHE) as f:
    org_data = json.load(f)

# Filter to organisms in cache with data
orgs_plot = [(k, org_data[k]) for k in ORGS_ORDER
             if k in org_data and len(org_data[k].get('co', [])) > 0]
print(f'Plotting {len(orgs_plot)} organisms')

PER_PAGE = 8
N_COLS   = 4
pages    = [orgs_plot[i:i+PER_PAGE] for i in range(0, len(orgs_plot), PER_PAGE)]

for pg_idx, page_orgs in enumerate(pages):
    n_orgs   = len(page_orgs)
    n_rows   = math.ceil(n_orgs / N_COLS)
    page_num = pg_idx + 1

    # ── Shared y-limits: use all values on this page ───────────────────────────
    all_vals = []
    for _, od in page_orgs:
        all_vals.extend(od.get('native', []))
        all_vals.extend(od.get('top10', []))
        all_vals.extend(od.get('co', []))
        all_vals.extend(od.get('co_foundation', []))
    pad  = (max(all_vals) - min(all_vals)) * 0.06
    y_lo = max(0.0, min(all_vals) - pad)
    y_hi = min(1.0, max(all_vals) + pad * 2)   # extra headroom for annotations

    fig = plt.figure(figsize=(14, n_rows * 4.5))
    gs_root = GridSpec(n_rows + 1, 1, figure=fig,
                       height_ratios=[1.0] * n_rows + [0.06],
                       hspace=0.62)

    axes_grid = []
    for row in range(n_rows):
        gs_row = GridSpecFromSubplotSpec(1, N_COLS, subplot_spec=gs_root[row], wspace=0.38)
        axes_grid.extend([fig.add_subplot(gs_row[c]) for c in range(N_COLS)])

    for idx, (org_key, od) in enumerate(page_orgs):
        ax = axes_grid[idx]

        data_map = {
            'All genes':      od.get('native', []),
            'Top 10%\ntAI':  od.get('top10',  []),
            'CO\nFoundation': od.get('co_foundation', []),
            'CO\nCSI-FT':    od.get('co', []),
        }
        active_conds = [c for c in CONDITIONS if data_map[c]]
        df_v = pd.DataFrame([
            {'Condition': cond, 'tAI': v}
            for cond in active_conds
            for v in data_map[cond]
        ])

        sns.violinplot(data=df_v, x='Condition', y='tAI',
                       order=active_conds,
                       palette=dict(zip(CONDITIONS, COLORS_V)),
                       hue='Condition', legend=False,
                       inner='box', cut=0, linewidth=0.8, ax=ax)

        # Shared y-axis
        ax.set_ylim(y_lo, y_hi)

        # Top-10% threshold line
        thresh = od.get('thresh', float('nan'))
        if math.isfinite(thresh):
            ax.axhline(thresh, color=C_TOP10, lw=1.3, ls='--', alpha=0.85, zorder=1)

        # n= labels at bottom
        for i, cond in enumerate(active_conds):
            n = len(data_map[cond])
            ax.text(i, y_lo + (y_hi - y_lo) * 0.01, f'n={n}',
                    ha='center', va='bottom', fontsize=7.0, color='#666')

        # CO CSI-FT mean + above/below indicator
        co_mean  = float(np.mean(od['co'])) if od.get('co') else float('nan')
        fnd_mean = float(np.mean(od['co_foundation'])) if od.get('co_foundation') else float('nan')
        ann_off  = (y_hi - y_lo) * 0.03
        near_band = (y_hi - y_lo) * 0.06

        def clear_offset(val):
            # Push the label further away if it would otherwise sit on the
            # dashed threshold line (avoids label/line collisions).
            if math.isfinite(thresh) and abs(val - thresh) < near_band:
                return ann_off * 3.2
            return ann_off

        csi_xi = active_conds.index('CO\nCSI-FT') if 'CO\nCSI-FT' in active_conds else 3
        fnd_xi = active_conds.index('CO\nFoundation') if 'CO\nFoundation' in active_conds else 2

        if math.isfinite(co_mean):
            ax.text(csi_xi, co_mean + clear_offset(co_mean), f'{co_mean:.3f}',
                    ha='center', va='bottom', fontsize=7.5,
                    color=C_CO, fontweight='bold')
            if math.isfinite(thresh):
                above = co_mean >= thresh
                sym, col = ('▲', '#117711') if above else ('▼', '#BB2222')
                ax.text(csi_xi + 0.45, thresh, sym,
                        ha='left', va='center', fontsize=8.5, color=col)

        if math.isfinite(fnd_mean):
            ax.text(fnd_xi, fnd_mean + clear_offset(fnd_mean), f'{fnd_mean:.3f}',
                    ha='center', va='bottom', fontsize=7.5,
                    color='#5599CC', fontweight='bold')

        label = od['label']
        ax.set_title(f'$\\it{{{label.replace(" ", "\\ ")}}}$',
                     fontsize=10.5, pad=5)
        ax.set_xlabel('')
        ax.set_ylabel(
            'tRNA Adaptation Index (tAI)' if idx % N_COLS == 0 else '',
            fontsize=9.5)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.9)
        ax.tick_params(axis='x', labelsize=8.5)


    for idx in range(n_orgs, n_rows * N_COLS):
        axes_grid[idx].set_visible(False)

    leg_labels = ['All genes', 'Top 10% tAI', 'CO Foundation (ep11)', 'CO CSI-FT (all39)']
    handles = [mpatches.Patch(color=c, label=l)
               for c, l in zip(COLORS_V, leg_labels)]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=10.0,
               frameon=False, bbox_to_anchor=(0.5, 0.0))


    for ext in ['png', 'pdf', 'svg']:
        out = BASE / f'figures/figS3_tai_p{page_num}.{ext}'
        kw  = {'dpi': 300, 'bbox_inches': 'tight'} if ext != 'svg' else {'format': 'svg', 'bbox_inches': 'tight'}
        fig.savefig(out, **kw)
    plt.close(fig)
    print(f'Page {page_num}: saved')

print('\nAll pages done.')
