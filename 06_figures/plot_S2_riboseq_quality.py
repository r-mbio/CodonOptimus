#!/usr/bin/env python3
"""Comprehensive Ribo-seq data quality diagnostic. Outputs figures/riboseq_quality_check.{png,pdf,svg}"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

BASE   = Path(__file__).resolve().parent.parent
RAW    = BASE / 'data' / 'raw'
OUT    = BASE / 'figures' / 'figS2_riboseq_quality.png'
OUTP   = BASE / 'figures' / 'figS2_riboseq_quality.pdf'
OUTS   = BASE / 'figures' / 'figS2_riboseq_quality.svg'

plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':       9,
    'axes.labelsize':  9,
    'axes.titlesize':  9.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth':  0.9,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'pdf.fonttype':     42,
})

# Simplified dataset labels for display
SRC_SHORT = {
    'riboseq_gse56622_plusglu+gse185286_wt+gse185458_wt': 'GSE56622+185286+185458',
    'riboseq_gse56622_plusglu':                            'GSE56622',
    'riboseq_gse56622_plusglu+gse185458_wt':              'GSE56622+185458',
    'riboseq_gse56622_plusglu+gse185286_wt':              'GSE56622+185286',
    'riboseq_gse185286_wt+gse185458_wt':                  'GSE185286+185458',
    'riboseq_gse185458_wt':                               'GSE185458',
    'riboseq_gse185286_wt':                               'GSE185286',
    'riboseq_prjna1335396_prjna906596':                    'PRJNA1335396\n+906596',
    'riboseq_gse126234_dk1042_lb_exponential':             'GSE126234\nDK1042 LB',
    'riboseq_srr32315787_88_normal_growth':                'SRR32315787+88\nnormal growth',
    'riboseq_gse159336':                                   'GSE159336\nAOX1 methanol',
}

ORG_ORDER  = ['s_cerevisiae', 'ecoli', 'bacillus', 'pichia']
ORG_LABELS = {
    's_cerevisiae': 'S. cerevisiae',
    'ecoli':        'E. coli',
    'bacillus':     'B. subtilis',
    'pichia':       'K. phaffii',
}
# Organism colors — distinct, colorblind-safe
COLORS = {
    's_cerevisiae': '#4C72B0',
    'ecoli':        '#CC8844',
    'bacillus':     '#3A9A5C',
    'pichia':       '#C44E52',
}

# ─── Load data ────────────────────────────────────────────────────────────────
data = {}
for org in ORG_ORDER:
    df = pd.read_csv(RAW / org / f'{org}_train.csv')
    data[org] = df[df['source'].str.startswith('riboseq_', na=False)].copy()


def panel_box(ax, lw=0.9):
    """Draw a full border around the panel."""
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(lw)


def add_label(ax, letter, x=-0.18, y=1.10):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=13, fontweight='bold', ha='left', va='bottom',
            color='#111')


# ─── Layout ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(4, 4, figure=fig,
                         hspace=0.68, wspace=0.40,
                         left=0.06, right=0.97, top=0.93, bottom=0.06)

# ─── Row 1 (A–D): Raw expression histograms ───────────────────────────────────
for ci, org in enumerate(ORG_ORDER):
    ax = fig.add_subplot(gs[0, ci])
    ev = data[org]['expression_level'].values
    ax.hist(ev, bins=40, color=COLORS[org], alpha=0.78,
            edgecolor='white', linewidth=0.35, zorder=2)
    ax.axvline(ev.mean(),   color='#222', lw=1.4, ls='--', zorder=3,
               label=f'mean={ev.mean():.1f}')
    ax.axvline(np.median(ev), color='#888', lw=1.2, ls=':', zorder=3,
               label=f'median={np.median(ev):.1f}')
    ax.set_title(
        f'$\\it{{{ORG_LABELS[org].replace(" ", "\\ ")}}}$  (n={len(ev):,})',
        fontsize=9.5, pad=5)
    ax.set_xlabel('Expression level (0–100)')
    if ci == 0:
        ax.set_ylabel('Gene count')
    ax.legend(fontsize=6.8, framealpha=0.85, edgecolor='#CCC',
              handlelength=1.2, loc='upper right')
    panel_box(ax)
    add_label(ax, chr(65 + ci))

# ─── Row 2 (E–H): Log₂ distribution ──────────────────────────────────────────
for ci, org in enumerate(ORG_ORDER):
    ax = fig.add_subplot(gs[1, ci])
    ev = data[org]['expression_level'].values
    ev_pos = ev[ev > 0.1]
    log_ev = np.log2(ev_pos + 1)
    skew = stats.skew(ev)
    ax.hist(log_ev, bins=35, color=COLORS[org], alpha=0.72,
            edgecolor='white', linewidth=0.35, zorder=2)
    _, p_norm = stats.normaltest(log_ev)
    verdict_col = '#2A8A2A' if p_norm > 0.05 else '#CC3333'
    verdict_txt = 'log-normal ✓' if p_norm > 0.05 else f'p={p_norm:.3f}'
    ax.text(0.97, 0.95, verdict_txt, transform=ax.transAxes,
            fontsize=7.5, ha='right', va='top', color=verdict_col,
            fontweight='bold')
    ax.set_title(fr'$\log_2$(expr+1)   skew={skew:.2f}', fontsize=9)
    ax.set_xlabel(r'$\log_2$(expression+1)')
    if ci == 0:
        ax.set_ylabel('Gene count')
    panel_box(ax)
    add_label(ax, chr(69 + ci))   # E, F, G, H

# ─── Row 3 (I, J): Dataset source composition + K.phaffii comparison ─────────
# Panel I — Ribo-seq source composition per organism (stacked bar)
ax_ba = fig.add_subplot(gs[2, :2])
# Use palette of blues/greens for sources
src_palette = ['#4C72B0', '#55A868', '#C44E52', '#DD8452', '#9B59B6',
               '#17BECF', '#BCBD22']
org_labels_short = ['S. cer.', 'E. coli', 'B. sub.', 'K. phaffii']
bar_bottoms  = np.zeros(4)
src_all = {}  # collect all sources for legend
for org in ORG_ORDER:
    for src in data[org]['source'].unique():
        src_all[src] = src_all.get(src, 0)
        src_all[src] += 1

# Plot stacked bars
xi = np.arange(4)
used_src = []
for si, (org, col) in enumerate(zip(ORG_ORDER, [COLORS[o] for o in ORG_ORDER])):
    src_counts = data[org]['source'].value_counts()
    bottom = 0
    sub_colors = [col] + ['#A8C4E0', '#80B8A8', '#F5B8A0', '#B0A0CC']
    for sj, (src, cnt) in enumerate(src_counts.items()):
        short = SRC_SHORT.get(src, src.replace('riboseq_', ''))
        label = f'{short} ({cnt:,})' if short not in [SRC_SHORT.get(s, s) for s in used_src] else None
        ax_ba.bar(si, cnt, bottom=bottom,
                  color=sub_colors[sj % len(sub_colors)],
                  alpha=0.85, edgecolor='white', lw=0.5,
                  label=label)
        if label:
            used_src.append(src)
        bottom += cnt
    ax_ba.text(si, bottom + 30, f'n={bottom:,}',
               ha='center', va='bottom', fontsize=8, fontweight='bold', color='#333')

ax_ba.set_xticks(xi)
ax_ba.set_xticklabels(org_labels_short, fontsize=9)
ax_ba.set_ylabel('Number of genes')
ax_ba.set_title('Ribo-seq dataset source composition per organism', fontsize=9.5)
ax_ba.legend(fontsize=6.8, framealpha=0.88, edgecolor='#CCC',
             loc='upper left', ncol=1, handlelength=1.0)
panel_box(ax_ba)
add_label(ax_ba, 'I', x=-0.06)

# Panel J — K. phaffii source comparison
ax_kp = fig.add_subplot(gs[2, 2:])
df_kp = data['pichia']
bins = np.linspace(0, 100, 41)
src_map = [
    ('riboseq_gse159336',                      '#C44E52', 'GSE159336: AOX1 methanol-induced (n=1,651)'),
    ('riboseq_srr32315787_88_normal_growth',   '#DD8452', 'SRR32315787+88: Normal YPD growth (n=2,632)'),
]
for src, col, lbl in src_map:
    sub = df_kp[df_kp['source'] == src]['expression_level'].values
    ax_kp.hist(sub, bins=bins, alpha=0.65, color=col,
               label=lbl, edgecolor='white', lw=0.35, zorder=2)
ev1 = df_kp[df_kp['source'] == 'riboseq_gse159336']['expression_level'].values
ev2 = df_kp[df_kp['source'] == 'riboseq_srr32315787_88_normal_growth']['expression_level'].values
if len(ev1) > 0 and len(ev2) > 0:
    ks2, ks2_p = stats.ks_2samp(ev1, ev2)
    ax_kp.text(0.97, 0.95, f'KS stat={ks2:.3f},  p={ks2_p:.2e}',
               transform=ax_kp.transAxes, fontsize=7.5, ha='right', va='top',
               bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                         edgecolor='#CCC', lw=0.7))
ax_kp.set_title('K. phaffii — Two Ribo-seq sources compared', fontsize=9.5)
ax_kp.set_xlabel('Expression level (0–100)')
ax_kp.set_ylabel('Gene count')
ax_kp.legend(fontsize=7.8, framealpha=0.90, edgecolor='#CCC')
panel_box(ax_kp)
add_label(ax_kp, 'J', x=-0.06)

# ─── Row 4 (K–N): QQ plots ────────────────────────────────────────────────────
for ci, org in enumerate(ORG_ORDER):
    ax = fig.add_subplot(gs[3, ci])
    ev = data[org]['expression_level'].values
    (quantiles, values), (slope, intercept, r) = stats.probplot(ev, dist='norm')
    ax.scatter(quantiles, values, s=4, alpha=0.35, color=COLORS[org],
               linewidths=0, zorder=3)
    xlim = ax.get_xlim()
    xx = np.linspace(xlim[0], xlim[1], 100)
    ax.plot(xx, slope * xx + intercept, color='#333', lw=1.2, ls='--', zorder=4)
    dr = np.percentile(ev, 90) / max(np.percentile(ev, 10), 0.1)
    ax.set_title(f'QQ vs Normal   r={r:.3f}', fontsize=9)
    ax.set_xlabel('Theoretical quantiles')
    if ci == 0:
        ax.set_ylabel('Sample quantiles')
    ax.text(0.97, 0.05, f'p90/p10 = {dr:.1f}×',
            transform=ax.transAxes, fontsize=7.5, ha='right', va='bottom',
            color='#555',
            bbox=dict(boxstyle='round,pad=0.22', facecolor='white',
                      edgecolor='#CCC', lw=0.7))
    panel_box(ax)
    add_label(ax, chr(75 + ci))   # K, L, M, N

# ─── Supertitle ───────────────────────────────────────────────────────────────

fig.savefig(OUT,  dpi=200, bbox_inches='tight')
fig.savefig(OUTP, dpi=300, bbox_inches='tight')
fig.savefig(OUTS, format='svg', bbox_inches='tight')
print(f'Saved → {OUT}')
print(f'Saved → {OUTP}')
print(f'Saved → {OUTS}')

# ─── Text summary ─────────────────────────────────────────────────────────────
print('\n=== QUALITY VERDICT ===')
for org in ORG_ORDER:
    ev = data[org]['expression_level'].values
    sk = stats.skew(ev)
    dr = np.percentile(ev, 90) / max(np.percentile(ev, 10), 0.1)
    n_zero = (ev == 0).sum()
    verdict = 'GOOD'
    if abs(sk) > 2.0:   verdict = f'WARN: skew={sk:.2f}'
    if dr < 2.0:         verdict = f'WARN: dyn_range={dr:.1f}x'
    if n_zero > len(ev) * 0.05: verdict = f'WARN: {n_zero} zeros'
    print(f'  {org}: n={len(ev)}, skew={sk:.2f}, dyn_range={dr:.1f}x, zeros={n_zero} → {verdict}')
