#!/usr/bin/env python3
"""
Supplementary Figure: N-terminal ramp analysis.
S. cerevisiae and K. phaffii — native high-expressing genes vs CO RS vs CT.
Shows eukaryote ramp pattern: body tAI > ramp tAI in highly expressed native genes.
"""

import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter1d
from pathlib import Path

BASE    = Path(__file__).resolve().parent.parent
TAI_DIR = BASE / 'data' / 'trna'
STOP_C  = {'TAA','TAG','TGA'}

# ── Style ───────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Helvetica','Arial','DejaVu Sans'],
    'font.size':        9,
    'axes.labelsize':   10,
    'axes.titlesize':   11,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'axes.linewidth':   0.9,
    'legend.fontsize':  8.5,
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'pdf.fonttype':     42,
})

C_HI   = '#C8860A'   # native high-expr — amber (matches Fig3)
C_LO   = '#AAAAAA'   # native low-expr  — grey
C_CO   = '#0072B2'   # CO RS — dark blue
C_CT   = '#888888'   # CT — grey
C_RAMP = '#F5A623'   # ramp highlight — amber

# ── Helpers ──────────────────────────────────────────────────────────────────────
def load_tai(path):
    df = pd.read_csv(path, sep='\t')
    df.columns = df.columns.str.strip()
    df['Codon'] = df['Codon'].str.strip().str.upper()
    return dict(zip(df['Codon'], df['tAI']))

def get_codons(dna, skip=0):
    dna = str(dna).upper().replace(' ','')
    if len(dna) >= 3 and dna[-3:] in STOP_C: dna = dna[:-3]
    c = []
    for i in range(0, len(dna)-2, 3):
        cdn = dna[i:i+3]
        if cdn in STOP_C or 'N' in cdn: break
        c.append(cdn)
    return c[skip:]

def parse_fasta(path):
    seqs = {}; name = None; buf = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if name: seqs[name] = ''.join(buf)
                name = line[1:]; buf = []
            else: buf.append(line.replace(' ',''))
    if name: seqs[name] = ''.join(buf)
    return seqs

def positional_profile(df_sub, dna_col, w, n_bins=100, min_len=80,
                       max_genes=500, seed=42):
    profiles = []
    sample = df_sub.sample(min(max_genes, len(df_sub)), random_state=seed)
    for _, row in sample.iterrows():
        codons = get_codons(str(row[dna_col]))
        if len(codons) < min_len: continue
        tai_v = np.array([w.get(c, 0.001) for c in codons])
        xs    = np.linspace(0, len(tai_v)-1, n_bins)
        profiles.append(np.interp(xs, np.arange(len(tai_v)), tai_v))
    if not profiles: return None, None, 0
    arr = np.array(profiles)
    return arr.mean(0), arr.std(0) / np.sqrt(len(arr)), len(profiles)

def tool_profile(seqs_dict, w, n_bins=100, sp_len=0):
    profiles = []
    for k, v in seqs_dict.items():
        codons = get_codons(v, skip=sp_len if len(get_codons(v)) > 150 else 0)
        if len(codons) < 20: continue
        tai_v = np.array([w.get(c, 0.001) for c in codons])
        xs    = np.linspace(0, len(tai_v)-1, n_bins)
        profiles.append(np.interp(xs, np.arange(len(tai_v)), tai_v))
    if not profiles: return None
    return np.mean(profiles, axis=0)

SMOOTH   = 9
N_BINS   = 100
TOP_PCT  = 10
BOT_PCT  = 10
BLG_SP   = 16

ct_all = parse_fasta(BASE/'other_optimizers/codontransformer/CodonTransformer_all_genes_real.fasta')

ORG_CFG = [
    {
        'org':      's_cerevisiae',
        'label':    'S. cerevisiae',
        'tai_file': 'saccharomyces_cerevisiae_tai.txt',
        'data':     'data/raw/s_cerevisiae/s_cerevisiae_train.csv',
        'co_fasta': 'CodonOptimus_s_cerevisiae.fasta',
        'ct_key':   's_cerevisiae',
        'has_ct':   True,
    },
    {
        'org':      'pichia',
        'label':    'K. phaffii',
        'tai_file': 'pichia_pastoris_gca_tai.txt',
        'data':     'data/raw/pichia/pichia_train.csv',
        'co_fasta': 'CodonOptimus_pichia.fasta',
        'ct_key':   None,
        'has_ct':   False,
    },
]

# ── Figure layout ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8),
                          gridspec_kw={'wspace': 0.34})

for ax, cfg in zip(axes, ORG_CFG):
    org   = cfg['org']
    label = cfg['label']
    w     = load_tai(TAI_DIR / cfg['tai_file'])

    # Load native genes
    df = pd.read_csv(BASE / cfg['data'])
    dna_col  = next(c for c in df.columns if 'dna' in c.lower())
    expr_col = next(c for c in df.columns if 'expression' in c.lower())
    df = df.dropna(subset=[dna_col, expr_col])
    df = df[df[expr_col] > 0].reset_index(drop=True)

    hi_thr = df[expr_col].quantile(1 - TOP_PCT/100)
    lo_thr = df[expr_col].quantile(BOT_PCT/100)
    df_hi  = df[df[expr_col] >= hi_thr]
    df_lo  = df[df[expr_col] <= lo_thr]

    mean_hi, se_hi, n_hi = positional_profile(df_hi, dna_col, w,
                                               n_bins=N_BINS, max_genes=500)
    mean_lo, se_lo, n_lo = positional_profile(df_lo, dna_col, w,
                                               n_bins=N_BINS, max_genes=300)

    # CO sequences (benchmark 4 proteins)
    co_seqs = parse_fasta(BASE/f'other_optimizers/benchmark/optimized_fastas/{cfg["co_fasta"]}')
    mean_co = tool_profile(co_seqs, w, n_bins=N_BINS, sp_len=BLG_SP)

    # CT sequences
    if cfg['has_ct']:
        ct_seqs = {k: v for k, v in ct_all.items()
                   if cfg['ct_key'] in k.lower()}
        mean_ct = tool_profile(ct_seqs, w, n_bins=N_BINS)
    else:
        mean_ct = None

    x  = np.linspace(1, 100, N_BINS)
    sm = lambda v: uniform_filter1d(v, SMOOTH)

    # Compute stats first
    ramp_hi = mean_hi[:30].mean(); body_hi = mean_hi[30:].mean()
    ratio_hi = body_hi / ramp_hi
    ramp_co  = mean_co[:30].mean() if mean_co is not None else np.nan
    body_co  = mean_co[30:].mean() if mean_co is not None else np.nan
    ratio_co = body_co / ramp_co if not np.isnan(ramp_co) else np.nan
    ratio_ct = None
    if mean_ct is not None:
        ratio_ct = mean_ct[30:].mean() / mean_ct[:30].mean()

    # Set y-limits before drawing
    all_curves = [sm(mean_hi), sm(mean_lo)]
    if mean_co is not None: all_curves.append(sm(mean_co))
    if mean_ct is not None: all_curves.append(sm(mean_ct))
    ymin = min(c.min() for c in all_curves) - 0.010
    ymax = max(c.max() for c in all_curves) + 0.055
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(1, 100)

    # Ramp shading
    ax.axvspan(1, 30, color=C_RAMP, alpha=0.10, zorder=0)
    ax.axvline(30, color=C_RAMP, lw=1.0, ls='--', alpha=0.65, zorder=1)

    # Plot lines
    ax.plot(x, sm(mean_hi), color=C_HI, lw=2.2, zorder=4,
            label=f'Native high-expr (top {TOP_PCT}%, n={n_hi})')
    ax.fill_between(x, sm(mean_hi - se_hi), sm(mean_hi + se_hi),
                    color=C_HI, alpha=0.13, zorder=3)
    ax.plot(x, sm(mean_lo), color=C_LO, lw=1.4, ls='--', zorder=2,
            label=f'Native low-expr (bot {BOT_PCT}%, n={n_lo})')
    ax.plot(x, sm(mean_co), color=C_CO, lw=2.0, zorder=5,
            label='CO Ribo-seq (4 proteins)')
    if mean_ct is not None:
        ax.plot(x, sm(mean_ct), color=C_CT, lw=2.0, ls='-.', zorder=5,
                label='CodonTransformer (4 proteins)')

    # Ratio annotations — placed at fixed transform positions.
    # Plain neutral text throughout; box border colour matches the
    # corresponding line colour (not a red/green good-bad signal).
    ann_col = '#333333'
    ax.text(0.15, 0.97,
            f'Native:\nbody/ramp = {ratio_hi:.3f}\n(>1 = ramp slower)',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=7.5, color=ann_col,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5FAF5',
                      edgecolor='#999999', lw=0.7, alpha=0.95))

    if not np.isnan(ratio_co):
        ax.text(0.55, 0.97,
                f'CO RS:\nbody/ramp = {ratio_co:.3f}',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=7.5, color=ann_col,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=C_CO, lw=0.8, alpha=0.95))

    if ratio_ct is not None:
        ax.text(0.87, 0.97,
                f'CT:\nbody/ramp = {ratio_ct:.3f}',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=7.5, color=ann_col,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=C_CT, lw=0.8, alpha=0.95))

    # Ramp label
    ax.text(15, ymin + 0.003, 'N-term\nramp', ha='center', va='bottom',
            fontsize=7, color='#8B5E00', style='italic')

    ax.set_xlabel('Normalised codon position (%)', fontsize=10)
    ax.set_ylabel('Per-codon tAI (decoding speed)', fontsize=10)
    ax.set_title(f'$\\it{{{label.replace(" ","\\ ")}}}$', fontsize=11, pad=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.9)
    ax.spines['bottom'].set_linewidth(0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.6)
    ax.legend(loc='lower right', frameon=True, framealpha=0.92,
              edgecolor='#cccccc', fontsize=8.5)

    panel = 'A' if org == 's_cerevisiae' else 'B'
    ax.text(-0.13, 1.04, panel, transform=ax.transAxes, fontsize=14,
            fontweight='bold', ha='left', va='bottom')

    if not cfg['has_ct']:
        ax.text(0.97, 0.38, 'CodonTransformer\nnot available\nfor K. phaffii',
                transform=ax.transAxes, ha='right', va='center',
                fontsize=7.5, color='#999', style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8F8F8',
                          edgecolor='#CCCCCC', lw=0.7))

    print(f'\n{label}:')
    print(f'  Native high-expr: ramp={ramp_hi:.4f} body={body_hi:.4f} ratio={ratio_hi:.4f}')
    if mean_co is not None:
        print(f'  CO RS:            ramp={ramp_co:.4f} body={body_co:.4f} ratio={body_co/ramp_co:.4f}')
    if mean_ct is not None:
        print(f'  CT:               ramp={mean_ct[:30].mean():.4f} body={mean_ct[30:].mean():.4f} ratio={mean_ct[30:].mean()/mean_ct[:30].mean():.4f}')


plt.tight_layout()
out_png = BASE/'figures/figS12_nterm_ramp.png'
out_pdf = BASE/'figures/figS12_nterm_ramp.pdf'
out_svg = BASE/'figures/figS12_nterm_ramp.svg'
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.savefig(out_svg, bbox_inches='tight')
print(f'\nSaved: {out_png}')
