#!/usr/bin/env python3
"""
Supplementary Figure — CSI benchmark for all 39 organisms.
4 conditions: All genes | Top 10% CSI | CO Foundation (ep11) | CO CSI-FT (all39)
Shared y-axis per page. Cache-based so generation is done once.
"""
import sys, json, math, random
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / '02_foundation_pretraining'))
sys.path.insert(0, str(_REPO_ROOT / '03_finetuning'))

import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import seaborn as sns
from pathlib import Path
from CAI import CAI, relative_adaptiveness
import torch

from train_mlm_industrial import (
    IndustrialMLM, AA_CODON_VOCAB, ID_TO_PAIR, ORG_TO_ID,
    CLS_TOKEN, PAD_TOKEN, _GENETIC_CODE, AA_MASK,
)

sns.set_style('ticks')
sns.set_context('paper', font_scale=1.5)

BASE      = Path(__file__).resolve().parent.parent
PRETRAIN  = BASE / 'data/pretrain/all_industrial_cds.tsv'
MODEL_CSI = BASE / 'models/industrial_mlm_csi_all39.pt'
MODEL_FND = BASE / 'models/industrial_mlm_ep11.pt'
CACHE     = BASE / 'results/figS_csi_all39_cache.json'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

STOP_C = {'TAA', 'TAG', 'TGA'}
N_SAMPLE, SEED = 500, 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

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
C_FOUND = '#A8D4F0'
C_CO    = '#0072B2'

CONDITIONS = ['All genes', 'Top 10%\nCSI', 'CO\nFoundation', 'CO\nCSI-FT']
COLORS_V   = [C_ALL, C_TOP10, C_FOUND, C_CO]

# ── Pretty organism labels ─────────────────────────────────────────────────────
ORG_LABEL = {
    'ecoli':                  'E. coli',
    'bacillus':               'B. subtilis',
    's_cerevisiae':           'S. cerevisiae',
    'pichia':                 'K. phaffii',
    'yarrowia':               'Y. lipolytica',
    'kluyveromyces_marxianus':'K. marxianus',
    'kluyveromyces':          'K. lactis',
    'ogataea':                'O. polymorpha',
    'rhodotorula':            'R. graminis',
    'scheffersomyces':        'S. stipitis',
    'ashbya':                 'A. gossypii',
    'trichoderma':            'T. reesei',
    'aspergillus_niger':      'A. niger',
    'aspergillus_oryzae':     'A. oryzae',
    'aspergillus_terreus':    'A. terreus',
    'aspergillus_fumigatus':  'A. fumigatus',
    'neurospora':             'N. crassa',
    'penicillium':            'P. rubens',
    'fusarium':               'F. graminearum',
    'rhizopus':               'R. delemar',
    'mucor':                  'M. circinelloides',
    'myceliophthora':         'M. thermophila',
    'chlamydomonas':          'C. reinhardtii',
    'phaeodactylum':          'P. tricornutum',
    'nannochloropsis':        'N. gaditana',
    'pseudomonas_putida':     'P. putida',
    'corynebacterium':        'C. glutamicum',
    'lactobacillus':          'L. acidophilus',
    'lactococcus':            'L. lactis',
    'streptomyces':           'S. coelicolor',
    'streptomyces_lividans':  'S. lividans',
    'gluconobacter':          'G. oxydans',
    'cupriavidus':            'C. necator',
    'brevibacillus':          'B. brevis',
    'bacillus_amyloliquefaciens': 'B. amyloliquefaciens',
    'bacillus_licheniformis': 'B. licheniformis',
    'clostridium':            'C. acetobutylicum',
    'cho':                    'CHO cells',
    'human':                  'H. sapiens',
}

# All 39 organisms in order (4 main first, then others)
ALL_ORGS = [
    'ecoli', 'bacillus', 's_cerevisiae', 'pichia',
    'yarrowia', 'kluyveromyces_marxianus', 'kluyveromyces', 'ogataea',
    'rhodotorula', 'scheffersomyces', 'ashbya',
    'trichoderma', 'aspergillus_niger', 'aspergillus_oryzae',
    'aspergillus_terreus', 'aspergillus_fumigatus',
    'neurospora', 'penicillium', 'fusarium', 'rhizopus',
    'mucor', 'myceliophthora', 'chlamydomonas', 'phaeodactylum', 'nannochloropsis',
    'pseudomonas_putida', 'corynebacterium', 'lactobacillus', 'lactococcus',
    'streptomyces', 'streptomyces_lividans', 'gluconobacter', 'cupriavidus',
    'brevibacillus', 'bacillus_amyloliquefaciens', 'bacillus_licheniformis',
    'clostridium', 'cho', 'human',
]

# ── Helpers ───────────────────────────────────────────────────────────────────
AA_TO_TIDS = {}
for (aa, cdn), tid in AA_CODON_VOCAB.items():
    AA_TO_TIDS.setdefault(aa, []).append(tid)


def dna_to_codons(dna):
    dna = str(dna).upper()
    if len(dna) >= 3 and dna[-3:] in STOP_C:
        dna = dna[:-3]
    return [dna[i:i+3] for i in range(0, len(dna)-2, 3)
            if 'N' not in dna[i:i+3] and dna[i:i+3] not in STOP_C]


def dna_to_aa(dna):
    codons = dna_to_codons(dna)
    aa = []
    for c in codons:
        a = _GENETIC_CODE.get(c, '')
        if not a or a == '*':
            break
        aa.append(a)
    return ''.join(aa)


def build_csi_weights(seqs):
    """Build CSI weights from a list of DNA sequences using relative_adaptiveness."""
    pool = []
    for dna in seqs:
        codons = dna_to_codons(str(dna))
        if len(codons) >= 30:
            pool.append(''.join(codons) + 'TAA')
    if not pool:
        return None
    try:
        return relative_adaptiveness(sequences=pool)
    except Exception as e:
        print(f'  WARNING: relative_adaptiveness failed: {e}')
        return None


def compute_csi(dna, weights):
    codons = dna_to_codons(str(dna))
    seq = ''.join(codons)
    if len(seq) < 9 or weights is None:
        return float('nan')
    try:
        return CAI(seq, weights)
    except Exception:
        return float('nan')


@torch.no_grad()
def co_generate(aa_seq, org_key, mdl):
    if org_key not in ORG_TO_ID:
        return None
    valid = [a for a in aa_seq.upper() if a in AA_MASK]
    if len(valid) < 30:
        return None
    tok = [CLS_TOKEN] + [AA_MASK[a] for a in valid]
    ids = torch.tensor([tok], dtype=torch.long, device=DEVICE)
    msk = torch.zeros(1, len(tok), dtype=torch.bool, device=DEVICE)
    oid = torch.tensor([ORG_TO_ID[org_key]], dtype=torch.long, device=DEVICE)
    logits = mdl(ids, msk, oid)
    result = []
    for pi, aa_ch in enumerate(valid):
        allowed = AA_TO_TIDS.get(aa_ch, [])
        pl = logits[0, pi + 1, :61].clone()
        if allowed:
            mask_l = torch.full((61,), float('-inf'), device=DEVICE)
            for t in allowed:
                if t < 61:
                    mask_l[t] = 0.0
            pl = pl + mask_l
        result.append(ID_TO_PAIR[pl.argmax().item()][1])
    return ''.join(result)


def load_model(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    mdl = IndustrialMLM(use_grad_ckpt=False)
    mdl.load_state_dict(ckpt['model'])
    return mdl.eval().to(DEVICE)


# ── Load models ───────────────────────────────────────────────────────────────
print('Loading CO CSI-FT model...')
model_csi = load_model(MODEL_CSI)
print('Loading CO Foundation model...')
model_fnd = load_model(MODEL_FND)

# ── Load pretrain data ────────────────────────────────────────────────────────
print('Loading pretrain sequences...')
df_all = pd.read_csv(PRETRAIN, sep='\t', usecols=['organism', 'dna'])
print(f'  Total rows: {len(df_all):,}')

# ── Load or init cache ────────────────────────────────────────────────────────
if CACHE.exists():
    with open(CACHE) as f:
        org_data = json.load(f)
    print(f'Loaded cache: {len(org_data)} organisms')
else:
    org_data = {}

# ── Generate per organism ─────────────────────────────────────────────────────
for org_key in ALL_ORGS:
    if org_key in org_data and 'co_foundation' in org_data[org_key]:
        print(f'  {org_key}: fully cached ✓')
        continue

    label = ORG_LABEL.get(org_key, org_key)
    print(f'\n── {label} ({org_key}) ──')

    sub = df_all[df_all['organism'] == org_key]['dna'].dropna().values
    valid = [d for d in sub if 150 * 3 <= len(str(d)) <= 600 * 3]
    print(f'  Valid seqs: {len(valid):,}')
    if len(valid) < 50:
        print('  SKIP — too few sequences')
        continue

    # Build CSI weights from ALL native sequences
    print(f'  Building CSI weights from {min(len(valid), 5000)} sequences...')
    rng = np.random.default_rng(SEED)
    ref_seqs = rng.choice(valid, size=min(5000, len(valid)), replace=False)
    weights = build_csi_weights(ref_seqs)
    if weights is None:
        print('  SKIP — CSI weights failed')
        continue

    # Native CSI (n=500)
    nat_sample = rng.choice(valid, size=min(N_SAMPLE, len(valid)), replace=False)
    nat_csi = [compute_csi(d, weights) for d in nat_sample]
    nat_csi = [x for x in nat_csi if math.isfinite(x)]

    # Top-10% CSI threshold and sample
    all_csi = [(d, compute_csi(d, weights)) for d in valid[:5000]]
    all_csi = [(d, c) for d, c in all_csi if math.isfinite(c)]
    thresh  = np.percentile([c for _, c in all_csi], 90)
    top10   = [c for _, c in all_csi if c >= thresh]

    # Pool for generation
    pool = rng.choice(valid, size=min(N_SAMPLE * 4, len(valid)), replace=False)

    # CO CSI-FT sequences
    if org_key in org_data and 'co' in org_data[org_key]:
        co_csi = org_data[org_key]['co']
        print(f'  CO CSI-FT: cached (n={len(co_csi)})')
    else:
        co_csi = []
        for dna in pool:
            if len(co_csi) >= N_SAMPLE:
                break
            aa = dna_to_aa(str(dna))
            if len(aa) < 30:
                continue
            co_dna = co_generate(aa, org_key, model_csi)
            if co_dna is None:
                continue
            c = compute_csi(co_dna, weights)
            if math.isfinite(c):
                co_csi.append(c)

    # CO Foundation sequences
    co_fnd = []
    rng2 = np.random.default_rng(SEED + 1)
    pool2 = rng2.choice(valid, size=min(N_SAMPLE * 4, len(valid)), replace=False)
    for dna in pool2:
        if len(co_fnd) >= N_SAMPLE:
            break
        aa = dna_to_aa(str(dna))
        if len(aa) < 30:
            continue
        co_dna = co_generate(aa, org_key, model_fnd)
        if co_dna is None:
            continue
        c = compute_csi(co_dna, weights)
        if math.isfinite(c):
            co_fnd.append(c)

    print(f'  Native:    n={len(nat_csi)}, mean={np.mean(nat_csi):.4f}')
    print(f'  Top 10%:   n={len(top10)},  thresh={thresh:.4f}')
    print(f'  CO CSI-FT: n={len(co_csi)}, mean={np.mean(co_csi):.4f}')
    print(f'  CO Found:  n={len(co_fnd)}, mean={np.mean(co_fnd):.4f}')

    org_data[org_key] = {
        'label':         label,
        'native':        nat_csi,
        'top10':         top10,
        'co':            co_csi,
        'co_foundation': co_fnd,
        'thresh':        float(thresh),
    }
    with open(CACHE, 'w') as f:
        json.dump(org_data, f)
    print(f'  Cache updated → {CACHE}')

# ── Plot — all organisms, 8 per page ─────────────────────────────────────────
orgs_plot = [(k, org_data[k]) for k in ALL_ORGS
             if k in org_data and len(org_data[k].get('co', [])) > 0]
print(f'\nPlotting {len(orgs_plot)} organisms ({math.ceil(len(orgs_plot)/8)} pages)...')

PER_PAGE = 8
N_COLS   = 4
pages    = [orgs_plot[i:i+PER_PAGE] for i in range(0, len(orgs_plot), PER_PAGE)]
n_pages  = len(pages)

for pg_idx, page_orgs in enumerate(pages):
    n_orgs   = len(page_orgs)
    n_rows   = math.ceil(n_orgs / N_COLS)
    page_num = pg_idx + 1

    # Shared y-limits for this page
    all_vals = []
    for _, od in page_orgs:
        all_vals.extend(od.get('native', []))
        all_vals.extend(od.get('top10', []))
        all_vals.extend(od.get('co', []))
        all_vals.extend(od.get('co_foundation', []))
    pad  = (max(all_vals) - min(all_vals)) * 0.06
    y_lo = max(0.0, min(all_vals) - pad)
    y_hi = min(1.0, max(all_vals) + pad * 2)

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
            'Top 10%\nCSI':  od.get('top10',  []),
            'CO\nFoundation': od.get('co_foundation', []),
            'CO\nCSI-FT':    od.get('co', []),
        }
        active = [c for c in CONDITIONS if data_map[c]]
        df_v = pd.DataFrame([
            {'Condition': cond, 'CSI': v}
            for cond in active for v in data_map[cond]
        ])

        sns.violinplot(data=df_v, x='Condition', y='CSI',
                       order=active,
                       palette=dict(zip(CONDITIONS, COLORS_V)),
                       hue='Condition', legend=False,
                       inner='box', cut=0, linewidth=0.8, ax=ax)

        thresh = od.get('thresh', float('nan'))
        if math.isfinite(thresh):
            ax.axhline(thresh, color=C_TOP10, lw=1.3, ls='--', alpha=0.85, zorder=1)

        ax.set_ylim(y_lo, y_hi)

        # n= labels
        for i, cond in enumerate(active):
            ax.text(i, y_lo + (y_hi - y_lo) * 0.01, f'n={len(data_map[cond])}',
                    ha='center', va='bottom', fontsize=7.0, color='#666')

        # Mean annotations + above/below indicator
        co_mean  = float(np.mean(od['co'])) if od.get('co') else float('nan')
        fnd_mean = float(np.mean(od['co_foundation'])) if od.get('co_foundation') else float('nan')
        ann_off  = (y_hi - y_lo) * 0.03
        near_band = (y_hi - y_lo) * 0.06

        def clear_offset(val):
            if math.isfinite(thresh) and abs(val - thresh) < near_band:
                return ann_off * 3.2
            return ann_off

        csi_xi = active.index('CO\nCSI-FT')      if 'CO\nCSI-FT'    in active else 3
        fnd_xi = active.index('CO\nFoundation')   if 'CO\nFoundation' in active else 2

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
        ax.set_ylabel('Codon Similarity Index (CSI)' if idx % N_COLS == 0 else '',
                      fontsize=9.5)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.9)
        ax.tick_params(axis='x', labelsize=8.5)
        if idx == 0:
            ax.text(-0.28, 1.12, 'A', transform=ax.transAxes,
                    fontsize=16, fontweight='bold')

    for idx in range(n_orgs, n_rows * N_COLS):
        axes_grid[idx].set_visible(False)

    leg_labels = ['All genes', 'Top 10% CSI', 'CO Foundation (ep11)', 'CO CSI-FT (all39)']
    handles = [mpatches.Patch(color=c, label=l)
               for c, l in zip(COLORS_V, leg_labels)]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=10.0,
               frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        f'Supplementary Figure — CodonOptimus CSI benchmark: all 39 organisms '
        f'(page {page_num} of {n_pages})\n'
        'CSI built from native pretrain sequences.  '
        'Shared y-axis per page.  ▲ CO above / ▼ CO below top-10% cutoff.',
        fontsize=10.5, style='italic', y=1.01,
    )

    for ext in ['png', 'pdf', 'svg']:
        out = BASE / f'figures/figS4_csi_p{page_num}.{ext}'
        kw = {'dpi': 300, 'bbox_inches': 'tight'} if ext != 'svg' \
             else {'format': 'svg', 'bbox_inches': 'tight'}
        fig.savefig(out, **kw)
    plt.close(fig)
    print(f'Page {page_num}: saved → figS4_csi_p{page_num}.png')

print('\nAll pages done.')
