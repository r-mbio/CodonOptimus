#!/usr/bin/env python3
"""
New Fig2: Combined tAI + CSI benchmark for 4 main organisms.
8 panels: A-D = tAI, E-H = CSI
4 conditions per panel: All genes | Top 10% | CO CSI-FT (all39) | CO RS (riboseq specialist)
S.cerevisiae has no RS model → shows CO CSI-FT for both CO columns.
"""
import sys, json, math, random
from pathlib import Path
from scipy import stats
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
sns.set_context('paper', font_scale=1.55)

BASE     = Path(__file__).resolve().parent.parent
PRETRAIN = BASE / 'data/pretrain/all_industrial_cds.tsv'
TAI_DIR  = BASE / 'data' / 'trna'
CACHE    = BASE / 'results/fig2_combined_cache.json'
OUT      = BASE / 'figures/fig2_tai_csi_combined.png'
OUTP     = BASE / 'figures/fig2_tai_csi_combined.pdf'
OUTS     = BASE / 'figures/fig2_tai_csi_combined.svg'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

STOP_C = {'TAA', 'TAG', 'TGA'}
N_SAMPLE, SEED = 500, 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

ORGS = ['ecoli', 'bacillus', 's_cerevisiae', 'pichia']
ORG_LABELS = {
    'ecoli':        'E. coli',
    'bacillus':     'B. subtilis',
    's_cerevisiae': 'S. cerevisiae',
    'pichia':       'K. phaffii',
}
TAI_FILES = {
    'ecoli':        'escherichia_coli_tai.txt',
    'bacillus':     'bacillus_subtilis_tai.txt',
    's_cerevisiae': 'saccharomyces_cerevisiae_tai.txt',
    'pichia':       'pichia_pastoris_gca_tai.txt',
}
RS_MODELS = {
    'ecoli':        BASE / 'models/industrial_mlm_all39_rs_ecoli.pt',
    'bacillus':     BASE / 'models/industrial_mlm_all39_rs_bacillus.pt',
    's_cerevisiae': BASE / 'models/industrial_mlm_all39_rs_s_cerevisiae.pt',
    'pichia':       BASE / 'models/industrial_mlm_all39_rs_pichia.pt',
}
CSI_MODEL = BASE / 'models/industrial_mlm_csi_all39.pt'
FND_MODEL = BASE / 'models/industrial_mlm_ep11.pt'

plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':       11,
    'axes.labelsize':  12,
    'axes.titlesize':  13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.linewidth':  1.0,
    'pdf.fonttype':    42,
})

C_ALL   = '#E69F00'   # amber        — all native genes
C_TOP10 = '#56B4E9'   # sky blue     — top 10% native
C_COFND = '#A8D4F0'   # very light blue — CO Foundation (ep11)
C_COCSI = '#009E73'   # green        — CO CSI-FT (Wong colorblind-safe)
C_CORS  = '#0072B2'   # dark blue    — CO RS (Wong colorblind-safe)

# 5-condition colors — all Wong/Okabe-Ito, colorblind-safe
COND_COLORS = [C_ALL, C_TOP10, C_COFND, C_COCSI, C_CORS]
CONDITIONS_TAI = ['All genes', 'Top 10%\ntAI',  'CO-F', 'CO-CSI-FT', 'CO-RS-FT']
CONDITIONS_CSI = ['All genes', 'Top 10%\nCSI',  'CO-F', 'CO-CSI-FT', 'CO-RS-FT']

# ── Helpers ────────────────────────────────────────────────────────────────────
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
    aa = []
    for c in dna_to_codons(dna):
        a = _GENETIC_CODE.get(c, '')
        if not a or a == '*': break
        aa.append(a)
    return ''.join(aa)


def load_tai_table(path):
    df = pd.read_csv(path, sep='\t')
    df.columns = df.columns.str.strip()
    df['Codon'] = df['Codon'].str.strip().str.upper()
    return dict(zip(df['Codon'], df['tAI']))


def compute_tai(dna, w):
    log_s, n = 0.0, 0
    for c in dna_to_codons(str(dna)):
        wt = w.get(c, 0)
        if wt > 0:
            log_s += math.log(wt); n += 1
    return math.exp(log_s / n) if n > 0 else float('nan')


def build_csi_weights(seqs):
    pool = []
    for dna in seqs:
        codons = dna_to_codons(str(dna))
        if len(codons) >= 30:
            pool.append(''.join(codons) + 'TAA')
    if not pool: return None
    try: return relative_adaptiveness(sequences=pool)
    except: return None


def compute_csi(dna, weights):
    seq = ''.join(dna_to_codons(str(dna)))
    if len(seq) < 9 or weights is None: return float('nan')
    try: return CAI(seq, weights)
    except: return float('nan')


@torch.no_grad()
def generate(aa_seq, org_key, mdl):
    if org_key not in ORG_TO_ID: return None
    valid = [a for a in aa_seq.upper() if a in AA_MASK]
    if len(valid) < 30: return None
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
            ml = torch.full((61,), float('-inf'), device=DEVICE)
            for t in allowed:
                if t < 61: ml[t] = 0.0
            pl = pl + ml
        result.append(ID_TO_PAIR[pl.argmax().item()][1])
    return ''.join(result)


def load_model(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    m = IndustrialMLM(use_grad_ckpt=False)
    m.load_state_dict(ckpt['model'])
    return m.eval().to(DEVICE)


# ── Load models ────────────────────────────────────────────────────────────────
print('Loading CO Foundation model (ep11)...')
model_fnd = load_model(FND_MODEL)
print('Loading CO CSI-FT model...')
model_csi = load_model(CSI_MODEL)

rs_models = {}
for org, path in RS_MODELS.items():
    print(f'Loading RS model for {org}...')
    rs_models[org] = load_model(path)

# ── Load pretrain data ─────────────────────────────────────────────────────────
print('Loading pretrain sequences...')
df_all = pd.read_csv(PRETRAIN, sep='\t', usecols=['organism', 'dna'])

# ── Load or build cache ────────────────────────────────────────────────────────
if CACHE.exists():
    with open(CACHE) as f:
        cache = json.load(f)
    print(f'Loaded cache: {list(cache.keys())}')
else:
    cache = {}

for org in ORGS:
    if org in cache and all(k in cache[org] for k in
                             ['nat_tai','top10_tai','nat_csi','top10_csi',
                              'co_fnd_tai','co_csi_tai','co_rs_tai',
                              'co_fnd_csi','co_csi_csi','co_rs_csi']):
        print(f'  {org}: fully cached ✓')
        continue

    label = ORG_LABELS[org]
    print(f'\n── {label} ({org}) ──')

    # Native sequences
    sub   = df_all[df_all['organism'] == org]['dna'].dropna().values
    valid = [d for d in sub if 150*3 <= len(str(d)) <= 600*3]
    print(f'  Valid seqs: {len(valid):,}')

    rng = np.random.default_rng(SEED)

    # tAI setup
    tai_path = TAI_DIR / TAI_FILES[org]
    tai_w    = load_tai_table(tai_path)

    # CSI setup — build from native
    ref_seqs = rng.choice(valid, size=min(5000, len(valid)), replace=False)
    csi_w    = build_csi_weights(ref_seqs)

    # Native samples
    nat_sample = rng.choice(valid, size=min(N_SAMPLE, len(valid)), replace=False)
    nat_tai = [compute_tai(d, tai_w) for d in nat_sample]
    nat_tai = [x for x in nat_tai if math.isfinite(x)]
    nat_csi = [compute_csi(d, csi_w) for d in nat_sample]
    nat_csi = [x for x in nat_csi if math.isfinite(x)]

    # Top-10% pools
    all_tai = [(d, compute_tai(d, tai_w)) for d in valid[:5000]]
    all_tai = [(d, t) for d, t in all_tai if math.isfinite(t)]
    thresh_tai = np.percentile([t for _, t in all_tai], 90)
    top10_tai  = [t for _, t in all_tai if t >= thresh_tai]

    all_csi = [(d, compute_csi(d, csi_w)) for d in valid[:5000]]
    all_csi = [(d, c) for d, c in all_csi if math.isfinite(c)]
    thresh_csi = np.percentile([c for _, c in all_csi], 90)
    top10_csi  = [c for _, c in all_csi if c >= thresh_csi]

    # Restore existing data if partially cached
    existing = cache.get(org, {})

    # Pools for generation
    pool  = rng.choice(valid, size=min(N_SAMPLE*4, len(valid)), replace=False)
    pool2 = np.random.default_rng(SEED+1).choice(valid, size=min(N_SAMPLE*4, len(valid)), replace=False)
    pool3 = np.random.default_rng(SEED+2).choice(valid, size=min(N_SAMPLE*4, len(valid)), replace=False)

    def gen_sequences(pool_seqs, mdl, n=N_SAMPLE):
        tai_l, csi_l = [], []
        for dna in pool_seqs:
            if len(tai_l) >= n: break
            aa = dna_to_aa(str(dna))
            if len(aa) < 30: continue
            gen = generate(aa, org, mdl)
            if gen is None: continue
            t = compute_tai(gen, tai_w)
            c = compute_csi(gen, csi_w)
            if math.isfinite(t): tai_l.append(t)
            if math.isfinite(c): csi_l.append(c)
        return tai_l, csi_l

    # CO Foundation
    if 'co_fnd_tai' in existing:
        co_fnd_tai, co_fnd_csi = existing['co_fnd_tai'], existing['co_fnd_csi']
        print(f'  CO Foundation: cached (n={len(co_fnd_tai)})')
    else:
        co_fnd_tai, co_fnd_csi = gen_sequences(pool3, model_fnd)

    # CO CSI-FT
    if 'co_csi_tai' in existing:
        co_csi_tai, co_csi_csi = existing['co_csi_tai'], existing['co_csi_csi']
        print(f'  CO CSI-FT: cached (n={len(co_csi_tai)})')
    else:
        co_csi_tai, co_csi_csi = gen_sequences(pool, model_csi)

    # CO RS
    if 'co_rs_tai' in existing:
        co_rs_tai, co_rs_csi = existing['co_rs_tai'], existing['co_rs_csi']
        print(f'  CO RS: cached (n={len(co_rs_tai)})')
    else:
        co_rs_tai, co_rs_csi = gen_sequences(pool2, rs_models[org])

    # Recompute native if not cached
    nat_tai  = existing.get('nat_tai',  nat_tai)
    nat_csi  = existing.get('nat_csi',  nat_csi)
    top10_tai = existing.get('top10_tai', top10_tai)
    top10_csi = existing.get('top10_csi', top10_csi)
    thresh_tai = existing.get('thresh_tai', float(thresh_tai))
    thresh_csi = existing.get('thresh_csi', float(thresh_csi))

    print(f'  nat tAI:        n={len(nat_tai)},  mean={np.mean(nat_tai):.4f}')
    print(f'  CO Foundation:  tAI={np.mean(co_fnd_tai):.4f}  CSI={np.mean(co_fnd_csi):.4f}')
    print(f'  CO CSI-FT:      tAI={np.mean(co_csi_tai):.4f}  CSI={np.mean(co_csi_csi):.4f}')
    print(f'  CO RS:          tAI={np.mean(co_rs_tai):.4f}   CSI={np.mean(co_rs_csi):.4f}')

    cache[org] = {
        'nat_tai':    nat_tai,    'top10_tai': top10_tai, 'thresh_tai': float(thresh_tai),
        'nat_csi':    nat_csi,    'top10_csi': top10_csi, 'thresh_csi': float(thresh_csi),
        'co_fnd_tai': co_fnd_tai, 'co_fnd_csi': co_fnd_csi,
        'co_csi_tai': co_csi_tai, 'co_rs_tai':  co_rs_tai,
        'co_csi_csi': co_csi_csi, 'co_rs_csi':  co_rs_csi,
    }
    with open(CACHE, 'w') as f:
        json.dump(cache, f)
    print(f'  Cache updated → {CACHE}')

# ── Compute shared y-limits across all organisms per metric row ────────────────
def row_ylims(keys):
    all_vals = []
    for org in ORGS:
        od = cache[org]
        for k in keys:
            all_vals.extend(od.get(k, []))
    rng = max(all_vals) - min(all_vals)
    # Extra headroom (30%) for significance brackets inside the border
    return max(0.0, min(all_vals) - rng * 0.04), min(1.0, max(all_vals) + rng * 0.30)

tai_ylo, tai_yhi = row_ylims(['nat_tai','top10_tai','co_fnd_tai','co_csi_tai','co_rs_tai'])
csi_ylo, csi_yhi = row_ylims(['nat_csi','top10_csi','co_fnd_csi','co_csi_csi','co_rs_csi'])

# ── Plot ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 11))
gs  = GridSpec(2, 4, figure=fig, hspace=0.62, wspace=0.30,
               left=0.07, right=0.97, top=0.90, bottom=0.10)

PANEL_LETTERS = list('ABCDEFGH')

for row, (metric, conditions, thresh_key, co_fnd_key, co_csi_key, co_rs_key, ylabel, ylo, yhi) in enumerate([
    ('tAI', CONDITIONS_TAI, 'thresh_tai', 'co_fnd_tai', 'co_csi_tai', 'co_rs_tai',
     'tRNA Adaptation Index (tAI)', tai_ylo, tai_yhi),
    ('CSI', CONDITIONS_CSI, 'thresh_csi', 'co_fnd_csi', 'co_csi_csi', 'co_rs_csi',
     'Codon Similarity Index (CSI)', csi_ylo, csi_yhi),
]):
    nat_key = 'nat_tai' if metric == 'tAI' else 'nat_csi'
    top_key = 'top10_tai' if metric == 'tAI' else 'top10_csi'

    for col, org in enumerate(ORGS):
        ax  = fig.add_subplot(gs[row, col])
        od  = cache[org]
        rs_label = 'CO\nCSI-FT*' if od.get('rs_is_csift') else 'CO\nRS'

        conds  = conditions   # 5 conditions — no special RS label needed (S.cer has real RS now)
        colors = COND_COLORS

        data_map = {
            conditions[0]: od[nat_key],
            conditions[1]: od[top_key],
            conditions[2]: od[co_fnd_key],
            conditions[3]: od[co_csi_key],
            conditions[4]: od[co_rs_key],
        }
        df_v = pd.DataFrame([
            {'Condition': c, metric: v}
            for c in conds for v in data_map[c]
        ])

        sns.violinplot(data=df_v, x='Condition', y=metric,
                       order=conds,
                       palette=dict(zip(conds, colors)),
                       hue='Condition', legend=False,
                       inner='box', cut=0, linewidth=0.9, ax=ax)

        # Apply shared y-limits
        ax.set_ylim(ylo, yhi)
        ann_off = (yhi - ylo) * 0.025

        thresh = od[thresh_key]
        ax.axhline(thresh, color=C_TOP10, lw=1.3, ls='--', alpha=0.80, zorder=1)

        # Vertical divider between native (0,1) and CO (2,3)
        ax.axvline(1.5, color='#888888', lw=1.0, ls='--', alpha=0.55, zorder=0)

        for i, c in enumerate(conds):
            n = len(data_map[c])
            ax.text(i, ylo + (yhi - ylo) * 0.01, f'n={n}',
                    ha='center', va='bottom', fontsize=8.5, color='#555')

        # CO mean annotations (Foundation, CSI-FT, RS)
        for xi, key, col_c in [(2, co_fnd_key, '#4488BB'),
                                (3, co_csi_key, C_COCSI),
                                (4, co_rs_key,  C_CORS)]:
            vals = od[key]
            m = np.mean(vals) if vals else float('nan')
            if math.isfinite(m) and ylo < m < (yhi - (yhi-ylo)*0.28):
                ax.text(xi, m + ann_off, f'{m:.3f}',
                        ha='center', va='bottom', fontsize=9,
                        color=col_c, fontweight='bold')

        # Significance: Top 10% vs each CO model — staggered, clipped inside axes
        top10_vals = data_map[conds[1]]
        span = yhi - ylo
        # Place 3 brackets in top 25% of y-range
        sig_heights = [yhi - span * 0.26,
                       yhi - span * 0.17,
                       yhi - span * 0.08]
        for (xi, key), bh in zip([(2, co_fnd_key), (3, co_csi_key), (4, co_rs_key)],
                                   sig_heights):
            co_vals = od[key]
            if len(top10_vals) > 1 and len(co_vals) > 1:
                _, p = stats.mannwhitneyu(top10_vals, co_vals, alternative='two-sided')
                sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
                x1, x2 = 1, xi
                tick = span * 0.008
                ax.plot([x1, x1, x2, x2],
                        [bh - tick, bh, bh, bh - tick],
                        color='#444', lw=1.0, clip_on=True, zorder=5)
                ax.text((x1 + x2) / 2, bh + tick * 0.5, sig,
                        ha='center', va='bottom', fontsize=10,
                        color='#222', fontweight='bold')

        label = ORG_LABELS[org]
        if row == 0:
            ax.set_title(f'$\\it{{{label.replace(" ", "\\ ")}}}$',
                         fontsize=11.5, pad=6)
        ax.set_xlabel('')
        ax.set_ylabel(ylabel if col == 0 else '', fontsize=10.5)
        ax.tick_params(axis='x', labelsize=8.5)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.9)

        letter = PANEL_LETTERS[row * 4 + col]
        ax.text(-0.20, 1.08, letter, transform=ax.transAxes,
                fontsize=16, fontweight='bold', ha='left', va='bottom')

# ── Legend ─────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(color=COND_COLORS[0], label='All genes (n=500)'),
    mpatches.Patch(color=COND_COLORS[1], label='Top 10% (n=500)'),
    mpatches.Patch(color=COND_COLORS[2], label='CO-F: Foundation (ep11, pre-trained, n=500)'),
    mpatches.Patch(color=COND_COLORS[3], label='CO-CSI-FT: all39, all layers unfrozen (n=500)'),
    mpatches.Patch(color=COND_COLORS[4], label='CO-RS-FT: Ribo-seq specialist, last 4 layers (n=500)'),
]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=10.0,
           frameon=False, bbox_to_anchor=(0.5, -0.02))


fig.suptitle(
    'CodonOptimus tAI and CSI benchmark — 4 main host organisms\n'
    'Rows A–D: tRNA Adaptation Index  |  Rows E–H: Codon Similarity Index',
    fontsize=12.5, y=0.97
)

fig.savefig(OUT,  dpi=200, bbox_inches='tight')
fig.savefig(OUTP, dpi=300, bbox_inches='tight')
fig.savefig(OUTS, format='svg', bbox_inches='tight')
print(f'\nSaved → {OUT}')
