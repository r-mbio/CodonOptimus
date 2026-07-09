#!/usr/bin/env python3
"""
Figure S2: Organism conditioning ablation.
Synonymous codon accuracy on held-out native genes WITH vs WITHOUT organism token.

Metric: exact codon match rate — fraction of codons in generated sequence
        that match the native (reference) codon. Since generation is constrained
        to synonymous codons, this purely measures organism-specific codon choice.

Model: industrial_mlm_ep11.pt (base pretrained model, before any FT)
Data:  200 held-out native genes per organism (random seed 42)
"""

import sys, random
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / '02_foundation_pretraining'))
sys.path.insert(0, str(_REPO_ROOT / '03_finetuning'))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

from train_mlm_industrial import (
    IndustrialMLM, AA_CODON_VOCAB, ID_TO_PAIR, ORG_TO_ID,
    CLS_TOKEN, PAD_TOKEN, _GENETIC_CODE, AA_MASK, AMINO_ACIDS,
)

BASE   = Path(__file__).resolve().parent.parent
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

STOP_C = {'TAA', 'TAG', 'TGA'}

_AA_TO_TIDS = {}
for (_aa, _cdn), _tid in AA_CODON_VOCAB.items():
    _AA_TO_TIDS.setdefault(_aa, []).append(_tid)


def dna2codons(dna):
    dna = str(dna).upper().replace(' ', '')
    if len(dna) >= 3 and dna[-3:] in STOP_C:
        dna = dna[:-3]
    codons = []
    for i in range(0, len(dna) - 2, 3):
        cdn = dna[i:i+3]
        if 'N' in cdn or cdn in STOP_C:
            break
        codons.append(cdn)
    return codons


def dna2aa(dna):
    aa = []
    for i in range(0, len(dna) - 2, 3):
        a = _GENETIC_CODE.get(dna[i:i+3].upper(), 'X')
        if a in ('*', 'X'):
            break
        aa.append(a)
    return ''.join(aa)


@torch.no_grad()
def co_greedy(aa_seq, org_id_tensor, model):
    """Generate codons given AA sequence and organism id tensor."""
    valid = [a for a in aa_seq.upper() if a in AA_MASK]
    tok   = [CLS_TOKEN] + [AA_MASK[a] for a in valid]
    ids   = torch.tensor([tok], dtype=torch.long, device=DEVICE)
    msk   = torch.zeros(1, len(tok), dtype=torch.bool, device=DEVICE)
    logits = model(ids, msk, org_id_tensor)
    result = []
    for pi, aa_ch in enumerate(valid):
        allowed = _AA_TO_TIDS.get(aa_ch, [])
        pl = logits[0, pi+1, :61].clone()
        if allowed:
            mask_l = torch.full((61,), float('-inf'), device=DEVICE)
            for t in allowed:
                if t < 61:
                    mask_l[t] = 0.0
            pl = pl + mask_l
        result.append(ID_TO_PAIR[pl.argmax().item()][1])
    return result


def codon_accuracy(generated, native):
    """Fraction of positions where generated codon == native codon."""
    n = min(len(generated), len(native))
    if n == 0:
        return float('nan')
    matches = sum(g == r for g, r in zip(generated[:n], native[:n]))
    return matches / n


# ── Load model ──────────────────────────────────────────────────────────────────
print('Loading model...')
ckpt = torch.load(BASE / 'models/industrial_mlm_ep11.pt',
                  map_location='cpu', weights_only=False)
cfg  = ckpt.get('config', {})
model = IndustrialMLM(
    vocab_size    = cfg.get('vocab_size',    83),
    d_model       = cfg.get('d_model',       512),
    nhead         = cfg.get('nhead',         8),
    num_layers    = cfg.get('num_layers',    12),
    dim_ffn       = cfg.get('dim_ffn',       2048),
    num_orgs      = cfg.get('num_orgs',      39),
    dropout       = 0.0,
    use_grad_ckpt = False,
)
model.load_state_dict(ckpt['model'])
model.to(DEVICE).eval()
print('Model loaded.')

# ── Load data ───────────────────────────────────────────────────────────────────
print('Loading data...')
df = pd.read_csv(BASE / 'data/pretrain/all_industrial_cds.tsv', sep='\t')
print(f'Total rows: {len(df)}')

ORG_CFG = [
    ('ecoli',        'E. coli',        'E.coli'),
    ('bacillus',     'B. subtilis',    'B.sub'),
    ('s_cerevisiae', 'S. cerevisiae',  'S.cer'),
    ('pichia',       'K. phaffii',     'K.phaffii'),
]

N_SAMPLE     = 200
MIN_CODONS   = 50
MAX_CODONS   = 400
RANDOM_SEED  = 42

results = {}   # org_key → {'with': [...], 'without': [...]}

for org_key, org_label, org_short in ORG_CFG:
    print(f'\n── {org_label} ──')
    sub = df[df['organism'] == org_key].copy()
    print(f'  Available: {len(sub)} sequences')
    if len(sub) == 0:
        print('  SKIP (no data)')
        continue

    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(len(sub))
    sub = sub.iloc[idx].reset_index(drop=True)

    org_id  = ORG_TO_ID[org_key]
    oid_t   = torch.tensor([org_id], dtype=torch.long, device=DEVICE)
    dummy_t = torch.tensor([0],      dtype=torch.long, device=DEVICE)  # placeholder

    acc_with    = []
    acc_without = []

    for _, row in sub.iterrows():
        if len(acc_with) >= N_SAMPLE:
            break
        dna = str(row['dna']).upper().replace(' ', '')
        native_codons = dna2codons(dna)
        if len(native_codons) < MIN_CODONS or len(native_codons) > MAX_CODONS:
            continue
        aa_seq = dna2aa(dna)
        if len(aa_seq) < MIN_CODONS:
            continue

        # Generate WITH organism token
        gen_with = co_greedy(aa_seq, oid_t, model)
        acc_with.append(codon_accuracy(gen_with, native_codons))

        # Generate WITHOUT organism token: zero out org_emb temporarily
        saved_weights = model.org_emb.weight.data.clone()
        model.org_emb.weight.data.zero_()
        gen_without = co_greedy(aa_seq, dummy_t, model)
        model.org_emb.weight.data.copy_(saved_weights)
        acc_without.append(codon_accuracy(gen_without, native_codons))

    results[org_key] = {
        'with':    np.array(acc_with),
        'without': np.array(acc_without),
        'label':   org_label,
        'short':   org_short,
    }
    mean_w  = np.mean(acc_with)
    mean_wo = np.mean(acc_without)
    delta   = mean_w - mean_wo
    t, p    = stats.wilcoxon(acc_with, acc_without)
    print(f'  n={len(acc_with)}, with={mean_w:.3f}, without={mean_wo:.3f}, '
          f'Δ={delta:+.3f}, p={p:.2e}')


# ── Plot ─────────────────────────────────────────────────────────────────────────
print('\nPlotting...')
fig, ax = plt.subplots(figsize=(7.5, 4.5))

COLOR_WITH    = '#0072B2'   # CO blue
COLOR_WITHOUT = '#888888'   # CT grey
JITTER        = 0.08

orgs_ordered = [k for k, _, _ in ORG_CFG if k in results]
n_orgs = len(orgs_ordered)
x_positions = np.arange(n_orgs)

for xi, org_key in enumerate(orgs_ordered):
    d = results[org_key]
    vals_w  = d['with']
    vals_wo = d['without']

    for color, vals, xoff in [
        (COLOR_WITHOUT, vals_wo, -0.18),
        (COLOR_WITH,    vals_w,  +0.18),
    ]:
        bp = ax.boxplot(
            vals,
            positions=[xi + xoff],
            widths=0.28,
            patch_artist=True,
            notch=False,
            showfliers=False,
            medianprops=dict(color='black', linewidth=2),
            boxprops=dict(facecolor=color, alpha=0.75, linewidth=1.2),
            whiskerprops=dict(color='#333333', linewidth=1.2),
            capprops=dict(color='#333333', linewidth=1.2),
        )
        # Individual data points (jitter)
        rng = np.random.default_rng(42)
        jx  = xi + xoff + rng.uniform(-JITTER, JITTER, len(vals))
        ax.scatter(jx, vals, color=color, alpha=0.25, s=8, zorder=3,
                   linewidths=0)

    # Significance annotation
    _, p = stats.wilcoxon(vals_w, vals_wo)
    ymax = max(np.percentile(vals_w, 95), np.percentile(vals_wo, 95)) + 0.02
    sig  = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    ax.text(xi, ymax + 0.015, sig, ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='#333333')

# Axes formatting
ax.set_xticks(x_positions)
ax.set_xticklabels([results[k]['label'] for k in orgs_ordered],
                   fontsize=11, style='italic')
ax.set_ylabel('Synonymous codon accuracy', fontsize=12)
ax.set_xlabel('')
ax.set_ylim(0.0, 0.65)
ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax.yaxis.set_tick_params(labelsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.8)

legend_handles = [
    mpatches.Patch(facecolor=COLOR_WITH,    alpha=0.75, label='With organism token'),
    mpatches.Patch(facecolor=COLOR_WITHOUT, alpha=0.75, label='Without organism token'),
]
ax.legend(handles=legend_handles, fontsize=10, frameon=False,
          loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=2)

plt.tight_layout()
out_png = BASE / 'figures/figS5_organism_conditioning.png'
out_pdf = BASE / 'figures/figS5_organism_conditioning.pdf'
out_svg = BASE / 'figures/figS5_organism_conditioning.svg'
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.savefig(out_svg, bbox_inches='tight')
print(f'Saved: {out_png}')

# Print summary table
print('\n── Summary ──')
print(f"{'Organism':<16} {'With org':>10} {'Without org':>12} {'Δ':>8} {'p-value':>12}")
for org_key in orgs_ordered:
    d = results[org_key]
    w  = np.mean(d['with'])
    wo = np.mean(d['without'])
    _, p = stats.wilcoxon(d['with'], d['without'])
    print(f"{d['label']:<16} {w:>10.3f} {wo:>12.3f} {w-wo:>+8.3f} {p:>12.2e}")
