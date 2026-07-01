#!/usr/bin/env python3
"""
Figure 2: CodonOptimus foundation model learns organism-specific codon landscapes
and enables dual-head prediction

Panel A: tSNE of 39-organism embeddings (coloured by phylogenetic group,
         dot size proportional to training corpus size)
Panel B: RSCU heatmap — synonymous codon usage across 39 organisms
         (hierarchically clustered; rows=codons, cols=organisms)
Panel C: Dual-head prediction accuracy — Expression R² and A-site Spearman r
         per organism (grouped bar chart)
"""

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / '02_foundation_pretraining'))
sys.path.insert(0, str(_REPO_ROOT / '03_finetuning'))

import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from collections import defaultdict
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

from train_mlm_industrial import IndustrialMLM, ORG_ORDER, ORG_TO_ID

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
CKPT = BASE / 'models' / 'industrial_mlm_pretrain.pt'
TSV  = BASE / 'data' / 'pretrain' / 'all_industrial_cds.tsv'
OUT  = BASE / 'figures' / 'figS1_organism_landscape.pdf'
OUTP = BASE / 'figures' / 'figS1_organism_landscape.png'
OUTS = BASE / 'figures' / 'figS1_organism_landscape.svg'

# ── Publication rcParams ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':         11,
    'axes.labelsize':    12,
    'axes.titlesize':    13,
    'xtick.labelsize':   10.5,
    'ytick.labelsize':   10.5,
    'legend.fontsize':   10,
    'axes.linewidth':    0.9,
    'xtick.major.width': 0.9,
    'ytick.major.width': 0.9,
    'xtick.major.size':  4.0,
    'ytick.major.size':  4.0,
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})

# ── Phylogenetic groups ───────────────────────────────────────────────────────
GROUPS = {
    'Bacteria': ['ecoli','bacillus','bacillus_licheniformis','bacillus_amyloliquefaciens',
                 'brevibacillus','lactococcus','lactobacillus','corynebacterium',
                 'pseudomonas_putida','gluconobacter','cupriavidus','streptomyces',
                 'streptomyces_lividans','clostridium'],
    'Yeast':    ['s_cerevisiae','pichia','yarrowia','kluyveromyces','kluyveromyces_marxianus',
                 'ogataea','scheffersomyces','ashbya','rhodotorula'],
    'Filamentous fungi': ['aspergillus_niger','aspergillus_oryzae','aspergillus_fumigatus',
                          'aspergillus_terreus','penicillium','trichoderma','fusarium',
                          'neurospora','rhizopus','mucor','myceliophthora'],
    'Microalgae': ['nannochloropsis','phaeodactylum','chlamydomonas'],
    'Mammalian':  ['cho','human'],
}
GROUP_COLORS = {
    'Bacteria':          '#D55E00',   # Wong vermillion
    'Yeast':             '#0072B2',   # Wong blue
    'Filamentous fungi': '#009E73',   # Wong green ✓
    'Microalgae':        '#E69F00',   # Wong amber
    'Mammalian':         '#CC79A7',   # Wong reddish purple
}
def org_to_group(org):
    for g, members in GROUPS.items():
        if org in members: return g
    return 'Other'

# ── Dual-head performance per organism (from ep45 training log) ───────────────
EXPR_R2 = {
    'ecoli':        0.654,
    'bacillus':     0.267,
    'pichia':       0.401,
    's_cerevisiae': 0.580,
}
ASITE_R = {
    'ecoli':        0.424,
    'pichia':       0.460,
    's_cerevisiae': 0.540,
    # bacillus: no riboseq data — encoder frozen preserves generation capability
}

# ── Key organism display names ────────────────────────────────────────────────
KEY_ORGS = {
    'pichia':        'K. phaffii',
    'ecoli':         'E. coli',
    'bacillus':      'B. subtilis',
    's_cerevisiae':  'S. cerevisiae',
    'cho':           'CHO',
    'human':         'Human',
    'yarrowia':      'Y. lipolytica',
    'corynebacterium':'C. glutamicum',
    'trichoderma':   'T. reesei',
    'aspergillus_niger': 'A. niger',
}

ORG_LABELS_PRETTY = {
    'pichia':             'K. phaffii',
    'ecoli':              'E. coli',
    'bacillus':           'B. subtilis',
    's_cerevisiae':       'S. cerevisiae',
    'cho':                'CHO',
    'human':              'Human',
    'yarrowia':           'Y. lipolytica',
    'corynebacterium':    'C. glutamicum',
    'trichoderma':        'T. reesei',
    'lactococcus':        'L. lactis',
    'pseudomonas_putida': 'P. putida',
    'aspergillus_niger':  'A. niger',
    'nannochloropsis':    'Nannochloropsis',
    'kluyveromyces':      'K. lactis',
    'kluyveromyces_marxianus': 'K. marxianus',
}

# ── Genetic code ──────────────────────────────────────────────────────────────
GENETIC_CODE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L',
    'CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M',
    'GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S',
    'CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T',
    'GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','CAT':'H','CAC':'H',
    'CAA':'Q','CAG':'Q','AAT':'N','AAC':'N',
    'AAA':'K','AAG':'K','GAT':'D','GAC':'D',
    'GAA':'E','GAG':'E','TGT':'C','TGC':'C','TGG':'W',
    'CGT':'R','CGC':'R','CGA':'R','CGG':'R','AGA':'R','AGG':'R',
    'AGT':'S','AGC':'S',
    'GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}
AA_CODONS = defaultdict(list)
for codon, aa in GENETIC_CODE.items():
    if aa not in ('*',):
        AA_CODONS[aa].append(codon)

SYN_CODONS = [c for c in GENETIC_CODE
              if GENETIC_CODE[c] not in ('M','W','*')]

def compute_rscu(dna_seqs, max_seqs=5000):
    counts    = defaultdict(int)
    aa_counts = defaultdict(int)
    seqs = dna_seqs[:max_seqs]
    for dna in seqs:
        for i in range(0, len(dna)-2, 3):
            codon = dna[i:i+3]
            aa    = GENETIC_CODE.get(codon)
            if aa and aa != '*':
                counts[codon] += 1
                aa_counts[aa] += 1
    rscu = {}
    for codon, aa in GENETIC_CODE.items():
        if aa == '*': continue
        syn = AA_CODONS[aa]
        n   = len(syn)
        freq = counts[codon] / aa_counts[aa] if aa_counts[aa] > 0 else 0
        rscu[codon] = freq * n
    return rscu


# ══════════════════════════════════════════════════════════════════════════════
print('Loading training counts...')
org_counts_df = pd.read_csv(TSV, sep='\t', usecols=['organism'])
ORG_COUNTS    = org_counts_df['organism'].value_counts().to_dict()

print('Loading model...')
model = IndustrialMLM()
ckpt  = torch.load(CKPT, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()

with torch.no_grad():
    org_ids = torch.arange(len(ORG_ORDER), dtype=torch.long)
    emb     = model.org_emb(org_ids).numpy()   # (39, 512)

# tSNE — with 39 points, skip PCA (it distorts distances at low n)
# Use perplexity=8 (rule of thumb: ~n/5), more iterations for stability
tsne = TSNE(n_components=2, perplexity=8, random_state=42,
            max_iter=3000, learning_rate='auto', init='pca')
xy   = tsne.fit_transform(emb)   # (39, 2)
# Normalize to ±10 range for clean axis labels and annotation offsets
xy   = (xy - xy.mean(0)) / xy.std(0) * 6.0

print('Computing RSCU...')
df_sample   = pd.read_csv(TSV, sep='\t', usecols=['dna','organism'])
rscu_matrix = {}
for org in ORG_ORDER:
    seqs = df_sample[df_sample['organism'] == org]['dna'].values
    rscu_matrix[org] = compute_rscu(seqs, max_seqs=5000)
    print(f'  {org}: {len(seqs)} seqs')

# RSCU dataframe (organisms × codons)
rscu_df = pd.DataFrame(rscu_matrix, index=SYN_CODONS).T

# ── Build figure ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 7.0))
gs  = fig.add_gridspec(1, 2, wspace=0.38, width_ratios=[1, 2.0])
ax1 = fig.add_subplot(gs[0])

# Panel B: dendrogram (top) + seaborn heatmap (bottom), nested gridspec
gs_b = gs[1].subgridspec(2, 1, height_ratios=[0.13, 1], hspace=0)
ax2d = fig.add_subplot(gs_b[0])
ax2  = fig.add_subplot(gs_b[1])

# ══════════════════════════════════════════════════════════════════════════════
# Panel A: tSNE
# ══════════════════════════════════════════════════════════════════════════════
HIGHLIGHT = {'pichia', 'ecoli', 'bacillus', 's_cerevisiae'}

for i, org in enumerate(ORG_ORDER):
    group = org_to_group(org)
    color = GROUP_COLORS.get(group, '#AAAAAA')
    count = ORG_COUNTS.get(org, 0)
    size  = min(80 + (count / 8000) * 40, 280)   # larger dots
    zord  = 5 if org in HIGHLIGHT else 3
    ew    = 1.5 if org in HIGHLIGHT else 0.5
    ec    = '#222' if org in HIGHLIGHT else 'white'

    ax1.scatter(xy[i, 0], xy[i, 1], s=size, c=color,
                alpha=0.90, edgecolors=ec, linewidths=ew, zorder=zord)

    if org in KEY_ORGS:
        name  = KEY_ORGS[org]
        label = f'$\\it{{{name.replace(" ", "\\ ")}}}$' if ' ' in name else name
        # offset proportional to normalized scale (±6 units)
        xoff = 1.2
        yoff = 1.0
        ax1.annotate(
            label,
            xy=(xy[i, 0], xy[i, 1]),
            xytext=(xy[i, 0] + xoff, xy[i, 1] + yoff),
            fontsize=8, ha='left', color='#222',
            arrowprops=dict(arrowstyle='-', lw=0.6, color='#888'),
        )

legend_handles = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor=c,
           markersize=8, label=g, markeredgecolor='#ccc', markeredgewidth=0.4)
    for g, c in GROUP_COLORS.items()
]
# Placed fully outside the plotted data area (below the axes) to guarantee
# no overlap with any scatter point, regardless of t-SNE layout on rerun.
leg1 = ax1.legend(handles=legend_handles, fontsize=9.5, ncol=3,
                   loc='upper center', bbox_to_anchor=(0.5, -0.10),
                   framealpha=1.0, edgecolor='#ddd', facecolor='white')
ax1.set_title('Organism embedding landscape\n'
              '(t-SNE; dot size proportional to training corpus)', fontsize=13)
ax1.set_xlabel('t-SNE dim 1', fontsize=12)
ax1.set_ylabel('t-SNE dim 2', fontsize=12)
for sp in ax1.spines.values(): sp.set_visible(True); sp.set_linewidth(0.9)
ax1.text(-0.14, 1.03, 'A', transform=ax1.transAxes,
         ha='left', va='bottom', fontsize=18, fontweight='bold')

# ══════════════════════════════════════════════════════════════════════════════
# Panel B: RSCU heatmap
# ══════════════════════════════════════════════════════════════════════════════
Z_org     = linkage(pdist(rscu_df.values, metric='euclidean'), method='ward')
org_order = [ORG_ORDER[i] for i in leaves_list(Z_org)]
n_orgs    = len(org_order)

aa_order    = ['F','L','I','V','S','P','T','A','Y','H','Q','N','K','D','E','C','R','G']
codon_order = []
for aa in aa_order:
    codon_order += sorted(AA_CODONS[aa])

rscu_plot = rscu_df.loc[org_order, codon_order].T   # codons × orgs (clustered order)
xlabels   = [ORG_LABELS_PRETTY.get(o, o.replace('_', ' ').capitalize()) for o in org_order]

# ── Dendrogram (ax2d) ─────────────────────────────────────────────────────────
# scipy leaf k at x = 10k+5;  seaborn col k center at x = k+0.5
# Both normalized: (k+0.5)/n  →  set ax2d.xlim=(0,n*10), ax2 xlim=(0,n) [seaborn default]
# Physical widths are equal (same gridspec column) so columns align exactly
dendrogram(Z_org, ax=ax2d, orientation='top',
           link_color_func=lambda _: '#444444', no_labels=True)
ax2d.set_xlim(0, n_orgs * 10)
ax2d.set_axis_off()

# ── Seaborn heatmap (ax2) ─────────────────────────────────────────────────────
norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=2.0)
sns.heatmap(
    rscu_plot,
    ax=ax2,
    cmap='RdBu_r',
    norm=norm,
    xticklabels=xlabels,
    yticklabels=True,
    linewidths=0,
    cbar=False,          # inset colorbar added below — keeps ax2 full width
)

ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=4.5, rotation=0)
ax2.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=5.5)
for tick, org in zip(ax2.get_xticklabels(), org_order):
    tick.set_color(GROUP_COLORS.get(org_to_group(org), '#444'))

# Amino acid group separators + side labels
y_pos = 0
for aa in aa_order:
    n = len(AA_CODONS[aa])
    if y_pos > 0:
        ax2.axhline(y_pos, color='white', linewidth=0.9)
    ax2.text(n_orgs + 0.3, y_pos + n / 2,
             aa, va='center', ha='left', fontsize=6, color='#333', fontweight='bold')
    y_pos += n

# Inset colorbar — does NOT narrow ax2 (critical for dendrogram alignment)
cax = ax2.inset_axes([1.01, 0.1, 0.025, 0.8])
sm  = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
cb  = plt.colorbar(sm, cax=cax)
cb.ax.tick_params(labelsize=7)
cb.set_label('RSCU', fontsize=8)

ax2d.set_title('Relative synonymous codon usage (RSCU)\nacross 39 industrial host organisms',
               fontsize=13, pad=4)
ax2d.text(-0.05, 1.35, 'B', transform=ax2d.transAxes,
          ha='left', va='bottom', fontsize=18, fontweight='bold')

# ── Save ──────────────────────────────────────────────────────────────────────
plt.suptitle(
    'Supplementary Figure S1 — CodonOptimus foundation model learns '
    'organism-specific codon landscapes reflecting evolutionary relationships',
    fontsize=12, style='italic', y=1.02)

fig.savefig(OUT,  dpi=300, bbox_inches='tight')
fig.savefig(OUTP, dpi=300, bbox_inches='tight')
fig.savefig(OUTS, format='svg', bbox_inches='tight')
print(f'\nSaved → {OUT}')
print(f'Saved → {OUTP}')
print(f'Saved → {OUTS}')
