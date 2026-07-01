#!/usr/bin/env python3
"""
Supplementary Figure: Problematic motif analysis.
Compares CO RS vs CT vs IDT vs WT on:
  A. Restriction enzyme sites per kb (cloning relevance)
  B. Internal Shine-Dalgarno sequences (prokaryotes: ribosome stalling)
  C. Rare codon clusters — CFD% (% codons below 10% max frequency)
  D. GC content % (mRNA stability / secondary structure)
"""

import numpy as np, pandas as pd, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
STOP_C = {'TAA','TAG','TGA'}

plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Helvetica','Arial','DejaVu Sans'],
    'font.size':        9,
    'axes.labelsize':   10,
    'axes.titlesize':   10.5,
    'xtick.labelsize':  8.5,
    'ytick.labelsize':  8.5,
    'axes.linewidth':   0.9,
    'legend.fontsize':  8,
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'pdf.fonttype':     42,
})

C_CO  = '#0072B2'
C_CT  = '#888888'
C_IDT = '#E69F00'
C_GS  = '#009E73'
C_TW  = '#D55E00'
C_WT  = '#BBBBBB'

TOOL_COLORS = {
    'CodonOptimus':     C_CO,
    'CodonTransformer': C_CT,
    'IDT':              C_IDT,
    'Genscript':        C_GS,
    'Twist':            C_TW,
    'WT':               C_WT,
}
TOOL_LABELS = {
    'CodonOptimus':     'CO',
    'CodonTransformer': 'CT',
    'IDT':              'IDT',
    'Genscript':        'GS',
    'Twist':            'Twist',
    'WT':               'WT',
}

# ── Restriction enzyme sites ────────────────────────────────────────────────────
RESTR_SITES = {
    'EcoRI':  'GAATTC',
    'BamHI':  'GGATCC',
    'HindIII':'AAGCTT',
    'NcoI':   'CCATGG',
    'XhoI':   'CTCGAG',
    'NheI':   'GCTAGC',
    'SalI':   'GTCGAC',
    'KpnI':   'GGTACC',
    'SacI':   'GAGCTC',
    'XbaI':   'TCTAGA',
    'PstI':   'CTGCAG',
    'NotI':   'GCGGCCGC',
}

def count_restriction_sites(dna):
    dna = str(dna).upper().replace(' ','')
    total = 0
    for seq in RESTR_SITES.values():
        total += len(re.findall(seq, dna))
        rc = seq.translate(str.maketrans('ATCG','TAGC'))[::-1]
        if rc != seq:
            total += len(re.findall(rc, dna))
    return total / (len(dna)/1000)   # per kb

# ── Internal Shine-Dalgarno sequences (prokaryotes) ────────────────────────────
# SD consensus: AAGG, GAGG, AGGAG, AAGGAG (complementary to 16S rRNA 3' end)
SD_PATTERNS = ['AAGGAG','AGGAGG','GAGGAG','AAGG','GAGG','AGGAG']

def count_internal_sd(dna):
    dna = str(dna).upper().replace(' ','')
    # Skip the first 6 codons (legitimate RBS region)
    body = dna[18:] if len(dna)>18 else dna
    total = 0
    for pat in SD_PATTERNS:
        total += len(re.findall(pat, body))
    return total / (len(body)/1000)  # per kb

# ── Load benchmark data ──────────────────────────────────────────────────────────
df_bench = pd.read_csv(BASE/'other_optimizers/benchmark/benchmark_results_tai.csv')

def parse_fasta(path):
    seqs={}; name=None; buf=[]
    with open(path) as f:
        for line in f:
            line=line.strip()
            if line.startswith('>'):
                if name: seqs[name]=''.join(buf)
                name=line[1:]; buf=[]
            else: buf.append(line.replace(' ',''))
    if name: seqs[name]=''.join(buf)
    return seqs

def get_idt_seq(org_file, prot):
    df = pd.read_csv(BASE/f'other_optimizers/idt/{org_file}')
    df.columns = df.columns.str.strip()
    row = df[df['Name'].str.lower().str.contains(prot.lower(), na=False)]
    return str(row['Optimized Sequence'].values[0]) if len(row)>0 else None

def get_gs_seq(org_file, prot):
    df = pd.read_excel(BASE/f'other_optimizers/genscript/{org_file}')
    row = df[df['Gene name'].str.lower().str.contains(prot.lower(), na=False)]
    return str(row['Optimized sequence'].values[0]) if len(row)>0 else None

def get_twist_seq(prot, org):
    twist_f = BASE/'other_optimizers/twist/Twist_optimized.fasta'
    seqs = parse_fasta(twist_f)
    return next((v for k,v in seqs.items() if prot.lower() in k.lower() and org.lower() in k.lower()), None)

ct_all = parse_fasta(BASE/'other_optimizers/codontransformer/CodonTransformer_all_genes_real.fasta')
wt_all = parse_fasta(BASE/'other_optimizers/benchmark/WT_sequences.fasta')

ORG_CFG = [
    ('ecoli',       'E. coli',       'IDT_result_E.coli.csv',       'Genscript_E.coli.xlsx',       'CodonOptimus_ecoli.fasta'),
    ('bacillus',    'B. subtilis',   'IDT_result_Bacillus.csv',     'Genscript_Bacillus.xlsx',     'CodonOptimus_bacillus.fasta'),
    ('s_cerevisiae','S. cerevisiae', 'IDT_result_Saccharomyces.csv','Genscript_Saccharomyces.xlsx','CodonOptimus_s_cerevisiae.fasta'),
    ('pichia',      'K. phaffii',    'IDT_result_pichia.csv',       'Genscript_Pichia.xlsx',       'CodonOptimus_pichia.fasta'),
]
PROTEINS = ['BLG','HSA','PHYA','XYN2']
PROK = {'ecoli','bacillus'}

# Collect per-sequence metrics for each tool × organism
records = []
for org, label, idt_f, gs_f, co_f in ORG_CFG:
    co_seqs = parse_fasta(BASE/f'other_optimizers/benchmark/optimized_fastas/{co_f}')

    for prot in PROTEINS:
        seqs_by_tool = {}

        # CO RS
        s = next((v for k,v in co_seqs.items() if prot.lower() in k.lower()), None)
        if s: seqs_by_tool['CodonOptimus'] = s

        # CT
        key = f'{prot}_CodonTransformer_{org}'
        if key in ct_all: seqs_by_tool['CodonTransformer'] = ct_all[key]

        # IDT
        s = get_idt_seq(idt_f, prot)
        if s: seqs_by_tool['IDT'] = s

        # Genscript
        try:
            s = get_gs_seq(gs_f, prot)
            if s: seqs_by_tool['Genscript'] = s
        except: pass

        # Twist
        s = get_twist_seq(prot, org)
        if s: seqs_by_tool['Twist'] = s

        # WT
        wt = next((v for k,v in wt_all.items() if prot.lower() in k.lower()), None)
        if wt: seqs_by_tool['WT'] = wt

        for tool, dna in seqs_by_tool.items():
            dna = str(dna).upper().replace(' ','')
            rs  = count_restriction_sites(dna)
            sd  = count_internal_sd(dna) if org in PROK else np.nan
            # GC and CFD from benchmark if available
            row_bench = df_bench[(df_bench['organism']==org) &
                                  (df_bench['optimizer']==tool) &
                                  (df_bench['protein']==prot.upper())]
            gc  = float(row_bench['gc'].values[0])       if len(row_bench)>0 else (dna.count('G')+dna.count('C'))/len(dna)*100
            cfd = float(row_bench['cfd_pct'].values[0])  if len(row_bench)>0 else np.nan
            records.append({'organism':org,'org_label':label,'tool':tool,
                            'protein':prot,'restr_per_kb':rs,
                            'sd_per_kb':sd,'gc':gc,'cfd':cfd})

df = pd.DataFrame(records)

# ── Figure ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5),
                          gridspec_kw={'hspace':0.44,'wspace':0.35})
axes = axes.flatten()

PANEL_CFG = [
    ('restr_per_kb', 'Restriction enzyme sites per kb',
     'A', 'Fewer = easier cloning', 'lower'),
    ('sd_per_kb',    'Internal Shine-Dalgarno motifs per kb\n(E. coli & B. subtilis only)',
     'B', 'Fewer = less ribosome stalling', 'lower'),
    ('cfd',          'Rare codon frequency (CFD %)',
     'C', 'Fewer = less rare-codon-induced stalling', 'lower'),
    ('gc',           'GC content (%)',
     'D', '40–60% optimal range for most expression systems', None),
]

TOOL_ORDER = ['WT','IDT','Twist','Genscript','CT','CodonOptimus']

for ax, (metric, ylabel, panel, note, direction) in zip(axes, PANEL_CFG):
    # Aggregate mean per tool × organism
    sub = df.dropna(subset=[metric])
    if sub.empty:
        ax.set_visible(False); continue

    # Per-organism grouped bars
    orgs   = ['ecoli','bacillus','s_cerevisiae','pichia']
    org_labels = ['E. coli','B. subtilis','S. cer.','K. phaffii']
    tools  = [t for t in TOOL_ORDER if t in sub['tool'].unique()]
    n_tools = len(tools)
    n_orgs  = len(orgs)
    bar_w  = 0.13
    grp_w  = n_tools * bar_w + 0.08
    x_grps = np.arange(n_orgs) * grp_w
    offsets= np.linspace(-(n_tools-1)/2, (n_tools-1)/2, n_tools) * bar_w

    for ti, tool in enumerate(tools):
        vals, errs = [], []
        for org in orgs:
            sub2 = sub[(sub['tool']==tool)&(sub['organism']==org)][metric]
            if len(sub2) == 0:
                vals.append(np.nan); errs.append(0)
            else:
                vals.append(sub2.mean()); errs.append(sub2.sem())
        xs = x_grps + offsets[ti]
        valid = [i for i,v in enumerate(vals) if not np.isnan(v)]
        if not valid: continue
        ax.bar([xs[i] for i in valid], [vals[i] for i in valid],
               width=bar_w*0.88, color=TOOL_COLORS[tool], alpha=0.85,
               yerr=[errs[i] for i in valid], capsize=2,
               error_kw=dict(lw=0.7, capthick=0.7),
               edgecolor='white', linewidth=0.3,
               label=TOOL_LABELS[tool], zorder=3)

    # GC optimal range shading
    if metric == 'gc':
        ax.axhspan(40, 60, color='#d4edda', alpha=0.4, zorder=0,
                   label='Optimal range (40–60%)')

    ax.set_xticks(x_grps)
    ax.set_xticklabels(org_labels, fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.9)
    ax.spines['bottom'].set_linewidth(0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.6)

    # Arrow indicating which direction is better
    if direction == 'lower':
        ax.text(0.98, 0.97, '↓ better', transform=ax.transAxes,
                ha='right', va='top', fontsize=8, color='#444', style='italic')

    ax.text(0.01, 0.97, note, transform=ax.transAxes,
            ha='left', va='top', fontsize=7.5, color='#555', style='italic')

    ax.text(-0.12, 1.04, panel, transform=ax.transAxes, fontsize=14,
            fontweight='bold', ha='left', va='bottom')

    ax.legend(fontsize=7.5, frameon=True, framealpha=0.92,
              edgecolor='#ccc', ncol=3, loc='upper center',
              bbox_to_anchor=(0.5, -0.14))

fig.suptitle('Supplementary Figure: Problematic sequence motif analysis across tools',
             fontsize=11, fontweight='bold', y=1.01)

plt.savefig(BASE/'figures/figS10_motif_avoidance.png', dpi=300, bbox_inches='tight')
plt.savefig(BASE/'figures/figS10_motif_avoidance.pdf', bbox_inches='tight')
plt.savefig(BASE/'figures/figS10_motif_avoidance.svg', bbox_inches='tight')
print('Saved.')

# Print summary table
print('\n=== CO vs CT vs IDT — mean per organism ===')
for metric, label, *_ in PANEL_CFG:
    sub = df.dropna(subset=[metric])
    if sub.empty: continue
    print(f'\n{label}:')
    piv = sub.groupby(['org_label','tool'])[metric].mean().unstack('tool').round(2)
    cols = [t for t in ['WT','IDT','Twist','Genscript','CodonTransformer','CodonOptimus'] if t in piv.columns]
    print(piv[cols].to_string())
