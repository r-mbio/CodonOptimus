#!/usr/bin/env python3
"""
Compute mRNA structural accessibility and Codon Pair Bias for benchmark sequences.

Metrics:
1. mrna_dg_5prime  — ΔG of first 90 nt (initiation window, ViennaRNA)
                     Less negative = LESS structured = BETTER ribosome access
                     Kudla et al. 2009 Science: 5' structure is #1 predictor of E. coli expression
2. mean_cpb        — Mean Codon Pair Bias = log2(obs_pair / expected_pair)
                     Higher = codon pairs more natural/common in organism genome
                     Coleman et al. 2008 Science: negative CPB reduces expression >1000x
3. mrna_dg_mean    — Mean ΔG across CDS (global folding propensity, secondary metric)
"""

import math, os, re, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import RNA   # ViennaRNA
from Bio import SeqIO

BASE      = Path(__file__).resolve().parent.parent
RAW_DIR   = BASE / 'data' / 'raw'
SEQ_STORE = BASE / 'other_optimizers'
BENCH_CSV = BASE / 'other_optimizers/benchmark/benchmark_results_tai.csv'
OUT_CSV   = BASE / 'other_optimizers/benchmark/benchmark_results_tai.csv'

WIN_NT   = 90   # 90 nt = 30 codons window for ΔG
AU_TAIL  = 90
STOP_CODONS = {'TAA', 'TAG', 'TGA'}

# ── Load genomic training sequences to build CPB tables ──────────────────────
print("Building CPB tables from genomic training data...")
cpb_tables = {}

for org in ['ecoli', 'bacillus', 's_cerevisiae', 'pichia']:
    pair_counts  = defaultdict(int)
    codon_counts = defaultdict(int)
    total_pairs  = 0
    n_seqs = 0

    for split in ['train', 'val']:
        p = RAW_DIR / org / f'{org}_{split}.csv'
        if not p.exists():
            continue
        df_r = pd.read_csv(p, usecols=['dna_sequence'])
        for dna in df_r['dna_sequence'].dropna():
            dna = dna.upper()
            n = len(dna) // 3
            codons = [dna[i*3:(i+1)*3] for i in range(n)]
            for ci in codons:
                codon_counts[ci] += 1
            for i in range(n - 1):
                pair_counts[(codons[i], codons[i+1])] += 1
                total_pairs += 1
            n_seqs += 1

    total_codons = sum(codon_counts.values())
    cps = {}
    for (c1, c2), cnt in pair_counts.items():
        obs = cnt / total_pairs
        exp = (codon_counts[c1] / total_codons) * (codon_counts[c2] / total_codons)
        if exp > 0 and obs > 0:
            cps[(c1, c2)] = math.log2(obs / exp)
        else:
            cps[(c1, c2)] = 0.0
    cpb_tables[org] = cps
    print(f"  {org}: {n_seqs} seqs, {len(cps)} codon pairs")


def compute_metrics(seq, org):
    """Compute mRNA dG and CPB metrics for a sequence."""
    seq = seq.upper().replace(' ', '').replace('\n', '')
    if seq[-3:] in STOP_CODONS:
        seq = seq[:-3]
    seq = seq[:len(seq) - len(seq) % 3]
    rna = seq.replace('T', 'U')
    n   = len(seq)
    n_codons = n // 3

    # 5' initiation window ΔG (first 90 nt)
    try:
        _, dg_5prime = RNA.fold(rna[:WIN_NT])
    except Exception:
        dg_5prime = float('nan')

    # Extended 5' ΔG (first 150 nt, includes more of coding sequence)
    try:
        _, dg_5prime_ext = RNA.fold(rna[:150])
    except Exception:
        dg_5prime_ext = float('nan')

    # Full-CDS mean ΔG (sampled every 10 codons)
    dg_vals = []
    for i in range(0, n_codons, 10):
        start = max(0, i*3 - WIN_NT//2)
        end   = min(n, start + WIN_NT)
        start = max(0, end - WIN_NT)
        try:
            _, dg = RNA.fold(rna[start:end])
            dg_vals.append(dg)
        except Exception:
            pass
    dg_mean = float(np.mean(dg_vals)) if dg_vals else float('nan')

    # 3' AU content
    tail = seq[max(0, n - AU_TAIL):]
    au_3prime = (tail.count('A') + tail.count('T')) / max(len(tail), 1)

    # Mean CPB
    cps_table = cpb_tables[org]
    codons = [seq[i*3:(i+1)*3] for i in range(n_codons)]
    cpb_vals = []
    for i in range(len(codons) - 1):
        c1, c2 = codons[i], codons[i+1]
        if c1 not in STOP_CODONS and c2 not in STOP_CODONS:
            cpb_vals.append(cps_table.get((c1, c2), 0.0))
    mean_cpb = float(np.mean(cpb_vals)) if cpb_vals else float('nan')

    # N-terminal CPB (first 30 codons)
    n_nterm = min(30, len(codons) - 1)
    nterm_cpb_vals = [cps_table.get((codons[i], codons[i+1]), 0.0)
                      for i in range(n_nterm) if codons[i] not in STOP_CODONS]
    nterm_cpb = float(np.mean(nterm_cpb_vals)) if nterm_cpb_vals else float('nan')

    return {
        'mrna_dg_5prime':     dg_5prime,
        'mrna_dg_5prime_ext': dg_5prime_ext,
        'mrna_dg_mean':       dg_mean,
        'mrna_au_3prime':     au_3prime,
        'mean_cpb':           mean_cpb,
        'nterm_cpb':          nterm_cpb,
    }


# ── Load all sequences ─────────────────────────────────────────────────────────
OPTIM_FASTA   = SEQ_STORE / 'benchmark/optimized_fastas'
TWIST_FASTA   = SEQ_STORE / 'twist/Twist_optimized.fasta'
IDT_DIR       = SEQ_STORE / 'idt'
GENSCRIPT_DIR = SEQ_STORE / 'genscript'
CT_FASTA      = SEQ_STORE / 'codontransformer/CodonTransformer_all_genes_real.fasta'
WT_FASTA      = SEQ_STORE / 'benchmark/WT_sequences.fasta'

ORG_ALIASES = {
    'e.coli': 'ecoli', 'ecoli': 'ecoli', 'e_coli': 'ecoli', 'escherichia': 'ecoli',
    'bacillus': 'bacillus', 'b.subtilis': 'bacillus', 'b_subtilis': 'bacillus',
    'saccharomyces': 's_cerevisiae', 's.cerevisiae': 's_cerevisiae',
    's_cerevisiae': 's_cerevisiae', 'cerevisiae': 's_cerevisiae',
    'pichia': 'pichia', 'k.phaffii': 'pichia', 'komagataella': 'pichia',
}
PROT_ALIASES = {'blg': 'BLG', 'hsa': 'HSA', 'phya': 'PHYA', 'xyn2': 'XYN2', 'xynii': 'XYN2'}

def norm_org(s):
    s = s.lower().strip()
    for k, v in ORG_ALIASES.items():
        if k in s: return v
    return None

def norm_prot(s):
    return PROT_ALIASES.get(s.lower(), s.upper())

seq_lookup = {}
for fname in os.listdir(OPTIM_FASTA):
    m = re.match(r'CodonOptimus_(\w+)\.fasta', fname)
    if not m: continue
    org = m.group(1)
    if org not in cpb_tables: continue
    for rec in SeqIO.parse(OPTIM_FASTA / fname, 'fasta'):
        prot = norm_prot(rec.id.split('_')[0])
        seq_lookup[(prot, org, 'CodonOptimus')] = str(rec.seq)

for rec in SeqIO.parse(WT_FASTA, 'fasta'):
    prot = norm_prot(rec.id.split('_')[0])
    for org in cpb_tables:
        seq_lookup[(prot, org, 'WT')] = str(rec.seq)

for fname in os.listdir(IDT_DIR):
    if not fname.endswith('.csv'): continue
    org = norm_org(fname.replace('IDT_result_', '').replace('.csv', ''))
    if org is None: continue
    df_idt = pd.read_csv(IDT_DIR / fname)
    df_idt.columns = df_idt.columns.str.strip()
    for _, row in df_idt.iterrows():
        prot = norm_prot(str(row['Name']))
        seq = str(row['Optimized Sequence']).replace(' ', '')
        seq_lookup[(prot, org, 'IDT')] = seq

for fname in os.listdir(GENSCRIPT_DIR):
    if not fname.endswith('.xlsx'): continue
    org = norm_org(fname.lower().replace('genscript_', '').replace('.xlsx',''))
    if org is None: continue
    df_gs = pd.read_excel(GENSCRIPT_DIR / fname)
    df_gs.columns = df_gs.columns.str.strip()
    seq_col = next((c for c in df_gs.columns if 'optimized' in c.lower()), None)
    name_col = 'Gene name' if 'Gene name' in df_gs.columns else '#'
    if seq_col is None: continue
    for _, row in df_gs.iterrows():
        prot = norm_prot(str(row[name_col]).strip())
        if prot in ('nan', ''): continue
        seq = str(row[seq_col]).replace(' ', '')
        if len(seq) > 30:
            seq_lookup[(prot, org, 'Genscript')] = seq

for rec in SeqIO.parse(TWIST_FASTA, 'fasta'):
    parts = rec.id.split('_')
    prot = norm_prot(parts[0])
    org = norm_org('_'.join(parts[2:]))
    if org:
        seq_lookup[(prot, org, 'Twist')] = str(rec.seq)

for rec in SeqIO.parse(CT_FASTA, 'fasta'):
    parts = rec.id.split('_')
    prot = norm_prot(parts[0])
    org = norm_org('_'.join(parts[2:]))
    if org:
        seq_lookup[(prot, org, 'CodonTransformer')] = str(rec.seq)

print(f"Sequences: {len(seq_lookup)}")

# ── Compute for benchmark ─────────────────────────────────────────────────────
df = pd.read_csv(BENCH_CSV)
df['protein'] = df['protein'].str.upper()

print("\nComputing mRNA structure + CPB for all sequences...")
new_cols = {c: [] for c in ['mrna_dg_5prime', 'mrna_dg_5prime_ext', 'mrna_dg_mean',
                              'mrna_au_3prime', 'mean_cpb', 'nterm_cpb']}
missing = []

for i, (_, row) in enumerate(df.iterrows()):
    prot, org, optim = row['protein'], row['organism'], row['optimizer']
    seq = seq_lookup.get((prot, org, optim))
    if seq is None:
        for c in new_cols:
            new_cols[c].append(float('nan'))
        missing.append((prot, org, optim))
    else:
        m = compute_metrics(seq, org)
        for c in new_cols:
            new_cols[c].append(m[c])
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(df)}")

for c, vals in new_cols.items():
    df[c] = vals

if missing:
    print(f"Missing: {missing}")

df.to_csv(OUT_CSV, index=False)
print(f"\nSaved → {OUT_CSV}")

# ── Results ───────────────────────────────────────────────────────────────────
ORG_LABELS = {'ecoli': 'E. coli', 'bacillus': 'B. subtilis',
              's_cerevisiae': 'S. cerevisiae', 'pichia': 'K. phaffii'}
TOOLS_ORDER = ['WT', 'IDT', 'Genscript', 'Twist', 'CodonTransformer', 'CodonOptimus']

print('\n' + '='*70)
print("5' mRNA ΔG (first 90 nt): HIGHER (less negative) = LESS structured = BETTER")
print('='*70)
print(df.groupby(['organism','optimizer'])['mrna_dg_5prime'].mean().unstack()
      [['WT','IDT','Genscript','Twist','CodonTransformer','CodonOptimus']].round(2))

print('\n' + '='*70)
print("MEAN CODON PAIR BIAS: HIGHER = more natural pairs = BETTER expression")
print('='*70)
print(df.groupby(['organism','optimizer'])['mean_cpb'].mean().unstack()
      [['WT','IDT','Genscript','Twist','CodonTransformer','CodonOptimus']].round(4))

print('\n' + '='*70)
print("CO vs CT summary (dG_5prime and CPB)")
print('='*70)
for org in ['ecoli', 'bacillus', 's_cerevisiae', 'pichia']:
    org_df = df[df['organism'] == org]
    co_dg  = org_df[org_df['optimizer']=='CodonOptimus']['mrna_dg_5prime'].mean()
    ct_dg  = org_df[org_df['optimizer']=='CodonTransformer']['mrna_dg_5prime'].mean()
    co_cpb = org_df[org_df['optimizer']=='CodonOptimus']['mean_cpb'].mean()
    ct_cpb = org_df[org_df['optimizer']=='CodonTransformer']['mean_cpb'].mean()
    dg_win = "CO" if co_dg > ct_dg else "CT"
    cpb_win = "CO" if co_cpb > ct_cpb else "CT"
    dg_pct = (co_dg - ct_dg) / abs(ct_dg) * 100 if ct_dg != 0 else 0
    cpb_diff = co_cpb - ct_cpb
    print(f"  {ORG_LABELS[org]:16s}: dG[{dg_win}] CO={co_dg:.2f} CT={ct_dg:.2f} ({dg_pct:+.1f}%)  "
          f"CPB[{cpb_win}] CO={co_cpb:.4f} CT={ct_cpb:.4f} (Δ={cpb_diff:+.4f})")
