#!/usr/bin/env python3
"""
Re-normalize ALL Ribo-seq training data using consistent TPM + log2 + winsorize(p99) + min-max.

WHY: Previous min-max normalization was not robust to outliers.
  - Top 1% genes in K.phaffii BAMs = 22.8% of reads → inflate log2 scale → compress rest
  - Fix: TPM (length-normalized) + clip outliers at p99 before normalizing

METHOD (same for all organisms):
  1. Compute TPM = (reads / gene_length_kb) / sum(rates) * 1e6
     OR use pre-computed RPKM/log2TPM where available (equivalent)
  2. log2(TPM + 1)
  3. Clip at p99 (winsorize) — removes extreme outliers
  4. Min-max normalize to [0, 100]

SOURCES:
  S.cer:   GSE56622 (RPKM) + GSE185286/458 (HTSeq counts + S.cer GFF lengths)
  E.coli:  BAMs (SRR35650607 + SRR22447282) — idxstats gives length + counts
  B.sub:   WIG files → per-gene sum → / gene_length_kb → TPM-like
  K.phaffii gse159336:  already log2TPM → winsorize + min-max (recompute)
  K.phaffii srr32315787_88: BAMs → idxstats → TPM
"""

import gzip, subprocess, shutil
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd

BASE    = Path(__file__).resolve().parent.parent
RAW_DIR = BASE / 'data' / 'raw'
LOG_FILE = BASE / 'RIBOSEQ_PROCESSING_LOG.txt'


def normalize_tpm(values_dict, winsorize_pct=99):
    """
    Given {gene: raw_value}, apply:
      1. log2(v + 1)
      2. clip at winsorize_pct percentile
      3. min-max to [0, 100]
    Returns {gene: normalized_0_100}
    """
    genes = list(values_dict.keys())
    vals  = np.array([values_dict[g] for g in genes], dtype=float)
    log2v = np.log2(vals + 1)
    clip  = np.percentile(log2v, winsorize_pct)
    log2v = np.clip(log2v, 0, clip)
    lo, hi = log2v.min(), log2v.max()
    norm  = 100.0 * (log2v - lo) / (hi - lo + 1e-9)
    return {g: float(norm[i]) for i, g in enumerate(genes)}


def dr(expr_dict_or_arr):
    """Dynamic range: p90/p10"""
    v = np.array(list(expr_dict_or_arr.values()) if isinstance(expr_dict_or_arr, dict)
                 else expr_dict_or_arr)
    return np.percentile(v, 90) / max(np.percentile(v, 10), 0.1)


# ─── 1. S. cerevisiae ─────────────────────────────────────────────────────────
print("=" * 60)
print("S. CEREVISIAE")
print("=" * 60)

RIBO_SCER = BASE / 'data' / 'riboseq_raw' / 's_cerevisiae'

# ── 1a. GSE56622 +Glu: already in RPKM ──
df56622 = pd.read_csv(RIBO_SCER / 's_cerevisiae_GSE56622_riboseq.txt', sep='\t')
COL_R1 = 'Ribo+Glu_RPKM-BY4741 - Replicate 1'
COL_R2 = 'Ribo+Glu_RPKM-BY4741 - Replicate 2'
assert COL_R1 in df56622.columns and COL_R2 in df56622.columns
df56622 = df56622[df56622['Systematic Name'].str.startswith('Y', na=False)].copy()
df56622['mean_rpkm'] = df56622[[COL_R1, COL_R2]].mean(axis=1)
rpkm56622 = df56622[df56622['mean_rpkm'] > 0.5].set_index('Systematic Name')['mean_rpkm'].to_dict()
expr_56622 = normalize_tpm(rpkm56622)
print(f"  GSE56622+Glu: {len(expr_56622)} genes, dyn_range={dr(expr_56622):.1f}x")

# ── 1b. GSE185286/458 WT: HTSeq counts + S.cer GFF lengths ──
# NCBI RefSeq annotation GCF_000146045.2 (S. cerevisiae R64).
# Download with: datasets download genome accession GCF_000146045.2 --include gff3
SCER_GFF = BASE / 'data' / 'raw_genomes' / 'saccharomyces_cerevisiae' / 'GCF_000146045.2_R64_genomic.gff'
gene_lengths_scer = {}
if SCER_GFF.exists():
    with open(SCER_GFF) as f:
        for line in f:
            if line.startswith('#') or '\t' not in line: continue
            p = line.split('\t')
            if len(p) < 9 or p[2] != 'gene': continue
            attrs = {a.split('=')[0]: a.split('=',1)[1].strip()
                     for a in p[8].split(';') if '=' in a}
            name = attrs.get('Name') or attrs.get('gene')
            if name and attrs.get('gene_biotype','') in ('protein_coding',''):
                gene_lengths_scer[name] = int(p[4]) - int(p[3]) + 1
    print(f"  S.cer GFF: {len(gene_lengths_scer)} genes with length info")
else:
    print("  WARNING: S.cer GFF not found — using count-only normalization for 185286/458")

def load_htseq_counts(files):
    """Load HTSeq count files, return {gene: total_counts}"""
    counts = {}
    for f in files:
        open_f = gzip.open if str(f).endswith('.gz') else open
        with open_f(f, 'rt') as fh:
            for line in fh:
                p = line.strip().split('\t')
                if len(p) < 3 or p[0].startswith('N_'): continue
                counts[p[0]] = counts.get(p[0], 0) + int(p[2])  # sense strand col
    return counts

def counts_to_tpm(counts, lengths):
    """Convert HTSeq counts + gene lengths to TPM"""
    rates = {}
    for g, c in counts.items():
        l = lengths.get(g, 0)
        if c > 0 and l > 50:  # require >50bp
            rates[g] = c / (l / 1000.0)
    total_rate = sum(rates.values())
    return {g: (r / total_rate) * 1e6 for g, r in rates.items()}

# GSE185286 WT (4 reps: GSM5610137-5610140)
GSE185286_DIR = RIBO_SCER / 'processed' / 'GSE185286'
wt_286 = sorted(GSE185286_DIR.glob('GSM561013[7-9]_ribo_wt_*.tsv.gz')) + \
         sorted(GSE185286_DIR.glob('GSM5610140_ribo_wt_*.tsv.gz'))
assert len(wt_286) == 4, f"Expected 4 WT files, got {len(wt_286)}"
counts_286 = load_htseq_counts(wt_286)
tpm_286 = counts_to_tpm(counts_286, gene_lengths_scer) if gene_lengths_scer else \
          {g: float(c) for g, c in counts_286.items() if c > 5}
expr_286 = normalize_tpm(tpm_286)
print(f"  GSE185286 WT:  {len(expr_286)} genes, dyn_range={dr(expr_286):.1f}x")

# GSE185458 WT (4 reps: GSM5615642-5615645)
GSE185458_DIR = RIBO_SCER / 'processed' / 'GSE185458'
wt_458 = sorted(GSE185458_DIR.glob('GSM561564[2-5]_ribo_wt_*.tsv.gz'))
assert len(wt_458) == 4, f"Expected 4 WT files, got {len(wt_458)}"
counts_458 = load_htseq_counts(wt_458)
tpm_458 = counts_to_tpm(counts_458, gene_lengths_scer) if gene_lengths_scer else \
          {g: float(c) for g, c in counts_458.items() if c > 5}
expr_458 = normalize_tpm(tpm_458)
print(f"  GSE185458 WT:  {len(expr_458)} genes, dyn_range={dr(expr_458):.1f}x")

# Merge: average per gene across sources present
all_scer_genes = set(expr_56622) | set(expr_286) | set(expr_458)
merged_scer = {}
for g in all_scer_genes:
    vals = [v for v in [expr_56622.get(g), expr_286.get(g), expr_458.get(g)] if v is not None]
    if vals: merged_scer[g] = float(np.mean(vals))
print(f"  Merged S.cer: {len(merged_scer)} genes, dyn_range={dr(merged_scer):.1f}x")

# Update CSV
df_scer = pd.read_csv(RAW_DIR / 's_cerevisiae' / 's_cerevisiae_train.csv')
df_scer['gene_key'] = df_scer['sequence_id'].str.replace('s_cerevisiae_', '', regex=False)
n_upd = 0
for i, row in df_scer.iterrows():
    if row['source'].startswith('riboseq_'):
        gk = row['gene_key']
        if gk in merged_scer:
            df_scer.at[i, 'expression_level'] = merged_scer[gk]
            n_upd += 1
df_scer = df_scer.drop(columns=['gene_key'])
df_scer.to_csv(RAW_DIR / 's_cerevisiae' / 's_cerevisiae_train.csv', index=False)
print(f"  Updated {n_upd} genes in CSV")


# ─── 2. E. coli ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("E. COLI")
print("=" * 60)

ECOLI_BAMS = [
    BASE / 'data' / 'riboseq_new' / 'ecoli' / 'SRR35650607_Aligned.sortedByCoord.out.bam',
    BASE / 'data' / 'riboseq_new' / 'ecoli' / 'SRR22447282_Aligned.sortedByCoord.out.bam',
]

gene_len_ec = {}; gene_reads_ec = {}
for bam in ECOLI_BAMS:
    res = subprocess.run(['samtools','idxstats',str(bam)], capture_output=True, text=True, check=True)
    for line in res.stdout.strip().split('\n'):
        p = line.split('\t')
        if len(p) < 4 or p[0] == '*': continue
        gene_len_ec[p[0]] = int(p[1])
        gene_reads_ec[p[0]] = gene_reads_ec.get(p[0], 0) + int(p[2])

# TPM
rates_ec = {g: gene_reads_ec[g] / (gene_len_ec[g]/1000.0)
            for g in gene_reads_ec if gene_reads_ec[g] >= 3 and gene_len_ec.get(g,0) > 0}
rate_sum_ec = sum(rates_ec.values())
tpm_ec = {g: r/rate_sum_ec*1e6 for g, r in rates_ec.items()}
expr_ec = normalize_tpm(tpm_ec)
print(f"  E.coli BAMs: {len(expr_ec)} genes, dyn_range={dr(expr_ec):.1f}x")

# Update CSV
df_ec = pd.read_csv(RAW_DIR / 'ecoli' / 'ecoli_train.csv')
df_ec['gene_key'] = df_ec['sequence_id'].str.replace('ecoli_', '', regex=False)
n_upd = 0
for i, row in df_ec.iterrows():
    gk = row['gene_key']
    if gk in expr_ec:
        df_ec.at[i, 'expression_level'] = expr_ec[gk]
        df_ec.at[i, 'source'] = 'riboseq_prjna1335396_prjna906596'
        n_upd += 1
df_ec = df_ec.drop(columns=['gene_key'])
df_ec.to_csv(RAW_DIR / 'ecoli' / 'ecoli_train.csv', index=False)
print(f"  Updated {n_upd} genes in CSV")


# ─── 3. B. subtilis ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("B. SUBTILIS")
print("=" * 60)

# NCBI RefSeq annotation GCF_000009045.1 (B. subtilis 168, ASM904v1).
# Download with: datasets download genome accession GCF_000009045.1 --include gff3
BSUB_GFF = BASE / 'data' / 'raw_genomes' / 'bacillus_subtilis' / 'GCF_000009045.1_ASM904v1_genomic.gff'
WIG_DIR_BS = BASE / 'data' / 'riboseq_raw' / 'bacillus' / 'processed' / 'GSE126234'

WT_REPS_BS = {
    'A': ('GSM3594054_DK1042_A_fwd.wig.gz','GSM3594054_DK1042_A_rev.wig.gz'),
    'B': ('GSM3594055_DK1042_B_fwd.wig.gz','GSM3594055_DK1042_B_rev.wig.gz'),
    'C': ('GSM3594056_DK1042_C_fwd.wig.gz','GSM3594056_DK1042_C_rev.wig.gz'),
}

# Get gene lengths from GFF
gene_info_bs = {}
with open(BSUB_GFF) as fh:
    for line in fh:
        if line.startswith('#') or '\t' not in line: continue
        p = line.rstrip('\n').split('\t')
        if len(p) < 9 or p[2] != 'gene': continue
        attrs = {a.split('=')[0]: a.split('=',1)[1].strip() for a in p[8].split(';') if '=' in a}
        if attrs.get('gene_biotype') != 'protein_coding': continue
        name = attrs.get('Name') or attrs.get('gene')
        if name:
            gene_info_bs[name] = {'start': int(p[3]), 'end': int(p[4]),
                                   'strand': p[6], 'length': int(p[4])-int(p[3])+1}

def load_wig(path):
    d = {}
    with gzip.open(path, 'rt') as f:
        for line in f:
            line = line.strip()
            if line.startswith(('track','#','variable','browser')): continue
            p = line.split()
            if len(p) == 2:
                try: d[int(p[0])] = d.get(int(p[0]),0) + int(p[1])
                except ValueError: pass
    return d

print("  Loading WIG files...")
rep_counts_bs = {}
for rep, (fwd_f, rev_f) in WT_REPS_BS.items():
    fwd = load_wig(WIG_DIR_BS / fwd_f)
    rev = load_wig(WIG_DIR_BS / rev_f)
    gc = {}
    for name, info in gene_info_bs.items():
        wig = fwd if info['strand'] == '+' else rev
        gc[name] = sum(wig.get(pos,0) for pos in range(info['start'], info['end']+1))
    rep_counts_bs[rep] = gc

# Average across reps → compute TPM-like
avg_counts_bs = {}
for gname in gene_info_bs:
    vals = [rep_counts_bs[r].get(gname, 0) for r in WT_REPS_BS]
    n_nz = sum(1 for v in vals if v > 0)
    mv = np.mean(vals)
    if mv > 3 and n_nz >= 2:
        avg_counts_bs[gname] = mv

# TPM: normalize by gene length
rates_bs = {g: avg_counts_bs[g] / (gene_info_bs[g]['length'] / 1000.0)
            for g in avg_counts_bs}
rate_sum_bs = sum(rates_bs.values())
tpm_bs = {g: r/rate_sum_bs*1e6 for g, r in rates_bs.items()}
expr_bs = normalize_tpm(tpm_bs)
print(f"  B.sub GSE126234: {len(expr_bs)} genes, dyn_range={dr(expr_bs):.1f}x")

# Update CSV (only genes matched)
df_bs = pd.read_csv(RAW_DIR / 'bacillus' / 'bacillus_train.csv')
df_bs['gene_key'] = df_bs['sequence_id'].str.replace('bacillus_', '', regex=False)
n_upd = 0
for i, row in df_bs.iterrows():
    if row['source'].startswith('riboseq_'):
        gk = row['gene_key']
        if gk in expr_bs:
            df_bs.at[i, 'expression_level'] = expr_bs[gk]
            n_upd += 1
df_bs = df_bs.drop(columns=['gene_key'])
df_bs.to_csv(RAW_DIR / 'bacillus' / 'bacillus_train.csv', index=False)
print(f"  Updated {n_upd} genes in CSV")


# ─── 4. K. phaffii ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("K. PHAFFII")
print("=" * 60)

PICHIA_RIBO = BASE / 'data' / 'riboseq_raw' / 'pichia' / 'pichia_GSE159336_riboseq.txt'
PICHIA_BAMS = [
    BASE / 'data' / 'bam' / 'pichia' / 'SRR32315787_Aligned.sortedByCoord.out.bam',
    BASE / 'data' / 'bam' / 'pichia' / 'SRR32315788_Aligned.sortedByCoord.out.bam',
]

# 4a. GSE159336: already log2TPM in the file
df_gse = pd.read_csv(PICHIA_RIBO, sep='\t')
df_gse['mean_l2tpm'] = df_gse[['Ribo.P4.l2tpm','Ribo.P6.l2tpm']].mean(axis=1)
# These are ALREADY log2TPM — just winsorize + min-max
l2tpm_vals = df_gse.set_index('Name')['mean_l2tpm'].to_dict()
# Use normalize_tpm but skip the log2 step (already log2)
# Manual: winsorize at p99, then min-max to [0-100]
genes_gse = list(l2tpm_vals.keys())
vals_gse   = np.array([l2tpm_vals[g] for g in genes_gse])
vals_gse   = np.clip(vals_gse, 0, np.percentile(vals_gse, 99))  # winsorize
lo_g, hi_g = vals_gse.min(), vals_gse.max()
norm_gse   = 100.0 * (vals_gse - lo_g) / (hi_g - lo_g + 1e-9)
expr_gse159336 = {genes_gse[i]: float(norm_gse[i]) for i in range(len(genes_gse))}
print(f"  GSE159336:  {len(expr_gse159336)} genes, dyn_range={dr(expr_gse159336):.1f}x")

# 4b. SRR32315787+88: BAMs → TPM
def map_pichia(b):
    s = b.replace('AT249_','')
    return 'GQ67_'+s[4:] if s.startswith('GQ67') else s

gene_len_kp = {}; gene_reads_kp = {}
for bam in PICHIA_BAMS:
    res = subprocess.run(['samtools','idxstats',str(bam)], capture_output=True, text=True, check=True)
    for line in res.stdout.strip().split('\n'):
        p = line.split('\t')
        if len(p) < 4 or p[0] == '*': continue
        gk = map_pichia(p[0])
        gene_len_kp[gk] = int(p[1])
        gene_reads_kp[gk] = gene_reads_kp.get(gk,0) + int(p[2])

rates_kp = {g: gene_reads_kp[g] / (gene_len_kp[g]/1000.0)
            for g in gene_reads_kp if gene_reads_kp[g] >= 3 and gene_len_kp.get(g,0) > 0}
rate_sum_kp = sum(rates_kp.values())
tpm_kp = {g: r/rate_sum_kp*1e6 for g, r in rates_kp.items()}
expr_srr = normalize_tpm(tpm_kp)
print(f"  SRR32315787+88: {len(expr_srr)} genes, dyn_range={dr(expr_srr):.1f}x")

# Update K.phaffii CSV
df_kp = pd.read_csv(RAW_DIR / 'pichia' / 'pichia_train.csv')
df_kp['gene_key'] = df_kp['sequence_id'].str.replace('pichia_', '', regex=False)
n_upd = 0
for i, row in df_kp.iterrows():
    gk = row['gene_key']
    src = row['source']
    if src == 'riboseq_gse159336' and gk in expr_gse159336:
        df_kp.at[i,'expression_level'] = expr_gse159336[gk]; n_upd += 1
    elif src == 'riboseq_srr32315787_88_normal_growth' and gk in expr_srr:
        df_kp.at[i,'expression_level'] = expr_srr[gk]; n_upd += 1
df_kp = df_kp.drop(columns=['gene_key'])
df_kp.to_csv(RAW_DIR / 'pichia' / 'pichia_train.csv', index=False)
print(f"  Updated {n_upd} genes in CSV")


# ─── Final summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL DYNAMIC RANGE SUMMARY")
print("=" * 60)
for org in ['s_cerevisiae','ecoli','bacillus','pichia']:
    df_f = pd.read_csv(RAW_DIR / org / f'{org}_train.csv')
    ribo = df_f[df_f['source'].str.startswith('riboseq_')]
    ev   = ribo['expression_level'].values
    print(f"  {org}: n={len(ev)} | dyn_range={np.percentile(ev,90)/max(np.percentile(ev,10),0.1):.1f}x"
          f" | mean={ev.mean():.1f} | zeros={( ev==0).sum()}")
    for src in sorted(ribo['source'].unique()):
        sv = ribo[ribo['source']==src]['expression_level'].values
        sdr = np.percentile(sv,90)/max(np.percentile(sv,10),0.1)
        flag = '⚠️' if sdr < 3.0 else '✅'
        print(f"    {flag} {src.replace('riboseq_','')[:38]}: n={len(sv)} dyn={sdr:.1f}x")

with open(LOG_FILE, 'a') as f:
    f.write(f"{date.today()} | TPM renormalization all 4 orgs | "
            f"log2(TPM+1)+winsorize_p99+minmax applied\n")

print("\nDone — all organisms renormalized with TPM + log2 + winsorize(p99) + min-max")
