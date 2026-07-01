#!/usr/bin/env python3
"""
Update B. subtilis training CSV with GSE126234 WT (DK1042) Ribo-seq.
Condition: LB medium, exponential phase OD600=0.3-0.4 — TRUE VEGETATIVE GROWTH.
Reference: Caballero et al. 2019, PLOS Genetics.

USE ONLY DK1042 (WT) replicates A, B, C.
DO NOT USE: DK2050 (Defp), DK5518, DK5955.
"""

import gzip, shutil
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd

BASE    = Path(__file__).resolve().parent.parent
RAW_DIR = BASE / 'data' / 'raw' / 'bacillus'
WIG_DIR = BASE / 'data' / 'riboseq_raw' / 'bacillus' / 'processed' / 'GSE126234'
# NCBI RefSeq annotation GCF_000009045.1 (B. subtilis 168, ASM904v1).
# Download with: datasets download genome accession GCF_000009045.1 --include gff3
GFF_PATH = BASE / 'data' / 'raw_genomes' / 'bacillus_subtilis' / 'GCF_000009045.1_ASM904v1_genomic.gff'
LOG_FILE = BASE / 'RIBOSEQ_PROCESSING_LOG.txt'
TRACKER  = BASE / 'RIBOSEQ_DATASET_PROCESSING.md'

TRAIN_CSV  = RAW_DIR / 'bacillus_train.csv'
BACKUP_CSV = RAW_DIR / f'bacillus_train_BACKUP_{date.today():%Y%m%d}.csv'

# ONLY DK1042 (WT) replicates
WT_REPS = {
    'A': ('GSM3594054_DK1042_A_fwd.wig.gz', 'GSM3594054_DK1042_A_rev.wig.gz'),
    'B': ('GSM3594055_DK1042_B_fwd.wig.gz', 'GSM3594055_DK1042_B_rev.wig.gz'),
    'C': ('GSM3594056_DK1042_C_fwd.wig.gz', 'GSM3594056_DK1042_C_rev.wig.gz'),
}

print("=" * 70)
print("UPDATE B. SUBTILIS RIBO-SEQ — GSE126234 DK1042 WT (LB EXPONENTIAL)")
print("=" * 70)

# Verify files
assert GFF_PATH.exists(), f"MISSING: {GFF_PATH}"
for rep, (fwd, rev) in WT_REPS.items():
    for fname in (fwd, rev):
        assert (WIG_DIR / fname).exists(), f"MISSING: {WIG_DIR/fname}"
print("[VERIFIED] All DK1042 A/B/C fwd+rev WIG files present")
print("[VERIFIED] Not using: DK2050, DK5518, DK5955")

# Step 1: Parse GFF — use 'gene' features which have actual gene names
print("\n[STEP 1] Parsing GFF for protein-coding gene coordinates")
genes = {}
with open(GFF_PATH) as fh:
    for line in fh:
        if line.startswith('#') or '\t' not in line: continue
        p = line.rstrip('\n').split('\t')
        if len(p) < 9 or p[2] != 'gene': continue
        attrs = {a.split('=')[0]: a.split('=',1)[1].strip()
                 for a in p[8].split(';') if '=' in a}
        if attrs.get('gene_biotype') != 'protein_coding': continue
        name = attrs.get('Name') or attrs.get('gene')
        if name:
            genes[name] = {'start': int(p[3]), 'end': int(p[4]), 'strand': p[6]}
print(f"  Genes: {len(genes)}, sample: {list(genes.keys())[:6]}")

# Step 2: Backup + load training CSV
shutil.copy2(TRAIN_CSV, BACKUP_CSV)
df_train = pd.read_csv(TRAIN_CSV)
df_train['gene_key'] = df_train['sequence_id'].str.replace('bacillus_', '', regex=False)
print(f"\n[STEP 2] Training CSV: {len(df_train)} rows")
print(df_train['source'].value_counts().to_string())

# Step 3: Load WIG files
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

print("\n[STEP 3] Loading WIG files and summing per gene")
rep_expr = {}
for rep, (fwd_f, rev_f) in WT_REPS.items():
    print(f"  Rep {rep}...", end=' ', flush=True)
    fwd = load_wig(WIG_DIR / fwd_f)
    rev = load_wig(WIG_DIR / rev_f)
    gc = {}
    for name, info in genes.items():
        wig = fwd if info['strand'] == '+' else rev
        gc[name] = sum(wig.get(pos,0) for pos in range(info['start'], info['end']+1))
    covered = sum(1 for v in gc.values() if v > 0)
    print(f"{covered} genes with reads")
    rep_expr[rep] = gc

# Average across reps
print("\n[STEP 4] Averaging across 3 reps and normalizing")
avg = {}
for gname in genes:
    vals = [rep_expr[r].get(gname, 0) for r in WT_REPS]
    n_nonzero = sum(1 for v in vals if v > 0)
    if np.mean(vals) > 5 and n_nonzero >= 2:
        avg[gname] = np.mean(vals)
print(f"  Genes passing filter (mean>5, >=2 reps): {len(avg)}")

# Normalize to 0-100
vals_arr = np.array(list(avg.values()))
log2_v   = np.log2(vals_arr + 1)
lo, hi   = log2_v.min(), log2_v.max()
norm_v   = 100 * (log2_v - lo) / (hi - lo + 1e-9)
expr_map = {g: float(v) for g, v in zip(avg.keys(), norm_v)}

# Step 5: Match to training CSV
# Training CSV gene_key format: 'dnaA', 'yrzE' — same as GFF Name
n_train_genes = set(df_train['gene_key'])
matched = {g: v for g, v in expr_map.items() if g in n_train_genes}
print(f"  Matched to training CSV: {len(matched)} genes")

# Step 6: Update CSV
updated = newly_added = unchanged = 0
new_exp, new_src = [], []
for _, row in df_train.iterrows():
    gk = row['gene_key']
    if gk in matched:
        new_exp.append(matched[gk])
        new_src.append('riboseq_gse126234_dk1042_lb_exponential')
        if row['source'] == 'no_expr': newly_added += 1
        else: updated += 1
    else:
        new_exp.append(row['expression_level'])
        new_src.append(row['source'])
        unchanged += 1

df_train['expression_level'] = new_exp
df_train['source'] = new_src
df_train = df_train.drop(columns=['gene_key'])
print(f"\n[STEP 5] Updated: {updated} | Newly covered: {newly_added} | Unchanged: {unchanged}")
print(df_train['source'].value_counts().to_string())

df_train.to_csv(TRAIN_CSV, index=False)
print(f"\n[SAVED] {TRAIN_CSV}")

# Update tracking
log_entry = (f"{date.today()} | B.sub GSE126234 DK1042 LB exponential | "
             f"Updated {updated}, added {newly_added} | 3 WT reps (A/B/C) WIG\n")
with open(LOG_FILE, 'a') as f:
    f.write(log_entry)

tracker_text = TRACKER.read_text()
tracker_text = tracker_text.replace(
    '| 6 | **MED** | B. subtilis | GSE126234 WT (DK1042) | LB exponential OD600=0.3-0.4, 3 reps | Download from GEO → `data/riboseq_raw/bacillus/processed/GSE126234/` | ⬜ PENDING |',
    f'| 6 | **MED** | B. subtilis | GSE126234 WT (DK1042) | LB exponential OD600=0.3-0.4, 3 reps | Download from GEO → `data/riboseq_raw/bacillus/processed/GSE126234/` | ✅ DONE {date.today()} |'
)
tracker_text = tracker_text.replace(
    '| (pending) | | | |',
    f'| {date.today()} | 6 | Updated {updated}, added {newly_added} | B.sub GSE126234 DK1042 LB exponential |\n| (pending) | | | |', 1
)
TRACKER.write_text(tracker_text)

print("\n" + "=" * 70)
print("DONE — B.subtilis updated with LB vegetative Ribo-seq (GSE126234)")
print("=" * 70)
