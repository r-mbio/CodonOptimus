#!/usr/bin/env python3
"""
Update S. cerevisiae training CSV with CORRECT Ribo-seq condition.

PROBLEM: Previously used GSE56622 GLUCOSE STARVATION condition (Ribo-Glu columns).
FIX: Use NORMAL GROWTH condition (Ribo+Glu columns, WITH glucose, BY4741 strain).

ALSO: Merge in additional WT samples from:
  - GSE185286: 4 WT replicates (ribo_wt_1 to ribo_wt_4)
  - GSE185458: 4 WT replicates (ribo_wt_1 to ribo_wt_4)

WHAT THIS DOES:
  1. Backs up s_cerevisiae_train.csv
  2. Replaces expression_level from -Glu → +Glu for all genes in GSE56622
  3. Adds expression_level for genes in GSE185286/185458 not already covered
  4. Updates source column to reflect correct condition

CRITICAL COLUMN GUARDS:
  GSE56622 USE: 'Ribo+Glu_RPKM-BY4741 - Replicate 1' and 'Ribo+Glu_RPKM-BY4741 - Replicate 2'
  GSE56622 DO NOT USE: 'Ribo-Glu_RPKM-BY4741' (starvation)
  GSE185286 USE: ribo_wt_*.tsv.gz files only (GSM5610137-5610140)
  GSE185458 USE: ribo_wt_*.tsv.gz files only (GSM5615642-5615645)
"""

import sys, gzip, shutil, json
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW_DIR = BASE / 'data' / 'raw' / 's_cerevisiae'
RIBO_DIR = BASE / 'data' / 'riboseq_raw' / 's_cerevisiae'
LOG_FILE = BASE / 'RIBOSEQ_PROCESSING_LOG.txt'
TRACKER  = BASE / 'RIBOSEQ_DATASET_PROCESSING.md'

TRAIN_CSV = RAW_DIR / 's_cerevisiae_train.csv'
BACKUP_CSV = RAW_DIR / f's_cerevisiae_train_BACKUP_{date.today():%Y%m%d}.csv'

print("=" * 70)
print("UPDATE S.CEREVISIAE RIBO-SEQ — CORRECT CONDITIONS")
print("=" * 70)

# ── Step 0: Verify we have the right files ────────────────────────────────────
gse56622_file = RIBO_DIR / 's_cerevisiae_GSE56622_riboseq.txt'
gse185286_dir = RIBO_DIR / 'processed' / 'GSE185286'
gse185458_dir = RIBO_DIR / 'processed' / 'GSE185458'

assert gse56622_file.exists(), f"MISSING: {gse56622_file}"
assert gse185286_dir.exists(), f"MISSING: {gse185286_dir}"
assert gse185458_dir.exists(), f"MISSING: {gse185458_dir}"

# Verify correct WT file names for GSE185286
wt_files_286 = sorted(gse185286_dir.glob('GSM561013[7-9]_ribo_wt_*.tsv.gz')) + \
               sorted(gse185286_dir.glob('GSM5610140_ribo_wt_*.tsv.gz'))
assert len(wt_files_286) == 4, f"Expected 4 WT files in GSE185286, got {len(wt_files_286)}: {[f.name for f in wt_files_286]}"
print(f"[VERIFIED] GSE185286 WT files: {[f.name for f in wt_files_286]}")

wt_files_458 = sorted(gse185458_dir.glob('GSM561564[2-5]_ribo_wt_*.tsv.gz'))
assert len(wt_files_458) == 4, f"Expected 4 WT files in GSE185458, got {len(wt_files_458)}: {[f.name for f in wt_files_458]}"
print(f"[VERIFIED] GSE185458 WT files: {[f.name for f in wt_files_458]}")

# ── Step 1: Backup ────────────────────────────────────────────────────────────
print(f"\n[STEP 1] Backing up training CSV → {BACKUP_CSV.name}")
shutil.copy2(TRAIN_CSV, BACKUP_CSV)
print(f"  Backup size: {BACKUP_CSV.stat().st_size / 1024:.1f} KB")

# ── Step 2: Load training CSV ─────────────────────────────────────────────────
print("\n[STEP 2] Loading training CSV")
df_train = pd.read_csv(TRAIN_CSV)
print(f"  Total rows: {len(df_train)}")
print(f"  Source distribution before update:")
print(df_train['source'].value_counts().to_string())

# Extract gene ID from sequence_id (format: s_cerevisiae_YGAL001C)
df_train['gene_id'] = df_train['sequence_id'].str.replace('s_cerevisiae_', '', regex=False)

# ── Step 3: GSE56622 +Glu (CORRECT condition) ─────────────────────────────────
print("\n[STEP 3] Loading GSE56622 +Glu columns (normal growth)")
df_gse56622 = pd.read_csv(gse56622_file, sep='\t')

# CRITICAL: use +Glu (normal growth), NOT -Glu (starvation)
COL_R1 = 'Ribo+Glu_RPKM-BY4741 - Replicate 1'
COL_R2 = 'Ribo+Glu_RPKM-BY4741 - Replicate 2'

# Verify columns exist
assert COL_R1 in df_gse56622.columns, f"Column not found: {COL_R1}"
assert COL_R2 in df_gse56622.columns, f"Column not found: {COL_R2}"
print(f"  [VERIFIED] Using columns:")
print(f"    {COL_R1}")
print(f"    {COL_R2}")
print(f"  [NOT USING] Ribo-Glu columns (glucose starvation)")

# Filter to yeast genes (systematic names start with Y)
df_56622 = df_gse56622[df_gse56622['Systematic Name'].str.startswith('Y', na=False)].copy()
df_56622['mean_rpkm'] = df_56622[[COL_R1, COL_R2]].mean(axis=1)
df_56622 = df_56622[df_56622['mean_rpkm'] > 0].copy()

print(f"  Genes with +Glu RPKM > 0: {len(df_56622)}")
print(f"  Genes with +Glu RPKM > 5: {(df_56622['mean_rpkm'] > 5).sum()}")

# Normalize to 0-100 (log2 scale, min-max)
df_56622['log2_rpkm'] = np.log2(df_56622['mean_rpkm'] + 1)
lo, hi = df_56622['log2_rpkm'].min(), df_56622['log2_rpkm'].max()
df_56622['expr_0_100'] = 100 * (df_56622['log2_rpkm'] - lo) / (hi - lo + 1e-9)
expr_56622 = df_56622.set_index('Systematic Name')['expr_0_100'].to_dict()
print(f"  Expression range: 0.0–100.0 (log2 min-max normalized)")

# ── Step 4: GSE185286 WT ─────────────────────────────────────────────────────
print("\n[STEP 4] Loading GSE185286 WT (4 reps, YPD exponential)")
counts_286 = {}
for f in wt_files_286:
    with gzip.open(f, 'rt') as fh:
        for line in fh:
            parts = line.strip().split('\t')
            if len(parts) < 3 or parts[0].startswith('N_'): continue
            gene = parts[0]
            # Use SENSE strand count (col index 2)
            try:
                count = int(parts[2])
            except (ValueError, IndexError):
                continue
            counts_286[gene] = counts_286.get(gene, 0) + count
# Average across 4 replicates
n_reps_286 = len(wt_files_286)
mean_286 = {g: v / n_reps_286 for g, v in counts_286.items()}
# Filter: mean count > 5 across reps
mean_286 = {g: v for g, v in mean_286.items() if v > 5}
print(f"  Genes with mean count > 5: {len(mean_286)}")

# Normalize to 0-100 (log2 scale)
vals_286 = np.array(list(mean_286.values()))
log2_286 = np.log2(vals_286 + 1)
lo, hi = log2_286.min(), log2_286.max()
expr_286_raw = np.log2(np.array([mean_286[g] for g in mean_286]) + 1)
norm_286 = 100 * (expr_286_raw - lo) / (hi - lo + 1e-9)
expr_286 = {g: float(v) for g, v in zip(mean_286.keys(), norm_286)}

# ── Step 5: GSE185458 WT ─────────────────────────────────────────────────────
print("\n[STEP 5] Loading GSE185458 WT (4 reps, YPD exponential)")
counts_458 = {}
for f in wt_files_458:
    with gzip.open(f, 'rt') as fh:
        for line in fh:
            parts = line.strip().split('\t')
            if len(parts) < 3 or parts[0].startswith('N_'): continue
            gene = parts[0]
            try:
                count = int(parts[2])
            except (ValueError, IndexError):
                continue
            counts_458[gene] = counts_458.get(gene, 0) + count
n_reps_458 = len(wt_files_458)
mean_458 = {g: v / n_reps_458 for g, v in counts_458.items()}
mean_458 = {g: v for g, v in mean_458.items() if v > 5}
print(f"  Genes with mean count > 5: {len(mean_458)}")

vals_458 = np.array([mean_458[g] for g in mean_458])
log2_458 = np.log2(vals_458 + 1)
lo, hi = log2_458.min(), log2_458.max()
norm_458 = 100 * (log2_458 - lo) / (hi - lo + 1e-9)
expr_458 = {g: float(v) for g, v in zip(mean_458.keys(), norm_458)}

# ── Step 6: Merge expression data ─────────────────────────────────────────────
# Priority: GSE56622+Glu (2 reps) → GSE185286 (4 reps) → GSE185458 (4 reps)
# For genes in multiple datasets: use the one with most replicates (185458 wins if in both)
print("\n[STEP 6] Merging expression levels — priority: 185458 > 185286 > 56622+Glu")

# Combined expression: average datasets for genes in multiple
def get_merged_expr(gene_id):
    vals = []
    sources = []
    if gene_id in expr_56622:
        vals.append(expr_56622[gene_id])
        sources.append('gse56622_plusglu')
    if gene_id in expr_286:
        vals.append(expr_286[gene_id])
        sources.append('gse185286_wt')
    if gene_id in expr_458:
        vals.append(expr_458[gene_id])
        sources.append('gse185458_wt')
    if not vals:
        return 0.0, 'no_expr'
    # Average all available
    return float(np.mean(vals)), '+'.join(sources)

# ── Step 7: Update training CSV ───────────────────────────────────────────────
print("\n[STEP 7] Updating training CSV")
updated = 0
newly_added = 0
unchanged = 0

new_expressions = []
new_sources = []

for _, row in df_train.iterrows():
    gene_id = row['gene_id']
    expr, src = get_merged_expr(gene_id)
    if expr > 0:
        new_expressions.append(expr)
        new_sources.append('riboseq_' + src)
        if row['source'] == 'no_expr':
            newly_added += 1
        else:
            updated += 1
    else:
        new_expressions.append(0.0)
        new_sources.append('no_expr')
        unchanged += 1

df_train['expression_level'] = new_expressions
df_train['source'] = new_sources
df_train = df_train.drop(columns=['gene_id'])  # remove temp column

print(f"  Genes updated (had old data): {updated}")
print(f"  Genes newly covered:          {newly_added}")
print(f"  Genes still no_expr:          {unchanged}")
print(f"  Total rows:                   {len(df_train)}")

# Verify no wrong condition
assert 'riboseq_gse56622' not in df_train['source'].values, \
    "ERROR: old riboseq_gse56622 (starvation) source still present!"
print(f"\n  [VERIFIED] No old 'riboseq_gse56622' starvation entries remain")
print(f"  Source distribution after update:")
print(df_train['source'].value_counts().to_string())

# ── Step 8: Save ─────────────────────────────────────────────────────────────
print(f"\n[STEP 8] Saving updated CSV")
df_train.to_csv(TRAIN_CSV, index=False)
print(f"  Saved: {TRAIN_CSV}")
print(f"  Rows: {len(df_train)}, non-expr: {(df_train['source']=='no_expr').sum()}")

# ── Step 9: Update tracking file ─────────────────────────────────────────────
log_entry = (
    f"{date.today()} | S.cer GSE56622+Glu+GSE185286+GSE185458 | "
    f"Updated {updated} genes, added {newly_added} new | "
    f"STARVATION REMOVED, using +Glu (normal growth)\n"
)
with open(LOG_FILE, 'a') as f:
    f.write(log_entry)

# Update tracker markdown
tracker_text = TRACKER.read_text()
tracker_text = tracker_text.replace(
    '| 1 | **HIGH** | S. cerevisiae | GSE56622 +Glu | Normal YPD growth (with glucose) BY4741 | `s_cerevisiae_GSE56622_riboseq.txt` — columns: `Ribo+Glu_RPKM-BY4741 - Replicate 1` & `Ribo+Glu_RPKM-BY4741 - Replicate 2` | ⬜ PENDING |',
    f'| 1 | **HIGH** | S. cerevisiae | GSE56622 +Glu | Normal YPD growth (with glucose) BY4741 | `s_cerevisiae_GSE56622_riboseq.txt` — columns: `Ribo+Glu_RPKM-BY4741 - Replicate 1` & `Ribo+Glu_RPKM-BY4741 - Replicate 2` | ✅ DONE {date.today()} |'
)
tracker_text = tracker_text.replace(
    '| 2 | **HIGH** | S. cerevisiae | GSE185286 WT | YPD exponential growth, WT BY4742, 4 reps | `processed/GSE185286/GSM5610137-5610140_ribo_wt_*.tsv.gz` — sense column (col 2, 0-indexed) | ⬜ PENDING |',
    f'| 2 | **HIGH** | S. cerevisiae | GSE185286 WT | YPD exponential growth, WT BY4742, 4 reps | `processed/GSE185286/GSM5610137-5610140_ribo_wt_*.tsv.gz` — sense column (col 2, 0-indexed) | ✅ DONE {date.today()} |'
)
tracker_text = tracker_text.replace(
    '| 3 | **HIGH** | S. cerevisiae | GSE185458 WT | YPD exponential growth, WT BY4742, 4 reps | `processed/GSE185458/GSM5615642-5615645_ribo_wt_*.tsv.gz` — sense column (col 2, 0-indexed) | ⬜ PENDING |',
    f'| 3 | **HIGH** | S. cerevisiae | GSE185458 WT | YPD exponential growth, WT BY4742, 4 reps | `processed/GSE185458/GSM5615642-5615645_ribo_wt_*.tsv.gz` — sense column (col 2, 0-indexed) | ✅ DONE {date.today()} |'
)
# Update completion log table
tracker_text = tracker_text.replace(
    '| (pending) | | | |',
    f'| {date.today()} | 1,2,3 | Updated {updated} genes, added {newly_added} new | S.cer: starvation→normal growth (GSE56622+Glu + GSE185286 + GSE185458 WT) |\n| (pending) | | | |'
)
TRACKER.write_text(tracker_text)
print(f"  Tracking file updated: {TRACKER.name}")

print("\n" + "=" * 70)
print("DONE — S.cerevisiae training data updated to NORMAL GROWTH condition")
print(f"  Backup at: {BACKUP_CSV.name}")
print("=" * 70)
