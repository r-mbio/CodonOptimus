#!/usr/bin/env python3
"""
Update E. coli training CSV with two additional normal-growth Ribo-seq datasets.

DATASETS (already aligned BAMs on disk):
  - PRJNA1335396 (SRR35650607): MG1655, normal growth, 45M unique reads, 31.5% mapping
  - PRJNA906596  (SRR22447282): MG1655, pH 7.6 CONTROL from acid stress study = NORMAL GROWTH

Both are aligned to per-gene pseudo-chromosomes (gene IDs = b0002, b0003...).
Use `samtools idxstats` to get mapped reads per gene.

WHAT THIS DOES:
  1. Backs up ecoli_train.csv
  2. Runs samtools idxstats on both BAMs
  3. Averages counts across the 2 datasets (after normalization)
  4. Updates expression_level in training CSV
  5. Marks status in tracking file
"""

import sys, subprocess, shutil
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd

BASE    = Path(__file__).resolve().parent.parent
RAW_DIR = BASE / 'data' / 'raw' / 'ecoli'
BAMS_DIR = BASE / 'data' / 'riboseq_new' / 'ecoli'
LOG_FILE = BASE / 'RIBOSEQ_PROCESSING_LOG.txt'
TRACKER  = BASE / 'RIBOSEQ_DATASET_PROCESSING.md'

TRAIN_CSV  = RAW_DIR / 'ecoli_train.csv'
BACKUP_CSV = RAW_DIR / f'ecoli_train_BACKUP_{date.today():%Y%m%d}.csv'

# BAM files to process
BAM_INFO = {
    'PRJNA1335396': {
        'bam': BAMS_DIR / 'SRR35650607_Aligned.sortedByCoord.out.bam',
        'label': 'SRR35650607 (MG1655, normal growth, 45M unique reads)',
        'condition': 'normal_growth_mg1655',
    },
    'PRJNA906596': {
        'bam': BAMS_DIR / 'SRR22447282_Aligned.sortedByCoord.out.bam',
        'label': 'SRR22447282 (MG1655, pH 7.6 control = normal growth, 7M unique reads)',
        'condition': 'normal_growth_ph76_control',
    },
}

print("=" * 70)
print("UPDATE E. COLI RIBO-SEQ — NEW NORMAL GROWTH DATASETS")
print("=" * 70)

# ── Step 0: Verify BAMs exist and are indexed ─────────────────────────────────
for acc, info in BAM_INFO.items():
    bam = info['bam']
    bai = Path(str(bam) + '.bai')
    assert bam.exists(), f"BAM missing: {bam}"
    assert bai.exists(), f"BAM index missing: {bai}"
    print(f"[VERIFIED] {acc}: {bam.name}")

# ── Step 1: Backup ────────────────────────────────────────────────────────────
print(f"\n[STEP 1] Backing up training CSV")
shutil.copy2(TRAIN_CSV, BACKUP_CSV)
print(f"  Backup: {BACKUP_CSV.name} ({BACKUP_CSV.stat().st_size/1024:.1f} KB)")

# ── Step 2: Load training CSV ─────────────────────────────────────────────────
print("\n[STEP 2] Loading E. coli training CSV")
df_train = pd.read_csv(TRAIN_CSV)
print(f"  Total rows: {len(df_train)}")
print(f"  Source distribution:\n{df_train['source'].value_counts().to_string()}")

# Gene ID: sequence_id = 'ecoli_b0002' → gene_key = 'b0002'
df_train['gene_key'] = df_train['sequence_id'].str.replace('ecoli_', '', regex=False)

# ── Step 3: Run samtools idxstats on each BAM ─────────────────────────────────
def get_counts_from_bam(bam_path):
    """Run samtools idxstats and return {gene_id: mapped_reads}"""
    result = subprocess.run(
        ['samtools', 'idxstats', str(bam_path)],
        capture_output=True, text=True, check=True
    )
    counts = {}
    for line in result.stdout.strip().split('\n'):
        parts = line.split('\t')
        if len(parts) < 3: continue
        gene_id = parts[0]
        if gene_id == '*': continue  # unmapped
        try:
            mapped = int(parts[2])
        except ValueError:
            continue
        counts[gene_id] = mapped
    return counts

all_counts = {}
for acc, info in BAM_INFO.items():
    print(f"\n[STEP 3a] samtools idxstats: {info['label']}")
    counts = get_counts_from_bam(info['bam'])
    total  = sum(counts.values())
    nonzero = sum(1 for v in counts.values() if v > 0)
    print(f"  Total mapped reads: {total:,}")
    print(f"  Genes with any reads: {nonzero}")
    # Convert to RPKM-like: reads per gene (length-normalized)
    all_counts[acc] = counts

# ── Step 4: Merge counts — normalize each BAM to RPM then average ─────────────
print("\n[STEP 4] Normalizing and merging counts from both BAMs")

# Get all gene IDs that appear in the training CSV
train_gene_keys = set(df_train['gene_key'].values)

# For each BAM: compute RPM (reads per million mapped), then log2 normalize to 0-100
merged_expr = {}  # gene_key → list of normalized expression values

for acc, counts in all_counts.items():
    total_mapped = sum(counts.values())
    if total_mapped == 0:
        print(f"  WARNING: {acc} has 0 mapped reads, skipping")
        continue
    rpm = {g: (v / total_mapped) * 1e6 for g, v in counts.items()}
    # Filter to genes in training CSV with at least 5 reads
    rpm_filtered = {g: v for g, v in rpm.items()
                    if g in train_gene_keys and counts[g] >= 5}
    print(f"  {acc}: {len(rpm_filtered)} training genes with ≥5 reads (RPM > 0)")
    # Log2 transform
    log2_vals = np.log2(np.array(list(rpm_filtered.values())) + 1)
    lo, hi = log2_vals.min(), log2_vals.max()
    norm_vals = 100 * (log2_vals - lo) / (hi - lo + 1e-9)
    for gene, norm_v in zip(rpm_filtered.keys(), norm_vals):
        merged_expr.setdefault(gene, []).append(float(norm_v))

# Average across datasets
final_expr = {g: float(np.mean(v)) for g, v in merged_expr.items()}
print(f"\n  Total genes with new expression data: {len(final_expr)}")

# ── Step 5: Update training CSV ───────────────────────────────────────────────
print("\n[STEP 5] Updating training CSV")
updated = 0
newly_added = 0
unchanged = 0

new_expressions = []
new_sources = []

for _, row in df_train.iterrows():
    gk = row['gene_key']
    if gk in final_expr:
        new_expressions.append(final_expr[gk])
        new_sources.append('riboseq_prjna1335396_prjna906596')
        if row['source'] == 'no_expr' or row['expression_level'] == 0:
            newly_added += 1
        else:
            updated += 1
    else:
        # Keep existing data unchanged
        new_expressions.append(row['expression_level'])
        new_sources.append(row['source'])
        unchanged += 1

df_train['expression_level'] = new_expressions
df_train['source'] = new_sources
df_train = df_train.drop(columns=['gene_key'])

print(f"  Genes updated (replaced existing): {updated}")
print(f"  Genes newly covered:               {newly_added}")
print(f"  Genes unchanged:                   {unchanged}")
print(f"  Source distribution after update:")
print(df_train['source'].value_counts().to_string())

# ── Step 6: Save ─────────────────────────────────────────────────────────────
print(f"\n[STEP 6] Saving updated CSV")
df_train.to_csv(TRAIN_CSV, index=False)
print(f"  Saved: {TRAIN_CSV}")

# ── Step 7: Update tracking file ─────────────────────────────────────────────
log_entry = (
    f"{date.today()} | E.coli PRJNA1335396+PRJNA906596 | "
    f"Updated {updated}, added {newly_added} new | "
    f"Normal growth MG1655, BAMs on disk\n"
)
with open(LOG_FILE, 'a') as f:
    f.write(log_entry)

tracker_text = TRACKER.read_text()
for item_num in ['4', '5']:
    tracker_text = tracker_text.replace(
        f'| {item_num} | **MED** | E. coli | PRJNA1335396 |',
        f'| {item_num} | **MED** | E. coli | PRJNA1335396 |'
    )
tracker_text = tracker_text.replace(
    '| 4 | **MED** | E. coli | PRJNA1335396 | Normal growth MG1655, BAMs on disk | `riboseq_new/ecoli/SRR35650607_*.bam` (45M unique reads, 31.5% mapping) | ⬜ PENDING |',
    f'| 4 | **MED** | E. coli | PRJNA1335396 | Normal growth MG1655, BAMs on disk | `riboseq_new/ecoli/SRR35650607_*.bam` (45M unique reads, 31.5% mapping) | ✅ DONE {date.today()} |'
)
tracker_text = tracker_text.replace(
    '| 5 | **MED** | E. coli | PRJNA906596 | pH 7.6 CONTROL (acid stress study) — is normal growth | `riboseq_new/ecoli/SRR22447282_*.bam` (7M unique reads, 30.3% mapping) | ⬜ PENDING |',
    f'| 5 | **MED** | E. coli | PRJNA906596 | pH 7.6 CONTROL (acid stress study) — is normal growth | `riboseq_new/ecoli/SRR22447282_*.bam` (7M unique reads, 30.3% mapping) | ✅ DONE {date.today()} |'
)
# Update completion log
tracker_text = tracker_text.replace(
    '| (pending) | | | |',
    f'| {date.today()} | 4,5 | Updated {updated}, added {newly_added} new | E.coli: new normal growth BAMs (PRJNA1335396 + PRJNA906596 pH7.6 control) |\n| (pending) | | | |',
    1
)
TRACKER.write_text(tracker_text)
print(f"  Tracking file updated")

print("\n" + "=" * 70)
print("DONE — E.coli training data updated with new normal growth datasets")
print("=" * 70)
