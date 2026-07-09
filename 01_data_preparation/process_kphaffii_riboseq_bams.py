#!/usr/bin/env python3
"""
Build K. phaffii A-site profiles and training CSV from Ribo-seq BAM files.
Dataset: SRR32315787, SRR32315788 — normal exponential growth.
Output: data/raw/pichia/pichia_train.csv with updated expression columns.
"""

import subprocess, shutil
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd

BASE     = Path(__file__).resolve().parent.parent
RAW_DIR  = BASE / 'data' / 'raw' / 'pichia'
BAM_DIR  = BASE / 'data' / 'bam' / 'pichia'
LOG_FILE = BASE / 'RIBOSEQ_PROCESSING_LOG.txt'
TRACKER  = BASE / 'RIBOSEQ_DATASET_PROCESSING.md'

TRAIN_CSV  = RAW_DIR / 'pichia_train.csv'
BACKUP_CSV = RAW_DIR / f'pichia_train_BACKUP_{date.today():%Y%m%d}.csv'

BAMS = [
    BAM_DIR / 'SRR32315787_Aligned.sortedByCoord.out.bam',
    BAM_DIR / 'SRR32315788_Aligned.sortedByCoord.out.bam',
]

print("=" * 70)
print("K. PHAFFII RIBO-SEQ PROCESSING — SRR32315787/88 (NORMAL GROWTH)")
print("=" * 70)

# Verify
for bam in BAMS:
    assert bam.exists(), f"MISSING: {bam}"
    assert Path(str(bam) + '.bai').exists(), f"BAM not indexed: {bam}"
    print(f"[VERIFIED] {bam.name}")

# Backup + load training CSV
shutil.copy2(TRAIN_CSV, BACKUP_CSV)
df_train = pd.read_csv(TRAIN_CSV)
df_train['gene_key'] = df_train['sequence_id'].str.replace('pichia_', '', regex=False)
print(f"\n[STEP 1] Training CSV: {len(df_train)} rows")
print(df_train['source'].value_counts().to_string())

# Gene ID mapping: AT249_GQ6702256 → GQ67_02256
def map_gene_id(bam_id):
    s = bam_id.replace('AT249_', '')   # GQ6702256
    if s.startswith('GQ67'):
        return 'GQ67_' + s[4:]         # GQ67_02256
    return s

# Run idxstats on both BAMs and merge
print("\n[STEP 2] Running samtools idxstats on both BAMs")
all_counts = {}
for bam in BAMS:
    result = subprocess.run(
        ['samtools', 'idxstats', str(bam)],
        capture_output=True, text=True, check=True
    )
    for line in result.stdout.strip().split('\n'):
        parts = line.split('\t')
        if len(parts) < 3 or parts[0] == '*': continue
        bam_id = parts[0]
        mapped = int(parts[2])
        gk = map_gene_id(bam_id)
        all_counts[gk] = all_counts.get(gk, 0) + mapped
    print(f"  {bam.name}: {sum(int(l.split(chr(9))[2]) for l in result.stdout.strip().split(chr(10)) if len(l.split(chr(9)))>=3 and l.split(chr(9))[0]!='*'):,} reads")

# Filter and normalize
total_reads = sum(all_counts.values())
rpm = {g: (v / total_reads) * 1e6 for g, v in all_counts.items() if v >= 5}
print(f"\n  Genes with ≥5 total reads: {len(rpm)}")

vals = np.array(list(rpm.values()))
log2_v = np.log2(vals + 1)
lo, hi = log2_v.min(), log2_v.max()
norm_v = 100 * (log2_v - lo) / (hi - lo + 1e-9)
expr_new = {g: float(v) for g, v in zip(rpm.keys(), norm_v)}

# Match to training CSV genes
train_gk = set(df_train['gene_key'])
matched = {g: v for g, v in expr_new.items() if g in train_gk}
print(f"  Matched to training CSV gene keys: {len(matched)}")

# Update CSV: only replace genes that are NOT already from riboseq_gse159336
# gse159336 is AOX1-driven (wet-lab validated) — keep as primary
print("\n[STEP 3] Updating training CSV")
updated = newly_added = kept_gse159336 = unchanged = 0
new_exp, new_src = [], []

for _, row in df_train.iterrows():
    gk = row['gene_key']
    if row['source'] == 'riboseq_gse159336':
        # Keep wet-lab validated AOX1 data
        new_exp.append(row['expression_level'])
        new_src.append(row['source'])
        kept_gse159336 += 1
    elif gk in matched:
        new_exp.append(matched[gk])
        new_src.append('riboseq_srr32315787_88_normal_growth')
        if row['source'] in ('no_expr', 'mmc3'):
            newly_added += 1
        else:
            updated += 1
    else:
        new_exp.append(row['expression_level'])
        new_src.append(row['source'])
        unchanged += 1

df_train['expression_level'] = new_exp
df_train['source'] = new_src
df_train = df_train.drop(columns=['gene_key'])

print(f"  Kept riboseq_gse159336 (AOX1, primary): {kept_gse159336}")
print(f"  Updated with new normal growth data:     {updated}")
print(f"  Newly covered (was no_expr/mmc3):        {newly_added}")
print(f"  Unchanged:                               {unchanged}")
print(f"  Source distribution:")
print(df_train['source'].value_counts().to_string())

df_train.to_csv(TRAIN_CSV, index=False)
print(f"\n[SAVED] {TRAIN_CSV}")

# Update tracking
log_entry = (f"{date.today()} | K.phaffii SRR32315787+88 normal growth | "
             f"Updated {updated}, added {newly_added}, kept {kept_gse159336} gse159336 | "
             f"YPD normal growth complement to AOX1\n")
with open(LOG_FILE, 'a') as f:
    f.write(log_entry)

tracker_text = TRACKER.read_text()
tracker_text = tracker_text.replace(
    '| 7 | **LOW** | K. phaffii | PRJNA669501 | YPD exponential OD600=2, GS115 | Download from SRA → align | ⬜ PENDING |',
    f'| 7 | **LOW** | K. phaffii | PRJNA669501/SRR32315787-88 | YPD normal growth, on-disk BAMs | `data/bam/pichia/SRR32315787+88_*.bam` | ✅ DONE {date.today()} |'
)
tracker_text = tracker_text.replace(
    '| (pending) | | | |',
    f'| {date.today()} | 7 | Updated {updated}, added {newly_added} | K.phaffii SRR32315787+88 normal growth (gse159336 kept as primary) |\n| (pending) | | | |',
    1
)
TRACKER.write_text(tracker_text)

print("\n" + "=" * 70)
print("DONE — K.phaffii updated with new normal growth Ribo-seq data")
print(f"  riboseq_gse159336 (AOX1/wet-lab) preserved for {kept_gse159336} genes")
print("=" * 70)
