#!/usr/bin/env python3
"""
Build E. coli A-site occupancy profiles from genuine in-vivo Ribo-seq BAMs.

Datasets used:
  - PRJNA906596 (SRR22447282, SRR22447285) — Ribo-seq, LB pH 7.6 control
  - PRJNA1335396 (SRR35650607) — Ribo-seq, normal exponential growth

Excluded (not ribosome footprint data):
  - GSE190954 — cell-free INRI-seq
  - PRJNA376419 — Rend-seq (RNA-seq; no ribosome footprint signal)

Output: data/asite_profiles/ecoli/{locus}.npy
"""

import json, subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.ndimage import gaussian_filter1d

BASE      = Path(__file__).resolve().parent.parent
TID_MAP_PATH = BASE / 'data' / 'star_index' / 'ecoli' / 'transcript_map.json'
BAM_DIR   = BASE / 'data' / 'riboseq_new' / 'ecoli'
ASITE_DIR = BASE / 'data' / 'asite_profiles' / 'ecoli'

# Only the genuine Ribo-seq BAMs (PRJNA906596 + PRJNA1335396).
# Excludes Rend-seq (PRJNA376419: SRR5278501/502, SRR6652810-813).
GENUINE_RIBOSEQ_BAMS = [
    'SRR22447282_Aligned.sortedByCoord.out.bam',  # PRJNA906596, pH7.6 control rep1
    'SRR22447285_Aligned.sortedByCoord.out.bam',  # PRJNA906596, pH7.6 control rep2
    'SRR35650607_Aligned.sortedByCoord.out.bam',  # PRJNA1335396, normal growth
]

OFFSETS_PROK = {
    20:12,21:12,22:12,23:12,24:12,25:12,26:12,27:12,28:12,
    29:13,30:13,31:13,32:14,33:14,34:15,35:15,36:15,
    37:15,38:15,39:15,40:15,
}
MIN_READS  = 50
MIN_CODONS = 30
SMOOTH_SIG = 1.0

def log(m): print(m, flush=True)

tid_map = json.load(open(TID_MAP_PATH))

bams = [BAM_DIR / b for b in GENUINE_RIBOSEQ_BAMS]
for b in bams:
    assert b.exists(), f"Missing BAM: {b}"
log(f'Using {len(bams)} genuine Ribo-seq BAMs: {[b.name for b in bams]}')

all_profiles = defaultdict(list)
all_read_counts = defaultdict(int)

for bam in bams:
    log(f'Processing {bam.name}...')
    gene_reads = defaultdict(list)
    gene_counts = defaultdict(int)
    total = 0

    cmd = ['samtools', 'view', '--no-PG', '-F', '2308', str(bam)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    for line in proc.stdout:
        f = line.split('\t')
        if len(f) < 10: continue
        ref = f[2]
        if ref == '*' or ref not in tid_map: continue
        pos = int(f[3]) - 1
        rlen = len(f[9])
        gene_reads[ref].append((rlen, pos))
        gene_counts[ref] += 1
        total += 1
    proc.wait()

    for ref, reads in gene_reads.items():
        if len(reads) < MIN_READS: continue
        info = tid_map[ref]
        n_codons = info['n_codons']
        if n_codons < MIN_CODONS: continue

        counts = np.zeros(n_codons, dtype=np.float32)
        for rlen, pos in reads:
            off = OFFSETS_PROK.get(rlen)
            if off is None: continue
            ci = (pos + off) // 3
            if 0 <= ci < n_codons:
                counts[ci] += 1
        mean = counts.mean()
        if mean > 0:
            counts /= mean
        all_profiles[ref].append(gaussian_filter1d(counts, SMOOTH_SIG).astype(np.float32))
        all_read_counts[ref] += gene_counts[ref]

    log(f'  -> {len(all_profiles)} genes accumulated so far')

# Average across the (up to 3) genuine BAMs per gene, full replacement (no blend with old file)
written = 0
for ref, profs in all_profiles.items():
    lens = [len(p) for p in profs]
    modal = max(set(lens), key=lens.count)
    arr = np.mean([p for p in profs if len(p) == modal], axis=0).astype(np.float32)
    np.save(ASITE_DIR / f'{ref}.npy', arr)
    written += 1

log(f'\nWritten {written} A-site profiles (genuine Ribo-seq only, GSE190954 and Rend-seq excluded)')
log(f'Genes with data from all 3 BAMs: {sum(1 for v in all_profiles.values() if len(v) == 3)}')
log(f'Genes with data from 2 BAMs:     {sum(1 for v in all_profiles.values() if len(v) == 2)}')
log(f'Genes with data from 1 BAM:      {sum(1 for v in all_profiles.values() if len(v) == 1)}')
