#!/usr/bin/env python3
"""
Download ALL strain assemblies for industrial microorganisms from NCBI.

Strategy:
  1. datasets summary genome taxon <name> --as-json-lines  → get accession IDs
  2. If total assemblies > max_assemblies, keep best N (sorted by completeness)
  3. datasets download genome accession <ids> --include cds  → download by accession

Filters each CDS to:
  - transl_table=1 only  (nuclear standard code, excludes mitochondria/plastids)
  - starts ATG, ends with stop codon, len 90-3000 nt, no ambiguous bases

Sequence cap: after dedup, randomly sample SEQ_CAP sequences per organism
(reviewer-defensible: "up to 50,000 unique CDS per organism")

Output: data/pretrain/{org}/cds.tsv         (dna, protein columns; uncapped)
        data/pretrain/all_industrial_cds.tsv (combined; capped at SEQ_CAP per org)

Usage:
    python3 download_industrial_cds.py                        # all organisms
    python3 download_industrial_cds.py --org ecoli            # single organism
    python3 download_industrial_cds.py --org ecoli --force    # force re-download
    python3 download_industrial_cds.py --combine-only         # just rebuild combined TSV
"""

import os, sys, re, gzip, zipfile, argparse, subprocess, time, json
from pathlib import Path
import pandas as pd

import shutil
_datasets_on_path = shutil.which('datasets')
if _datasets_on_path is None:
    raise RuntimeError(
        "NCBI 'datasets' CLI not found on PATH. Install it with "
        "'conda install -c conda-forge ncbi-datasets-cli' (or see "
        "https://www.ncbi.nlm.nih.gov/datasets/docs/v2/download-and-install/) "
        "before running this script."
    )
DATASETS_BIN = Path(_datasets_on_path)
BASE    = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / 'data' / 'pretrain'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_CAP = 100_000       # max sequences per organism in combined training set

# Assembly level priority for sorting when capping (lower = better)
LEVEL_RANK = {'complete': 0, 'chromosome': 1, 'scaffold': 2, 'contig': 3}

# (scientific_name, max_assemblies, assembly_source, assembly_levels)
# max_assemblies=None → download ALL available annotated assemblies
# max_assemblies=N    → cap at N (sorted by completeness, take best N)
INDUSTRIAL_ORGS = {
    # ── Original 20 small/specialized organisms — download ALL assemblies ──
    'pichia':                ('Komagataella',                None, 'all',    'complete,chromosome,scaffold,contig'),
    'yarrowia':              ('Yarrowia lipolytica',         None, 'all',    'complete,chromosome,scaffold'),
    'trichoderma':           ('Trichoderma reesei',            None, 'all',    'complete,chromosome,scaffold,contig'),
    'corynebacterium':       ('Corynebacterium glutamicum',  None, 'all',    'complete,chromosome,scaffold'),
    'kluyveromyces':         ('Kluyveromyces lactis',        None, 'all',    'complete,chromosome,scaffold'),
    'ogataea':               ('Ogataea polymorpha',          None, 'all',    'complete,chromosome,scaffold'),
    'aspergillus_niger':     ('Aspergillus niger',           None, 'all',    'complete,chromosome,scaffold'),
    'aspergillus_oryzae':    ('Aspergillus oryzae',          None, 'all',    'complete,chromosome,scaffold'),
    'bacillus_licheniformis':('Bacillus licheniformis',      None, 'all',    'complete,chromosome,scaffold'),
    's_cerevisiae':          ('Saccharomyces cerevisiae',    None, 'all',    'complete,chromosome,scaffold'),
    'lactococcus':           ('Lactococcus lactis',          None, 'all',    'complete,chromosome,scaffold'),
    'pseudomonas_putida':    ('Pseudomonas putida',          None, 'all',    'complete,chromosome,scaffold'),
    'streptomyces':          ('Streptomyces coelicolor',     None, 'all',    'complete,chromosome,scaffold'),
    'kluyveromyces_marxianus':('Kluyveromyces marxianus',    None, 'all',    'complete,chromosome,scaffold,contig'),
    'penicillium':           ('Penicillium',                 None, 'all',    'complete,chromosome,scaffold,contig'),  # expanded to full genus
    'aspergillus_terreus':   ('Aspergillus terreus',         None, 'all',    'complete,chromosome,scaffold,contig'),
    'rhodotorula':           ('Rhodotorula toruloides',      None, 'all',    'complete,chromosome,scaffold,contig'),
    'scheffersomyces':       ('Scheffersomyces stipitis',    None, 'all',    'complete,chromosome,scaffold,contig'),
    'ashbya':                ('Ashbya gossypii',             None, 'all',    'complete,chromosome,scaffold,contig'),
    'myceliophthora':        ('Myceliophthora thermophila',  None, 'all',    'complete,chromosome,scaffold,contig'),
    # ── Large taxa — expanded from 10; sequence cap controls training balance ──
    'bacillus':              ('Bacillus subtilis',           None, 'RefSeq', 'complete,chromosome'),   # was: 10
    'ecoli':                 ('Escherichia coli',            500,  'RefSeq', 'complete'),               # was: 10
    # ── New organisms ─────────────────────────────────────────────────────────
    # Bacteria
    'bacillus_amyloliquefaciens': ('Bacillus amyloliquefaciens', 200, 'RefSeq', 'complete,chromosome'),
    'lactobacillus':         ('Lactobacillus plantarum',     200,  'RefSeq', 'complete,chromosome'),
    'streptomyces_venezuelae':('Streptomyces venezuelae',    None, 'all',    'complete,chromosome,scaffold,contig'),
    'streptomyces_lividans': ('Streptomyces lividans',       None, 'all',    'complete,chromosome,scaffold,contig'),
    'gluconobacter':         ('Gluconobacter oxydans',       None, 'all',    'complete,chromosome,scaffold,contig'),
    'cupriavidus':           ('Cupriavidus necator',         None, 'all',    'complete,chromosome,scaffold,contig'),
    'clostridium':           ('Clostridium acetobutylicum',  None, 'all',    'complete,chromosome,scaffold,contig'),
    'brevibacillus':         ('Brevibacillus brevis',        None, 'all',    'complete,chromosome,scaffold,contig'),
    # Fungi
    'aspergillus_fumigatus': ('Aspergillus fumigatus',       None, 'all',    'complete,chromosome,scaffold,contig'),
    'fusarium':              ('Fusarium oxysporum',          None, 'all',    'complete,chromosome,scaffold,contig'),
    'neurospora':            ('Neurospora crassa',           None, 'all',    'complete,chromosome,scaffold,contig'),
    'rhizopus':              ('Rhizopus delemar',            None, 'all',    'complete,chromosome,scaffold,contig'),
    'mucor':                 ('Mucor circinelloides',        None, 'all',    'complete,chromosome,scaffold,contig'),
    # Algae / microalgae
    'nannochloropsis':       ('Nannochloropsis',             None, 'all',    'complete,chromosome,scaffold,contig'),
    'phaeodactylum':         ('Phaeodactylum tricornutum',   None, 'all',    'complete,chromosome,scaffold,contig'),
    'chlamydomonas':         ('Chlamydomonas reinhardtii',   None, 'all',    'complete,chromosome,scaffold,contig'),
    # Mammalian cell lines
    'cho':                   ('Cricetulus griseus',          None, 'all',    'complete,chromosome,scaffold'),  # CHO cells
    'human':                 ('Homo sapiens',                1,    'RefSeq', 'complete'),                      # HEK293
}

STOP_CODONS = {'TAA', 'TAG', 'TGA'}
VALID_BASES  = set('ATCG')

def log(m): print(m, flush=True)

def is_valid_cds(dna: str) -> bool:
    dna = dna.upper().replace('U','T')
    if len(dna) < 90 or len(dna) > 3000: return False
    if len(dna) % 3 != 0: return False
    if not dna.startswith('ATG'): return False
    if dna[-3:] not in STOP_CODONS: return False
    if not all(b in VALID_BASES for b in dna): return False
    return True

def parse_cds_fna(text: str) -> list:
    """Parse NCBI cds_from_genomic.fna. Keeps only transl_table=1."""
    results = []
    header, seq_parts = '', []

    def flush(h, s):
        if not s: return
        dna = ''.join(s).upper().replace('U','T')
        tt  = re.search(r'\[transl_table=(\d+)\]', h)
        if tt and int(tt.group(1)) != 1: return
        prot = re.search(r'\[translation=([A-Z*]+)\]', h)
        protein = prot.group(1).rstrip('*') if prot else ''
        if is_valid_cds(dna):
            results.append((dna, protein))

    for line in text.splitlines():
        if line.startswith('>'):
            flush(header, seq_parts)
            header, seq_parts = line, []
        else:
            seq_parts.append(line.strip())
    flush(header, seq_parts)
    return results

def get_accessions(sci_name: str, max_assemblies, assembly_source: str,
                   assembly_levels: str) -> list:
    """
    Query NCBI summary to get assembly accessions.
    Returns list of accession strings, sorted best-first, capped at max_assemblies.
    """
    cmd = [str(DATASETS_BIN), 'summary', 'genome', 'taxon', sci_name,
           '--assembly-level', assembly_levels,
           '--annotated',
           '--as-json-lines']
    if assembly_source != 'all':
        cmd += ['--assembly-source', assembly_source]
    if max_assemblies is not None:
        cmd += ['--limit', str(max_assemblies * 3)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f'  SUMMARY ERROR: {result.stderr[:300]}')
        return []

    assemblies = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            reports = rec.get('reports', [rec])
            for r in reports:
                acc   = r.get('accession', '')
                level = r.get('assembly_info', {}).get('assembly_level', 'contig').lower()
                if acc:
                    assemblies.append((LEVEL_RANK.get(level, 9), acc))
        except json.JSONDecodeError:
            continue

    assemblies.sort()
    accs = [a for _, a in assemblies]
    if max_assemblies is not None:
        accs = accs[:max_assemblies]
    return accs

def download_org(org_name: str, sci_name: str, max_assemblies,
                 assembly_source: str, assembly_levels: str,
                 force: bool = False) -> pd.DataFrame:
    org_dir  = OUT_DIR / org_name
    org_dir.mkdir(exist_ok=True)
    tsv_path = org_dir / 'cds.tsv'

    if tsv_path.exists() and not force:
        df = pd.read_csv(tsv_path, sep='\t')
        log(f'  {org_name}: loaded existing {len(df):,} sequences')
        return df

    # Step 1: get accession list
    log(f'  Fetching assembly list for {sci_name}...')
    accs = get_accessions(sci_name, max_assemblies, assembly_source, assembly_levels)
    if not accs:
        log(f'  ERROR: no assemblies found for {sci_name}')
        return pd.DataFrame(columns=['dna','protein'])

    log(f'  Found {len(accs)} assemblies → downloading CDS...')
    log(f'  Accessions: {", ".join(accs[:5])}{"..." if len(accs) > 5 else ""}')

    # Step 2: download by accession using --inputfile (avoids command line length limit)
    zip_path = org_dir / f'{org_name}_cds.zip'
    acc_file = org_dir / 'accessions.txt'
    acc_file.write_text('\n'.join(accs))

    cmd = [str(DATASETS_BIN), 'download', 'genome', 'accession',
           '--inputfile', str(acc_file),
           '--include', 'cds',
           '--filename', str(zip_path)]

    log(f'  Command: datasets download genome accession --inputfile <{len(accs)} accs> --include cds')
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        log(f'  ERROR: {result.stderr[:300]}')
        return pd.DataFrame(columns=['dna','protein'])

    if not zip_path.exists():
        log(f'  ERROR: zip not created')
        return pd.DataFrame(columns=['dna','protein'])

    zip_size_mb = zip_path.stat().st_size / 1e6
    log(f'  Downloaded {zip_size_mb:.0f} MB in {elapsed:.0f}s  →  parsing CDS...')

    pairs = []
    try:
        with zipfile.ZipFile(zip_path) as z:
            cds_files = [n for n in z.namelist()
                         if 'cds_from_genomic' in n and
                         (n.endswith('.fna') or n.endswith('.fna.gz'))]
            log(f'  Found {len(cds_files)} CDS files in zip')
            for fname in cds_files:
                data = z.read(fname)
                if fname.endswith('.gz'):
                    data = gzip.decompress(data)
                text = data.decode('utf-8', errors='ignore')
                pairs.extend(parse_cds_fna(text))
    except Exception as e:
        log(f'  Parse error: {e}')
        return pd.DataFrame(columns=['dna','protein'])

    seen = set(); unique = []
    for dna, prot in pairs:
        if dna not in seen:
            seen.add(dna); unique.append((dna, prot))

    log(f'  {len(pairs):,} raw CDS → {len(unique):,} unique (transl_table=1, 90-3000 nt)')

    df = pd.DataFrame(unique, columns=['dna','protein'])
    df.to_csv(tsv_path, sep='\t', index=False)
    log(f'  Saved → {tsv_path}')
    zip_path.unlink()
    if acc_file.exists():
        acc_file.unlink()
    return df

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--org',          default=None, choices=list(INDUSTRIAL_ORGS.keys()))
parser.add_argument('--force',        action='store_true', help='Re-download even if cds.tsv exists')
parser.add_argument('--combine-only', action='store_true', help='Skip downloads; just rebuild combined TSV')
args = parser.parse_args()

orgs = {args.org: INDUSTRIAL_ORGS[args.org]} if args.org else INDUSTRIAL_ORGS

log('=' * 65)
log('Industrial CDS Download — ALL strains, nuclear sequences only')
log(f'Sequence cap per organism: {SEQ_CAP:,}')
log('=' * 65)

all_dfs = []

if not args.combine_only:
    for org_name, (sci_name, max_assemblies, src, levels) in orgs.items():
        log(f'\n{"="*50}')
        log(f'  {org_name.upper()}  ({sci_name})')
        cap_str = f'cap={max_assemblies}' if max_assemblies else 'ALL'
        log(f'  assemblies: {cap_str}  source={src}  levels={levels}')
        df = download_org(org_name, sci_name, max_assemblies, src, levels, force=args.force)
        if len(df) > 0:
            df['organism'] = org_name
            all_dfs.append(df)
else:
    # Load all existing cds.tsv files
    for org_name in INDUSTRIAL_ORGS:
        tsv_path = OUT_DIR / org_name / 'cds.tsv'
        if tsv_path.exists():
            df = pd.read_csv(tsv_path, sep='\t')
            df['organism'] = org_name
            all_dfs.append(df)
            log(f'  {org_name}: {len(df):,} sequences')
        else:
            log(f'  {org_name}: no cds.tsv found, skipping')

if all_dfs and (not args.org or args.combine_only):
    # Apply per-organism sequence cap (reproducible random sample)
    # Skip when --org is used (single-org download; run --combine-only to rebuild combined TSV)
    capped_dfs = []
    for df in all_dfs:
        org = df['organism'].iloc[0]
        n_raw = len(df)
        if n_raw > SEQ_CAP:
            df = df.sample(n=SEQ_CAP, random_state=42)
            log(f'  {org:<28} {SEQ_CAP:>8,} sequences  (capped from {n_raw:,})')
        else:
            log(f'  {org:<28} {n_raw:>8,} sequences')
        capped_dfs.append(df)

    combined = pd.concat(capped_dfs, ignore_index=True)
    out_path = OUT_DIR / 'all_industrial_cds.tsv'
    combined.to_csv(out_path, sep='\t', index=False)

    log(f'\n{"="*65}')
    log('COMBINED DATASET (after SEQ_CAP)')
    log(f'{"="*65}')
    for org in combined['organism'].unique():
        n = (combined['organism'] == org).sum()
        log(f'  {org:<28} {n:>8,} sequences')
    log(f'  {"TOTAL":<28} {len(combined):>8,} sequences')
    log(f'  Saved → {out_path}')
