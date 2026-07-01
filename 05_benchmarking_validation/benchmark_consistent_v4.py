#!/usr/bin/env python3
"""
Consistent benchmark v4 — all tools scored with:
  • CSI       : genomic codon table (data/codon_tables/{org}_codon_usage.json)
  • ExprScore : per-organism dual-head specialists (same model for all)

CHANGES vs v3:
  CO Ribo-seq generator: per-organism all39+RS models
    E.coli    → models/industrial_mlm_all39_rs_ecoli.pt
    K.phaffii → models/industrial_mlm_all39_rs_pichia.pt
    B.sub     → models/industrial_mlm_all39_rs_bacillus.pt  (NEW: LB Ribo-seq FT)
    S.cer     → models/industrial_mlm_csi_all39.pt  (no valid Ribo-seq)
  CO CSI-FT: still uses industrial_mlm_csi_all39.pt (single 39-org model)

Tools: CO RS (all39+RS), CO Foundation, CO CSI-FT (all39), CT, IDT, Twist, Genscript
Output: results/benchmark_consistent_v4.csv
"""

import sys, json, random
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / '02_foundation_pretraining'))
sys.path.insert(0, str(_REPO_ROOT / '03_finetuning'))

import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
import openpyxl
from CAI import CAI, relative_adaptiveness

from train_mlm_industrial import (
    IndustrialMLM, AA_CODON_VOCAB, ID_TO_PAIR, ORG_TO_ID,
    CLS_TOKEN, PAD_TOKEN, _GENETIC_CODE, AA_MASK,
)
from finetune_dual_head_specialist import (
    DualHeadModel, AsiteHead, ExpressionHead,
)

BASE   = Path(__file__).resolve().parent.parent
OUT    = BASE / 'results/benchmark_consistent_v4.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STOP_C = {'TAA','TAG','TGA'}
print(f'Device: {DEVICE}')

# ── Codon utilities ────────────────────────────────────────────────────────────
_AA_TO_TIDS = {}
for (_aa, _cdn), _tid in AA_CODON_VOCAB.items():
    _AA_TO_TIDS.setdefault(_aa, []).append(_tid)

def codon_split(dna):
    dna = str(dna).upper().replace(' ','')
    if len(dna) >= 3 and dna[-3:] in STOP_C: dna = dna[:-3]
    return [dna[i:i+3] for i in range(0, len(dna)-2, 3)
            if dna[i:i+3] not in STOP_C and 'N' not in dna[i:i+3]]

def dna2aa(dna):
    aa = []
    for i in range(0, len(dna)-2, 3):
        a = _GENETIC_CODE.get(dna[i:i+3].upper(), 'X')
        if a in ('*','X'): break
        aa.append(a)
    return ''.join(aa)

def parse_fasta(path):
    seqs = {}; name = None; buf = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if name: seqs[name] = ''.join(buf)
                name = line[1:]; buf = []
            else: buf.append(line.replace(' ',''))
    if name: seqs[name] = ''.join(buf)
    return seqs

# ── Genomic CSI weights ────────────────────────────────────────────────────────
CODON_TABLE_FILES = {
    'ecoli':        'ecoli_codon_usage.json',
    'bacillus':     'bacillus_codon_usage.json',
    's_cerevisiae': 's_cerevisiae_codon_usage.json',
    'pichia':       'pichia_codon_usage.json',
}
print('Building genomic CSI weights...')
CSI_WEIGHTS = {}
for org, fname in CODON_TABLE_FILES.items():
    with open(BASE / 'data/codon_tables' / fname) as f:
        tbl = json.load(f)
    pool = []
    for aa, codons in tbl.items():
        if aa == '*': continue
        total = sum(codons.values())
        for cod, freq in codons.items():
            pool.extend([cod] * max(1, round(freq * 1000)))
    CSI_WEIGHTS[org] = relative_adaptiveness(sequences=[''.join(pool) + 'TAA'])
    print(f'  {org}: OK')

def calc_csi(dna, org):
    codons = codon_split(str(dna).replace(' ','').upper())
    seq = ''.join(codons)
    if len(seq) < 3: return float('nan')
    try: return CAI(seq, CSI_WEIGHTS[org])
    except: return float('nan')

# ── Model loading ──────────────────────────────────────────────────────────────
def load_dual_head(path):
    ckpt    = torch.load(path, map_location=DEVICE, weights_only=False)
    encoder = IndustrialMLM(use_grad_ckpt=False)
    model   = DualHeadModel(encoder).to(DEVICE)
    sd = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(sd, strict=False)
    return model.eval()

def load_mlm(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    m    = IndustrialMLM(use_grad_ckpt=False)
    sd   = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
    if any(k.startswith('encoder.') for k in sd):
        sd = {k.replace('encoder.',''):v for k,v in sd.items() if k.startswith('encoder.')}
    m.load_state_dict(sd, strict=False)
    return m.eval().to(DEVICE)

print('\nLoading per-organism dual-head models...')
DH = {}
DH_PATHS = {
    'ecoli':        BASE/'models/dual_head_ep11_ecoli.pt',
    'bacillus':     BASE/'models/dual_head_pretrain_bacillus.pt',
    's_cerevisiae': BASE/'models/dual_head_pretrain_s_cerevisiae.pt',
    'pichia':       BASE/'models/dual_head_ep11_pichia.pt',
}
for org, path in DH_PATHS.items():
    DH[org] = load_dual_head(path)
    print(f'  {org}: loaded')

print('\nLoading generator models...')
MLM_PRETRAIN = load_mlm(BASE/'models/industrial_mlm_pretrain.pt')
# v4: per-organism all39+RS for E.coli and K.phaffii; all39 for B.sub and S.cer
MLM_RS = {
    'ecoli':        load_mlm(BASE/'models/industrial_mlm_all39_rs_ecoli.pt'),
    'bacillus':     load_mlm(BASE/'models/industrial_mlm_all39_rs_bacillus.pt'),
    's_cerevisiae': load_mlm(BASE/'models/industrial_mlm_csi_all39.pt'),
    'pichia':       load_mlm(BASE/'models/industrial_mlm_all39_rs_pichia.pt'),
}
# v4: single all39 model as CO CSI-FT (39-org, all layers, genomic ref)
CSI_ALL39 = load_mlm(BASE/'models/industrial_mlm_csi_all39.pt')
CSI_MODELS = {org: CSI_ALL39 for org in ('ecoli','bacillus','s_cerevisiae','pichia')}
print('  all loaded')

# ── ExprScore from DNA using per-org dual-head ─────────────────────────────────
@torch.no_grad()
def score_expr(dna, org):
    tokens = [CLS_TOKEN]
    for i in range(0, len(dna)-2, 3):
        cdn = dna[i:i+3].upper()
        aa  = _GENETIC_CODE.get(cdn, '?')
        if aa in ('*','?'): break
        tok = AA_CODON_VOCAB.get((aa, cdn), PAD_TOKEN)
        tokens.append(tok)
    if len(tokens) < 4: return float('nan')
    ids  = torch.tensor([tokens], dtype=torch.long,  device=DEVICE)
    mask = torch.zeros(1, len(tokens), dtype=torch.bool, device=DEVICE)
    oid  = torch.tensor([ORG_TO_ID[org]], dtype=torch.long, device=DEVICE)
    _, expr = DH[org](ids, mask, oid)
    return float(expr)

# ── Greedy sequence generation ─────────────────────────────────────────────────
@torch.no_grad()
def co_greedy(aa_seq, org, model):
    valid = [a for a in aa_seq.upper() if a in AA_MASK]
    tok   = [CLS_TOKEN] + [AA_MASK[a] for a in valid]
    ids   = torch.tensor([tok], dtype=torch.long, device=DEVICE)
    msk   = torch.zeros(1, len(tok), dtype=torch.bool, device=DEVICE)
    oid   = torch.tensor([ORG_TO_ID[org]], dtype=torch.long, device=DEVICE)
    logits = model(ids, msk, oid)
    result = []
    for pi, aa_ch in enumerate(valid):
        allowed = _AA_TO_TIDS.get(aa_ch, [])
        pl = logits[0, pi+1, :61].clone()
        if allowed:
            mask_l = torch.full((61,), float('-inf'), device=DEVICE)
            for t in allowed:
                if t < 61: mask_l[t] = 0.0
            pl = pl + mask_l
        result.append(ID_TO_PAIR[pl.argmax().item()][1])
    return ''.join(result)

# ── Protein AAs ───────────────────────────────────────────────────────────────
BLG_SIGNAL_LEN = 16

print('\nLoading protein AA sequences...')
wt_fa = parse_fasta(BASE/'other_optimizers/benchmark/WT_sequences.fasta')
PROTEINS = {}
PROT_NAMES = ['BLG','HSA','PHYA','XYN2']
for header, dna in wt_fa.items():
    for pname in PROT_NAMES:
        if pname.lower() in header.lower():
            PROTEINS[pname] = dna2aa(dna.replace(' ','').upper())
            print(f'  {pname} (from WT): {len(PROTEINS[pname])} AA')
            break
co_ecoli = parse_fasta(BASE/'other_optimizers/benchmark/optimized_fastas/CodonOptimus_ecoli.fasta')
for name, dna in co_ecoli.items():
    for pname in PROT_NAMES:
        if pname.lower() in name.lower() and pname not in PROTEINS:
            aa_full = dna2aa(dna.replace(' ','').upper())
            PROTEINS[pname] = aa_full[BLG_SIGNAL_LEN:] if pname == 'BLG' else aa_full
            print(f'  {pname} (from CO, trimmed): {len(PROTEINS[pname])} AA')
            break
print('  Protein lengths:', {p: len(v) for p, v in PROTEINS.items()})

ORG_CFG = [
    ('ecoli',        'E. coli'),
    ('bacillus',     'B. subtilis'),
    ('s_cerevisiae', 'S. cerevisiae'),
    ('pichia',       'K. phaffii'),
]

# ── Load existing DNA sequences ───────────────────────────────────────────────
print('\nLoading existing sequences...')

CO_RS_SEQS = {}
for org, _ in ORG_CFG:
    fa = parse_fasta(BASE/f'other_optimizers/benchmark/optimized_fastas/CodonOptimus_{org}.fasta')
    for name, dna in fa.items():
        for pname in PROT_NAMES:
            if pname.lower() in name.lower():
                dna_clean = dna.replace(' ','').upper()
                if pname == 'BLG' and len(codon_split(dna_clean)) == 178:
                    dna_clean = dna_clean[BLG_SIGNAL_LEN*3:]
                CO_RS_SEQS[(pname, org)] = dna_clean

CT_SEQS = {}
ct_fa = parse_fasta(BASE/'other_optimizers/codontransformer/CodonTransformer_all_genes_real.fasta')
for header, dna in ct_fa.items():
    hl = header.lower()
    for pname in PROT_NAMES:
        if pname.lower() in hl or (pname=='PHYA' and 'phya' in hl):
            for org, _ in ORG_CFG:
                if org in hl or (org=='s_cerevisiae' and 's_cerevisiae' in hl):
                    dna_clean = dna.replace(' ','').upper()
                    if pname == 'BLG' and len(codon_split(dna_clean)) == 178:
                        dna_clean = dna_clean[BLG_SIGNAL_LEN*3:]
                    CT_SEQS[(pname, org)] = dna_clean

IDT_SEQS = {}
IDT_FILES = {
    'ecoli':'IDT_result_E.coli.csv','bacillus':'IDT_result_Bacillus.csv',
    's_cerevisiae':'IDT_result_Saccharomyces.csv','pichia':'IDT_result_pichia.csv',
}
for org, fname in IDT_FILES.items():
    df_idt = pd.read_csv(BASE/'other_optimizers/idt'/fname)
    df_idt.columns = [c.strip() for c in df_idt.columns]
    seq_col  = next((c for c in df_idt.columns if 'Optimized' in c), None)
    name_col = next((c for c in df_idt.columns if 'Name' in c), None)
    if seq_col is None or name_col is None: continue
    for _, row in df_idt.iterrows():
        name = str(row[name_col]).strip().upper()
        dna  = str(row[seq_col]).replace(' ','').upper()
        for pname in PROT_NAMES:
            if pname in name:
                IDT_SEQS[(pname, org)] = dna

TWIST_SEQS = {}
twist_fa = parse_fasta(BASE/'other_optimizers/twist/Twist_optimized.fasta')
ORG_MAP_TW = {'pichia':'pichia','e.coli':'ecoli','bacillus':'bacillus',
               'saccharomyces':'s_cerevisiae'}
for header, dna in twist_fa.items():
    hl = header.lower()
    for pname in PROT_NAMES:
        if pname.lower() in hl:
            for alias, org in ORG_MAP_TW.items():
                if alias in hl:
                    TWIST_SEQS[(pname, org)] = dna.replace(' ','').upper()

GS_SEQS = {}
GS_FILES = {
    'ecoli':'Genscript_E.coli.xlsx','bacillus':'Genscript_Bacillus.xlsx',
    's_cerevisiae':'Genscript_Saccharomyces.xlsx','pichia':'Genscript_Pichia.xlsx',
}
for org, fname in GS_FILES.items():
    wb = openpyxl.load_workbook(BASE/'other_optimizers/genscript'/fname)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if not rows: continue
    hdr = [str(c).strip().lower() if c else '' for c in rows[0]]
    name_col = next((i for i,h in enumerate(hdr) if 'gene' in h), None)
    seq_col  = next((i for i,h in enumerate(hdr) if 'optimized' in h and 'sequence' in h), None)
    if seq_col is None:
        seq_col = next((i for i,h in enumerate(hdr) if 'optimized' in h), None)
    if name_col is None or seq_col is None: continue
    for row in rows[1:]:
        if not row[name_col]: continue
        name = str(row[name_col]).strip().upper()
        dna  = str(row[seq_col]).replace(' ','').upper() if row[seq_col] else ''
        for pname in PROT_NAMES:
            if pname in name:
                GS_SEQS[(pname, org)] = dna

print(f'  CO RS:{len(CO_RS_SEQS)}  CT:{len(CT_SEQS)}  IDT:{len(IDT_SEQS)}  '
      f'Twist:{len(TWIST_SEQS)}  GS:{len(GS_SEQS)}')

# ── Score all ─────────────────────────────────────────────────────────────────
records = []

def score_one(pname, org, org_label, optimizer, dna):
    if not dna or len(dna) < 9: return
    dna = dna.replace(' ','').upper()
    csi  = calc_csi(dna, org)
    expr = score_expr(dna, org)
    records.append({'protein':pname,'organism':org,'org_label':org_label,
                    'optimizer':optimizer,'csi':round(csi,6),'expr_score':round(expr,6)})
    print(f'  {pname}/{org} | {optimizer:22s}  CSI={csi:.4f}  Expr={expr:.4f}')

print('\n=== Scoring ===')
for org, org_label in ORG_CFG:
    print(f'\n--- {org_label} ---')
    for pname in PROT_NAMES:
        aa = PROTEINS.get(pname)
        if not aa: continue
        score_one(pname, org, org_label, 'CO Ribo-seq',
                  co_greedy(aa, org, MLM_RS[org]))
        score_one(pname, org, org_label, 'CO Foundation',
                  co_greedy(aa, org, MLM_PRETRAIN))
        score_one(pname, org, org_label, 'CO CSI-FT',
                  co_greedy(aa, org, CSI_MODELS[org]))
        score_one(pname, org, org_label, 'CodonTransformer',
                  CT_SEQS.get((pname, org), ''))
        score_one(pname, org, org_label, 'IDT',
                  IDT_SEQS.get((pname, org), ''))
        score_one(pname, org, org_label, 'Twist',
                  TWIST_SEQS.get((pname, org), ''))
        score_one(pname, org, org_label, 'Genscript',
                  GS_SEQS.get((pname, org), ''))

df = pd.DataFrame(records)
df.to_csv(OUT, index=False)
print(f'\nSaved → {OUT}  ({len(df)} rows)')

print('\n=== CSI by organism + optimizer ===')
print(df.groupby(['organism','optimizer'])['csi'].mean().unstack().round(3).to_string())
print('\n=== ExprScore by organism + optimizer ===')
print(df.groupby(['organism','optimizer'])['expr_score'].mean().unstack().round(3).to_string())
