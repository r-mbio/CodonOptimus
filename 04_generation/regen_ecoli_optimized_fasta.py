#!/usr/bin/env python3
"""
Regenerate optimized_fastas/CodonOptimus_ecoli.fasta using the corrected
industrial_mlm_all39_rs_ecoli.pt (real in-vivo top-10% gene selection).

Mirrors the exact co_greedy() generation logic already verified in
benchmark_consistent_v4.py, for the 4 benchmark proteins only.
"""
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / '02_foundation_pretraining'))
sys.path.insert(0, str(_REPO_ROOT / '03_finetuning'))

import torch
from train_mlm_industrial import (
    IndustrialMLM, AA_CODON_VOCAB, ID_TO_PAIR, ORG_TO_ID, AA_MASK, CLS_TOKEN, _GENETIC_CODE,
)

BASE = Path(__file__).resolve().parent.parent
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_AA_TO_TIDS = {}
for (aa, cdn), tid in AA_CODON_VOCAB.items():
    _AA_TO_TIDS.setdefault(aa, []).append(tid)

def dna2aa(dna):
    aa = []
    for i in range(0, len(dna)-2, 3):
        a = _GENETIC_CODE.get(dna[i:i+3].upper(), 'X')
        if a in ('*', 'X'): break
        aa.append(a)
    return ''.join(aa)

def parse_fasta(path):
    seqs = {}; name=None; buf=[]
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if name: seqs[name] = ''.join(buf)
                name = line[1:]; buf = []
            else: buf.append(line.replace(' ',''))
    if name: seqs[name] = ''.join(buf)
    return seqs

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

def load_mlm(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    m = IndustrialMLM(use_grad_ckpt=False)
    sd = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
    if any(k.startswith('encoder.') for k in sd):
        sd = {k.replace('encoder.', ''): v for k, v in sd.items() if k.startswith('encoder.')}
    m.load_state_dict(sd, strict=False)
    return m.eval().to(DEVICE)

print('Loading corrected E. coli RS-FT model...')
model_rs = load_mlm(BASE / 'models/industrial_mlm_all39_rs_ecoli.pt')

print('Loading WT protein sequences...')
wt_fa = parse_fasta(BASE / 'other_optimizers/benchmark/WT_sequences.fasta')
PROT_NAMES = ['BLG', 'HSA', 'PHYA', 'XYN2']
PROTEINS = {}
for header, dna in wt_fa.items():
    for pname in PROT_NAMES:
        if pname.lower() in header.lower():
            PROTEINS[pname] = dna2aa(dna.replace(' ', '').upper())
            print(f'  {pname}: {len(PROTEINS[pname])} AA')

out_path = BASE / 'other_optimizers/benchmark/optimized_fastas/CodonOptimus_ecoli.fasta'
backup = out_path.with_suffix('.fasta.bak_gse190954_cellfree')
if out_path.exists():
    import shutil
    shutil.copy(out_path, backup)
    print(f'Backed up old FASTA -> {backup.name}')

with open(out_path, 'w') as f:
    for pname in PROT_NAMES:
        if pname not in PROTEINS:
            print(f'  WARNING: {pname} not found in WT_sequences.fasta, skipping')
            continue
        dna = co_greedy(PROTEINS[pname], 'ecoli', model_rs)
        f.write(f'>{pname}_CodonOptimus_ecoli\n')
        for i in range(0, len(dna), 60):
            f.write(dna[i:i+60] + '\n')
        print(f'  {pname}: generated {len(dna)} nt')

print(f'\nSaved -> {out_path}')
