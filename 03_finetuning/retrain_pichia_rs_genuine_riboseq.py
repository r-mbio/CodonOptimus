#!/usr/bin/env python3
"""
Retrain K. phaffii RS-FT generator model from genuine, AOX1-matched Ribo-seq.

The original script that produced industrial_mlm_all39_rs_pichia.pt is no
longer present in this repository (lost during cleanup); only the checkpoint
survived. This script reconstructs that training step cleanly so the
pipeline is fully reproducible from source.

Data source: pichia_train.csv, filtered to source == 'riboseq_gse159336'
(AOX1-driven recombinant expression, the system matched to the K. phaffii
wet-lab validation in this study). The mmc3 source (incompatible scale,
identified as contaminated — see project history) and the
srr32315787_88_normal_growth complement (a different physiological
condition, not AOX1-induced) are both excluded, matching the same filter
already used for the K. phaffii dual-head model.

Everything else (architecture, hyperparameters, training procedure) follows
the same pattern as the E. coli / B. subtilis RS-FT scripts in this repo.

Output: models/industrial_mlm_all39_rs_pichia.pt
"""

import sys, math, random, shutil
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / '02_foundation_pretraining'))
sys.path.insert(0, str(_REPO_ROOT / '03_finetuning'))

import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from train_mlm_industrial import (
    IndustrialMLM, AA_CODON_VOCAB, ID_TO_PAIR, ORG_TO_ID, AA_MASK,
    CLS_TOKEN, PAD_TOKEN, _GENETIC_CODE,
)

BASE    = Path(__file__).resolve().parent.parent
CKPT_IN = BASE / 'models/industrial_mlm_csi_all39.pt'
OUT     = BASE / 'models/industrial_mlm_all39_rs_pichia.pt'
BACKUP  = BASE / 'models/old_models/industrial_mlm_all39_rs_pichia_prev.pt'

ORG          = 'pichia'
TOP_PCT      = 90
UNFREEZE     = 4
MASK_PROB    = 0.15
MAX_LEN      = 600
BATCH_SIZE   = 64
LR           = 1e-5
EPOCHS       = 30
WARMUP_STEPS = 100
LABEL_SMOOTH = 0.05
SEED         = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STOP_C = {'TAA', 'TAG', 'TGA'}
PAIR_TO_ID = {(aa, cdn): tid for (aa, cdn), tid in AA_CODON_VOCAB.items()}
print(f'Device: {DEVICE}')

if OUT.exists():
    BACKUP.parent.mkdir(exist_ok=True)
    shutil.copy(OUT, BACKUP)
    print(f'Backed up old model -> {BACKUP.name}')

# ── Load genuine AOX1-matched expression ───────────────────────────────────────
print('\nLoading pichia_train.csv (riboseq_gse159336 only)...')
df = pd.read_csv(BASE / 'data/raw/pichia/pichia_train.csv')
df = df.set_index('sequence_id')
df = df[df['source'] == 'riboseq_gse159336']
df = df.dropna(subset=['expression_level'])
print(f'  Genes (AOX1-matched, genuine Ribo-seq): {len(df)}')

# ── Select top-10% by expression_level ─────────────────────────────────────────
thresh = df['expression_level'].quantile(TOP_PCT / 100)
df_hi  = df[df['expression_level'] >= thresh]
print(f'\nTop-10% expression threshold: {thresh:.2f}')
print(f'Selected: {len(df_hi)} genes  '
      f'(mean expr = {df_hi["expression_level"].mean():.1f} vs all = {df["expression_level"].mean():.1f})')

# ── Tokenise ───────────────────────────────────────────────────────────────────
def dna_to_tokens(dna):
    dna = str(dna).upper()
    if len(dna) >= 3 and dna[-3:] in STOP_C: dna = dna[:-3]
    tokens = []
    for i in range(0, len(dna) - 2, 3):
        c = dna[i:i+3]
        if c in STOP_C or 'N' in c: break
        aa = _GENETIC_CODE.get(c, '')
        if not aa or aa == '*': break
        tid = PAIR_TO_ID.get((aa, c))
        if tid is None: break
        tokens.append(tid)
    return tokens

org_id  = ORG_TO_ID[ORG]
dna_col = 'dna_sequence'
samples = []
for _, row in df_hi.iterrows():
    toks = dna_to_tokens(str(row[dna_col]))
    if 20 <= len(toks) <= MAX_LEN:
        samples.append((toks, org_id))
print(f'\nTokenised sequences: {len(samples)}')

random.shuffle(samples)

class CodonDataset(Dataset):
    def __init__(self, s): self.s = s
    def __len__(self): return len(self.s)
    def __getitem__(self, i): return self.s[i]

def collate_fn(batch):
    max_len = max(len(t) for t, _ in batch) + 1
    I, L, A, O = [], [], [], []
    for toks, oid in batch:
        seq = [CLS_TOKEN] + list(toks); m = []; l = []
        for i, t in enumerate(seq):
            if i == 0: m.append(t); l.append(-100); continue
            if random.random() < MASK_PROB:
                m.append(AA_MASK[ID_TO_PAIR[t][0]]); l.append(t)
            else: m.append(t); l.append(-100)
        pad = max_len - len(seq)
        I.append(m + [PAD_TOKEN] * pad); L.append(l + [-100] * pad)
        A.append([False] * len(seq) + [True] * pad); O.append(oid)
    return {'input_ids': torch.tensor(I, dtype=torch.long),
            'labels':    torch.tensor(L, dtype=torch.long),
            'attn_mask': torch.tensor(A, dtype=torch.bool),
            'org_ids':   torch.tensor(O, dtype=torch.long)}

loader = DataLoader(CodonDataset(samples), batch_size=BATCH_SIZE,
                    shuffle=True, collate_fn=collate_fn,
                    num_workers=2, pin_memory=True)
print(f'Batches/epoch: {len(loader)}  Epochs: {EPOCHS}  Total steps: {EPOCHS * len(loader):,}')

# ── Model: CSI backbone -> unfreeze last 4 layers ──────────────────────────────
ckpt  = torch.load(CKPT_IN, map_location='cpu', weights_only=False)
model = IndustrialMLM(use_grad_ckpt=False)
model.load_state_dict(ckpt['model'])
model.to(DEVICE)

for p in model.parameters(): p.requires_grad_(False)
model.org_emb.weight.requires_grad_(True)
for layer in model.layers[len(model.layers) - UNFREEZE:]:
    for p in layer.parameters(): p.requires_grad_(True)
for p in model.head.parameters(): p.requires_grad_(True)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_p   = sum(p.numel() for p in model.parameters())
print(f'\nTrainable: {trainable:,}/{total_p:,} ({trainable/total_p*100:.1f}%) — last {UNFREEZE} layers + head')

params      = [p for p in model.parameters() if p.requires_grad]
optimizer   = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
total_steps = EPOCHS * len(loader)

def lr_lambda(step):
    if step < WARMUP_STEPS: return step / max(WARMUP_STEPS, 1)
    prog = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
    return max(0.05, 0.5 * (1 + math.cos(math.pi * prog)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
loss_fn   = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=LABEL_SMOOTH)

best_loss = float('inf'); model.train()
print('\n=== Training ===')
for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0; n_tok = 0
    for batch in loader:
        ids  = batch['input_ids'].to(DEVICE)
        lbls = batch['labels'].to(DEVICE)
        attn = batch['attn_mask'].to(DEVICE)
        oids = batch['org_ids'].to(DEVICE)
        logits = model(ids, attn, oids)
        B, L, V = logits.shape
        loss = loss_fn(logits.reshape(B * L, V), lbls.reshape(B * L))
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step(); scheduler.step()
        mask = (lbls != -100)
        total_loss += loss.item() * mask.sum().item()
        n_tok += mask.sum().item()
    avg = total_loss / max(n_tok, 1)
    ppl = math.exp(avg) if avg < 20 else float('inf')
    lr  = scheduler.get_last_lr()[0]
    print(f'Epoch {epoch:3d}/{EPOCHS}  loss={avg:.4f}  ppl={ppl:.3f}  lr={lr:.2e}')
    if avg < best_loss:
        best_loss = avg
        torch.save({'model': model.state_dict(), 'epoch': epoch,
                    'loss': avg, 'source': 'riboseq_gse159336_AOX1',
                    'n_genes': len(samples)}, OUT)

print(f'\nBest loss: {best_loss:.4f}')
print(f'Saved -> {OUT}')
