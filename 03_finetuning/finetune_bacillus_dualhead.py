#!/usr/bin/env python3
"""
B. subtilis dual-head fine-tuning (expression MLP + A-site 1D-CNN).

Fine-tunes expression and A-site prediction heads on B. subtilis Ribo-seq data
(GSE249448 WT0, 2 replicates, r=0.934). Last 2 transformer layers are unfrozen;
expression head uses log1p-TPM targets.

Output: models/dual_head_pretrain_bacillus.pt
"""

import sys, random, math, shutil
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / '02_foundation_pretraining'))
sys.path.insert(0, str(_REPO_ROOT / '03_finetuning'))

import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from train_mlm_industrial import (
    IndustrialMLM, AA_CODON_VOCAB, ID_TO_PAIR, ORG_TO_ID,
    CLS_TOKEN, PAD_TOKEN, _GENETIC_CODE,
)
from finetune_dual_head_specialist import DualHeadModel

BASE   = Path(__file__).resolve().parent.parent
OUT    = BASE / 'models/dual_head_pretrain_bacillus.pt'
BACKUP = BASE / 'models/old_models/dual_head_pretrain_bacillus_gse126234.pt'

EPOCHS   = 80
BATCH    = 48
LR       = 5e-5
UNFREEZE = 2      # fewer layers → less overfitting with small dataset
MAX_LEN  = 512
VAL_FRAC = 0.15
SEED     = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STOP_C = {'TAA','TAG','TGA'}
print(f'Device: {DEVICE}')

if OUT.exists():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(OUT, BACKUP)
    print(f'Backed up → {BACKUP.name}')

# ── Load GSE249448 WT0 ─────────────────────────────────────────────────────────
print('\nLoading GSE249448 WT0 TPM...')
df_gse = pd.read_excel(
    BASE / 'data/riboseq_raw/bacillus/processed/GSE249448/GSE249448_countsTPM_CDS_Ribo.xlsx'
)
df_gse['wt0_mean'] = (df_gse['WT0r_1'] + df_gse['WT0r_2']) / 2
df_gse = df_gse.set_index('Name')[['wt0_mean']]

# ── Join with training CSV ─────────────────────────────────────────────────────
df = pd.read_csv(BASE / 'data/raw/bacillus/bacillus_train.csv')
df['gene_name'] = df['sequence_id'].str.replace('bacillus_', '')
df = df.set_index('gene_name')
df = df.join(df_gse, how='inner').dropna(subset=['wt0_mean'])
df = df[df['wt0_mean'] > 0]  # exclude zero-expressed genes

# Log1p transform → normalise to [0,1]
df['log_tpm']   = np.log1p(df['wt0_mean'])
lmin, lmax      = df['log_tpm'].min(), df['log_tpm'].max()
df['expr_norm'] = (df['log_tpm'] - lmin) / (lmax - lmin + 1e-6)
print(f'  Matched genes: {len(df)}  TPM range: {df.wt0_mean.min():.1f}–{df.wt0_mean.max():.0f}')

# ── Build dataset ──────────────────────────────────────────────────────────────
items = []
for gene_id, row in df.iterrows():
    dna    = str(row['dna_sequence']).strip().upper()
    tokens = [CLS_TOKEN]
    for i in range(0, len(dna)-2, 3):
        cdn = dna[i:i+3].upper()
        aa  = _GENETIC_CODE.get(cdn, '?')
        if aa in ('*','?'): break
        tokens.append(AA_CODON_VOCAB.get((aa, cdn), PAD_TOKEN))
        if len(tokens) >= MAX_LEN: break
    if len(tokens) < 11: continue
    items.append({'tokens': tokens, 'expr': float(row['expr_norm'])})

print(f'  Valid items: {len(items)}')
random.shuffle(items)
n_val   = max(10, int(len(items) * VAL_FRAC))
val_it  = items[:n_val]
train_it = items[n_val:]
print(f'  Train: {len(train_it)}  Val: {len(val_it)}')

class BacDataset(Dataset):
    def __init__(self, it): self.it = it
    def __len__(self): return len(self.it)
    def __getitem__(self, i): return self.it[i]

def collate(batch):
    max_len = max(len(x['tokens']) for x in batch)
    ids, masks, exprs, oids = [], [], [], []
    for x in batch:
        t = x['tokens']; pad = max_len - len(t)
        ids.append(t + [PAD_TOKEN]*pad)
        masks.append([False]*len(t) + [True]*pad)
        exprs.append(x['expr'])
        oids.append(ORG_TO_ID['bacillus'])
    return {
        'input_ids': torch.tensor(ids,   dtype=torch.long),
        'attn_mask': torch.tensor(masks, dtype=torch.bool),
        'expr':      torch.tensor(exprs, dtype=torch.float32),
        'org_ids':   torch.tensor(oids,  dtype=torch.long),
    }

train_dl = DataLoader(BacDataset(train_it), batch_size=BATCH,
                      shuffle=True,  collate_fn=collate, num_workers=2, pin_memory=True)
val_dl   = DataLoader(BacDataset(val_it),   batch_size=BATCH,
                      shuffle=False, collate_fn=collate, num_workers=2, pin_memory=True)
total_steps = EPOCHS * len(train_dl)
print(f'  Batches/epoch: {len(train_dl)}  Total steps: {total_steps:,}')

# ── Model (pretrain backbone, unfreeze only 2 layers) ─────────────────────────
ckpt  = torch.load(BASE/'models/industrial_mlm_pretrain.pt', map_location='cpu', weights_only=False)
enc   = IndustrialMLM(use_grad_ckpt=False)
enc.load_state_dict(ckpt['model'])
model = DualHeadModel(enc).to(DEVICE)

for p in model.parameters(): p.requires_grad_(False)
# Only unfreeze last 2 transformer layers + expr_head + org_emb
for layer in model.encoder.layers[len(model.encoder.layers)-UNFREEZE:]:
    for p in layer.parameters(): p.requires_grad_(True)
for p in model.encoder.head.parameters():     p.requires_grad_(True)
for p in model.encoder.org_emb.parameters():  p.requires_grad_(True)
for p in model.expr_head.parameters():        p.requires_grad_(True)
# A-site head stays frozen (78% of profiles unreliable for Bacillus)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_p   = sum(p.numel() for p in model.parameters())
print(f'\nTrainable: {trainable:,}/{total_p:,} ({trainable/total_p*100:.1f}%)')
print(f'UNFREEZE={UNFREEZE} layers  LR={LR}  (reduced from 4/3e-4 to avoid overfit)')

params    = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0.05)

def lr_lambda(step):
    warmup=200
    if step<warmup: return step/max(warmup,1)
    prog=(step-warmup)/max(total_steps-warmup,1)
    return max(0.05, 0.5*(1+math.cos(math.pi*prog)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
mse_loss  = nn.MSELoss()

def run_epoch(dl, train=True):
    model.train(train)
    preds, targets, total_loss = [], [], 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in dl:
            ids  = batch['input_ids'].to(DEVICE)
            attn = batch['attn_mask'].to(DEVICE)
            oids = batch['org_ids'].to(DEVICE)
            tgt  = batch['expr'].to(DEVICE)
            _, pred_expr = model(ids, attn, oids)
            loss = mse_loss(pred_expr, tgt)
            if train:
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step(); scheduler.step()
            preds.extend(pred_expr.detach().cpu().tolist())
            targets.extend(tgt.cpu().tolist())
            total_loss += loss.item() * len(tgt)
    avg = total_loss / max(len(preds), 1)
    r2  = float(np.corrcoef(preds, targets)[0,1]**2) if len(preds)>2 else 0.0
    return avg, r2

print('\n=== Training (GSE249448 WT0, expression-only) ===')
best_r2 = -1.0

for epoch in range(1, EPOCHS+1):
    tr_loss, tr_r2 = run_epoch(train_dl, train=True)
    vl_loss, vl_r2 = run_epoch(val_dl,   train=False)
    lr_now = scheduler.get_last_lr()[0]
    print(f'Epoch {epoch:3d}/{EPOCHS}  '
          f'train={tr_loss:.4f} R²={tr_r2:.4f}  '
          f'val={vl_loss:.4f} R²={vl_r2:.4f}  lr={lr_now:.2e}')
    if vl_r2 > best_r2:
        best_r2 = vl_r2
        torch.save({'model':  model.state_dict(),
                    'epoch':  epoch, 'val_r2': vl_r2,
                    'source': 'GSE249448_WT0',
                    'backbone':'pretrain', 'org':'bacillus'}, OUT)
        print(f'          ✓ saved  val_R²={best_r2:.4f}')

print(f'\nBest val R²: {best_r2:.4f}  (old R²=0.334)')
print(f'Saved → {OUT}')
