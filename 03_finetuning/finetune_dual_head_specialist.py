#!/usr/bin/env python3
"""
Per-organism dual-head specialist training.

Trains one dual-head model per organism with:
  - Encoder: ep11 OR pretrain (passed via --backbone)
  - Organism: one of ecoli / bacillus / s_cerevisiae / pichia (passed via --org)
  - Unfrozen: last 4 transformer layers + both prediction heads
  - Data: ALL Ribo-seq genes for that organism (not filtered to top-10%)
  - Loss: A-site Spearman + expression R² (balanced)

Usage (run 8 in parallel on H100):
  python3 finetune_dual_head_specialist.py --backbone ep11 --org ecoli
  python3 finetune_dual_head_specialist.py --backbone pretrain --org ecoli
  ... etc
"""

import sys, json, argparse, random, math
from pathlib import Path
from collections import defaultdict

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / '02_foundation_pretraining'))
sys.path.insert(0, str(_REPO_ROOT / '03_finetuning'))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr, pearsonr

from train_mlm_industrial import (
    IndustrialMLM, AA_CODON_VOCAB, ID_TO_PAIR, ORG_TO_ID,
    CLS_TOKEN, PAD_TOKEN, _GENETIC_CODE, AA_MASK, VOCAB_SIZE,
)

BASE      = Path(__file__).resolve().parent.parent
ASITE_DIR = BASE / 'data' / 'asite_profiles'
RAW_DIR   = BASE / 'data' / 'raw'

BACKBONE_CKPTS = {
    'ep11':    BASE / 'models' / 'industrial_mlm_ep11.pt',
    'pretrain':BASE / 'models' / 'industrial_mlm_pretrain.pt',
}

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--backbone', choices=['ep11','pretrain'], required=True)
    p.add_argument('--org',      required=True,
                   choices=['ecoli','bacillus','s_cerevisiae','pichia'])
    p.add_argument('--epochs',   type=int,   default=45)
    p.add_argument('--batch',    type=int,   default=48)
    p.add_argument('--lr',       type=float, default=3e-4)
    p.add_argument('--unfreeze', type=int,   default=4,
                   help='Number of transformer layers to unfreeze')
    p.add_argument('--w_asite',  type=float, default=1.0)
    p.add_argument('--w_expr',   type=float, default=1.0)
    p.add_argument('--max_len',  type=int,   default=512)
    p.add_argument('--seed',     type=int,   default=42)
    return p.parse_args()

# ── Heads (identical to finetune_dual_head.py) ────────────────────────────────
class AsiteHead(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.conv1 = nn.Conv1d(d, 256, 9, padding=4); self.norm1 = nn.LayerNorm(256)
        self.conv2 = nn.Conv1d(256, 64, 5, padding=2); self.norm2 = nn.LayerNorm(64)
        self.conv3 = nn.Conv1d(64,  32, 3, padding=1); self.norm3 = nn.LayerNorm(32)
        self.proj  = nn.Linear(32, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv1d): nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.gelu(self.norm1(self.conv1(x).transpose(1,2)).transpose(1,2))
        x = F.gelu(self.norm2(self.conv2(x).transpose(1,2)).transpose(1,2))
        x = F.gelu(self.norm3(self.conv3(x).transpose(1,2)).transpose(1,2))
        return self.proj(x.transpose(1,2)).squeeze(-1)

class ExpressionHead(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)

    def forward(self, x, attn_mask):
        valid  = (~attn_mask).unsqueeze(-1).float()
        pooled = (x * valid).sum(1) / valid.sum(1).clamp(min=1)
        return self.net(pooled).squeeze(-1)

class DualHeadModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder    = encoder
        self.asite_head = AsiteHead(512)
        self.expr_head  = ExpressionHead(512)

    def _get_hidden(self, input_ids, attn_mask, org_ids):
        enc = self.encoder
        B, L = input_ids.shape
        x = enc.token_emb(input_ids) + enc.org_emb(org_ids).unsqueeze(1)
        rs = enc.rope.sin[:L].unsqueeze(0).unsqueeze(0)
        rc = enc.rope.cos[:L].unsqueeze(0).unsqueeze(0)
        for layer in enc.layers:
            x = layer(x, rs, rc, attn_mask)
        return enc.norm(x)

    def forward(self, input_ids, attn_mask, org_ids):
        hidden     = self._get_hidden(input_ids, attn_mask, org_ids)
        codon_repr = hidden[:, 1:, :]
        codon_mask = attn_mask[:, 1:]
        return self.asite_head(codon_repr), self.expr_head(codon_repr, codon_mask)

# ── Data loading ──────────────────────────────────────────────────────────────
def load_locus_map(org):
    p = ASITE_DIR / org / 'locus_mapping.json'
    return json.load(open(p)) if p.exists() else {}

def normalise_asite(occ, dom):
    eps = 1e-6
    log_occ = np.log(occ + eps)
    reliable = dom > 0.5
    # Always standardise — use reliable positions if available, else all positions
    ref = log_occ[reliable] if reliable.sum() >= 5 else log_occ
    mu  = ref.mean()
    std = max(ref.std(), 0.5)   # floor std at 0.5 — prevents blow-up on near-constant genes
    normed = (log_occ - mu) / std
    return np.clip(normed, -10.0, 10.0)  # hard clip — no target > 10 SD

def load_org_data(org, max_len, val_frac=0.1, seed=42):
    rng      = random.Random(seed)
    org_id   = ORG_TO_ID[org]
    lmap     = load_locus_map(org)
    csv_path = RAW_DIR / org / f'{org}_train.csv'

    df = pd.read_csv(csv_path)
    # Keep only genes with Ribo-seq data
    if 'source' in df.columns:
        df = df[df['source'] != 'no_expr'].reset_index(drop=True)
        # K.phaffii: use only valid Ribo-seq sources (gse159336 = AOX1 wet-lab validated,
        # srr32315787_88 = normal YPD growth). mmc3 is removed from CSV — no longer needed.
        if org == 'pichia':
            import os
            if os.environ.get('PICHIA_GSE159336_ONLY'):
                df = df[df['source'] == 'riboseq_gse159336'].reset_index(drop=True)
                print(f'[{org}] GSE159336-only mode: {len(df)} genes')
            else:
                valid_pichia = df['source'].str.startswith('riboseq_')
                df = df[valid_pichia].reset_index(drop=True)
                print(f'[{org}] Valid Ribo-seq sources: {df["source"].value_counts().to_dict()}')

    # Normalise expression per organism to [0, 1]
    expr_vals  = df['expression_level'].values.astype(np.float32)
    expr_min   = expr_vals.min()
    expr_range = expr_vals.max() - expr_min + 1e-6
    df['expr_norm'] = (expr_vals - expr_min) / expr_range

    items = []
    for _, row in df.iterrows():
        gene_id    = str(row['sequence_id']).replace(f'{org}_', '')
        dna        = str(row['dna_sequence']).strip().upper()
        expr_level = float(row['expr_norm'])

        # Tokenise DNA
        tokens = [CLS_TOKEN]
        codons = []
        for i in range(0, len(dna) - 2, 3):
            cdn = dna[i:i+3]
            aa  = _GENETIC_CODE.get(cdn, '?')
            if aa in ('*', '?'): break
            tok = AA_CODON_VOCAB.get((aa, cdn), PAD_TOKEN)
            tokens.append(tok)
            codons.append(cdn)
            if len(codons) >= max_len - 1: break
        if len(codons) < 10: continue

        # A-site profile
        asite_norm = None
        gid_look   = lmap.get(gene_id, gene_id)
        for suffix in ['', f'_{gene_id}']:
            for fname in [f'{gid_look}.npy', f'b{gid_look}.npy']:
                p = ASITE_DIR / org / fname
                if p.exists():
                    try:
                        occ = np.load(p)
                        dom_p = ASITE_DIR / org / fname.replace('.npy', '_domain.npy')
                        dom   = np.load(dom_p) if dom_p.exists() else np.ones_like(occ)
                        asite_norm = normalise_asite(occ, dom)
                    except: pass
                    break
            if asite_norm is not None: break

        items.append({
            'tokens':     tokens,
            'n_codons':   len(codons),
            'expr':       expr_level,
            'asite':      asite_norm,
            'org_id':     org_id,
        })

    rng.shuffle(items)
    n_val  = max(1, int(len(items) * val_frac))
    return items[n_val:], items[:n_val]

class DualHeadDataset(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate_fn(batch, max_len):
    L = min(max_len + 1, max(len(it['tokens']) for it in batch))
    input_ids, attn_mask, org_ids, exprs, asites, asite_masks = [], [], [], [], [], []
    for it in batch:
        toks = it['tokens'][:L]
        pad  = L - len(toks)
        input_ids.append(toks + [PAD_TOKEN]*pad)
        attn_mask.append([False]*len(toks) + [True]*pad)
        org_ids.append(it['org_id'])
        exprs.append(it['expr'])

        n = it['n_codons']
        if it['asite'] is not None:
            a = it['asite'][:n]
            m = [False]*len(a)
            a_pad = max(0, L-1-len(a))
            asites.append(list(a) + [0.0]*a_pad)
            asite_masks.append(m + [True]*a_pad)
        else:
            asites.append([0.0]*(L-1))
            asite_masks.append([True]*(L-1))

    return {
        'input_ids':   torch.tensor(input_ids,   dtype=torch.long),
        'attn_mask':   torch.tensor(attn_mask,   dtype=torch.bool),
        'org_ids':     torch.tensor(org_ids,     dtype=torch.long),
        'expr':        torch.tensor(exprs,       dtype=torch.float32),
        'asite':       torch.tensor(asites,      dtype=torch.float32),
        'asite_mask':  torch.tensor(asite_masks, dtype=torch.bool),
    }

# ── Loss helpers ──────────────────────────────────────────────────────────────
def asite_loss(pred, target, mask):
    valid = ~mask
    if valid.sum() < 2: return torch.tensor(0.0, device=pred.device, requires_grad=True)
    p = pred[valid]; t = target[valid]
    return F.mse_loss(p, t)

def expr_loss(pred, target):
    return F.mse_loss(pred, target)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    import os as _os
    _suffix = '_gse159336only' if (_os.environ.get('PICHIA_GSE159336_ONLY') and args.org == 'pichia') else ''
    out_path = BASE / f'models/dual_head_{args.backbone}_{args.org}{_suffix}.pt'
    log_path = BASE / f'logs/dual_head_{args.backbone}_{args.org}{_suffix}.log'
    print(f'[{args.backbone}/{args.org}] Device: {device}')
    print(f'[{args.backbone}/{args.org}] Output: {out_path}')

    # Load encoder
    ckpt_path = BACKBONE_CKPTS[args.backbone]
    ckpt      = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    encoder   = IndustrialMLM(use_grad_ckpt=False)
    encoder.load_state_dict(ckpt['model'])
    print(f'[{args.backbone}/{args.org}] Encoder loaded (ep{ckpt.get("epoch",0)+1})')

    model = DualHeadModel(encoder).to(device)

    # Freeze all encoder → unfreeze last N layers + heads
    for p in model.encoder.parameters():
        p.requires_grad_(False)

    n_layers = len(model.encoder.layers)
    for layer in model.encoder.layers[n_layers - args.unfreeze:]:
        for p in layer.parameters():
            p.requires_grad_(True)
    model.encoder.norm.weight.requires_grad_(True)
    model.encoder.norm.bias.requires_grad_(True)

    for p in model.asite_head.parameters(): p.requires_grad_(True)
    for p in model.expr_head.parameters():  p.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f'[{args.backbone}/{args.org}] Trainable: {trainable:,}/{total_p:,} '
          f'({trainable/total_p*100:.1f}%) — last {args.unfreeze} enc layers + heads')

    # Load data
    print(f'[{args.backbone}/{args.org}] Loading Ribo-seq data...')
    train_items, val_items = load_org_data(args.org, args.max_len, seed=args.seed)
    print(f'[{args.backbone}/{args.org}] Train: {len(train_items)}  Val: {len(val_items)}')

    has_asite = sum(1 for it in train_items if it['asite'] is not None)
    print(f'[{args.backbone}/{args.org}] Genes with A-site: {has_asite}/{len(train_items)}')

    col_fn   = lambda b: collate_fn(b, args.max_len)
    train_dl = DataLoader(DualHeadDataset(train_items), batch_size=args.batch,
                          shuffle=True, collate_fn=col_fn, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(DualHeadDataset(val_items),   batch_size=args.batch,
                          shuffle=False,collate_fn=col_fn, num_workers=2)

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(train_dl)
    warmup      = min(200, total_steps // 10)

    def lr_lambda(step):
        if step < warmup: return step / max(warmup, 1)
        prog = (step - warmup) / max(total_steps - warmup, 1)
        return max(0.05, 0.5*(1+math.cos(math.pi*prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log_lines = []
    def log(msg): print(msg); log_lines.append(msg)

    log(f'Training: backbone={args.backbone} org={args.org} '
        f'epochs={args.epochs} unfreeze={args.unfreeze} lr={args.lr}')

    best_val_r2 = -999.0

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0; n_steps = 0

        for batch in train_dl:
            ids  = batch['input_ids'].to(device)
            attn = batch['attn_mask'].to(device)
            oids = batch['org_ids'].to(device)
            expr = batch['expr'].to(device)
            asit = batch['asite'].to(device)
            amsk = batch['asite_mask'].to(device)

            asite_pred, expr_pred = model(ids, attn, oids)

            la = asite_loss(asite_pred, asit, amsk) * args.w_asite
            le = expr_loss(expr_pred, expr)         * args.w_expr
            loss = la + le

            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step(); scheduler.step()

            total_loss += loss.item(); n_steps += 1

        avg_loss = total_loss / max(n_steps, 1)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        all_expr_pred, all_expr_true = [], []
        all_asite_pred, all_asite_true = [], []

        with torch.no_grad():
            for batch in val_dl:
                ids  = batch['input_ids'].to(device)
                attn = batch['attn_mask'].to(device)
                oids = batch['org_ids'].to(device)
                asit = batch['asite']
                amsk = batch['asite_mask']

                asite_pred, expr_pred = model(ids, attn, oids)

                all_expr_pred.extend(expr_pred.cpu().tolist())
                all_expr_true.extend(batch['expr'].tolist())

                valid = ~amsk
                ap = asite_pred.cpu()[valid]; at = asit[valid]
                if len(ap) > 0:
                    all_asite_pred.extend(ap.tolist())
                    all_asite_true.extend(at.tolist())

        r2 = pearsonr(all_expr_pred, all_expr_true)[0]**2 if len(all_expr_pred)>1 else 0
        sr = spearmanr(all_asite_pred, all_asite_true)[0] if len(all_asite_pred)>1 else 0

        msg = (f'[{args.backbone}/{args.org}] Ep {epoch:3d}/{args.epochs}  '
               f'loss={avg_loss:.4f}  expr_R²={r2:.4f}  asite_Sp={sr:.4f}  '
               f'lr={scheduler.get_last_lr()[0]:.2e}')
        log(msg)

        if r2 > best_val_r2:
            best_val_r2 = r2
            torch.save({
                'model':    model.state_dict(),
                'epoch':    epoch,
                'backbone': args.backbone,
                'org':      args.org,
                'val_r2':   r2,
                'val_sp':   sr,
                'unfreeze': args.unfreeze,
            }, out_path)
            log(f'  ✓ saved  R²={r2:.4f}')

    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, 'w') as f: f.write('\n'.join(log_lines))
    print(f'[{args.backbone}/{args.org}] Done → {out_path}  best_R²={best_val_r2:.4f}')

if __name__ == '__main__':
    main()
