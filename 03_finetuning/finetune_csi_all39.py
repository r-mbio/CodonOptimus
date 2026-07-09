#!/usr/bin/env python3
"""
CSI fine-tune using ALL 39 organisms in the pre-training corpus.

Speed optimisations vs original:
  - 39 organisms processed in PARALLEL (multiprocessing.Pool, up to 48 workers)
  - CSI computed with fast numpy geometric mean instead of slow CAI library loop
  - TSV loaded once with pandas, split by organism in memory

Pipeline:
  1. Load data/pretrain/all_industrial_cds.tsv (1.8M seqs, 39 orgs)
  2. Per organism (parallel): build genomic CSI weights, take top-10% by CSI
  3. Pool all top-10% seqs across 39 orgs (~174K total)
  4. Fine-tune ALL layers (CT-style), batch=2048, grad_ckpt, LR=1e-5

Input:   models/industrial_mlm_ep11.pt
Output:  models/industrial_mlm_csi_all39.pt
         logs/finetune_csi_all39.log
"""

import sys, math, random, time
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / '02_foundation_pretraining'))
sys.path.insert(0, str(_REPO_ROOT / '03_finetuning'))

import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from CAI import relative_adaptiveness

from train_mlm_industrial import (
    IndustrialMLM, AA_CODON_VOCAB, ID_TO_PAIR, ORG_TO_ID, ORG_ORDER,
    AA_MASK, CLS_TOKEN, PAD_TOKEN, _GENETIC_CODE,
)

BASE     = Path(__file__).resolve().parent.parent
TSV_PATH = BASE / 'data/pretrain/all_industrial_cds.tsv'
CKPT_IN  = BASE / 'models/industrial_mlm_ep11.pt'
OUT      = BASE / 'models/industrial_mlm_csi_all39.pt'
LOG      = BASE / 'logs/finetune_csi_all39.log'

TOP_PCT      = 90
MASK_PROB    = 0.15
BATCH_SIZE   = 512
LR           = 1e-5
EPOCHS       = 10
WARMUP_STEPS = 500
LABEL_SMOOTH = 0.05
SEED         = 42
MIN_CODONS   = 20
MAX_CODONS   = 600
N_WORKERS    = min(48, cpu_count())
USE_BF16     = True   # H100 native bf16 — 2× faster, same quality

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}  |  CPU workers: {N_WORKERS}')

STOP_C     = {'TAA','TAG','TGA'}
PAIR_TO_ID = {(aa,cdn): tid for (aa,cdn),tid in AA_CODON_VOCAB.items()}

# ── Codon utilities (module-level so picklable for multiprocessing) ─────────────
def codon_split(dna):
    dna = str(dna).upper()
    if len(dna) >= 3 and dna[-3:] in STOP_C: dna = dna[:-3]
    return [dna[i:i+3] for i in range(0, len(dna)-2, 3)
            if dna[i:i+3] not in STOP_C and 'N' not in dna[i:i+3]]

def dna_to_tokens(dna):
    dna = str(dna).upper()
    if len(dna) >= 3 and dna[-3:] in STOP_C: dna = dna[:-3]
    tokens = []
    for i in range(0, len(dna)-2, 3):
        c = dna[i:i+3]
        if c in STOP_C or 'N' in c: break
        aa = _GENETIC_CODE.get(c, '')
        if not aa or aa == '*': break
        tid = PAIR_TO_ID.get((aa, c))
        if tid is None: break
        tokens.append(tid)
    return tokens

def process_organism(args):
    """
    Per-organism worker: build CSI weights, fast-compute CSI, select top-10%, tokenize.
    Runs in a separate process — must be picklable (no CUDA, no module-level state).
    """
    org, dna_list, org_id, top_pct, min_codons, max_codons, seed = args
    rng = random.Random(seed)

    # Build genomic CSI weights from ALL sequences (same as relative_adaptiveness)
    aa_cod = defaultdict(lambda: defaultdict(int))
    for dna in dna_list:
        for cod in codon_split(dna):
            aa = _GENETIC_CODE.get(cod)
            if aa and aa not in ('*', 'X'):
                aa_cod[aa][cod] += 1

    # relative_adaptiveness: weight = count / max_count per AA group
    raw_weights = {}
    for aa, counts in aa_cod.items():
        mx = max(counts.values())
        for cod, cnt in counts.items():
            raw_weights[cod] = cnt / mx

    # Pre-compute log weights for fast geometric mean
    log_w = {c: np.log(max(w, 1e-10)) for c, w in raw_weights.items()}

    # Fast CSI: geometric mean of per-codon weights (numpy, no CAI library)
    csi_vals = np.empty(len(dna_list), dtype=np.float32)
    for i, dna in enumerate(dna_list):
        codons = codon_split(dna)
        if not codons:
            csi_vals[i] = np.nan
            continue
        log_sum = sum(log_w.get(c, -23.0) for c in codons)
        csi_vals[i] = np.exp(log_sum / len(codons))

    valid_mask = ~np.isnan(csi_vals)
    if valid_mask.sum() == 0:
        return org, len(dna_list), 0, 0, float('nan'), []

    thresh = np.percentile(csi_vals[valid_mask], top_pct)
    top_mask = valid_mask & (csi_vals >= thresh)

    samples = []
    for i, dna in enumerate(dna_list):
        if not top_mask[i]: continue
        toks = dna_to_tokens(dna)
        if min_codons <= len(toks) <= max_codons:
            samples.append((toks, org_id))

    return org, len(dna_list), int(top_mask.sum()), len(samples), float(thresh), samples

# ── Load TSV once ──────────────────────────────────────────────────────────────
print(f'\nLoading {TSV_PATH.name} ...')
t0 = time.time()
df = pd.read_csv(TSV_PATH, sep='\t', usecols=['dna', 'organism'])
df = df.dropna(subset=['dna']).reset_index(drop=True)
print(f'  {len(df):,} sequences loaded in {time.time()-t0:.1f}s')

# Group by organism — fast in-memory split
org_groups = {org: grp['dna'].tolist() for org, grp in df.groupby('organism')}
del df  # free 3GB RAM immediately
print(f'  {len(org_groups)} organisms found')

# ── Parallel CSI computation across all 39 organisms ──────────────────────────
args_list = [
    (org, org_groups[org], ORG_TO_ID[org], TOP_PCT, MIN_CODONS, MAX_CODONS, SEED)
    for org in ORG_ORDER if org in org_groups and org in ORG_TO_ID
]

print(f'\nComputing CSI in parallel ({N_WORKERS} workers, {len(args_list)} organisms)...')
t1 = time.time()

log_lines = []
all_samples = []
org_stats   = {}

with Pool(N_WORKERS) as pool:
    for result in pool.imap_unordered(process_organism, args_list):
        org, total, top10, valid, thresh, samples = result
        all_samples.extend(samples)
        org_stats[org] = {'total': total, 'top10': top10, 'valid': valid, 'thresh': thresh}
        msg = (f'  {org:30s}  total={total:6,}  top10%={top10:5,}  '
               f'valid_toks={valid:5,}  CSI_thresh={thresh:.4f}')
        print(msg); log_lines.append(msg)

print(f'\nCSI computation done in {time.time()-t1:.1f}s')
total_msg = f'Total training sequences: {len(all_samples):,} from {len(org_stats)} organisms'
print(total_msg); log_lines.append(total_msg)

# ── Dataset ────────────────────────────────────────────────────────────────────
class CodonDataset(Dataset):
    def __init__(self, s): self.s = s
    def __len__(self): return len(self.s)
    def __getitem__(self, i): return self.s[i]

def collate_fn(batch):
    max_len = max(len(t) for t,_ in batch) + 1
    I, L, A, O = [], [], [], []
    for toks, oid in batch:
        seq = [CLS_TOKEN] + list(toks); m = []; l = []
        for i, t in enumerate(seq):
            if i == 0: m.append(t); l.append(-100); continue
            if random.random() < MASK_PROB:
                m.append(AA_MASK[ID_TO_PAIR[t][0]]); l.append(t)
            else: m.append(t); l.append(-100)
        pad = max_len - len(seq)
        I.append(m + [PAD_TOKEN]*pad); L.append(l + [-100]*pad)
        A.append([False]*len(seq) + [True]*pad); O.append(oid)
    return {'input_ids': torch.tensor(I, dtype=torch.long),
            'labels':    torch.tensor(L, dtype=torch.long),
            'attn_mask': torch.tensor(A, dtype=torch.bool),
            'org_ids':   torch.tensor(O, dtype=torch.long)}

# BucketBatchSampler — groups sequences of similar length to minimise padding waste
class BucketBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, bucket_width=30):
        self.lengths = lengths; self.batch_size = batch_size; self.bucket_width = bucket_width
    def __iter__(self):
        noise = [random.randint(0, self.bucket_width) for _ in self.lengths]
        keys  = [l + n for l, n in zip(self.lengths, noise)]
        idx   = sorted(range(len(self.lengths)), key=lambda i: keys[i])
        batches = [idx[i:i+self.batch_size] for i in range(0, len(idx), self.batch_size)]
        random.shuffle(batches)
        for b in batches: yield b
    def __len__(self): return math.ceil(len(self.lengths) / self.batch_size)

lengths  = [len(t) for t, _ in all_samples]
sampler  = BucketBatchSampler(lengths, BATCH_SIZE)
loader   = DataLoader(CodonDataset(all_samples), batch_sampler=sampler,
                      collate_fn=collate_fn, num_workers=8, pin_memory=True)
loader_msg = f'DataLoader: {len(loader)} batches/epoch  ({BATCH_SIZE} seqs/batch, bucketed)'
print(loader_msg); log_lines.append(loader_msg)

# ── Model ──────────────────────────────────────────────────────────────────────
print(f'\nLoading {CKPT_IN.name} ...')
ckpt  = torch.load(CKPT_IN, map_location='cpu', weights_only=False)
model = IndustrialMLM(use_grad_ckpt=False)
model.load_state_dict(ckpt['model'])
model.to(DEVICE)
torch.compile(model, mode='reduce-overhead')   # H100 kernel fusion — ~1.5× faster

total_params = sum(p.numel() for p in model.parameters())
model_msg = f'Trainable: {total_params:,} (100%) — full fine-tune, all 39 organisms, bf16+compile'
print(model_msg); log_lines.append(model_msg)

optimizer   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scaler      = torch.amp.GradScaler('cuda', enabled=USE_BF16)
total_steps = EPOCHS * len(loader)

def lr_lambda(step):
    if step < WARMUP_STEPS: return step / max(WARMUP_STEPS, 1)
    prog = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
    return max(0.05, 0.5 * (1 + math.cos(math.pi * prog)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
loss_fn   = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=LABEL_SMOOTH)

# ── Train ──────────────────────────────────────────────────────────────────────
print(f'\nTraining {EPOCHS} epochs ...')
log_lines.append(f'Training: {EPOCHS} epochs | LR={LR} | batch={BATCH_SIZE}')
best_loss = float('inf'); model.train()

for epoch in range(1, EPOCHS+1):
    t_ep = time.time()
    total_loss, n_tok = 0.0, 0
    for batch in loader:
        ids  = batch['input_ids'].to(DEVICE); lbls = batch['labels'].to(DEVICE)
        attn = batch['attn_mask'].to(DEVICE); oids = batch['org_ids'].to(DEVICE)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=USE_BF16):
            logits = model(ids, attn, oids); B, L, V = logits.shape
            loss   = loss_fn(logits.reshape(B*L, V), lbls.reshape(B*L))
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        mask = (lbls != -100)
        total_loss += loss.item() * mask.sum().item(); n_tok += mask.sum().item()

    avg = total_loss / max(n_tok, 1); ppl = math.exp(avg) if avg < 20 else float('inf')
    msg = (f'Ep {epoch:3d}/{EPOCHS}  loss={avg:.4f}  ppl={ppl:.3f}  '
           f'lr={scheduler.get_last_lr()[0]:.2e}  ({time.time()-t_ep:.0f}s)')
    print(msg); log_lines.append(msg)

    if avg < best_loss:
        best_loss = avg
        torch.save({
            'model': model.state_dict(), 'epoch': epoch,
            'csi_all39': True, 'n_orgs': len(org_stats),
            'n_seqs': len(all_samples), 'ref': 'genomic_all_genes',
        }, OUT)
        saved_msg = f'  ✓ saved'; print(saved_msg); log_lines.append(saved_msg)

LOG.parent.mkdir(exist_ok=True)
with open(LOG, 'w') as f: f.write('\n'.join(log_lines))
print(f'\nDone → {OUT}')
