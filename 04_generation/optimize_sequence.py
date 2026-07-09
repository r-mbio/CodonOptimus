#!/usr/bin/env python3
"""
CodonOptimus: optimize a codon sequence and evaluate with dual-head predictions.

Given an amino acid sequence and target organism:
  1. Generates optimized codon sequence (foundation model — greedy argmax)
  2. Predicts per-codon A-site occupancy + gene-level expression (dual-head)
  3. Computes codon optimization metrics: CSI, CFD%, %MinMax, GC%
  4. Optionally compares against native DNA sequence

Usage:
  python3 optimize_sequence.py --aa_seq MKVLF... --org ecoli
  python3 optimize_sequence.py --aa_seq MKVLF... --org pichia --native_dna ATGAAA...
  python3 optimize_sequence.py --fasta protein.faa --org s_cerevisiae
  python3 optimize_sequence.py --fasta protein.faa --org ecoli --save_dna out.fna
"""

import sys, argparse, json, math, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / '02_foundation_pretraining'))
from train_mlm_industrial import (
    IndustrialMLM, AA_CODON_VOCAB, ID_TO_PAIR, ORG_TO_ID,
    CLS_TOKEN, PAD_TOKEN, _GENETIC_CODE, AA_MASK,
    STOP_CODONS, AMINO_ACIDS,
)

try:
    from CAI import CAI, relative_adaptiveness
    HAS_CAI = True
except ImportError:
    HAS_CAI = False

BASE           = Path(__file__).resolve().parent.parent
TABLE_DIR       = BASE / 'data' / 'codon_tables'

# Per-organism generator model: RS-FT where ribosome-profiling fine-tuning
# exists for that organism, else the all-39-organism CSI-FT model (the
# correct general-purpose default — NOT the foundation-only checkpoint).
RS_FT_GENERATOR = {
    'ecoli':    BASE / 'models' / 'industrial_mlm_all39_rs_ecoli.pt',
    'bacillus': BASE / 'models' / 'industrial_mlm_all39_rs_bacillus.pt',
    'pichia':   BASE / 'models' / 'industrial_mlm_all39_rs_pichia.pt',
}
CSI_FT_GENERATOR = BASE / 'models' / 'industrial_mlm_csi_all39.pt'

# Per-organism dual-head specialist (expression + A-site prediction).
# Only these four organisms have ribosome-profiling-supervised dual heads;
# all others have no dual-head model — ExprScore/A-site predictions are
# skipped for them rather than silently using a mismatched model.
DUAL_HEAD_SPECIALIST = {
    'ecoli':        BASE / 'models' / 'dual_head_ep11_ecoli.pt',
    'bacillus':     BASE / 'models' / 'dual_head_pretrain_bacillus.pt',
    'pichia':       BASE / 'models' / 'dual_head_ep11_pichia.pt',
    's_cerevisiae': BASE / 'models' / 'dual_head_pretrain_s_cerevisiae.pt',
}


def default_generator_ckpt(org):
    """RS-FT if this organism has ribosome-profiling supervision, else CSI-FT."""
    return RS_FT_GENERATOR.get(org, CSI_FT_GENERATOR)


def default_dual_head_ckpt(org):
    """Per-organism dual-head specialist, or None if unavailable for this organism."""
    return DUAL_HEAD_SPECIALIST.get(org)

ORG_LABELS = {
    'ecoli':             'E. coli',
    'bacillus':          'B. subtilis',
    's_cerevisiae':      'S. cerevisiae',
    'pichia':            'K. phaffii',
    'yarrowia':          'Y. lipolytica',
    'human':             'Human',
    'aspergillus_niger': 'A. niger',
    'trichoderma':       'T. reesei',
    'cho':               'CHO',
}

ORG_TABLES = {
    'ecoli':             'ecoli_codon_usage.json',
    'bacillus':          'bacillus_codon_usage.json',
    's_cerevisiae':      's_cerevisiae_codon_usage.json',
    'pichia':            'pichia_codon_usage.json',
    'yarrowia':          'yarrowia_codon_usage.json',
    'human':             'human_codon_usage.json',
    'aspergillus_niger': 'aspergillus_codon_usage.json',
    'trichoderma':       'trichoderma_codon_usage.json',
}


# ═══════════════════════════════════════════════════════════════════════════════
# Dual-head architecture (must match finetune_dual_head.py exactly)
# ═══════════════════════════════════════════════════════════════════════════════

class AsiteHead(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, 256, kernel_size=9, padding=4)
        self.norm1 = nn.LayerNorm(256)
        self.conv2 = nn.Conv1d(256, 64,  kernel_size=5, padding=2)
        self.norm2 = nn.LayerNorm(64)
        self.conv3 = nn.Conv1d(64,  32,  kernel_size=3, padding=1)
        self.norm3 = nn.LayerNorm(32)
        self.proj  = nn.Linear(32, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.gelu(self.norm1(self.conv1(x).transpose(1,2)).transpose(1,2))
        x = F.gelu(self.norm2(self.conv2(x).transpose(1,2)).transpose(1,2))
        x = F.gelu(self.norm3(self.conv3(x).transpose(1,2)).transpose(1,2))
        x = x.transpose(1, 2)
        return self.proj(x).squeeze(-1)   # (B, L)


class ExpressionHead(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 64),      nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, attn_mask):
        valid  = (~attn_mask).unsqueeze(-1).float()
        pooled = (x * valid).sum(1) / valid.sum(1).clamp(min=1)
        return self.net(pooled).squeeze(-1)   # (B,)


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
        rope_sin = enc.rope.sin[:L].unsqueeze(0).unsqueeze(0)
        rope_cos = enc.rope.cos[:L].unsqueeze(0).unsqueeze(0)
        for layer in enc.layers:
            x = layer(x, rope_sin, rope_cos, attn_mask)
        return enc.norm(x)

    def forward(self, input_ids, attn_mask, org_ids):
        hidden     = self._get_hidden(input_ids, attn_mask, org_ids)
        codon_repr = hidden[:, 1:, :]
        codon_mask = attn_mask[:, 1:]
        return self.asite_head(codon_repr), self.expr_head(codon_repr, codon_mask)


# ═══════════════════════════════════════════════════════════════════════════════
# Sequence encoding
# ═══════════════════════════════════════════════════════════════════════════════

def encode_dna(dna, max_len=512):
    """DNA string → (token_list, codon_list). Returns (None, None) if too short."""
    tokens = [CLS_TOKEN]
    codons = []
    for i in range(0, len(dna) - 2, 3):
        cdn = dna[i:i+3].upper()
        aa  = _GENETIC_CODE.get(cdn, '?')
        if aa in ('*', '?'):
            break
        tok = AA_CODON_VOCAB.get((aa, cdn), PAD_TOKEN)
        tokens.append(tok)
        codons.append(cdn)
        if len(codons) >= max_len - 1:
            break
    if len(codons) < 5:
        return None, None
    return tokens, codons


def encode_aa_as_masks(aa_seq):
    """AA string → masked token list for foundation model generation."""
    valid = [aa for aa in aa_seq.upper() if aa in AA_MASK]
    return [CLS_TOKEN] + [AA_MASK[aa] for aa in valid], valid


def to_tensor(tokens, org_id, device):
    ids      = torch.tensor([tokens], dtype=torch.long, device=device)
    attn     = torch.zeros(1, len(tokens), dtype=torch.bool, device=device)
    org_ids  = torch.tensor([org_id], dtype=torch.long, device=device)
    return ids, attn, org_ids


# ═══════════════════════════════════════════════════════════════════════════════
# Generation
# ═══════════════════════════════════════════════════════════════════════════════

# Per-AA → list of valid codon token IDs (built once at import)
_AA_TO_TOKEN_IDS = {}
for (_aa, _cdn), _tid in AA_CODON_VOCAB.items():
    _AA_TO_TOKEN_IDS.setdefault(_aa, []).append(_tid)


def _apply_synonymous_mask(logits_1L83, valid_aas):
    """Zero out non-synonymous codon logits per position (in-place, returns logits)."""
    L83 = logits_1L83[0]   # (L, 83)
    for pos, aa in enumerate(valid_aas):
        allowed = _AA_TO_TOKEN_IDS.get(aa, [])
        if allowed:
            mask = torch.full((83,), float('-inf'), device=L83.device)
            mask[allowed] = 0.0
            L83[pos + 1] = L83[pos + 1] + mask   # +1 for CLS
    return logits_1L83


def _sample_top_p(probs_1d, top_p):
    """Nucleus (top-p) sampling from a 1D probability vector."""
    sorted_p, sorted_idx = torch.sort(probs_1d, descending=True)
    cumsum = torch.cumsum(sorted_p, dim=0)
    sorted_p[cumsum - sorted_p > top_p] = 0.0
    sorted_p /= sorted_p.sum()
    chosen = torch.multinomial(sorted_p, 1)
    return sorted_idx[chosen].item()


def generate_codons(foundation_model, aa_seq, org, device,
                    temperature=0.0, top_p=0.95):
    """
    Predict codons for aa_seq in org.
    temperature=0 → greedy argmax (deterministic).
    temperature>0 → stochastic sampling with synonymous masking and nucleus filtering.
    Returns (dna_string, codon_list, valid_aas).
    """
    tokens, valid_aas = encode_aa_as_masks(aa_seq)
    ids, attn, org_ids = to_tensor(tokens, ORG_TO_ID[org], device)
    with torch.no_grad():
        logits = foundation_model(ids, attn, org_ids)   # (1, L, 83)

    # Always enforce synonymous constraint
    logits = _apply_synonymous_mask(logits, valid_aas)

    if temperature == 0.0:
        best_ids = logits[0, 1:, :61].argmax(dim=-1)
        codons   = [ID_TO_PAIR[t.item()][1] for t in best_ids]
    else:
        codons = []
        for pos in range(len(valid_aas)):
            logit_row = logits[0, pos + 1, :61] / temperature
            probs     = torch.softmax(logit_row, dim=0)
            tok       = _sample_top_p(probs, top_p)
            codons.append(ID_TO_PAIR[tok][1])

    dna = ''.join(codons)
    return dna, codons, valid_aas


# ═══════════════════════════════════════════════════════════════════════════════
# Dual-head predictions
# ═══════════════════════════════════════════════════════════════════════════════

def predict_dual(dual_model, tokens, org_id, device):
    """Run dual-head on a token list. Returns (asite_profile np.array, expr_score float).
    Returns (empty array, nan) if no dual-head model is available for this organism."""
    if dual_model is None:
        return np.array([]), float('nan')
    ids, attn, org_ids = to_tensor(tokens, org_id, device)
    with torch.no_grad():
        asite, expr = dual_model(ids, attn, org_ids)
    return asite[0].cpu().numpy(), expr[0].item()


# ═══════════════════════════════════════════════════════════════════════════════
# Codon optimization metrics (ported from evaluate_codonoptimus.py)
# ═══════════════════════════════════════════════════════════════════════════════

def load_codon_table(org):
    fname = ORG_TABLES.get(org)
    if not fname:
        return None
    path = TABLE_DIR / fname
    return json.load(open(path)) if path.exists() else None


def _table_to_cai_weights(codon_table):
    codons_pool = []
    for aa, codons in codon_table.items():
        for codon, freq in codons.items():
            count = max(1, round(freq * 1000))
            codons_pool.extend([codon] * count)
    random.seed(42)
    random.shuffle(codons_pool)
    ref_seq = ''.join(codons_pool) + 'TAA'
    return relative_adaptiveness(sequences=[ref_seq])


def _table_to_codon_frequencies(codon_table):
    return {aa: (list(d.keys()), list(d.values())) for aa, d in codon_table.items()}


def calc_csi(codon_list, codon_table):
    if not HAS_CAI or not codon_list:
        return float('nan')
    dna     = ''.join(codon_list).upper()
    weights = _table_to_cai_weights(codon_table)
    try:
        return CAI(dna, weights)
    except Exception:
        return float('nan')


def get_cfd(codon_list, codon_table, threshold=0.3):
    codon_freq = _table_to_codon_frequencies(codon_table)
    codon2freq = {
        codon: freq / max(freqs)
        for aa, (codons, freqs) in codon_freq.items()
        for codon, freq in zip(codons, freqs)
        if max(freqs) > 0
    }
    rare = sum(1 for c in codon_list if codon2freq.get(c.upper(), 1.0) < threshold)
    return 100.0 * rare / len(codon_list) if codon_list else 0.0


def get_minmax_mean(codon_list, codon_table, window_size=18):
    codon_freq  = _table_to_codon_frequencies(codon_table)
    codon2amino = {c: aa for aa, (codons, _) in codon_freq.items() for c in codons}
    codons = [c.upper() for c in codon_list]
    values = []
    for i in range(len(codons) - window_size + 1):
        window = codons[i:i + window_size]
        Actual = Max = Min = Avg = 0.0
        for codon in window:
            aa = codon2amino.get(codon)
            if aa is None:
                continue
            freqs = codon_freq[aa][1]
            idx   = codon_freq[aa][0].index(codon) if codon in codon_freq[aa][0] else 0
            f     = codon_freq[aa][1][idx]
            Actual += f; Max += max(freqs); Min += min(freqs)
            Avg    += sum(freqs) / len(freqs)
        Actual /= window_size; Max /= window_size
        Min    /= window_size; Avg /= window_size
        if Max - Avg > 0:
            pct = ((Actual - Avg) / (Max - Avg)) * 100
        elif Avg - Min > 0:
            pct = -((Avg - Actual) / (Avg - Min)) * 100
        else:
            pct = 0.0
        values.append(pct)
    return float(np.mean(values)) if values else float('nan')


def calc_gc(codon_list):
    seq = ''.join(codon_list).upper()
    if not seq:
        return float('nan')
    return 100.0 * sum(1 for b in seq if b in 'GC') / len(seq)


def verify_synonymous(aa_seq, codon_list):
    errors = 0
    for aa, codon in zip(aa_seq.upper(), codon_list):
        if aa not in AMINO_ACIDS:
            continue
        pred_aa = _GENETIC_CODE.get(codon.upper(), '?')
        if pred_aa != aa:
            errors += 1
    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# FASTA parsing
# ═══════════════════════════════════════════════════════════════════════════════

def read_fasta(path):
    """Returns list of (header, sequence) tuples."""
    records = []
    header, seq = None, []
    for line in open(path):
        line = line.rstrip()
        if line.startswith('>'):
            if header is not None:
                records.append((header, ''.join(seq)))
            header = line[1:]
            seq    = []
        else:
            seq.append(line)
    if header is not None:
        records.append((header, ''.join(seq)))
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument('--aa_seq',     help='Amino acid sequence string')
    grp.add_argument('--fasta',      help='FASTA file with amino acid sequence(s)')

    p.add_argument('--org',          required=True,
                   help=f'Target organism. Available: {", ".join(ORG_TO_ID.keys())}')
    p.add_argument('--native_dna',   default=None,
                   help='Native CDS DNA for comparison (optional)')
    p.add_argument('--foundation_ckpt', default=None,
                   help='Override the generator checkpoint. Default: RS-FT for '
                        'organisms with ribosome-profiling supervision (E. coli, '
                        'B. subtilis, K. phaffii), else the all-39-organism CSI-FT '
                        'model. Pass industrial_mlm_pretrain.pt explicitly for the '
                        'foundation-only model.')
    p.add_argument('--dual_head_ckpt',  default=None,
                   help='Override the dual-head checkpoint. Default: the '
                        'per-organism specialist for E. coli, B. subtilis, '
                        'K. phaffii, or S. cerevisiae; unavailable (skipped) for '
                        'any other organism.')
    p.add_argument('--max_len',      type=int,   default=512)
    p.add_argument('--n_seqs',       type=int,   default=1,
                   help='Generate N sequences and keep the one with highest CSI '
                        '(requires --temperature > 0 for diversity)')
    p.add_argument('--temperature',  type=float, default=0.0,
                   help='Sampling temperature (0 = greedy argmax, 0.2-0.5 = stochastic)')
    p.add_argument('--top_p',        type=float, default=0.95,
                   help='Nucleus sampling threshold (only used when temperature > 0)')
    p.add_argument('--save_dna',     default=None,
                   help='Save optimized DNA to FASTA file')
    p.add_argument('--save_report',  default=None,
                   help='Save metrics and candidate rankings to a CSV file')
    p.add_argument('--device',       default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def evaluate_sequence(label, codon_list, tokens, org_id, codon_table,
                       dual_model, device):
    """Compute all metrics for one sequence. Returns dict."""
    res = {'label': label, 'n_codons': len(codon_list)}

    # Codon optimization metrics
    if codon_table:
        res['csi']    = calc_csi(codon_list, codon_table)
        res['cfd']    = get_cfd(codon_list, codon_table)
        res['minmax'] = get_minmax_mean(codon_list, codon_table)
    else:
        res['csi'] = res['cfd'] = res['minmax'] = float('nan')
    res['gc'] = calc_gc(codon_list)

    # Dual-head predictions
    asite_profile, expr_score = predict_dual(dual_model, tokens, org_id, device)
    n = min(len(asite_profile), len(codon_list))
    res['expr_score']      = expr_score
    res['asite_mean']      = float(np.mean(asite_profile[:n])) if n > 0 else float('nan')
    res['asite_std']       = float(np.std(asite_profile[:n])) if n > 0 else float('nan')
    res['asite_profile']   = asite_profile[:n]

    return res


def print_report(results, org, native_res=None):
    label = ORG_LABELS.get(org, org)
    w = 72
    print()
    print('=' * w)
    print(f"  CodonOptimus — Sequence Optimization Report")
    print(f"  Organism: {label}")
    print('=' * w)

    if not HAS_CAI:
        print("  [Note] CAI library not found — CSI and CFD scores unavailable")

    def fmt(v, decimals=3):
        if math.isnan(v):
            return 'N/A'
        if abs(v) < 0.001 and v != 0.0:
            return f"{v:.2e}"
        return f"{v:.{decimals}f}"

    rows = [results] if native_res is None else [native_res, results]

    # Header
    print(f"\n  {'Metric':<28}  ", end='')
    for r in rows:
        print(f"  {r['label']:>14}", end='')
    if len(rows) == 2:
        print(f"  {'Change':>10}", end='')
    print()
    print(f"  {'-'*(28 + 18*len(rows) + (12 if len(rows)==2 else 0))}")

    def row(name, key, fmt_fn, better='↑'):
        vals = [r[key] for r in rows]
        print(f"  {name:<28}  ", end='')
        for v in vals:
            print(f"  {fmt_fn(v):>14}", end='')
        if len(rows) == 2 and better != '' and not math.isnan(vals[0]) and not math.isnan(vals[1]):
            delta = vals[1] - vals[0]
            indicator = better if delta > 0 else ('↓' if better == '↑' else '↑')
            print(f"  {delta:+.3f} {indicator:>2}", end='')
        print()

    row('Sequence length (codons)', 'n_codons',    lambda v: f"{int(v)}", better='')
    row('GC content (%)',           'gc',           lambda v: fmt(v,1))
    row('CSI (Codon Sim. Index)',   'csi',          lambda v: fmt(v,3), better='↑')
    row('CFD% (rare codons)',       'cfd',          lambda v: fmt(v,1), better='↓')
    row('%MinMax (mean)',           'minmax',       lambda v: fmt(v,1), better='↑')
    print()
    row('Predicted expression',     'expr_score',   lambda v: fmt(v,4), better='↑')
    row('A-site occ. mean (z)',     'asite_mean',   lambda v: fmt(v,4))
    row('A-site occ. std (z)',      'asite_std',    lambda v: fmt(v,4), better='↓')

    print()
    print(f"  Notes:")
    print(f"    CSI:         Codon Similarity Index — higher = better match to organism codon usage")
    print(f"    CFD%:        % codons below 30% frequency — lower = fewer rare codons")
    print(f"    %MinMax:     Sliding-window codon usage score — higher = more optimal")
    print(f"    Expression:  Model-predicted translation output [0–1 org-relative scale]")
    print(f"    A-site std:  Profile uniformity — lower = smoother ribosome movement")
    print()


def main():
    args = get_args()
    device = torch.device(args.device)

    if args.org not in ORG_TO_ID:
        print(f"ERROR: '{args.org}' not in ORG_TO_ID. Available organisms:")
        for o in sorted(ORG_TO_ID.keys()):
            print(f"  {o}")
        sys.exit(1)

    # ── Load AA sequence(s) ────────────────────────────────────────────────────
    if args.aa_seq:
        proteins = [('input', args.aa_seq.strip().upper())]
    else:
        records  = read_fasta(args.fasta)
        proteins = [(hdr, seq.upper()) for hdr, seq in records]
        print(f"Loaded {len(proteins)} sequence(s) from {args.fasta}")

    # ── Resolve generator/dual-head checkpoints for this organism ──────────────
    generator_ckpt = Path(args.foundation_ckpt) if args.foundation_ckpt else default_generator_ckpt(args.org)
    dual_head_path = Path(args.dual_head_ckpt) if args.dual_head_ckpt else default_dual_head_ckpt(args.org)

    # ── Load generator model ───────────────────────────────────────────────────
    print(f"\nLoading generator model: {generator_ckpt}")
    f_ckpt      = torch.load(generator_ckpt, map_location='cpu', weights_only=False)
    foundation  = IndustrialMLM(use_grad_ckpt=False)
    sd = f_ckpt.get('model', f_ckpt)
    foundation.load_state_dict(sd)
    foundation.to(device).eval()
    ep = f_ckpt.get('epoch', 0) + 1
    m  = f_ckpt.get('val_metrics', {})
    print(f"  Epoch {ep}  val_ppl={m.get('ppl',0):.2f}  top1={m.get('top1',0):.1f}%  "
          f"syn={m.get('syn_acc',0):.1f}%  params={sum(p.numel() for p in foundation.parameters())/1e6:.1f}M")

    # ── Load dual-head model (if available for this organism) ─────────────────
    dual = None
    if dual_head_path is not None and Path(dual_head_path).exists():
        print(f"Loading dual-head model:  {dual_head_path}")
        dh_ckpt   = torch.load(dual_head_path, map_location='cpu', weights_only=False)
        encoder2  = IndustrialMLM(use_grad_ckpt=False)
        dual      = DualHeadModel(encoder2)
        dual.load_state_dict(dh_ckpt.get('model', dh_ckpt), strict=False)
        dual.to(device).eval()
        print(f"  val_r2={dh_ckpt.get('val_r2', float('nan')):.4f}  "
              f"val_sp={dh_ckpt.get('val_sp', float('nan')):.4f}")
    else:
        print(f"No dual-head model available for '{args.org}' — "
              f"ExprScore/A-site predictions will be skipped (NaN).")

    codon_table = load_codon_table(args.org)
    if codon_table is None:
        print(f"  [Warning] No codon table found for '{args.org}' — CSI/CFD/MinMax unavailable")

    org_id = ORG_TO_ID[args.org]

    # ── Optional native DNA ────────────────────────────────────────────────────
    native_res = None
    if args.native_dna:
        tokens_n, codons_n = encode_dna(args.native_dna, args.max_len)
        if tokens_n is None:
            print("WARNING: native DNA could not be encoded (too short or bad codons)")
        else:
            print(f"\nEncoded native DNA: {len(codons_n)} codons")
            native_res = evaluate_sequence(
                'Native', codons_n, tokens_n, org_id, codon_table, dual, device)

    # ── Process each protein ───────────────────────────────────────────────────
    save_seqs   = []
    report_rows = []   # for --save_report
    for prot_name, aa_seq in proteins:
        print(f"\n{'─'*60}")
        print(f"Protein: {prot_name}  ({len(aa_seq)} AA)")

        # Generate optimized sequence(s)
        candidates = []
        n_gen = args.n_seqs if args.temperature > 0 else 1
        for i in range(n_gen):
            dna_c, codons_c, vaas_c = generate_codons(
                foundation, aa_seq, args.org, device,
                temperature=args.temperature, top_p=args.top_p)
            tokens_c, codons_c2 = encode_dna(dna_c, args.max_len)
            if tokens_c is None:
                continue
            csi_c  = calc_csi(codons_c2, codon_table) if codon_table else float('nan')
            cfd_c  = get_cfd(codons_c2, codon_table) if codon_table else float('nan')
            _, expr_c = predict_dual(dual, tokens_c, org_id, device)
            candidates.append((dna_c, codons_c, vaas_c, tokens_c, codons_c2, csi_c, cfd_c, expr_c))

        if not candidates:
            print("  ERROR: failed to generate valid sequence — skipping")
            continue

        # Rank by ExprScore if dual-head available, else by CSI
        has_expr = not math.isnan(candidates[0][7])
        rank_key = (lambda x: x[7]) if has_expr else (lambda x: x[5] if not math.isnan(x[5]) else -1)
        candidates.sort(key=rank_key, reverse=True)

        if n_gen > 1:
            rank_by = 'ExprScore' if has_expr else 'CSI'
            print(f"\n  Ranking {len(candidates)} candidates by {rank_by}:")
            print(f"  {'Rank':>4}  {'ExprScore':>10}  {'CSI':>6}  {'CFD%':>5}")
            print(f"  {'-'*36}")
            for rank, cand in enumerate(candidates, 1):
                marker = ' ◀ selected' if rank == 1 else ''
                expr_s = f"{cand[7]:10.4f}" if not math.isnan(cand[7]) else f"{'N/A':>10}"
                csi_s  = f"{cand[5]:6.3f}"  if not math.isnan(cand[5]) else f"{'N/A':>6}"
                cfd_s  = f"{cand[6]:5.1f}"  if not math.isnan(cand[6]) else f"{'N/A':>5}"
                print(f"  {rank:>4}  {expr_s}  {csi_s}  {cfd_s}{marker}")
            print()

        opt_dna, opt_codons, valid_aas, tokens_o, codons_o, *_ = candidates[0]

        # Verify synonymous accuracy (should always be 100% — mask enforced during generation)
        n_err = verify_synonymous(''.join(valid_aas), opt_codons)
        if n_err:
            raise RuntimeError(
                f"AA-level alignment FAILED: {n_err} non-synonymous codon(s) in "
                f"output. This should never happen — please file a bug report."
            )
        print(f"  Synonymous accuracy: 100% ({len(opt_codons)} codons)")

        opt_res = evaluate_sequence(
            'Optimized', codons_o, tokens_o, org_id, codon_table, dual, device)

        print_report(opt_res, args.org, native_res)

        # Show optimized DNA
        print(f"  Optimized DNA ({len(opt_dna)} bp):")
        for i in range(0, len(opt_dna), 60):
            print(f"    {opt_dna[i:i+60]}")
        print()

        save_seqs.append((prot_name, opt_dna))

        # Collect rows for report CSV — one row per candidate, all metrics
        if args.save_report:
            for rank, cand in enumerate(candidates, 1):
                dna_r, codons_r, _, tokens_r, codons_r2, csi_r, cfd_r, expr_r = cand
                gc_r     = calc_gc(codons_r2)
                mm_r     = get_minmax_mean(codons_r2, codon_table) if codon_table else float('nan')
                asite_r, _ = predict_dual(dual, tokens_r, org_id, device)
                n_r      = min(len(asite_r), len(codons_r2))
                report_rows.append({
                    'protein':       prot_name,
                    'organism':      args.org,
                    'rank':          rank,
                    'recommended':   'YES' if rank == 1 else 'no',
                    'expr_score':    round(expr_r, 4)   if not math.isnan(expr_r)  else '',
                    'csi':           round(csi_r,  4)   if not math.isnan(csi_r)   else '',
                    'cfd_pct':       round(cfd_r,  2)   if not math.isnan(cfd_r)   else '',
                    'gc_pct':        round(gc_r,   2)   if not math.isnan(gc_r)    else '',
                    'minmax':        round(mm_r,   2)   if not math.isnan(mm_r)    else '',
                    'asite_mean':    round(float(np.mean(asite_r[:n_r])), 4) if n_r > 0 else '',
                    'asite_std':     round(float(np.std( asite_r[:n_r])), 4) if n_r > 0 else '',
                    'n_codons':      len(codons_r2),
                    'dna':           dna_r,
                })

    # ── Save optimized DNA ─────────────────────────────────────────────────────
    if args.save_dna:
        org_label = ORG_LABELS.get(args.org, args.org)
        with open(args.save_dna, 'w') as f:
            for name, dna in save_seqs:
                f.write(f">{name} | CodonOptimus optimized | {org_label}\n")
                for i in range(0, len(dna), 60):
                    f.write(dna[i:i+60] + '\n')
        print(f"Saved optimized DNA    → {args.save_dna}")

    # ── Save report CSV ────────────────────────────────────────────────────────
    if args.save_report and report_rows:
        import csv
        fields = ['protein', 'organism', 'rank', 'recommended',
                  'expr_score', 'csi', 'cfd_pct', 'gc_pct', 'minmax',
                  'asite_mean', 'asite_std', 'n_codons', 'dna']
        with open(args.save_report, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(report_rows)
        print(f"Saved metrics report   → {args.save_report}  ({len(report_rows)} rows)")


if __name__ == '__main__':
    main()
