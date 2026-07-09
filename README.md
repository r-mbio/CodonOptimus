# CodonOptimus

AI-driven codon optimisation for 39 industrial organisms, with ribosome-profiling
supervision and dual-head expression/A-site prediction.

Associated manuscript: **"CodonOptimus: a foundation model for industrial codon
optimisation with ribosome-profiling supervision and experimental validation"**
(Nucleic Acids Research, 2026).

---

## Quick start — optimise a protein sequence

```bash
# 1. Clone and install dependencies (see Requirements below)
git clone https://github.com/r-mbio/CodonOptimus.git
cd CodonOptimus

# 2. Download model weights and reference data (see Data & Models below)

# 3a. Optimise a single sequence inline
python3 04_generation/optimize_sequence.py \
    --aa_seq MKVLATVFLAVSAAVNG \
    --org ecoli

# 3b. Optimise a FASTA file with multiple proteins, saving the optimised DNA
python3 04_generation/optimize_sequence.py \
    --fasta examples/benchmark_proteins.faa \
    --org pichia \
    --save_dna pichia_optimised.fna
```

Example FASTA files are provided in `examples/`:

| File | Contents |
|---|---|
| `examples/benchmark_proteins.faa` | 4 benchmark proteins (BLG, HSA, PHYA, XYN2) — the proteins used in the paper |
| `examples/quick_test.faa` | 3 short sequences (6–31 AA) for a fast smoke test |

**Output per sequence:** optimised CDS, CSI, CFD%, %MinMax, GC%, predicted expression
score, and per-codon A-site occupancy profile (last two only for organisms with
dual-head models: E. coli, B. subtilis, K. phaffii, S. cerevisiae).

**Synonymous guarantee:** the model hard-constrains every predicted codon to encode
the correct amino acid before argmax/sampling — AA-level alignment is enforced by
construction and verified post-generation (raises `RuntimeError` on any mismatch,
so incorrect output is impossible to miss).

All 39 supported organism keys (training data spans all publicly available
assemblies of the taxon; organisms with Ribo-seq supervision are marked †; ‡ = Ribo-seq feeds dual-head model only, not the sequence generator):

| `--org` key | Scientific name | Common strain / notes |
|---|---|---|
| `ecoli` † | *Escherichia coli* | K-12 MG1655 (RS-FT); 500 RefSeq complete assemblies |
| `bacillus` † | *Bacillus subtilis* | 168 (RS-FT); all RefSeq complete assemblies |
| `s_cerevisiae` †‡ | *Saccharomyces cerevisiae* | S288C; all assemblies |
| `pichia` † | *Komagataella phaffii* | CBS 7435 (RS-FT, GSE159336 AOX1); all *Komagataella* assemblies |
| `yarrowia` | *Yarrowia lipolytica* | W29 / CLIB89; all assemblies |
| `trichoderma` | *Trichoderma reesei* | RUT-C30; all assemblies |
| `corynebacterium` | *Corynebacterium glutamicum* | ATCC 13032; all assemblies |
| `kluyveromyces` | *Kluyveromyces lactis* | CBS 2359; all assemblies |
| `kluyveromyces_marxianus` | *Kluyveromyces marxianus* | CBS 712; all assemblies |
| `ogataea` | *Ogataea polymorpha* | CBS 4732; all assemblies |
| `scheffersomyces` | *Scheffersomyces stipitis* | CBS 6054; all assemblies |
| `ashbya` | *Ashbya gossypii* | ATCC 10895; all assemblies |
| `rhodotorula` | *Rhodotorula toruloides* | NP11; all assemblies |
| `aspergillus_niger` | *Aspergillus niger* | CBS 513.88; all assemblies |
| `aspergillus_oryzae` | *Aspergillus oryzae* | RIB40; all assemblies |
| `aspergillus_terreus` | *Aspergillus terreus* | NIH2624; all assemblies |
| `aspergillus_fumigatus` | *Aspergillus fumigatus* | Af293; all assemblies |
| `penicillium` | *Penicillium* sp. | Genus-level (multiple species); all assemblies |
| `myceliophthora` | *Myceliophthora thermophila* | ATCC 42464; all assemblies |
| `neurospora` | *Neurospora crassa* | OR74A; all assemblies |
| `fusarium` | *Fusarium oxysporum* | Fo5176; all assemblies |
| `rhizopus` | *Rhizopus delemar* | 99-880; all assemblies |
| `mucor` | *Mucor circinelloides* | CBS 277.49; all assemblies |
| `bacillus_licheniformis` | *Bacillus licheniformis* | ATCC 14580; all assemblies |
| `bacillus_amyloliquefaciens` | *Bacillus amyloliquefaciens* | DSM 7; ≤200 RefSeq assemblies |
| `brevibacillus` | *Brevibacillus brevis* | 47; all assemblies |
| `lactococcus` | *Lactococcus lactis* | MG1363; all assemblies |
| `lactobacillus` | *Lactobacillus plantarum* | WCFS1; ≤200 RefSeq assemblies |
| `streptomyces` | *Streptomyces coelicolor* | A3(2); all assemblies |
| `streptomyces_lividans` | *Streptomyces lividans* | TK24; all assemblies |
| `pseudomonas_putida` | *Pseudomonas putida* | KT2440; all assemblies |
| `gluconobacter` | *Gluconobacter oxydans* | 621H; all assemblies |
| `cupriavidus` | *Cupriavidus necator* | H16; all assemblies |
| `clostridium` | *Clostridium acetobutylicum* | ATCC 824; all assemblies |
| `nannochloropsis` | *Nannochloropsis* sp. | Genus-level; all assemblies |
| `phaeodactylum` | *Phaeodactylum tricornutum* | CCAP 1055/1; all assemblies |
| `chlamydomonas` | *Chlamydomonas reinhardtii* | CC-503; all assemblies |
| `cho` | *Cricetulus griseus* (CHO) | CHO-K1; all assemblies |
| `human` | *Homo sapiens* | GRCh38 (1 RefSeq assembly) |

---

## Requirements

### Python ≥ 3.10

Install with pip or conda:

```bash
pip install torch>=2.6 numpy>=2.0 pandas>=2.0 scipy>=1.13 \
            scikit-learn>=1.4 matplotlib>=3.8 seaborn>=0.12 \
            biopython>=1.80 openpyxl>=3.1 ViennaRNA python-codon-adaptation-index
```

A GPU (CUDA ≥ 12.4) is strongly recommended for training; inference runs on CPU
but is slower.

### System tools (data preparation only)

- `samtools` ≥ 1.17 — for processing Ribo-seq BAM files
- `ncbi-datasets-cli` — for downloading reference genome CDS; install with
  `conda install -c conda-forge ncbi-datasets-cli`

### Reference genome annotations (data preparation only)

Two NCBI RefSeq GFF3 files are needed before running the data-preparation scripts;
download them once and place them at the paths shown:

```bash
# B. subtilis 168 (ASM904v1)
datasets download genome accession GCF_000009045.1 --include gff3
unzip ncbi_dataset.zip
mv ncbi_dataset/data/GCF_000009045.1/genomic.gff \
   data/raw_genomes/bacillus_subtilis/GCF_000009045.1_ASM904v1_genomic.gff

# S. cerevisiae R64
datasets download genome accession GCF_000146045.2 --include gff3
unzip ncbi_dataset.zip
mv ncbi_dataset/data/GCF_000146045.2/genomic.gff \
   data/raw_genomes/saccharomyces_cerevisiae/GCF_000146045.2_R64_genomic.gff
```

---

## Data & models

Model weights, processed training data, benchmark results, and A-site profiles
are deposited at **https://doi.org/10.6084/m9.figshare.32847818** (PolyForm Noncommercial License 1.0.0).

Download and unpack so the repository looks like this:

```
CodonOptimus/
├── models/
│   ├── industrial_mlm_pretrain.pt              # foundation model
│   ├── industrial_mlm_ep11.pt                  # generation checkpoint (epoch 11)
│   ├── industrial_mlm_csi_all39.pt             # CSI-FT model (all 39 organisms)
│   ├── industrial_mlm_all39_rs_ecoli.pt        # RS-FT generator — E. coli
│   ├── industrial_mlm_all39_rs_bacillus.pt     # RS-FT generator — B. subtilis
│   ├── industrial_mlm_all39_rs_pichia.pt       # RS-FT generator — K. phaffii
│   ├── industrial_mlm_all39_rs_s_cerevisiae.pt # RS-FT generator — S. cerevisiae (Fig 2 only)
│   ├── dual_head_ep11_ecoli.pt                 # dual-head predictor — E. coli
│   ├── dual_head_pretrain_bacillus.pt          # dual-head predictor — B. subtilis
│   ├── dual_head_ep11_pichia.pt                # dual-head predictor — K. phaffii
│   └── dual_head_pretrain_s_cerevisiae.pt      # dual-head predictor — S. cerevisiae
├── data/
│   ├── pretrain/all_industrial_cds.tsv         # 1.8M CDS sequences used for pretraining
│   ├── raw/{ecoli,bacillus,pichia,s_cerevisiae}/   # per-organism train/val CSVs
│   ├── asite_profiles/{ecoli,pichia,s_cerevisiae,bacillus}/  # per-gene .npy profiles
│   ├── trna/                                   # tAI weights per organism
│   ├── codon_tables/                           # RSCU tables for CSI computation
│   └── star_index/ecoli/transcript_map.json
├── results/
│   ├── benchmark_consistent_v4.csv             # generated by Stage 05; needed by Fig 4 / S8
│   ├── fig2_combined_cache.json                # generated by plot_Fig2_tai_csi.py; needed by plot_S7
│   ├── figS_tai_other_orgs_cache.json          # pre-computed; needed by plot_S3
│   ├── figS_csi_all39_cache.json               # pre-computed; needed by plot_S4
│   └── OLD_RESULTS/
│       ├── large_benchmark_combined.csv        # pre-computed; needed by plot_S9
│       └── native_cpb_100genes.csv             # pre-computed; needed by plot_S11
└── other_optimizers/
    ├── benchmark/
    │   ├── WT_sequences.fasta                  # wild-type protein sequences
    │   ├── benchmark_results_tai.csv           # CodonOptimus benchmark results
    │   ├── tai_top10_cache.json                # tAI cache for top-10% genes
    │   └── optimized_fastas/                   # per-tool optimised FASTA files
    ├── codontransformer/                       # CodonTransformer sequences (for comparison)
    ├── idt/                                    # IDT-optimised sequences
    ├── twist/                                  # Twist-optimised sequences
    └── genscript/                              # GenScript-optimised sequences
```

**Run-order note for Stage 06:** `plot_S7_bacillus_atbias.py` reads the cache
written by `plot_Fig2_tai_csi.py` — run Fig 2 first. `plot_Fig4` and `plot_S8`
read `results/benchmark_consistent_v4.csv` — run Stage 05 first. All other
pre-computed files in `results/` are included in the data deposit and do not
need to be regenerated.

---

## Pipeline overview

Each numbered directory is a self-contained stage. Run them in order to
reproduce the paper from raw data; or start at stage 04 with the released model
weights to just generate optimised sequences.

```
01_data_preparation/   — download CDS, process Ribo-seq BAMs, build A-site profiles
02_foundation_pretraining/ — pretrain the 37.9M-parameter codon-level MLM
03_finetuning/         — CSI fine-tuning (all 39 orgs), RS-FT (per org), dual-head
04_generation/         — generate optimised sequences; regenerate benchmark FASTAs
05_benchmarking_validation/ — compute CPB, tAI, CSI, ExprScore vs competitor tools
06_figures/            — reproduce all main and supplementary figures
```

### Stage 01 — Data preparation

```bash
# Download genome CDS from NCBI for all 39 organisms
python3 01_data_preparation/download_industrial_cds.py

# Build normalised Ribo-seq expression targets
python3 01_data_preparation/normalize_all_riboseq_tpm.py

# Build E. coli A-site profiles from genuine in-vivo Ribo-seq BAMs
# (BAMs must be downloaded from SRA: SRR35650607, SRR22447282, SRR22447285)
python3 01_data_preparation/rebuild_ecoli_asite_genuine_riboseq.py

# Similarly for other organisms:
python3 01_data_preparation/update_bacillus_riboseq_gse126234.py
python3 01_data_preparation/update_kphaffii_riboseq_newbams.py
python3 01_data_preparation/update_scer_riboseq_correct_condition.py
python3 01_data_preparation/update_ecoli_riboseq_new_bams.py
```

### Stage 02 — Foundation pretraining

```bash
# Trains on all 1.8M CDS sequences across 39 organisms (~80 epochs, H100 ~48 h)
python3 02_foundation_pretraining/train_mlm_industrial.py
```

### Stage 03 — Fine-tuning

```bash
# CSI fine-tuning (all 39 organisms simultaneously)
python3 03_finetuning/finetune_csi_all39.py

# RS-FT generators (one per organism with Ribo-seq data)
python3 03_finetuning/finetune_ecoli_rs.py
python3 03_finetuning/finetune_bacillus_rs.py
python3 03_finetuning/finetune_pichia_rs.py

# Dual-head fine-tuning (expression MLP + A-site CNN)
python3 03_finetuning/finetune_dual_head_specialist.py --org ecoli
python3 03_finetuning/finetune_dual_head_specialist.py --org s_cerevisiae
python3 03_finetuning/finetune_dual_head_specialist.py --org pichia
python3 03_finetuning/finetune_bacillus_dualhead.py
```

### Stage 04 — Generation

```bash
# Generate an optimised coding sequence for any protein:
python3 04_generation/optimize_sequence.py --aa_seq <SEQUENCE> --org <ORGANISM>

# Regenerate E. coli benchmark FASTAs after any model update:
python3 04_generation/regen_ecoli_optimized_fasta.py
```

### Stage 05 — Benchmarking

```bash
# Compute all CPB / tAI / CSI / ExprScore metrics (writes benchmark_results_tai.csv)
python3 05_benchmarking_validation/compute_mrna_cpb_metrics.py

# Cross-tool comparison table
python3 05_benchmarking_validation/benchmark_consistent_v4.py
```

### Stage 06 — Figures

```bash
python3 06_figures/plot_Fig2_tai_csi.py          # Figure 2
python3 06_figures/plot_Fig3_cpb.py              # Figure 3
python3 06_figures/plot_Fig4_ctes_exprscore_wetlab.py  # Figure 4
python3 06_figures/plot_S*.py                    # Supplementary figures
```

---

## Comparison tools (benchmarking baselines)

CodonOptimus is benchmarked against the following codon optimisation tools in Stage 05.
Pre-generated sequences for each tool are deposited in `other_optimizers/` (see data deposit).

| Tool | Reference | Notes |
|---|---|---|
| **CodonTransformer** | Roodgar et al. 2024, *Nat. Mach. Intell.* | Transformer-based, 164 organisms |
| **IDT** (Freq.-based) | Integrated DNA Technologies codon optimiser | Frequency-based commercial tool |
| **Twist Bioscience** | Twist Bioscience codon optimiser | Frequency-based commercial tool |
| **GenScript** | GenScript OptimumGene™ | Proprietary scoring function |
| **Wild-type (WT)** | NCBI RefSeq native CDS | Unoptimised baseline |

---

## Ribo-seq data sources

| Organism | GEO / BioProject | Condition | Use |
|---|---|---|---|
| E. coli | PRJNA1335396 + PRJNA906596 | Normal growth, MG1655 | dual-head + RS-FT |
| K. phaffii | GSE159336 + SRR32315787/88 | AOX1-induced + YPD growth | dual-head + RS-FT |
| S. cerevisiae | GSE56622 + GSE185286 + GSE185458 | YPD growth | dual-head |
| B. subtilis | GSE126234 + GSE249448 | LB exponential + exponential growth | expression training + dual-head + RS-FT |

---

## Citation

```
Sathyamoorthy R. et al. (2026). CodonOptimus: a ribosome-profiling-supervised
foundation model for codon optimisation and recombinant protein production in
microbial cell factories. Nucleic Acids Research, [doi to be added].
```

---

## Licence

CodonOptimus is released under the **PolyForm Noncommercial License 1.0.0** (see
[LICENSE](LICENSE)).

- **Academic and non-profit research:** free to use, modify, and distribute.
- **Commercial use** (for-profit companies, commercial products or services):
  contact M.Puri@massey.ac.nz for licensing.
