<p align="center">
  <h1 align="center">🚀 ProRL: Effective Reinforcement Learning for Proactive Recommendation via Rectified Policy Gradient Estimation</h1>
  <p align="center">
    <em>Official implementation — ICML 2026</em>
  </p>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-data-preparation">Data</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-training">Training</a> •
  <a href="#-evaluation">Evaluation</a>
</p>

---

## 📋 Overview

**ProRL** is a framework for **Proactive Recommendation** that combines semantic-ID item representations with reinforcement learning. The model learns to generate item trajectories that gradually steer users toward a target item while jointly optimizing several objectives:

- **IoI (Increase of Interest)** — increase in the probability of the user engaging with the target item.
- **IoR (Increase of Rank)** — improvement in the ranking of the target item.
- **CTR (Click-Through Rate)** — predicted click probability of the recommended intermediate items.

### Key Features

- 🎯 **Multi-objective reward** — jointly optimizes IoI, IoR and CTR with configurable weights.
- 🔄 **Rectified policy gradient (ProRL)** — stable RL training with KL-divergence regularization toward the pretrained reference policy.
- 📊 **Semantic-ID tokenization** — items are represented as short codes from a learned codebook.
- ⚡ **Distributed training** — multi-GPU training via 🤗 Accelerate.

---

## 🔧 Installation

### Requirements

- Python ≥ 3.11
- CUDA ≥ 12.4 (we tested on 4× GPUs)
- PyTorch ≥ 1.12

### Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/ProRL.git
cd ProRL

# Install PyTorch (CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Core dependencies
pip install transformers==4.45.2
pip install accelerate==1.0.1
pip install sentence_transformers
pip install tensorboard
pip install recbole

# RecBole pulls in a newer numpy — pin to 1.26.0
pip uninstall -y numpy
pip install numpy==1.26.0
```

---

## 📦 Data Preparation

### Expected layout

Place all datasets under `datasets/` at the project root. Each dataset folder must contain the four files below:

```
datasets/
├── ml-1m/
│   ├── ml-1m.train                       # training sequences   (JSON)
│   ├── ml-1m.val                         # validation sequences (JSON)
│   ├── ml-1m.test                        # test sequences       (JSON)
│   ├── ml-1m.datamaps                    # ID & attribute maps  (JSON)
│   └── qwen3-embedding-8b-pca.sem_ids    # semantic IDs         (JSON)
├── Steam/
│   └── ... (same structure, prefix "Steam")
└── Books/
    └── ... (same structure, prefix "Books")
```

> The semantic-ID file name is controlled by `token_prefix` / `token_suffix` in the config (default `qwen3-embedding-8b-pca.sem_ids`).

### Raw data sources

| Dataset | Description | Link |
|---------|-------------|------|
| ML-1M | MovieLens-1M | [Download](https://grouplens.org/datasets/movielens/1m/) |
| Steam | Steam Video Games | [Download](https://cseweb.ucsd.edu//~jmcauley/datasets.html#steam_data/) |
| Books | Amazon Books | [Download](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) |

### SASRec evaluator checkpoints

The RL stage uses a pretrained **SASRec** model (via [RecBole](https://github.com/RUCAIBox/RecBole)) as a reward model / evaluator. Before launching ProRL, train SASRec on each dataset and place the checkpoints at the paths expected by RecBole:

```
ckpt/
├── SASRec-ml-1m-sas.pth
├── SASRec-steam-merged.pth
└── SASRec-amazon-books.pth
```

The matching evaluator configs are provided in `config/`:
- `config/ml-1m-sas_sasrec_config.yaml`
- `config/steam-merged_sasrec_config.yaml`
- `config/amazon-books_sasrec_config.yaml`

> The mapping from our dataset name to the RecBole dataset name is fixed inside `evaluator.py`:
> `ml-1m → ml-1m-sas`, `Steam → steam-merged`, `Books → amazon-books`.

---

## 🚀 Quick Start

All training is launched through ready-to-use shell scripts in `scripts/`. They handle the `accelerate` launch, paths and hyperparameters for you.

### Pretrain a single dataset

```bash
# ML-1M
bash scripts/Pretrain/run_ml1m_pretrain.sh

# Steam
bash scripts/Pretrain/run_steam_pretrain.sh

# Amazon Books
bash scripts/Pretrain/run_books_pretrain.sh
```

### Pretrain all three datasets sequentially

```bash
bash scripts/run_pretrain.sh
```

### ProRL fine-tuning of a single dataset

```bash
# ML-1M
bash scripts/RL/run_ml1m_prorl.sh

# Steam
bash scripts/RL/run_steam_prorl.sh

# Amazon Books
bash scripts/RL/run_books_prorl.sh
```

### ProRL fine-tuning on all three datasets sequentially

```bash
bash scripts/run_prorl.sh
```

> ⚠️ Before launching any RL script, open it and set `--pretrained_ckpt` to the actual `.pth` file produced by your pretraining run (under `ckpt/<dataset>/<run-id>/<run-id>.pth`).

---

## 🏋️ Training

### Stage 1 — Pretraining

Each pretrain script runs `proactive_pretrain.py` through `accelerate` on 4 GPUs by default:

```bash
PYTHONNOUSERSITE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m accelerate.commands.launch \
  --config_file ./config/rec_config.yaml \
  --main_process_port 16086 \
  --num_processes 4 \
  ./proactive_pretrain.py \
  --dataset ml-1m \
  --config_file ./config/ptconfig.yaml
```

Outputs are written under `ckpt/<dataset>/<timestamp-hash>/` and logs under `run_logs/`. The trainer automatically saves the best checkpoint on the validation metric (`IoI_max@10` by default).

### Stage 2 — ProRL Fine-tuning

ProRL fine-tunes the pretrained policy with a rectified policy-gradient objective and a KL penalty against the frozen reference policy. The corresponding script for ML-1M looks like:

```bash
PYTHONNOUSERSITE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m accelerate.commands.launch \
  --config_file ./config/rec_config.yaml \
  --main_process_port 16086 \
  --num_processes 4 \
  ./Proactive_RL_prorl.py \
  --dataset ml-1m \
  --config_file ./config/prorl.yaml \
  --pretrained_ckpt ./ckpt/ml-1m/<your-pretrain-run>/<your-pretrain-run>.pth \
  --mode prorl \
  --prorl_beta 1e-2 \
  --prorl_lr 1e-4 \
  --prorl_gamma 1 \
  --prorl_epochs 50 \
  --reward_weight_ctr 1.0 \
  --reward_weight_ioi 1.0 \
  --reward_weight_ior 1.0
```

#### Key CLI arguments (`Proactive_RL_prorl.py`)

| Argument | Description | Default (see `config/prorl.yaml`) |
|----------|-------------|-----------------------------------|
| `--dataset` | One of `ml-1m`, `Steam`, `Books` | — (required) |
| `--config_file` | Path to the ProRL YAML config | — (required) |
| `--pretrained_ckpt` | Path to the Stage-1 `.pth` checkpoint | — (required) |
| `--mode` | `prorl` for training, `eval` for evaluation-only | — (required) |
| `--prorl_beta` | KL-divergence penalty coefficient β | `1e-2` |
| `--prorl_lr` | RL learning rate | dataset-specific (see scripts) |
| `--prorl_gamma` | Discount factor γ for cumulative rewards | `1.0` |
| `--prorl_epochs` | Number of RL training epochs | `50` |
| `--prorl_num_samples` | Rollout samples per prompt (group size) | `16` |
| `--reward_weight_ctr` | Weight of the CTR reward term | `1.0` |
| `--reward_weight_ioi` | Weight of the IoI reward term | `1.0` |
| `--reward_weight_ior` | Weight of the IoR reward term | `1.0` |

#### Hyperparameters we used per dataset

| Dataset | `prorl_lr` | `prorl_beta` | `reward_weight_ctr` | `reward_weight_ioi` | `reward_weight_ior` |
|---------|-----------|-------------|--------------------|--------------------|--------------------|
| ML-1M | `1e-4` | `1e-2` | `1.0` | `1.0` | `1.0` |
| Steam | `1e-5` | `1e-2` | `0.1` | `1.0` | `1.0` |
| Books | `5e-4` | `1e-2` | `1.0` | `1.0` | `1.0` |

These match the defaults baked into `scripts/RL/run_<dataset>_prorl.sh`.

---

## 📊 Evaluation

To evaluate a ProRL checkpoint without further training, run the same entry point with `--mode eval`:

```bash
PYTHONNOUSERSITE=1 \
CUDA_VISIBLE_DEVICES=0 \
python -m accelerate.commands.launch \
  --num_processes 1 \
  ./Proactive_RL_prorl.py \
  --dataset ml-1m \
  --config_file ./config/prorl.yaml \
  --pretrained_ckpt ./ckpt/ml-1m/<your-prorl-run>/<your-prorl-run>.pth \
  --mode eval
```

### Reported metrics

| Metric | Description |
|--------|-------------|
| `IoI@K` | Increase of Interest at top-K trajectory length |
| `IoR@K` | Increase of Rank at top-K trajectory length |
| `CTR@K` | Average click-through rate over the top-K trajectory |
| `Coherence@K` | Trajectory coherence based on item attributes |

Top-K values default to `[1, 5, 10]` (see `config/prorl.yaml`).

---

## 🎛️ Configuration Reference

### Model architecture (T5 backbone) — `config/ptconfig.yaml` / `config/prorl.yaml`

```yaml
num_layers: 3
num_decoder_layers: 3
d_model: 128
d_ff: 512
num_heads: 4
d_kv: 64
dropout_rate: 0.1
activation_function: relu
```

### Semantic-ID tokenizer

```yaml
n_codebooks: 3
codebook_size: 256
expand_final: True
token_prefix: "qwen3-embedding-8b-pca"
token_suffix: "sem_ids"
```

### Accelerate launcher — `config/rec_config.yaml`

```yaml
distributed_type: MULTI_GPU
mixed_precision: bf16
num_processes: 2     # overridden on the CLI by --num_processes
```

---

## 📁 Project Structure

```
ProRL/
├── config/                              # YAML configs
│   ├── ptconfig.yaml                    # Pretraining config
│   ├── prorl.yaml                       # ProRL config
│   ├── rec_config.yaml                  # Accelerate launch config
│   ├── ml-1m-sas_sasrec_config.yaml     # RecBole evaluator configs
│   ├── steam-merged_sasrec_config.yaml
│   ├── amazon-books_sasrec_config.yaml
│   └── *_gru4rec_config.yaml            # Alternative GRU4Rec evaluators
│
├── scripts/                             # Launcher scripts (entry points)
│   ├── run_pretrain.sh                  # Run all pretrain scripts in sequence
│   ├── run_prorl.sh                     # Run all RL scripts in sequence
│   ├── Pretrain/
│   │   ├── run_ml1m_pretrain.sh
│   │   ├── run_steam_pretrain.sh
│   │   └── run_books_pretrain.sh
│   └── RL/
│       ├── run_ml1m_prorl.sh
│       ├── run_steam_prorl.sh
│       └── run_books_prorl.sh
│
├── datasets/                            # Datasets go here (you create this)
├── ckpt/                                # Checkpoints (auto-created)
├── run_logs/                            # Training logs   (auto-created)
├── tensorboard/                         # TensorBoard logs (auto-created)
│
├── proactive_pretrain.py                # Stage-1 entry point
├── Proactive_RL_prorl.py                # Stage-2 (ProRL) entry point
├── model.py                             # PRARec model (T5 backbone)
├── trainer.py                           # Stage-1 trainer
├── trainer_RL_prorl.py                  # Stage-2 (ProRL) trainer
├── tokenizer.py                         # Semantic-ID tokenizer
├── dataset.py                           # ProactiveRecDataset
├── collator.py                          # Train / RL collators
├── data_utils.py                        # Dataset / dataloader helpers
├── evaluator.py                         # Reward model + metric computation
├── utils.py                             # General utilities
└── README.md
```

---

## 🙏 Acknowledgments

- [RecBole](https://github.com/RUCAIBox/RecBole) — sequential recommendation baselines and the SASRec evaluator.
- [Hugging Face Transformers](https://github.com/huggingface/transformers) — T5 implementation.
- [Hugging Face Accelerate](https://github.com/huggingface/accelerate) — distributed training.