# Dynamic Token Revival — Training/Evaluation Code

- Paper / preprint: <LINK>
- Official page / project: <LINK>
- Weights / checkpoints: <LINK>

## Contents
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Downloading the Repository](#downloading-the-repository)
- [Installation with uv (reproducible environment)](#installation-with-uv-reproducible-environment)
- [Training (`train.py`)](#training-trainpy)
- [Evaluation (`evaluate.py`)](#evaluation-evaluatepy)
- [Configuration Reference (YAML)](#configuration-reference-yaml)
- [Tools (`tools/`)](#tools-tools)

## Repository Structure

- Main Scripts
	- [train.py](train.py): training with PyTorch Lightning + pruning/revival.
	- [evaluate.py](evaluate.py): accuracy evaluation and selection policy sweeps.
	- [train_config.yaml](train_config.yaml): training configuration.
	- [eval_config.yaml](eval_config.yaml): evaluation configuration.
- Source Code
	- [core/](core/): pruning/revival foundations and `TokenManager`.
	- [models/](models/): models (DeiT/ViT) with RKT architecture.
	- [pruning/](pruning/): pruning heuristics.
	- [revival/](revival/): revival heuristics.
	- [mydatasets/](mydatasets/): *DataModules*.
- Analysis and figures
	- [tools/](tools/): notebooks and scripts for visualization/statistics.
- Artifacts (generated)
	- `checkpoints/`: checkpoints saved during training.
	- `lightning_logs/`: Lightning logs.

### Quick Map (file by file)

- Root
	- [.python-version](.python-version): target Python version.
	- [pyproject.toml](pyproject.toml): project dependencies.
	- [uv.lock](uv.lock): `uv` lockfile.
	- [train.py](train.py): training script (reads `train_config.yaml`).
	- [evaluate.py](evaluate.py): evaluation script (reads `eval_config.yaml`).
	- [train_config.yaml](train_config.yaml): training parameters.
	- [eval_config.yaml](eval_config.yaml): evaluation parameters.

- [core/](core/)
	- [core/token_manager.py](core/token_manager.py): pruning+revival orchestration per block.
	- [core/pruning_base.py](core/pruning_base.py): pruning base interface and logic.
	- [core/revival_base.py](core/revival_base.py): revival base interface and logic.

- [models/](models/)
	- [models/__init__.py](models/__init__.py): `load_model()` (resolve names like `deit_small`).
	- [models/rkt.py](models/rkt.py): main implementation `VisionTransformerRKT` (pruning+revival in attention).
	- [models/deit.py](models/deit.py): DeiT -> RKT wrapper (tiny/small/base sizes).

- [mydatasets/](mydatasets/)
	- [mydatasets/__init__.py](mydatasets/__init__.py): `load_dataset()`.
	- [mydatasets/imagenet1k_dataset.py](mydatasets/imagenet1k_dataset.py): ImageNet-1k DataModule.

- [pruning/](pruning/)
	- [pruning/heuristic_pruner.py](pruning/heuristic_pruner.py): `C1`–`C4` heuristics for token scoring.

- [revival/](revival/)
	- [revival/affinity_revivor.py](revival/affinity_revivor.py): `C1`–`C4` heuristics to select tokens to revive.

- [tools/](tools/)
	- [tools/token_visualization.ipynb](tools/token_visualization.ipynb): block-wise visualization (masks/overlay).
	- [tools/token_stats.ipynb](tools/token_stats.ipynb): event matrix (active/pruned/revived) and figure.
	- [tools/pruning_scheds.py](tools/pruning_scheds.py): PDF generation of schedules.
	- `tools/imgs/`: example images for visualizations.

## Requirements

- Linux/macOS/Windows.
- Python: >= 3.10 (see [.python-version](.python-version)).
- GPU (optional but recommended) for fast training/evaluation.
- Prepared dataset (see `dataset.data_dir` in YAML).

Expected data structure (summary):

```
imagenet/
├── train/
│   ├── n01440764/          # Class 1 (ej. Tench)
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   ├── n01443537/          # Class 2 (ej. Goldfish)
│   │   ├── image_005.jpg
│   │   └── ...
│   └── ...                 # Rest of the 1,000 classes
├── val/
│   ├── n01440764/          # Class 1
│   │   ├── image_101.jpg
│   │   └── ...
│   ├── n01443537/          # Class 2
│   │   ├── image_105.jpg
│   │   └── ...
│   └── ...
└── test/
    ├── n01440764/          # Class 1 (optional, sometimes without labels)
    ├── n01443537/          # Class 2
    └── ...
```

## Downloading the Repository

```bash
git clone https://github.com/LightVED-prhlt/DTR.git
cd DTR
```

## Installation with uv (reproducible environment)

This project uses `uv` + [pyproject.toml](pyproject.toml) + [uv.lock](uv.lock).

### 1) Install uv

Option A (recommended by uv, Linux/macOS):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Option B (if you use `pipx`):

```bash
pipx install uv
```

Check that it works:

```bash
uv --version
```

### 2) Create the environment and install dependencies

From the project root (where `pyproject.toml` is located):

```bash
uv sync --frozen
```

- This creates `.venv/` and installs dependencies pinned in `uv.lock`.
- If you want to run without activating the venv, always use `uv run ...`.

## Training (`train.py`)

Run training with the default configuration:

```bash
uv run python train.py train_config.yaml
```

### Outputs

- Checkpoints: `checkpoints/<run_name>/` (generated automatically).
- The experiment name is composed of:
	- `model.name`, `run.simulate_revival`, `run.distill.mode`, `run.lr`, `pruning.prune_ratio`, `revival.revive_ratio` + timestamp.

### Note on W&B

Training uses `WandbLogger`.

- If you don't want to upload anything: `export WANDB_MODE=offline`
- If you want to log normally: `wandb login` (or define `WANDB_API_KEY`).

## Evaluation (`evaluate.py`)

Basic execution:

```bash
uv run python evaluate.py eval_config.yaml
```

The logic depends on `run.mode` inside [eval_config.yaml](eval_config.yaml). Supported values by the script:

- `inference only`: validates once and shows Top-1.
- `selection policy evaluation`: searches for the best `hold_percent` for a budget and a scheduler (values are hardcoded inside the script).
- `selection policy grid search`: iterates over several budgets and schedulers (hardcoded) and searches for the best `hold_percent`.
- `pruning heuristic grid search`: evaluates several pruning heuristics (`C1`–`C4`) with revival disabled (`can_tokens_revive: false`) for various budgets (defined in `TOKENS_PER_BLOCK`).
- `revival heuristic grid search`: evaluates several revival heuristics (`C1`–`C4`) with fixed pruning (`pruning_criterion: C2`) for various budgets (defined in `TOKENS_PER_BLOCK`).

Important: if `run.mode` does not match one of the above values, the script will not execute any branch.

Example: for a simple validation, set `run.mode: "inference only"` in `eval_config.yaml`.

## Configuration Reference (YAML)

This section lists **all** existing keys in the repo's YAMLs and what they affect in the code.

### `train_config.yaml`

- `seed` (int): global seed (`lightning.seed_everything`).

- `model`:
	- `name` (str): format `deit_<tiny|small|base>` or `vit_small`.
	- `pretrained` (bool): if `true`, loads timm weights (if `checkpoint_path` is empty).
	- `checkpoint_path` (str): path to `.ckpt` (if not empty, this checkpoint is loaded).
	- `image_size` (int): input size (used by DataModules).

- `run`:
	- `batch_size` (int): batch size.
	- `lr` (float): AdamW learning rate.
	- `max_epochs` (int): epochs.
	- `log_every_n_steps` (int): Lightning logging frequency.
	- `accumulate_grad_batches` (int): gradient accumulation.
	- `val_check_interval` (float): how often to validate (percentage of an epoch, e.g., `0.25`).
	- `wandb_project` (str): W&B project name.
	- `can_tokens_revive` (bool): activates revival logic inside the model (if `false`, only pruning).
	- `simulate_revival` (bool):
		- if `true`, converts (`prune_ratio`, `revive_ratio`) into a **list of tokens per block** to fix the budget per block.
		- if `false`, uses `pruning.prune_ratio` as a fixed fraction.
	- `distill` (dict): distillation against a fixed teacher (`deit_small_patch16_224.fb_in1k`).
		- `mode` (str): `none` | `soft` | `hard`.
		- `weight` (float): weight of the distillation loss.
		- `temp` (float): temperature (only `soft`).

- `pruning`:
	- `prune_ratio` (float): fraction of **alive spatial tokens** kept per block (0–1).
	- `pruning_criterion` (str): pruner criterion `C1`–`C4` (see below).

- `revival`:
	- `revive_ratio` (float): fraction of dead tokens to revive per block (0–1).
	- `revival_criterion` (str): revivor criterion `C1`–`C4` (see below).

- `dataset`:
	- `name` (str): `imagenet1k`.
	- `data_dir` (str): path to the dataset.

- `mixup` (augmentation for training):
	- `mixup_alpha` (float)
	- `cutmix_alpha` (float)
	- `cutmix_minmax` (list|null)
	- `prob` (float)
	- `switch_prob` (float)
	- `mode` (str): e.g., `batch`.
	- `label_smoothing` (float)
	- `num_classes` (int): **not used directly** (classes come from the dataset/model).

### `eval_config.yaml`

- `model`:
	- `name`, `pretrained`, `checkpoint_path`, `image_size`: same as in training.

- `run`:
	- `mode` (str): see supported modes in the evaluation section.
	- `batch_size` (int): batch size.
	- `can_tokens_revive` (bool): activates revival inside the model.
	- `lr` (float): **not currently used** in `evaluate.py`.

- `pruning`:
	- `prune_ratio` (float | list[int]):
		- float: fraction to keep per block.
		- list[int]: number of spatial tokens to keep per block (length 12).
	- `pruning_criterion` (str): `C1`–`C4`.

- `revival`:
	- `revive_ratio` (float | list[int]):
		- float: fraction of dead tokens to revive.
		- list[int]: number of tokens to revive per block (length 12).
	- `revival_criterion` (str): `C1`–`C4`.

- `dataset`:
	- `name`, `data_dir`: same as in training.

- `mixup`: exists in the YAML but **is not used** in `evaluate.py`.
	- `mixup_alpha` (float)
	- `cutmix_alpha` (float)
	- `cutmix_minmax` (list|null)
	- `prob` (float)
	- `switch_prob` (float)
	- `mode` (str)
	- `label_smoothing` (float)
	- `num_classes` (int)

### Available Criteria

- Pruning (`pruning.pruning_criterion` in [pruning/heuristic_pruner.py](pruning/heuristic_pruner.py)):
	- `C1`: attention to CLS.
	- `C2`: incoming contribution (sum over queries).
	- `C3`: outgoing contribution (sum over keys).
	- `C4`: outgoing entropy (`-entropy` is used).

- Revival (`revival.revival_criterion` in [revival/affinity_revivor.py](revival/affinity_revivor.py)):
	- `C1`: cosine similarity with CLS.
	- `C2`: affinity with alive tokens (maximum similarity).
	- `C3`: reconstruction-like projection.
	- `C4`: L2 norm of the embedding.

## Tools (`tools/`)

The [tools/](tools/) folder contains notebooks/scripts for analysis and figures (many paths are hardcoded to local directories; adjust them to your machine):

- Notebooks
	- [tools/token_visualization.ipynb](tools/token_visualization.ipynb): visualizes the pruning evolution per block overlaid on images (exports PDFs as `pruning_evolution_*.pdf`). Uses [tools/imgs/](tools/imgs/) as an example.
	- [tools/token_stats.ipynb](tools/token_stats.ipynb): executes a forward pass saving events (active/pruned/revived) and draws an event matrix (exports `token_stats.pdf`).

- Scripts
	- [tools/pruning_scheds.py](tools/pruning_scheds.py): generates figures of pruning schedules and an 80/20-like decomposition (exports `pruning_schedules*.pdf`).

## DynamicViT

Read the README.md inside the [dynamicvit/](dynamicvit/) folder.

## EViT

Read the README.md inside the [evit/](evit/) folder.
