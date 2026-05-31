# MultiGeo-DTA: Multi-modal Geometric Deep Learning Enables Drug-Target Affinity Prediction with Robustness and Generalization

MultiGeoDTA is a ***multimodal neural network*** that integrates protein pocket 3D structure, drug 3D structure, protein full sequence, protein pocket sequence, and drug SMILES sequence information to predict drug-target affinity.

![MultiGeoDTA architecture](assets/MultiGeoDTA.png)

## Project layout

```
MultiGeoDTA/
├── assets/                      # Figures (e.g. MultiGeoDTA.png)
├── src/multigeodta/             # Installable package
│   ├── cli.py                   # Unified CLI (train / evaluate / screen)
│   ├── config/                  # YAML loading
│   ├── data/                    # Datasets, featurizers, task registry
│   ├── models/                  # DTA model & GVP
│   ├── training/                # Trainer & experiment loop
│   ├── inference/               # Virtual screening
│   ├── metrics/                 # Regression metrics
│   └── utils/
├── configs/
│   ├── base.yaml
│   └── tasks/                   # Per-benchmark YAML (+ pdbbind_v2016_robustness/)
├── scripts/
│   ├── install.sh               # Environment setup
│   ├── download_data.sh         # Hugging Face dataset download
│   ├── build_zinc_vs_dataset.sh # ZINC virtual-screening prep
│   ├── dataset/                 # Raw PDBBind preprocessing
│   ├── virtual_screening/
│   │   └── docking/             # Molecular docking demo
│   └── lib/
├── requirements/                # base.txt, cuda118.txt
├── data/                        # Downloaded datasets (gitignored)
├── outputs/                     # Checkpoints & logs (gitignored)
├── run_MultiGeoDTA.py           # Legacy shim → multigeodta train/evaluate
├── run_vs.py                    # Legacy shim → multigeodta screen
├── test_MultiGeoDTA.py          # Legacy test entry
├── pyproject.toml
└── environment.yml
```

## Quick install

**Recommended** — one script installs PyTorch 2.1 + cu118, DGL, PyG, and mamba wheels in the correct order:

```bash
cd /path/to/MultiGeoDTA
bash scripts/install.sh
conda activate multigeodta
export MULTIGEODTA_DATA_DIR=/path/to/MultiGeoDTA/data
```

Other CUDA / PyTorch versions: pick matching wheels from:

- https://github.com/state-spaces/mamba/releases
- https://github.com/Dao-AILab/causal-conv1d/releases

## Data

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/download_datasets_and_model_weights.sh
export MULTIGEODTA_DATA_DIR=/path/to/MultiGeoDTA/data
```

## Gold Standard Benchmark Results
### PDBBind v2016
![Benchmark_PDBBind_v2016](assets/Benchmark_PDBBind_v2016.png)

### PDBBind v2020
![Benchmark_PDBBind_v2020](assets/Benchmark_PDBBind_v2020.png)

## Usage

All commands use the unified CLI (`python -m multigeodta`). Each benchmark has a YAML under `configs/tasks/`; override hyperparameters via CLI flags when needed.

### PDBBind v2016

```bash
# Train
python -m multigeodta train --config configs/tasks/pdbbind_v2016.yaml

# Evaluate
python -m multigeodta evaluate --config configs/tasks/pdbbind_v2016.yaml \
  --model_file pdbbind_v2016 --output_dir outputs/pdbbind_v2016
```

### PDBBind v2020

```bash
# Train
python -m multigeodta train --config configs/tasks/pdbbind_v2020.yaml

# Evaluate
python -m multigeodta evaluate --config configs/tasks/pdbbind_v2020.yaml \
  --model_file pdbbind_v2020 --output_dir outputs/pdbbind_v2020
```

### PDBBind v2021 (similarity split)

Similarity-based splits with ligand/protein novelty settings (`new_compound`, `new_protein`, `new_new`) and Tanimoto thresholds (`0.3`–`0.6`). Default config uses `new_new` at threshold `0.5`.

```bash
# Train (default: new_new / 0.5)
python -m multigeodta train --config configs/tasks/pdbbind_v2021_similarity.yaml

# Train another split
python -m multigeodta train --task pdbbind_v2021_similarity \
  --split_method new_protein --thre 0.4 \
  --output_dir outputs/pdbbind_v2021_similarity/new_protein/0.4

# Evaluate
python -m multigeodta evaluate --config configs/tasks/pdbbind_v2021_similarity.yaml \
  --model_file pdbbind_v2021_similarity/new_new/0.5 \
  --output_dir outputs/pdbbind_v2021_similarity/new_new/0.5
```

### PDBBind v2021 (time split)

```bash
# Train
python -m multigeodta train --config configs/tasks/pdbbind_v2021_time.yaml

# Evaluate
python -m multigeodta evaluate --config configs/tasks/pdbbind_v2021_time.yaml \
  --model_file pdbbind_v2021_time --output_dir outputs/pdbbind_v2021_time
```

### LP-PDBBind

```bash
# Train
python -m multigeodta train --config configs/tasks/lp_pdbbind.yaml

# Evaluate
python -m multigeodta evaluate --config configs/tasks/lp_pdbbind.yaml \
  --model_file lp_pdbbind --output_dir outputs/lp_pdbbind
```

### PDBBind v2016 robustness benchmark

Robustness CSVs live under `data/pdbbind_v2016/pdbbind_v2016_robustness_test/`.  
Only the **training set** is perturbed; **validation and test** use the standard PDBBind v2016 splits.

| Type | Variants | Description |
|------|----------|-------------|
| Missing samples | `missing_0.2` … `missing_0.8` | Randomly drop 20%–80% of training rows |
| Label noise | `noised_scale_0.2` … `noised_scale_1.0` | Add Gaussian noise: `label + scale × N(0,1)` |

See [`data/pdbbind_v2016/pdbbind_v2016_robustness_test/README.md`](data/pdbbind_v2016/pdbbind_v2016_robustness_test/README.md) for file details and regeneration scripts.

```bash
# Train one variant (YAML per variant under configs/tasks/pdbbind_v2016_robustness/)
python -m multigeodta train \
  --config configs/tasks/pdbbind_v2016_robustness/noised_scale_0.4.yaml

# Or
python -m multigeodta train --task pdbbind_v2016_robustness \
  --variant missing_0.6 \
  --output_dir outputs/pdbbind_v2016_robustness/missing_0.6

# Evaluate
python -m multigeodta evaluate \
  --config configs/tasks/pdbbind_v2016_robustness/noised_scale_0.4.yaml \
  --model_file pdbbind_v2016_robustness/noised_scale_0.4 \
  --output_dir outputs/pdbbind_v2016_robustness/noised_scale_0.4

# Run all 9 variants
for cfg in configs/tasks/pdbbind_v2016_robustness/*.yaml; do
  python -m multigeodta train --config "$cfg"
done
```

### Virtual Screening

Screen ZINC compounds against a target (protein sequence + pocket positions). Default config uses a CB1R example and checkpoints trained on PDBBind v2020.

```bash
# Screen
python -m multigeodta screen --config configs/tasks/zinc_vs.yaml \
  --model_file pdbbind_v2020 --output_dir outputs/zinc --device 0
```

See [`data/zinc/README.md`](data/zinc/README.md) for the full ZINC download, preprocessing, and new-target workflow. One-command build: `bash scripts/build_zinc_vs_dataset.sh`.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MULTIGEODTA_DATA_DIR` | `./data` or `./create_dataset` | Dataset root |
| `MULTIGEODTA_OUTPUT_DIR` | `./outputs` | Checkpoints & logs |

## Citation

If you use this code, please cite the MultiGeo-DTA paper and contact Yazi Li (liyazi126@126.com) for questions.

## Contact

GitHub issues or liyazi126@126.com
