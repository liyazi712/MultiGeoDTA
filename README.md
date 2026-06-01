# MultiGeo-DTA: Multi-modal Geometric Deep Learning Enables Drug-Target Affinity Prediction with Robustness and Generalization

MultiGeoDTA is a ***multimodal neural network*** that integrates protein pocket 3D structure, drug 3D structure, protein full sequence, protein pocket sequence, and drug SMILES sequence information to predict drug-target affinity.

![MultiGeoDTA architecture](assets/MultiGeoDTA.png)

## 1. Project layout

```
MultiGeoDTA/
├── assets/                              # Figures (e.g. MultiGeoDTA.png, benchmark plots)
├── docs/                                # Detailed documentation (MultiGeoDTA.md)
├── src/multigeodta/                     # Installable package
│   ├── cli.py                           # Unified CLI (train / evaluate / screen)
│   ├── config/                          # YAML loading
│   ├── data/                            # Datasets, featurizers, task registry
│   ├── models/                          # DTA model & GVP
│   ├── training/                        # Trainer & experiment loop
│   ├── inference/                       # Virtual screening
│   ├── metrics/                         # Regression metrics
│   └── utils/
├── configs/
│   ├── base.yaml
│   └── tasks/                           # Per-benchmark YAML
├── scripts/
│   ├── install.sh                       # Environment setup
│   ├── download_datasets_and_model_weights.sh  # Hugging Face data & checkpoints
│   ├── run_predict_pipeline.sh          # End-to-end affinity prediction
│   ├── build_zinc_vs_dataset.sh         # ZINC virtual-screening prep
│   ├── smoke_install_and_train.sh       # Smoke test
│   ├── upload_to_hf.sh                  # Upload artifacts to Hugging Face
│   ├── predict_affinity_from_sequence/  # New-target prediction pipeline
│   │   ├── predict_structure.py         # ESMFold2 + DoGSite3 pocket detection
│   │   ├── structure_pipeline.py        # Structure generation helpers
│   │   └── predict_affinity.py          # MultiGeo-DTA ensemble inference
│   ├── dataset/                         # Raw PDBBind preprocessing
│   ├── virtual_screening/
│   │   ├── docking/                     # Molecular docking demo
│   │   └── filter_vs_results.ipynb
│   └── lib/
│       └── install_deps.sh              # Dependency installer (used by install.sh)
├── requirements/                        # base.txt, cuda118.txt
├── data/                                # Downloaded datasets (gitignored)
├── outputs/                             # Checkpoints, logs & user predictions (gitignored)
│   └── user_request_results/            # Timestamped outputs from run_predict_pipeline.sh
├── pyproject.toml
└── environment.yml
```

## 2. Quick install

**Recommended** — one script installs PyTorch 2.1 + cu118, DGL, PyG, and mamba wheels in the correct order:

```bash
cd MultiGeoDTA
bash scripts/install.sh
conda activate multigeodta
```

Other CUDA / PyTorch versions: pick matching wheels from:

- [https://github.com/state-spaces/mamba/releases](https://github.com/state-spaces/mamba/releases)
- [https://github.com/Dao-AILab/causal-conv1d/releases](https://github.com/Dao-AILab/causal-conv1d/releases)

## 3. Download preprocessed data and trained model weights

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/download_datasets_and_model_weights.sh
export MULTIGEODTA_DATA_DIR=/path/to/MultiGeoDTA/data
```

## 4. End-to-end drug-target affinity prediction for new data (Use trained model weights on PDBBind v2020 dataset for prediction)

### Step 1: Create a new virtual environment for ESMFold2

```bash
conda create -n esmfold2 python=3.12 -y
conda activate esmfold2
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install "esm@git+https://github.com/Biohub/esm.git@main"
export HF_ENDPOINT=https://hf-mirror.com   # download ESMFold2 model weights
```

### Step2: Run end-to-end prediction

```bash
bash scripts/run_predict_pipeline.sh -p "GENFMDIECFMVLNPSQQLAIAVLSLTLGTFTVLENLLVLCVILHSRSLRCRPSYHFIGSLAVADLLGSVIFVYSFIDFHVFHRKDSRNVFLFKLGGVTASFTASVGSLFLAAIDRYISIHRPLAYKRIVTRPKAVVAFCLMWTIAIVIAVLPLLGWNCEKLQSVCSDIFPHIDKTYLMFWIGVVSVLLLFIVYAYMYILWKAHSHAVAKALIVYGSTTGNTEYTAETIARELADAGYEVDSRDAASVEAGGLFEGFDLVLLGCSTWGDDSIELQDDFIPLFDSLEETGAQGRKVACFGCGDSSWEYFCGAVDAIEEKLKNLGAEIVQDGLRIDGDPRAARDDIVGWAHDVRGAIPDQARMDIELAKTLVLILVVLIICWGPLLAIMVYDVFGKMNKLIKTVFAFCSMLCLLNSTVNPIIYALRSKDLRHAFRSMFPS" -s "Cc1nn(CCCNC(=O)c2ccco2)c(C)c1Br"
```

#### Note: -p: full protein sequence; -s: SMILES sequence.

## 5. Gold Standard Benchmark Results

### PDBBind v2016

![Benchmark_PDBBind_v2016](assets/Benchmark_PDBBind_v2016.png)

### PDBBind v2020

![Benchmark_PDBBind_v2020](assets/Benchmark_PDBBind_v2020.png)

## 6. Other Usage

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


| Type            | Variants                                | Description                                  |
| --------------- | --------------------------------------- | -------------------------------------------- |
| Missing samples | `missing_0.2` … `missing_0.8`           | Randomly drop 20%–80% of training rows       |
| Label noise     | `noised_scale_0.2` … `noised_scale_1.0` | Add Gaussian noise: `label + scale × N(0,1)` |


See `[data/pdbbind_v2016/pdbbind_v2016_robustness_test/README.md](data/pdbbind_v2016/pdbbind_v2016_robustness_test/README.md)` for file details and regeneration scripts.

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

See `[data/zinc/README.md](data/zinc/README.md)` for the full ZINC download, preprocessing, and new-target workflow. One-command build: `bash scripts/build_zinc_vs_dataset.sh`.

### Environment variables


| Variable                 | Default                        | Description        |
| ------------------------ | ------------------------------ | ------------------ |
| `MULTIGEODTA_DATA_DIR`   | `./data` or `./create_dataset` | Dataset root       |
| `MULTIGEODTA_OUTPUT_DIR` | `./outputs`                    | Checkpoints & logs |


## 7. Detailed project documentation

If you need to make improvements based on MultiGeoDTA or want to gain a deeper understanding of the working mechanism of MultiGeoDTA, please read the more detailed project documentation.`[docs/MultiGeoDTA.md](docs/MultiGeoDTA.md)`

## 8. Citation

If you use this code, please cite the MultiGeo-DTA paper and contact Yazi Li ([yazi_li@tongji.edu.cn](mailto:yazi_li@tongji.edu.cn)) for questions.

## 9. Contact

GitHub issues or [yazi_li@tongji.edu.cn](mailto:yazi_li@tongji.edu.cn)