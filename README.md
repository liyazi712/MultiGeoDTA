# MultiGeo-DTA
- [MultiGeo-DTA](#MultiGeo-DTA)
  - [Overview](#overview)
  - [Installation Guide](#Installation-Guide)
  - [Dependencies](#dependencies)
  - [Data availability](#Data-availability)
  - [Test](#Test)
  - [Train from scratch](#Train-from-scratch)
  - [Other usages](#other-usages)
  - [Contact](#contact)

## Overview
MultiGeo-DTA is a multimodal neural network that integrates structure and sequence information to predict compound–protein binding affinity.

## Installation Guide
```bash
git clone https://github.com/liyazi712/MultiGeo-DTA.git
cd MultiGeo-DTA
```
## Dependencies
This package is tested with Python 3.8 and CUDA 11.8 on Ubuntu 20.04, with access to an Nvidia V100 GPU (32GB RAM), AMD EPYC 7443 CPU (2.85 GHz), and 512G RAM. Run the following to create a conda environment and install the required Python packages (modify `pytorch-cuda=11.8` according to your CUDA version). 
```bash
conda create -n MultiGeo-DTA python=3.8
conda activate MultiGeo-DTA

# (pip or conda, select one)
conda install pytorch==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

conda install -c dglteam/label/th21_cu118 dgl

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html
pip install rdkit pyyaml scikit-learn torch_geometric pandas joblib
pip install causal_conv1d-1.4.0
pip install mamba_ssm-2.2.2
```
Install causal_conv1d and mamba_ssm may has error, then you can download the .whl files and install as following：
```bash
pip install causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```

The example provides versions of causal_conv1d and mamba_ssm compatible with PyTorch 2.1.0 (CUDA 11.8) and Python 3.8.
If your setup differs, ensure PyTorch, PyTorch Geometric, and DGL versions are aligned. For other versions, 
download causal_conv1d and mamba_ssm from their respective GitHub releases:
- mamba_ssm: [https://github.com/state-spaces/mamba/releases](https://github.com/state-spaces/mamba/releases)
- causal_conv1d: [https://github.com/Dao-AILab/causal-conv1d/releases](https://github.com/Dao-AILab/causal-conv1d/releases)

common error: 
1. OSError:  libcusparse.so.11: cannot open shared object file: No such file or directory
2. solution:  conda install -c nvidia cudatoolkit=11.8

Running the above lines of `conda install` should be sufficient to install all  MultiGeo-DTA's required packages (and their dependencies).
## Data availability
1. Download open source data from Hugging Face Dataset. (Because of the official website's limitation, PDBBind v2021 dataset will open source after the official website open source them)
    ```bash
    pip install -U huggingface_hub
    export HF_ENDPOINT="https://hf-mirror.com"
    cd create_dataset
    huggingface-cli download laddymo/MultiGeoDTA --repo-type dataset --local-dir MultiGeoDTA --local-dir-use-symlinks False
    ```

## Test
1. PDBBind_v2016
```
python test_MultiGeoDTA.py --task pdbbind_v2016 --output_dir ./MultiGeoDTA/output/pdbbind_v2016 --model_file pdbbind_v2016
```

2. PDBBind_v2020
```
python test_MultiGeoDTA.py --task pdbbind_v2020 --output_dir ./MultiGeoDTA/output/pdbbind_v2020 --model_file pdbbind_v2020
```

3. PDBBind_v2021_time
```
python test_MultiGeoDTA.py --task pdbbind_v2021_time --output_dir ./MultiGeoDTA/output/pdbbind_v2021_time --model_file pdbbind_v2021_time
```

4. PDBBind_v2021_similarity
```
python test_MultiGeoDTA.py --task pdbbind_v2021_similarity --output_dir ./MultiGeoDTA/output/pdbbind_v2021_similarity/new_new/0.5 --model_file pdbbind_v2021_similarity/new_new/0.5 --split_method new_new --thre 0.5 
```

5. LP-PDBBind
```
python test_MultiGeoDTA.py --task lp_pdbbind --output_dir ./MultiGeoDTA/output/lp_pdbbind --model_file lp_pdbbind
```

5. ZINC(virtual screening)
```
python run_vs.py --output_dir ./MultiGeoDTA/output/zinc --model_file pdbbind_v2020 --device 0
```

note: split_method: new_new, new_compound, new_protein; thre: 0.3, 0.4, 0.5, 0.6, modify output_dir and model_file according to split_method and thre)

## Train from scratch
1. PDBBind_v2016
```
python run_MultiGeoDTA.py --task pdbbind_v2016 --output_dir ./MultiGeoDTA/output/pdbbind_v2016
```

2. PDBBind_v2020
```
python run_MultiGeoDTA.py --task pdbbind_v2020 --output_dir ./MultiGeoDTA/output/pdbbind_v2020
```

3. PDBBind_v2021_time
```
python run_MultiGeoDTA.py --task pdbbind_v2021_time --output_dir ./MultiGeoDTA/output/pdbbind_v2021_time
```

4. PDBBind_v2021_similarity
```
python run_MultiGeoDTA.py --task pdbbind_v2021_similarity --output_dir ./MultiGeoDTA/output/pdbbind_v2021_similarity/new_new/0.5 --split_method new_new --thre 0.5
```
note: split_method: new_new, new_compound, new_protein; thre: 0.3, 0.4, 0.5, 0.6, modify output_dir and model_file according to split_method and thre)

5. LP-PDBBind
```
python run_MultiGeoDTA.py --task lp_pdbbind --output_dir ./MultiGeoDTA/output/lp_pdbbind
```

## Other usages
1. missing_dataset. 
    train:
    ```
    python run_MultiGeoDTA.py --task pdbbind_v2016 --output_dir ./MultiGeoDTA/output/pdbbind_v2016_robustness/missing_0.2
    ```
    test:
    ```
    python test_MultiGeoDTA.py --task pdbbind_v2016 --output_dir ./MultiGeoDTA/output/pdbbind_v2016_robustness/missing_0.2 --model_file pdbbind_v2016_robustness/missing_0.2
    ```

2. noise_label. 
    train:
    ```
    python run_MultiGeoDTA.py --task pdbbind_v2016 --output_dir ./MultiGeoDTA/output/pdbbind_v2016_robustness/noise_0.2
    ```
    test:
    ```
    python test_MultiGeoDTA.py --task pdbbind_v2016 --output_dir ./MultiGeoDTA/output/pdbbind_v2016_robustness/noise_0.2 --model_file pdbbind_v2016_robustness/noise_0.2
    ```

## Contact
Please submit GitHub issues or contact Yazi Li (liyazi126@126.com) for any questions related to the source code.

