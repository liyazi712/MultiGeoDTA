# MultiGeoDTA 安装指南

## 系统要求

- **操作系统**: Linux (Ubuntu 18.04+), macOS, Windows
- **Python**: 3.8 - 3.11
- **CUDA**: 11.7+ (可选，用于GPU加速)
- **内存**: 至少 16GB RAM
- **GPU**: NVIDIA GPU with 8GB+ VRAM (推荐用于训练)

## 方法一：使用自动安装脚本（推荐）

```bash
cd MultiGeoDTA_cc_fixed

# 使用CUDA 11.8 (默认)
bash setup_env.sh multigeodta 118

# 或者使用CUDA 12.1
bash setup_env.sh multigeodta 121

# 或者使用CPU版本
bash setup_env.sh multigeodta cpu
```

## 方法二：手动安装

### Step 1: 创建虚拟环境

使用 conda:
```bash
conda create -n multigeodta python=3.8
conda activate multigeodta
```

或使用 venv:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
.\venv\Scripts\activate   # Windows
```

### Step 2: 安装 PyTorch

根据你的CUDA版本选择:

```bash
# CUDA 11.8
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: 安装 PyTorch Geometric

```bash
pip install torch_geometric

# 安装依赖扩展 (根据PyTorch和CUDA版本)
# CUDA 11.8
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# CPU
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

### Step 4: 安装 DGL

```bash
# CUDA 11.8
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html

# CPU
pip install dgl -f https://data.dgl.ai/wheels/repo.html
```

### Step 5: 安装 RDKit

```bash
pip install rdkit
```

### Step 6: 安装 Mamba (可选但推荐)

```bash
pip install mamba_ssm causal_conv1d
```

**注意**: Mamba需要CUDA支持。如果安装失败，系统会自动使用LSTM作为替代方案。

### Step 7: 安装其他依赖

```bash
pip install numpy scipy pandas scikit-learn tqdm pyyaml joblib tensorboard
```

### Step 8: 安装 MultiGeoDTA

```bash
cd MultiGeoDTA_cc_fixed
pip install -e .
```

## 验证安装

```bash
python -c "
from multigeodta import DTAModel, DTATrainer
from multigeodta.configs import get_default_config
from multigeodta.utils import evaluation_metrics

print('All imports successful!')

# 创建模型
model = DTAModel()
print(f'Model parameters: {model.get_num_parameters():,}')

# 创建配置
config = get_default_config('pdbbind_v2016')
print(f'Config task: {config.task}')

print('Installation verified successfully!')
"
```

## 常见问题

### 1. torch_scatter 安装失败

确保PyTorch版本和CUDA版本匹配：
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 2. Mamba 安装失败

Mamba需要CUDA编译环境。如果安装失败：
- 确保安装了CUDA toolkit
- 确保 nvcc 可用: `nvcc --version`
- 如果无法安装，代码会自动使用LSTM作为替代

### 3. DGL 安装失败

尝试使用conda安装：
```bash
conda install -c dglteam dgl-cuda11.8
```

### 4. 内存不足

- 减小 batch_size
- 使用梯度累积
- 使用混合精度训练

## 数据准备

下载预处理数据集：
```bash
# 从 Hugging Face 下载
# huggingface-cli download your-username/multigeodta-data --local-dir ./data

# 或者手动准备数据，参考 create_dataset/ 目录
```

## 快速测试

```bash
# 测试模型前向传播（不需要数据）
python -c "
import torch
from multigeodta import DTAModel

model = DTAModel()
print('Model created successfully!')
print(f'Parameters: {model.get_num_parameters():,}')
"
```
