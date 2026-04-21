#!/bin/bash
# LongVU 快速安装脚本 - 一键完成环境搭建
# 使用方法：bash quick_start.sh

set -e  # 遇到错误立即停止

echo "============================================================"
echo "LongVU 快速安装脚本"
echo "============================================================"
echo ""

# 检查 Python 版本
echo "[1/4] 检查 Python 环境..."
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到 Python3，请先安装 Python 3.10"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Python 版本：$PYTHON_VERSION"

if [ "$PYTHON_VERSION" != "3.10" ]; then
    echo "⚠️  建议使用 Python 3.10，当前版本：$PYTHON_VERSION"
fi

# 检查 conda
echo ""
echo "[2/4] 检查 Conda 环境..."
if ! command -v conda &> /dev/null; then
    echo "❌ 未找到 Conda，请先安装 Miniconda："
    echo "   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "   bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi
echo "✓ Conda 已安装"

# 创建/激活环境
echo ""
echo "[3/4] 创建 Conda 环境..."
ENV_NAME="longvu"

if conda env list | grep -q "^$ENV_NAME "; then
    echo "✓ 环境 $ENV_NAME 已存在"
else
    echo "正在创建环境 $ENV_NAME ..."
    conda create -n $ENV_NAME python=3.10 -y
fi

echo "激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME
echo "✓ 环境激活成功"

# 安装依赖
echo ""
echo "[4/4] 安装依赖包..."
if [ -f "requirements.txt" ]; then
    echo "正在安装 requirements.txt..."
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo "✓ 依赖安装完成"
else
    echo "❌ 未找到 requirements.txt"
    exit 1
fi

# 检查 GPU
echo ""
echo "============================================================"
echo "检查 GPU 支持..."
if nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "✓ 检测到 $GPU_COUNT 个 GPU: $GPU_NAME"
    
    # 测试 PyTorch CUDA
    python3 -c "import torch; print('✓ PyTorch CUDA 可用:', torch.cuda.is_available())"
else
    echo "⚠️  未检测到 NVIDIA GPU，将无法使用 CUDA 加速"
fi

echo ""
echo "============================================================"
echo "🎉 安装完成！"
echo "============================================================"
echo ""
echo "下一步操作："
echo "1. 下载模型权重到 ./checkpoints/ 目录"
echo "   cd ./checkpoints"
echo "   git lfs install"
echo "   git clone https://huggingface.co/Vision-CAIR/LongVU_Qwen2_7B"
echo ""
echo "2. 运行测试脚本"
echo "   conda activate longvu"
echo "   python test_inference.py"
echo ""
echo "详细说明请查看：REPRODUCTION_GUIDE.md"
echo "============================================================"
