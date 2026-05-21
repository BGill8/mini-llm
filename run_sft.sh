#!/bin/bash
#SBATCH -A eecs
#SBATCH -p eecs
#SBATCH --gres=gpu:1
#SBATCH --exclude=cn-gpu3
#SBATCH -c 4
#SBATCH --time=36:00:00

# Navigate to your specific mini-llm directory
cd ~/hpc-share/LMM-hw/mini-llm

# Activate your environment
source /nfs/stak/users/gillb3/hpc-share/Cell-Segmentation-Deep-Learning/.venv/bin/activate

# Set up HuggingFace cache locally to avoid home directory quota issues
mkdir -p data/cache
export HF_HOME=$PWD/data/cache
export HF_DATASETS_CACHE=$PWD/data/cache

# Environment and GPU Sanity Checks
echo "=========================================================="
hostname
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
nvidia-smi
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPUs Available: {torch.cuda.device_count()}')"
echo "=========================================================="

# Run the SFT script
# (Make sure 'chkpts/pretrain_85m_5000.pt' exists by the time you run this!)
python train_sft.py --checkpoint chkpts/pretrain_85m_5000.pt