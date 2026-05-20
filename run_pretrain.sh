# run_pretrain.sh
#!/bin/bash
#SBATCH -A eecs
#SBATCH -p eecs
#SBATCH --gres=gpu:1
#SBATCH -c 8

export HF_HOME=~/hpc-share/AI539_S2026/A3/data/cache
cd ~/hpc-share/AI539_S2026/A3
source ../env/bin/activate
python train_pre.py