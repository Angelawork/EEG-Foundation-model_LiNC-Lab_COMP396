#!/bin/bash

#SBATCH --job-name=EntireDataset_eval
#SBATCH --output=out/EntireDataset_eval_%j.out
#SBATCH --time=100:00:00
#SBATCH --gpus-per-task=rtx8000:1
#SBATCH --cpus-per-task=6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=30G

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# Load any modules and activate your Python environment here
module load python/3.10 
# module load cuda/12.3.2/cudnn/8.9
# module load python/3.9
 

cd /home/mila/q/qingchen.hu/EEG_comp396/EEG-Foundation-model_LiNC-Lab_COMP396

# install or activate requirements
if ! [ -d "$SLURM_TMPDIR/env/" ]; then
    virtualenv $SLURM_TMPDIR/env/
    source $SLURM_TMPDIR/env/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source $SLURM_TMPDIR/env/bin/activate
fi

export TF_ENABLE_ONEDNN_OPTS=0
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/extras/CUPTI/lib64
# echo $LD_LIBRARY_PATH

python basic_benchmark.py