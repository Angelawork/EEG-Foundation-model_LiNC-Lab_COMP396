#!/bin/bash

#SBATCH --job-name=EEG_benchmark
#SBATCH --output=out/EEG_benchmark_%j.out
#SBATCH --error=err/EEG_benchmark_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
 #SBATCH --gpus-per-task=rtx8000:1
 #SBATCH --cpus-per-task=6
 #SBATCH --ntasks-per-node=1
#SBATCH --mem=30G

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# Load any modules and activate your Python environment here
module load python/3.10 

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

# mkdir -p $SLURM_TMPDIR/mne_data
# export MNE_DATA='/network/scratch/q/qingchen.hu/mne_data'
# echo $MNE_DATA
python main.py