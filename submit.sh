#!/bin/bash -l

#SBATCH -J PIS  #job name
#SBATCH -o results/7b-gen-ncc.out
#SBATCH -p gpu-all      #queue used
#SBATCH --gres gpu:1    #number of gpus needed, default is 0
#SBATCH -c 1            #number of CPUs needed, default is 1
#SBATCH --mem=32G    #amount of memory needed, default is 4096 MB per core

#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=tkhang@hbku.edu.qa

module load cuda11.8/toolkit/11.8.0
conda activate pis

model="7b"
data="python-piss-my-name"
r=16

python main.py --pname "7b-gen-ncc" \
    --data $data \
    --model $model \
    --lora_r $r \
    --dout 0.1 \
    --epochs 20 \
    --bs 2

