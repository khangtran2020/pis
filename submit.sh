#!/bin/bash -l

#SBATCH -J PIS  #job name
#SBATCH -o results/7b-gen-ncc-8.out
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
data="synthetic-piss"
r=8
perc=50

python main.py --pname "7b-syn50-8" \
    --data $data \
    --model $model \
    --lora_r $r \
    --pperc $perc \
    --dout 0.1 \
    --epochs 1 \
    --bs 2

# def get_cuda_runtime_lib_paths(candidate_paths: Set[Path]) -> Set[Path]:
#     paths = set()
#     for libname in CUDA_RUNTIME_LIBS:
#         for path in candidate_paths:
#             try:
#                 if (path / libname).is_file():
#                     paths.add(path / libname)
#             except PermissionError:
#                 continue
#     return paths