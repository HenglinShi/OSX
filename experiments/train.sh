#!/bin/bash
#SBATCH -A berzelius-2024-331
#SBATCH --gpus=1
#SBATCH -t 3-00:00:00
#SBATCH -e train_smil.e
#SBATCH -o train_smil.o

module load PyTorch
export LD_LIBRARY_PATH=/software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/:$LD_LIBRARY_PATH
export PYTHONPATH=/home/x_hensh/.local/lib/python3.10/site-packages:$PPYTHONPATH


python ../main/train.py \
            --gpu 0 \
            --lr 1e-4 \
            --exp_name output/train_smil/ \
            --end_epoch 140 \
            --pretrained_model_path ../pretrained_models/osx_l.pth.tar \
            --ima_benchmark \
            --train_batch_size 16 \
            --continue \
            --decoder_setting normal \
            --model_type smil_h