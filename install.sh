#!/bin/bash
#pip install openmim --user
#mim install mmcv-full==1.7.1 --user
pip install -r requirements.txt --user
cd main/transformer_utils && python setup.py install --user
#conda install -y -c conda-forge ffmpeg