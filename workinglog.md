```
module load PyTorch/2.3.0-python-3.10-hpc1
export PYTHONPATH=/home/x_hensh/.local/lib/python3.10/site-packages:$PPYTHONPATH
export LD_LIBRARY_PATH=/software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/:$LD_LIBRARY_PATH
#export LIBRARY_PATH=/software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/:$LIBRARY_PATH
```         




```
python test.py --gpu 0 --exp_name output/train_setting1/ --pretrained_model_path ../output/train_setting1/model_dump/snapshot_13.pth.tar --testset EHF

python test.py --gpu 0 --exp_name output/train_setting2/ --pretrained_model_path ../output/train_setting2/model_dump/snapshot_139.pth.tar --testset AGORA --agora_benchmark --test_batch_size 64 --decoder_setting wo_decoder

```


python train.py --gpu 0 --lr 1e-4 --exp_name output/train_test/ --end_epoch 140 --pretrained_model_path ../pretrained_models/osx_l.pth.tar  --ima_benchmark --train_batch_size 16 --decoder_setting normal --model_type smil_h --continue 

python demo.py --gpu 0 --img_path ../../../data/Youtube-Infant-Body-Parsing/frames/1000107000376.jpg --pretrained_model_path  ../pretrained_models/osx_l.pth.tar --decoder_setting normal --model_type smil_h --output_folder ../output/lu