PYTHONPATH has been set to /home/x_hensh/.local/lib/python3.10/site-packages:: 
Please do not modify PYTHONPATH while using this module. 
 
PYTHONPATH has been set to
/software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages:/home/x_hensh/.local/lib/python3.10/site-packages:: 
Please do not modify PYTHONPATH while using this module. 
 
/home/x_hensh/.local/lib/python3.10/site-packages/mmcv/__init__.py:20: UserWarning: On January 1, 2023, MMCV will release v2.0.0, in which it will remove components related to the training process and add a data transformation module. In addition, it will rename the package names mmcv to mmcv-lite and mmcv-full to mmcv. See https://github.com/open-mmlab/mmcv/blob/master/docs/en/compatibility.md for more details.
  warnings.warn(
/home/x_hensh/.local/lib/python3.10/site-packages/timm/models/layers/__init__.py:48: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
[92m02-03 13:17:11[0m Creating dataset...
[92m02-03 13:17:11[0m Creating graph and optimizer...
[92m02-03 13:17:30[0m Load checkpoint from ../pretrained_models/osx_l.pth.tar
[92m02-03 13:17:30[0m set gpu_ids to 0
[92m02-03 13:17:30[0m set num_gpus to 1
[92m02-03 13:17:30[0m set lr to 0.0001
[92m02-03 13:17:30[0m set debug to False
[92m02-03 13:17:30[0m set continue_train to True
[92m02-03 13:17:30[0m set exp_name to output/train_smil/
[92m02-03 13:17:30[0m set num_thread to 16
[92m02-03 13:17:30[0m set train_batch_size to 16
[92m02-03 13:17:30[0m set encoder_setting to osx_l
[92m02-03 13:17:30[0m set decoder_setting to normal
[92m02-03 13:17:30[0m set end_epoch to 140
[92m02-03 13:17:30[0m set pretrained_model_path to ../pretrained_models/osx_l.pth.tar
[92m02-03 13:17:30[0m set agora_benchmark to False
[92m02-03 13:17:30[0m set ubody_benchmark to False
[92m02-03 13:17:30[0m set ima_benchmark to True
[92m02-03 13:17:30[0m set model_type to smil_h
[92m02-03 13:17:30[0m set output_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_smil/
[92m02-03 13:17:30[0m set model_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_smil/model_dump
[92m02-03 13:17:30[0m set vis_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_smil/vis
[92m02-03 13:17:30[0m set log_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_smil/log
[92m02-03 13:17:30[0m set code_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_smil/code
[92m02-03 13:17:30[0m set result_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_smil/result
[92m02-03 13:17:30[0m set encoder_config_file to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../main/transformer_utils/configs/osx/encoder/body_encoder_large.py
[92m02-03 13:17:30[0m set encoder_pretrained_model_path to ../pretrained_models/osx_vit_l.pth
[92m02-03 13:17:30[0m set feat_dim to 1024
[92m02-03 13:17:30[0m set trainset_3d to []
[92m02-03 13:17:30[0m set trainset_2d to ['IMA']
[92m02-03 13:17:30[0m set testset to IMA
/home/x_hensh/.local/lib/python3.10/site-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3190.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
[92m02-03 13:18:39[0m Epoch 0/140 itr 99/271: lr: 9.99983e-05 speed: 0.58(0.58s r0.00)s/itr 0.04h/epoch loss_joint_proj: 0.3849
[92m02-03 13:19:37[0m Epoch 0/140 itr 199/271: lr: 9.99932e-05 speed: 0.58(0.58s r0.00)s/itr 0.04h/epoch loss_joint_proj: 0.2748
[92m02-03 13:20:26[0m Write snapshot into /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_smil/model_dump/snapshot_0.pth.tar
[92m02-03 13:21:27[0m Epoch 1/140 itr 99/271: lr: 9.99768e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.2398
[92m02-03 13:22:25[0m Epoch 1/140 itr 199/271: lr: 9.99625e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.3080
[92m02-03 13:24:06[0m Epoch 2/140 itr 99/271: lr: 9.99305e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.3104
[92m02-03 13:25:04[0m Epoch 2/140 itr 199/271: lr: 9.99071e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.2547
[92m02-03 13:26:46[0m Epoch 3/140 itr 99/271: lr: 9.98595e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.2054
[92m02-03 13:27:44[0m Epoch 3/140 itr 199/271: lr: 9.9827e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1873
[92m02-03 13:29:25[0m Epoch 4/140 itr 99/271: lr: 9.97639e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1834
[92m02-03 13:30:23[0m Epoch 4/140 itr 199/271: lr: 9.97222e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.2038
[92m02-03 13:32:04[0m Epoch 5/140 itr 99/271: lr: 9.96436e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.2094
[92m02-03 13:33:02[0m Epoch 5/140 itr 199/271: lr: 9.95929e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.2296
[92m02-03 13:34:43[0m Epoch 6/140 itr 99/271: lr: 9.94988e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1643
[92m02-03 13:35:41[0m Epoch 6/140 itr 199/271: lr: 9.9439e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.2200
[92m02-03 13:37:22[0m Epoch 7/140 itr 99/271: lr: 9.93295e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1890
[92m02-03 13:38:20[0m Epoch 7/140 itr 199/271: lr: 9.92606e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1987
[92m02-03 13:40:02[0m Epoch 8/140 itr 99/271: lr: 9.91358e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1582
[92m02-03 13:41:00[0m Epoch 8/140 itr 199/271: lr: 9.90578e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1788
[92m02-03 13:42:41[0m Epoch 9/140 itr 99/271: lr: 9.89177e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1551
[92m02-03 13:43:39[0m Epoch 9/140 itr 199/271: lr: 9.88308e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1579
[92m02-03 13:45:20[0m Epoch 10/140 itr 99/271: lr: 9.86755e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1757
[92m02-03 13:46:18[0m Epoch 10/140 itr 199/271: lr: 9.85797e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1897
[92m02-03 13:47:13[0m Write snapshot into /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_smil/model_dump/snapshot_10.pth.tar
[92m02-03 13:48:14[0m Epoch 11/140 itr 99/271: lr: 9.84092e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1709
[92m02-03 13:49:12[0m Epoch 11/140 itr 199/271: lr: 9.83045e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1416
[92m02-03 13:50:54[0m Epoch 12/140 itr 99/271: lr: 9.81189e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1857
[92m02-03 13:51:52[0m Epoch 12/140 itr 199/271: lr: 9.80054e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1810
[92m02-03 13:53:33[0m Epoch 13/140 itr 99/271: lr: 9.78049e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1640
[92m02-03 13:54:31[0m Epoch 13/140 itr 199/271: lr: 9.76826e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1584
[92m02-03 13:56:12[0m Epoch 14/140 itr 99/271: lr: 9.74672e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1301
[92m02-03 13:57:10[0m Epoch 14/140 itr 199/271: lr: 9.73361e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1473
[92m02-03 13:58:52[0m Epoch 15/140 itr 99/271: lr: 9.7106e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1624
[92m02-03 13:59:50[0m Epoch 15/140 itr 199/271: lr: 9.69663e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1472
[92m02-03 14:01:31[0m Epoch 16/140 itr 99/271: lr: 9.67215e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1432
[92m02-03 14:02:29[0m Epoch 16/140 itr 199/271: lr: 9.65733e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1624
[92m02-03 14:04:10[0m Epoch 17/140 itr 99/271: lr: 9.6314e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1465
[92m02-03 14:05:08[0m Epoch 17/140 itr 199/271: lr: 9.61572e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1191
[92m02-03 14:06:49[0m Epoch 18/140 itr 99/271: lr: 9.58835e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1846
[92m02-03 14:07:47[0m Epoch 18/140 itr 199/271: lr: 9.57183e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1632
[92m02-03 14:09:29[0m Epoch 19/140 itr 99/271: lr: 9.54303e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1533
[92m02-03 14:10:27[0m Epoch 19/140 itr 199/271: lr: 9.52568e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1492
[92m02-03 14:12:08[0m Epoch 20/140 itr 99/271: lr: 9.49547e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1678
[92m02-03 14:13:06[0m Epoch 20/140 itr 199/271: lr: 9.47729e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1242
[92m02-03 14:13:56[0m Write snapshot into /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_smil/model_dump/snapshot_20.pth.tar
[92m02-03 14:14:57[0m Epoch 21/140 itr 99/271: lr: 9.44569e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1750
[92m02-03 14:15:55[0m Epoch 21/140 itr 199/271: lr: 9.42669e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1431
[92m02-03 14:17:36[0m Epoch 22/140 itr 99/271: lr: 9.39371e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1493
[92m02-03 14:18:34[0m Epoch 22/140 itr 199/271: lr: 9.3739e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1594
[92m02-03 14:20:15[0m Epoch 23/140 itr 99/271: lr: 9.33956e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1790
[92m02-03 14:21:13[0m Epoch 23/140 itr 199/271: lr: 9.31895e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1148
[92m02-03 14:22:55[0m Epoch 24/140 itr 99/271: lr: 9.28326e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1808
[92m02-03 14:23:53[0m Epoch 24/140 itr 199/271: lr: 9.26187e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1571
[92m02-03 14:25:34[0m Epoch 25/140 itr 99/271: lr: 9.22485e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1371
[92m02-03 14:26:32[0m Epoch 25/140 itr 199/271: lr: 9.20268e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1357
[92m02-03 14:28:14[0m Epoch 26/140 itr 99/271: lr: 9.16435e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1523
[92m02-03 14:29:12[0m Epoch 26/140 itr 199/271: lr: 9.14142e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1171
[92m02-03 14:30:53[0m Epoch 27/140 itr 99/271: lr: 9.1018e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1587
[92m02-03 14:31:51[0m Epoch 27/140 itr 199/271: lr: 9.07811e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1532
[92m02-03 14:33:32[0m Epoch 28/140 itr 99/271: lr: 9.03722e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1356
[92m02-03 14:34:31[0m Epoch 28/140 itr 199/271: lr: 9.01279e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1342
[92m02-03 14:36:12[0m Epoch 29/140 itr 99/271: lr: 8.97064e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1146
[92m02-03 14:37:10[0m Epoch 29/140 itr 199/271: lr: 8.94549e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1437
[92m02-03 14:38:52[0m Epoch 30/140 itr 99/271: lr: 8.90211e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1463
[92m02-03 14:39:50[0m Epoch 30/140 itr 199/271: lr: 8.87624e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1620
[92m02-03 14:40:39[0m Write snapshot into /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_smil/model_dump/snapshot_30.pth.tar
[92m02-03 14:41:40[0m Epoch 31/140 itr 99/271: lr: 8.83165e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1450
[92m02-03 14:42:38[0m Epoch 31/140 itr 199/271: lr: 8.80508e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1320
[92m02-03 14:44:19[0m Epoch 32/140 itr 99/271: lr: 8.75931e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1351
[92m02-03 14:45:17[0m Epoch 32/140 itr 199/271: lr: 8.73204e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1440
[92m02-03 14:46:58[0m Epoch 33/140 itr 99/271: lr: 8.68511e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1237
[92m02-03 14:47:56[0m Epoch 33/140 itr 199/271: lr: 8.65716e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1299
[92m02-03 14:49:37[0m Epoch 34/140 itr 99/271: lr: 8.60909e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1393
[92m02-03 14:50:35[0m Epoch 34/140 itr 199/271: lr: 8.58048e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1213
[92m02-03 14:52:16[0m Epoch 35/140 itr 99/271: lr: 8.53129e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1338
[92m02-03 14:53:15[0m Epoch 35/140 itr 199/271: lr: 8.50203e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1173
[92m02-03 14:54:56[0m Epoch 36/140 itr 99/271: lr: 8.45175e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1385
[92m02-03 14:55:54[0m Epoch 36/140 itr 199/271: lr: 8.42186e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1331
[92m02-03 14:57:35[0m Epoch 37/140 itr 99/271: lr: 8.37051e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1474
[92m02-03 14:58:33[0m Epoch 37/140 itr 199/271: lr: 8.34e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1277
[92m02-03 15:00:15[0m Epoch 38/140 itr 99/271: lr: 8.28762e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1252
[92m02-03 15:01:13[0m Epoch 38/140 itr 199/271: lr: 8.2565e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1365
[92m02-03 15:02:54[0m Epoch 39/140 itr 99/271: lr: 8.2031e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1196
[92m02-03 15:03:52[0m Epoch 39/140 itr 199/271: lr: 8.1714e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1259
[92m02-03 15:05:33[0m Epoch 40/140 itr 99/271: lr: 8.11701e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1353
[92m02-03 15:06:31[0m Epoch 40/140 itr 199/271: lr: 8.08473e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1291
[92m02-03 15:07:20[0m Write snapshot into /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_smil/model_dump/snapshot_40.pth.tar
[92m02-03 15:08:21[0m Epoch 41/140 itr 99/271: lr: 8.02938e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1697
[92m02-03 15:09:19[0m Epoch 41/140 itr 199/271: lr: 7.99655e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.2398
[92m02-03 15:10:59[0m Epoch 42/140 itr 99/271: lr: 7.94027e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1171
[92m02-03 15:11:57[0m Epoch 42/140 itr 199/271: lr: 7.9069e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0987
[92m02-03 15:13:38[0m Epoch 43/140 itr 99/271: lr: 7.84971e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1066
[92m02-03 15:14:36[0m Epoch 43/140 itr 199/271: lr: 7.81581e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1360
[92m02-03 15:16:17[0m Epoch 44/140 itr 99/271: lr: 7.75775e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1138
[92m02-03 15:17:15[0m Epoch 44/140 itr 199/271: lr: 7.72335e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1081
[92m02-03 15:18:57[0m Epoch 45/140 itr 99/271: lr: 7.66444e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1038
[92m02-03 15:19:55[0m Epoch 45/140 itr 199/271: lr: 7.62955e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1166
[92m02-03 15:21:37[0m Epoch 46/140 itr 99/271: lr: 7.56983e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0978
[92m02-03 15:22:36[0m Epoch 46/140 itr 199/271: lr: 7.53446e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1223
[92m02-03 15:24:17[0m Epoch 47/140 itr 99/271: lr: 7.47395e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0937
[92m02-03 15:25:15[0m Epoch 47/140 itr 199/271: lr: 7.43813e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1111
[92m02-03 15:26:57[0m Epoch 48/140 itr 99/271: lr: 7.37686e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1213
[92m02-03 15:27:55[0m Epoch 48/140 itr 199/271: lr: 7.3406e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1069
[92m02-03 15:29:37[0m Epoch 49/140 itr 99/271: lr: 7.27861e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1194
[92m02-03 15:30:35[0m Epoch 49/140 itr 199/271: lr: 7.24193e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0944
[92m02-03 15:32:16[0m Epoch 50/140 itr 99/271: lr: 7.17924e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0936
[92m02-03 15:33:14[0m Epoch 50/140 itr 199/271: lr: 7.14217e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1107
[92m02-03 15:34:05[0m Write snapshot into /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_smil/model_dump/snapshot_50.pth.tar
[92m02-03 15:35:06[0m Epoch 51/140 itr 99/271: lr: 7.07881e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0992
[92m02-03 15:36:04[0m Epoch 51/140 itr 199/271: lr: 7.04136e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0849
[92m02-03 15:37:46[0m Epoch 52/140 itr 99/271: lr: 6.97737e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1119
[92m02-03 15:38:44[0m Epoch 52/140 itr 199/271: lr: 6.93955e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1086
[92m02-03 15:40:25[0m Epoch 53/140 itr 99/271: lr: 6.87496e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0856
[92m02-03 15:41:23[0m Epoch 53/140 itr 199/271: lr: 6.8368e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1052
[92m02-03 15:43:04[0m Epoch 54/140 itr 99/271: lr: 6.77164e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1204
[92m02-03 15:44:02[0m Epoch 54/140 itr 199/271: lr: 6.73315e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1061
[92m02-03 15:45:43[0m Epoch 55/140 itr 99/271: lr: 6.66746e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1015
[92m02-03 15:46:41[0m Epoch 55/140 itr 199/271: lr: 6.62867e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0974
[92m02-03 15:48:22[0m Epoch 56/140 itr 99/271: lr: 6.56247e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1191
[92m02-03 15:49:20[0m Epoch 56/140 itr 199/271: lr: 6.52339e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1201
[92m02-03 15:51:01[0m Epoch 57/140 itr 99/271: lr: 6.45673e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0891
[92m02-03 15:51:59[0m Epoch 57/140 itr 199/271: lr: 6.41738e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1168
[92m02-03 15:53:41[0m Epoch 58/140 itr 99/271: lr: 6.35028e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1220
[92m02-03 15:54:39[0m Epoch 58/140 itr 199/271: lr: 6.31069e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1111
[92m02-03 15:56:20[0m Epoch 59/140 itr 99/271: lr: 6.24318e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1126
[92m02-03 15:57:19[0m Epoch 59/140 itr 199/271: lr: 6.20336e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0922
[92m02-03 15:59:00[0m Epoch 60/140 itr 99/271: lr: 6.13549e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1118
[92m02-03 15:59:58[0m Epoch 60/140 itr 199/271: lr: 6.09546e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0937
[92m02-03 16:00:48[0m Write snapshot into /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_smil/model_dump/snapshot_60.pth.tar
[92m02-03 16:01:49[0m Epoch 61/140 itr 99/271: lr: 6.02725e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0983
[92m02-03 16:02:46[0m Epoch 61/140 itr 199/271: lr: 5.98704e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0806
[92m02-03 16:04:28[0m Epoch 62/140 itr 99/271: lr: 5.91853e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1023
[92m02-03 16:05:26[0m Epoch 62/140 itr 199/271: lr: 5.87815e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0965
[92m02-03 16:07:07[0m Epoch 63/140 itr 99/271: lr: 5.80937e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0975
[92m02-03 16:08:05[0m Epoch 63/140 itr 199/271: lr: 5.76884e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0962
[92m02-03 16:09:47[0m Epoch 64/140 itr 99/271: lr: 5.69983e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1119
[92m02-03 16:10:45[0m Epoch 64/140 itr 199/271: lr: 5.65917e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0912
[92m02-03 16:12:26[0m Epoch 65/140 itr 99/271: lr: 5.58997e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0885
[92m02-03 16:13:24[0m Epoch 65/140 itr 199/271: lr: 5.5492e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1177
[92m02-03 16:15:05[0m Epoch 66/140 itr 99/271: lr: 5.47983e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0885
[92m02-03 16:16:03[0m Epoch 66/140 itr 199/271: lr: 5.43899e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0872
[92m02-03 16:17:45[0m Epoch 67/140 itr 99/271: lr: 5.36948e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1006
[92m02-03 16:18:44[0m Epoch 67/140 itr 199/271: lr: 5.32857e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0833
[92m02-03 16:20:25[0m Epoch 68/140 itr 99/271: lr: 5.25898e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0995
[92m02-03 16:21:23[0m Epoch 68/140 itr 199/271: lr: 5.21802e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0911
[92m02-03 16:23:04[0m Epoch 69/140 itr 99/271: lr: 5.14836e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1161
[92m02-03 16:24:02[0m Epoch 69/140 itr 199/271: lr: 5.10738e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1030
[92m02-03 16:25:43[0m Epoch 70/140 itr 99/271: lr: 5.0377e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1054
[92m02-03 16:26:41[0m Epoch 70/140 itr 199/271: lr: 4.99672e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1037
[92m02-03 16:27:31[0m Write snapshot into /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_smil/model_dump/snapshot_70.pth.tar
[92m02-03 16:28:32[0m Epoch 71/140 itr 99/271: lr: 4.92705e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1098
[92m02-03 16:29:30[0m Epoch 71/140 itr 199/271: lr: 4.88608e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0771
[92m02-03 16:31:11[0m Epoch 72/140 itr 99/271: lr: 4.81645e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1137
[92m02-03 16:32:09[0m Epoch 72/140 itr 199/271: lr: 4.77552e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0904
[92m02-03 16:33:50[0m Epoch 73/140 itr 99/271: lr: 4.70598e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0922
[92m02-03 16:34:48[0m Epoch 73/140 itr 199/271: lr: 4.6651e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1049
[92m02-03 16:36:30[0m Epoch 74/140 itr 99/271: lr: 4.59567e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0950
[92m02-03 16:37:28[0m Epoch 74/140 itr 199/271: lr: 4.55487e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0892
[92m02-03 16:39:09[0m Epoch 75/140 itr 99/271: lr: 4.48559e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0743
[92m02-03 16:40:07[0m Epoch 75/140 itr 199/271: lr: 4.44489e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0894
[92m02-03 16:41:48[0m Epoch 76/140 itr 99/271: lr: 4.3758e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0928
[92m02-03 16:42:47[0m Epoch 76/140 itr 199/271: lr: 4.33522e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0903
[92m02-03 16:44:28[0m Epoch 77/140 itr 99/271: lr: 4.26634e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0979
[92m02-03 16:45:27[0m Epoch 77/140 itr 199/271: lr: 4.2259e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0910
[92m02-03 16:47:09[0m Epoch 78/140 itr 99/271: lr: 4.15727e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0708
[92m02-03 16:48:07[0m Epoch 78/140 itr 199/271: lr: 4.11699e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1034
[92m02-03 16:49:48[0m Epoch 79/140 itr 99/271: lr: 4.04865e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0848
[92m02-03 16:50:46[0m Epoch 79/140 itr 199/271: lr: 4.00855e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1012
[92m02-03 16:52:28[0m Epoch 80/140 itr 99/271: lr: 3.94053e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0913
[92m02-03 16:53:25[0m Epoch 80/140 itr 199/271: lr: 3.90062e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0912
[92m02-03 16:54:15[0m Write snapshot into /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_smil/model_dump/snapshot_80.pth.tar
[92m02-03 16:55:15[0m Epoch 81/140 itr 99/271: lr: 3.83296e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1019
[92m02-03 16:56:13[0m Epoch 81/140 itr 199/271: lr: 3.79328e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0816
[92m02-03 16:57:54[0m Epoch 82/140 itr 99/271: lr: 3.72601e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0876
[92m02-03 16:58:53[0m Epoch 82/140 itr 199/271: lr: 3.68656e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1021
[92m02-03 17:00:34[0m Epoch 83/140 itr 99/271: lr: 3.61971e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0952
[92m02-03 17:01:32[0m Epoch 83/140 itr 199/271: lr: 3.58052e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0904
[92m02-03 17:03:13[0m Epoch 84/140 itr 99/271: lr: 3.51413e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0803
[92m02-03 17:04:11[0m Epoch 84/140 itr 199/271: lr: 3.47522e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0997
[92m02-03 17:05:52[0m Epoch 85/140 itr 99/271: lr: 3.40932e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0880
[92m02-03 17:06:50[0m Epoch 85/140 itr 199/271: lr: 3.3707e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0805
[92m02-03 17:08:32[0m Epoch 86/140 itr 99/271: lr: 3.30532e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0941
[92m02-03 17:09:30[0m Epoch 86/140 itr 199/271: lr: 3.26703e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0733
[92m02-03 17:11:13[0m Epoch 87/140 itr 99/271: lr: 3.2022e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0911
[92m02-03 17:12:11[0m Epoch 87/140 itr 199/271: lr: 3.16424e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0801
[92m02-03 17:13:52[0m Epoch 88/140 itr 99/271: lr: 3.1e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0918
[92m02-03 17:14:50[0m Epoch 88/140 itr 199/271: lr: 3.0624e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0839
[92m02-03 17:16:32[0m Epoch 89/140 itr 99/271: lr: 2.99878e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0976
[92m02-03 17:17:30[0m Epoch 89/140 itr 199/271: lr: 2.96155e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0852
[92m02-03 17:19:11[0m Epoch 90/140 itr 99/271: lr: 2.89858e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0762
[92m02-03 17:20:10[0m Epoch 90/140 itr 199/271: lr: 2.86174e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0827
[92m02-03 17:20:59[0m Write snapshot into /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_smil/model_dump/snapshot_90.pth.tar
[92m02-03 17:22:00[0m Epoch 91/140 itr 99/271: lr: 2.79946e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0688
[92m02-03 17:22:58[0m Epoch 91/140 itr 199/271: lr: 2.76303e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0779
[92m02-03 17:24:39[0m Epoch 92/140 itr 99/271: lr: 2.70146e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0874
[92m02-03 17:25:37[0m Epoch 92/140 itr 199/271: lr: 2.66546e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0846
[92m02-03 17:27:19[0m Epoch 93/140 itr 99/271: lr: 2.60464e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0758
[92m02-03 17:28:17[0m Epoch 93/140 itr 199/271: lr: 2.56909e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0764
[92m02-03 17:29:59[0m Epoch 94/140 itr 99/271: lr: 2.50904e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0708
[92m02-03 17:30:57[0m Epoch 94/140 itr 199/271: lr: 2.47395e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0832
[92m02-03 17:32:39[0m Epoch 95/140 itr 99/271: lr: 2.41471e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.1050
[92m02-03 17:33:37[0m Epoch 95/140 itr 199/271: lr: 2.3801e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0865
[92m02-03 17:35:19[0m Epoch 96/140 itr 99/271: lr: 2.32169e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0669
[92m02-03 17:36:17[0m Epoch 96/140 itr 199/271: lr: 2.28759e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0792
[92m02-03 17:37:58[0m Epoch 97/140 itr 99/271: lr: 2.23004e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0803
[92m02-03 17:38:57[0m Epoch 97/140 itr 199/271: lr: 2.19645e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0978
[92m02-03 17:40:38[0m Epoch 98/140 itr 99/271: lr: 2.1398e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0687
[92m02-03 17:41:36[0m Epoch 98/140 itr 199/271: lr: 2.10674e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0838
[92m02-03 17:43:19[0m Epoch 99/140 itr 99/271: lr: 2.05101e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0767
[92m02-03 17:44:17[0m Epoch 99/140 itr 199/271: lr: 2.01851e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0674
[92m02-03 17:45:59[0m Epoch 100/140 itr 99/271: lr: 1.96373e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0777
[92m02-03 17:46:57[0m Epoch 100/140 itr 199/271: lr: 1.93179e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0830
[92m02-03 17:47:47[0m Write snapshot into /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_smil/model_dump/snapshot_100.pth.tar
[92m02-03 17:48:48[0m Epoch 101/140 itr 99/271: lr: 1.87798e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0837
[92m02-03 17:49:46[0m Epoch 101/140 itr 199/271: lr: 1.84662e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0740
[92m02-03 17:51:28[0m Epoch 102/140 itr 99/271: lr: 1.79382e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0582
[92m02-03 17:52:27[0m Epoch 102/140 itr 199/271: lr: 1.76306e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0772
[92m02-03 17:54:08[0m Epoch 103/140 itr 99/271: lr: 1.71129e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0798
[92m02-03 17:55:06[0m Epoch 103/140 itr 199/271: lr: 1.68114e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0797
[92m02-03 17:56:48[0m Epoch 104/140 itr 99/271: lr: 1.63043e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0731
[92m02-03 17:57:46[0m Epoch 104/140 itr 199/271: lr: 1.60091e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0766
[92m02-03 17:59:28[0m Epoch 105/140 itr 99/271: lr: 1.55127e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0823
[92m02-03 18:00:25[0m Epoch 105/140 itr 199/271: lr: 1.5224e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0861
[92m02-03 18:02:07[0m Epoch 106/140 itr 99/271: lr: 1.47387e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0621
[92m02-03 18:03:05[0m Epoch 106/140 itr 199/271: lr: 1.44565e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0772
[92m02-03 18:04:47[0m Epoch 107/140 itr 99/271: lr: 1.39825e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0673
[92m02-03 18:05:45[0m Epoch 107/140 itr 199/271: lr: 1.3707e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0680
[92m02-03 18:07:27[0m Epoch 108/140 itr 99/271: lr: 1.32445e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0840
[92m02-03 18:08:25[0m Epoch 108/140 itr 199/271: lr: 1.29759e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0579
[92m02-03 18:10:07[0m Epoch 109/140 itr 99/271: lr: 1.25252e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0954
[92m02-03 18:11:05[0m Epoch 109/140 itr 199/271: lr: 1.22636e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0687
[92m02-03 18:12:47[0m Epoch 110/140 itr 99/271: lr: 1.18249e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0707
[92m02-03 18:13:45[0m Epoch 110/140 itr 199/271: lr: 1.15704e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0708
[92m02-03 18:14:35[0m Write snapshot into /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_smil/model_dump/snapshot_110.pth.tar
[92m02-03 18:15:35[0m Epoch 111/140 itr 99/271: lr: 1.11439e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0656
[92m02-03 18:16:34[0m Epoch 111/140 itr 199/271: lr: 1.08967e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0777
[92m02-03 18:18:16[0m Epoch 112/140 itr 99/271: lr: 1.04826e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0690
[92m02-03 18:19:14[0m Epoch 112/140 itr 199/271: lr: 1.02427e-05 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0744
[92m02-03 18:20:56[0m Epoch 113/140 itr 99/271: lr: 9.84127e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0666
[92m02-03 18:21:54[0m Epoch 113/140 itr 199/271: lr: 9.60888e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0675
[92m02-03 18:23:36[0m Epoch 114/140 itr 99/271: lr: 9.22026e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0832
[92m02-03 18:24:34[0m Epoch 114/140 itr 199/271: lr: 8.99548e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0715
[92m02-03 18:26:15[0m Epoch 115/140 itr 99/271: lr: 8.61989e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0657
[92m02-03 18:27:13[0m Epoch 115/140 itr 199/271: lr: 8.40283e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0616
[92m02-03 18:28:55[0m Epoch 116/140 itr 99/271: lr: 8.04046e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0791
[92m02-03 18:29:53[0m Epoch 116/140 itr 199/271: lr: 7.83122e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0606
[92m02-03 18:31:35[0m Epoch 117/140 itr 99/271: lr: 7.48224e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0756
[92m02-03 18:32:33[0m Epoch 117/140 itr 199/271: lr: 7.28094e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0697
[92m02-03 18:34:14[0m Epoch 118/140 itr 99/271: lr: 6.94553e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0728
[92m02-03 18:35:12[0m Epoch 118/140 itr 199/271: lr: 6.75226e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0571
[92m02-03 18:36:53[0m Epoch 119/140 itr 99/271: lr: 6.43059e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0715
[92m02-03 18:37:51[0m Epoch 119/140 itr 199/271: lr: 6.24544e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0636
[92m02-03 18:39:32[0m Epoch 120/140 itr 99/271: lr: 5.93767e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0637
[92m02-03 18:40:30[0m Epoch 120/140 itr 199/271: lr: 5.76075e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0576
[92m02-03 18:41:22[0m Write snapshot into /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_smil/model_dump/snapshot_120.pth.tar
[92m02-03 18:42:23[0m Epoch 121/140 itr 99/271: lr: 5.46703e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0490
[92m02-03 18:43:21[0m Epoch 121/140 itr 199/271: lr: 5.29841e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0714
[92m02-03 18:45:03[0m Epoch 122/140 itr 99/271: lr: 5.01889e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0739
[92m02-03 18:46:01[0m Epoch 122/140 itr 199/271: lr: 4.85867e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0657
[92m02-03 18:47:42[0m Epoch 123/140 itr 99/271: lr: 4.59349e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0709
[92m02-03 18:48:40[0m Epoch 123/140 itr 199/271: lr: 4.44174e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0731
[92m02-03 18:50:22[0m Epoch 124/140 itr 99/271: lr: 4.19103e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0567
[92m02-03 18:51:20[0m Epoch 124/140 itr 199/271: lr: 4.04784e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0649
[92m02-03 18:53:02[0m Epoch 125/140 itr 99/271: lr: 3.81172e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0751
[92m02-03 18:54:00[0m Epoch 125/140 itr 199/271: lr: 3.67715e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0657
[92m02-03 18:55:41[0m Epoch 126/140 itr 99/271: lr: 3.45574e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0817
[92m02-03 18:56:39[0m Epoch 126/140 itr 199/271: lr: 3.32986e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0540
[92m02-03 18:58:20[0m Epoch 127/140 itr 99/271: lr: 3.12328e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0609
[92m02-03 18:59:18[0m Epoch 127/140 itr 199/271: lr: 3.00615e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0654
[92m02-03 19:01:00[0m Epoch 128/140 itr 99/271: lr: 2.8145e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0557
[92m02-03 19:01:59[0m Epoch 128/140 itr 199/271: lr: 2.70618e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0734
[92m02-03 19:03:41[0m Epoch 129/140 itr 99/271: lr: 2.52955e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0714
[92m02-03 19:04:39[0m Epoch 129/140 itr 199/271: lr: 2.43009e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0610
[92m02-03 19:06:21[0m Epoch 130/140 itr 99/271: lr: 2.26858e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0708
[92m02-03 19:07:19[0m Epoch 130/140 itr 199/271: lr: 2.17803e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0657
[92m02-03 19:08:14[0m Write snapshot into /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_smil/model_dump/snapshot_130.pth.tar
[92m02-03 19:09:15[0m Epoch 131/140 itr 99/271: lr: 2.03172e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0540
[92m02-03 19:10:14[0m Epoch 131/140 itr 199/271: lr: 1.95013e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0779
[92m02-03 19:11:56[0m Epoch 132/140 itr 99/271: lr: 1.81908e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0740
[92m02-03 19:12:54[0m Epoch 132/140 itr 199/271: lr: 1.74649e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0736
[92m02-03 19:14:35[0m Epoch 133/140 itr 99/271: lr: 1.63077e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0600
[92m02-03 19:15:34[0m Epoch 133/140 itr 199/271: lr: 1.56722e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0753
[92m02-03 19:17:15[0m Epoch 134/140 itr 99/271: lr: 1.46689e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0564
[92m02-03 19:18:14[0m Epoch 134/140 itr 199/271: lr: 1.41241e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0657
[92m02-03 19:19:56[0m Epoch 135/140 itr 99/271: lr: 1.32751e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0583
[92m02-03 19:20:55[0m Epoch 135/140 itr 199/271: lr: 1.28213e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0514
[92m02-03 19:22:37[0m Epoch 136/140 itr 99/271: lr: 1.21272e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0872
[92m02-03 19:23:35[0m Epoch 136/140 itr 199/271: lr: 1.17645e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0912
[92m02-03 19:25:17[0m Epoch 137/140 itr 99/271: lr: 1.12256e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0637
[92m02-03 19:26:16[0m Epoch 137/140 itr 199/271: lr: 1.09543e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0773
[92m02-03 19:27:57[0m Epoch 138/140 itr 99/271: lr: 1.05708e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0675
[92m02-03 19:28:56[0m Epoch 138/140 itr 199/271: lr: 1.03909e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0678
[92m02-03 19:30:38[0m Epoch 139/140 itr 99/271: lr: 1.01631e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0681
[92m02-03 19:31:36[0m Epoch 139/140 itr 199/271: lr: 1.00748e-06 speed: 0.59(0.58s r0.01)s/itr 0.04h/epoch loss_joint_proj: 0.0616
[92m02-03 19:32:26[0m Write snapshot into /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_smil/model_dump/snapshot_139.pth.tar
