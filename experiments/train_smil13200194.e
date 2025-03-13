PYTHONPATH has been set to
/software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages 
Please do not modify PYTHONPATH while using this module. 
 
/home/x_hensh/.local/lib/python3.10/site-packages/mmcv/__init__.py:20: UserWarning: On January 1, 2023, MMCV will release v2.0.0, in which it will remove components related to the training process and add a data transformation module. In addition, it will rename the package names mmcv to mmcv-lite and mmcv-full to mmcv. See https://github.com/open-mmlab/mmcv/blob/master/docs/en/compatibility.md for more details.
  warnings.warn(
/home/x_hensh/.local/lib/python3.10/site-packages/timm/models/layers/__init__.py:48: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
[92m03-13 00:21:15[0m Creating dataset...
[92m03-13 00:21:15[0m Creating graph and optimizer...
[92m03-13 00:21:24[0m Load checkpoint from ../pretrained_models/osx_l.pth.tar
[92m03-13 00:21:24[0m set lr to 0.0001
[92m03-13 00:21:24[0m set debug to False
[92m03-13 00:21:24[0m set continue_train to True
[92m03-13 00:21:24[0m set device to cuda
[92m03-13 00:21:24[0m set gpu_ids to ['0']
[92m03-13 00:21:24[0m set exp_name to output/train_kpt1dsr1_p3drender_2/
[92m03-13 00:21:24[0m set num_thread to 16
[92m03-13 00:21:24[0m set train_batch_size to 16
[92m03-13 00:21:24[0m set encoder_setting to osx_l
[92m03-13 00:21:24[0m set decoder_setting to normal
[92m03-13 00:21:24[0m set end_epoch to 140
[92m03-13 00:21:24[0m set pretrained_model_path to ../pretrained_models/osx_l.pth.tar
[92m03-13 00:21:24[0m set agora_benchmark to False
[92m03-13 00:21:24[0m set ubody_benchmark to False
[92m03-13 00:21:24[0m set ima_benchmark to True
[92m03-13 00:21:24[0m set model_type to smil_h
[92m03-13 00:21:24[0m set output_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_2/
[92m03-13 00:21:24[0m set model_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_2/model_dump
[92m03-13 00:21:24[0m set vis_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_2/vis
[92m03-13 00:21:24[0m set log_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_2/log
[92m03-13 00:21:24[0m set code_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_2/code
[92m03-13 00:21:24[0m set result_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_2/result
[92m03-13 00:21:24[0m set encoder_config_file to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../main/transformer_utils/configs/osx/encoder/body_encoder_large.py
[92m03-13 00:21:24[0m set encoder_pretrained_model_path to ../pretrained_models/osx_vit_l.pth
[92m03-13 00:21:24[0m set feat_dim to 1024
[92m03-13 00:21:24[0m set trainset_3d to []
[92m03-13 00:21:24[0m set trainset_2d to ['IMA']
[92m03-13 00:21:24[0m set testset to IMA
/proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../common/utils/transforms.py:80: UserWarning: Using torch.cross without specifying the dim arg is deprecated.
Please either pass the dim explicitly or simply use torch.linalg.cross.
The default value of dim will change to agree with that of linalg.cross in a future release. (Triggered internally at /opt/conda/conda-bld/pytorch_1712608935911/work/aten/src/ATen/native/Cross.cpp:62.)
  b3 = torch.cross(b1, b2)
/software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages/torch/functional.py:512: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /opt/conda/conda-bld/pytorch_1712608935911/work/aten/src/ATen/native/TensorShape.cpp:3587.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
[92m03-13 00:23:56[0m Epoch 0/140 itr 99/271: lr: 9.99983e-05 speed: 1.34(1.34s r0.00)s/itr 0.10h/epoch loss_joint_proj: 5.0530 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9097
[92m03-13 00:26:11[0m Epoch 0/140 itr 199/271: lr: 9.99932e-05 speed: 1.34(1.34s r0.00)s/itr 0.10h/epoch loss_joint_proj: 5.9596 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9192
[92m03-13 00:27:51[0m Write snapshot into /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_2/model_dump/snapshot_0.pth.tar
[92m03-13 00:30:14[0m Epoch 1/140 itr 99/271: lr: 9.99768e-05 speed: 1.36(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 6.9999 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9173
[92m03-13 00:32:28[0m Epoch 1/140 itr 199/271: lr: 9.99625e-05 speed: 1.36(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 7.6178 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9283
[92m03-13 00:36:21[0m Epoch 2/140 itr 99/271: lr: 9.99305e-05 speed: 1.36(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 6.0148 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9288
[92m03-13 00:38:35[0m Epoch 2/140 itr 199/271: lr: 9.99071e-05 speed: 1.36(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.7784 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9207
[92m03-13 00:42:30[0m Epoch 3/140 itr 99/271: lr: 9.98595e-05 speed: 1.36(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.4599 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9223
[92m03-13 00:44:44[0m Epoch 3/140 itr 199/271: lr: 9.9827e-05 speed: 1.36(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.9903 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9103
[92m03-13 00:48:40[0m Epoch 4/140 itr 99/271: lr: 9.97639e-05 speed: 1.36(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 6.5974 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9231
[92m03-13 00:50:55[0m Epoch 4/140 itr 199/271: lr: 9.97222e-05 speed: 1.36(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 6.1617 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9259
[92m03-13 00:54:51[0m Epoch 5/140 itr 99/271: lr: 9.96436e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.7677 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9022
[92m03-13 00:57:05[0m Epoch 5/140 itr 199/271: lr: 9.95929e-05 speed: 1.36(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 7.4755 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9461
[92m03-13 01:01:00[0m Epoch 6/140 itr 99/271: lr: 9.94988e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.9002 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9165
[92m03-13 01:03:15[0m Epoch 6/140 itr 199/271: lr: 9.9439e-05 speed: 1.36(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.5546 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9233
[92m03-13 01:07:11[0m Epoch 7/140 itr 99/271: lr: 9.93295e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.8124 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9386
[92m03-13 01:09:25[0m Epoch 7/140 itr 199/271: lr: 9.92606e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 6.5643 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9096
[92m03-13 01:13:21[0m Epoch 8/140 itr 99/271: lr: 9.91358e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.7651 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9099
[92m03-13 01:15:36[0m Epoch 8/140 itr 199/271: lr: 9.90578e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 6.3333 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9337
[92m03-13 01:19:31[0m Epoch 9/140 itr 99/271: lr: 9.89177e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.9761 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9159
[92m03-13 01:21:45[0m Epoch 9/140 itr 199/271: lr: 9.88308e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 8.0243 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9367
[92m03-13 01:25:41[0m Epoch 10/140 itr 99/271: lr: 9.86755e-05 speed: 1.37(1.34s r0.03)s/itr 0.10h/epoch loss_joint_proj: 3.8954 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9222
[92m03-13 01:27:56[0m Epoch 10/140 itr 199/271: lr: 9.85797e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.8989 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9116
[92m03-13 01:29:37[0m Write snapshot into /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_2/model_dump/snapshot_10.pth.tar
[92m03-13 01:31:58[0m Epoch 11/140 itr 99/271: lr: 9.84092e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.6183 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9010
[92m03-13 01:34:13[0m Epoch 11/140 itr 199/271: lr: 9.83045e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.5137 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9274
[92m03-13 01:38:08[0m Epoch 12/140 itr 99/271: lr: 9.81189e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.8906 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9220
[92m03-13 01:40:22[0m Epoch 12/140 itr 199/271: lr: 9.80054e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.7511 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9295
[92m03-13 01:44:17[0m Epoch 13/140 itr 99/271: lr: 9.78049e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.5226 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9112
[92m03-13 01:46:31[0m Epoch 13/140 itr 199/271: lr: 9.76826e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.1481 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9263
[92m03-13 01:50:28[0m Epoch 14/140 itr 99/271: lr: 9.74672e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.5269 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9307
[92m03-13 01:52:42[0m Epoch 14/140 itr 199/271: lr: 9.73361e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 9.2213 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9350
[92m03-13 01:56:37[0m Epoch 15/140 itr 99/271: lr: 9.7106e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.0275 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9300
[92m03-13 01:58:52[0m Epoch 15/140 itr 199/271: lr: 9.69663e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.5988 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9166
[92m03-13 02:02:47[0m Epoch 16/140 itr 99/271: lr: 9.67215e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.8408 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9186
[92m03-13 02:05:02[0m Epoch 16/140 itr 199/271: lr: 9.65733e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.5342 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9046
[92m03-13 02:08:56[0m Epoch 17/140 itr 99/271: lr: 9.6314e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.3207 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9125
[92m03-13 02:11:11[0m Epoch 17/140 itr 199/271: lr: 9.61572e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.5649 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9226
[92m03-13 02:15:05[0m Epoch 18/140 itr 99/271: lr: 9.58835e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.6593 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9280
[92m03-13 02:17:20[0m Epoch 18/140 itr 199/271: lr: 9.57183e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.1232 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9173
[92m03-13 02:21:14[0m Epoch 19/140 itr 99/271: lr: 9.54303e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.9319 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9410
[92m03-13 02:23:29[0m Epoch 19/140 itr 199/271: lr: 9.52568e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 6.1214 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9238
[92m03-13 02:27:25[0m Epoch 20/140 itr 99/271: lr: 9.49547e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.1344 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9316
[92m03-13 02:29:40[0m Epoch 20/140 itr 199/271: lr: 9.47729e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.6738 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9106
[92m03-13 02:31:21[0m Write snapshot into /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_2/model_dump/snapshot_20.pth.tar
[92m03-13 02:33:42[0m Epoch 21/140 itr 99/271: lr: 9.44569e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.7932 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9020
[92m03-13 02:35:56[0m Epoch 21/140 itr 199/271: lr: 9.42669e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.6116 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9329
[92m03-13 02:39:52[0m Epoch 22/140 itr 99/271: lr: 9.39371e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 6.8677 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9251
[92m03-13 02:42:06[0m Epoch 22/140 itr 199/271: lr: 9.3739e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.7373 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9252
[92m03-13 02:46:00[0m Epoch 23/140 itr 99/271: lr: 9.33956e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 6.6683 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9199
[92m03-13 02:48:15[0m Epoch 23/140 itr 199/271: lr: 9.31895e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.5599 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9380
[92m03-13 02:52:09[0m Epoch 24/140 itr 99/271: lr: 9.28326e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.3570 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9249
[92m03-13 02:54:24[0m Epoch 24/140 itr 199/271: lr: 9.26187e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.8866 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9218
[92m03-13 02:58:18[0m Epoch 25/140 itr 99/271: lr: 9.22485e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.5786 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9241
[92m03-13 03:00:33[0m Epoch 25/140 itr 199/271: lr: 9.20268e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.8524 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9200
[92m03-13 03:04:27[0m Epoch 26/140 itr 99/271: lr: 9.16435e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.4197 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9245
[92m03-13 03:06:41[0m Epoch 26/140 itr 199/271: lr: 9.14142e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 6.5795 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9079
[92m03-13 03:10:37[0m Epoch 27/140 itr 99/271: lr: 9.1018e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.7383 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9269
[92m03-13 03:12:51[0m Epoch 27/140 itr 199/271: lr: 9.07811e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.6783 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9181
[92m03-13 03:16:48[0m Epoch 28/140 itr 99/271: lr: 9.03722e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.1483 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9101
[92m03-13 03:19:02[0m Epoch 28/140 itr 199/271: lr: 9.01279e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.4384 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9370
[92m03-13 03:22:57[0m Epoch 29/140 itr 99/271: lr: 8.97064e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 6.6415 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9267
[92m03-13 03:25:11[0m Epoch 29/140 itr 199/271: lr: 8.94549e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 7.5294 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9407
[92m03-13 03:29:09[0m Epoch 30/140 itr 99/271: lr: 8.90211e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.8955 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9144
[92m03-13 03:31:23[0m Epoch 30/140 itr 199/271: lr: 8.87624e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 6.0239 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9152
[92m03-13 03:33:05[0m Write snapshot into /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_2/model_dump/snapshot_30.pth.tar
[92m03-13 03:35:27[0m Epoch 31/140 itr 99/271: lr: 8.83165e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.9724 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9164
[92m03-13 03:37:42[0m Epoch 31/140 itr 199/271: lr: 8.80508e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.5136 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9266
[92m03-13 03:41:38[0m Epoch 32/140 itr 99/271: lr: 8.75931e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.8851 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9123
[92m03-13 03:43:53[0m Epoch 32/140 itr 199/271: lr: 8.73204e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.2061 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9088
[92m03-13 03:47:50[0m Epoch 33/140 itr 99/271: lr: 8.68511e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.4147 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9178
[92m03-13 03:50:04[0m Epoch 33/140 itr 199/271: lr: 8.65716e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.5475 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9407
[92m03-13 03:54:00[0m Epoch 34/140 itr 99/271: lr: 8.60909e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.2755 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9147
[92m03-13 03:56:15[0m Epoch 34/140 itr 199/271: lr: 8.58048e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.1310 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9254
[92m03-13 04:00:11[0m Epoch 35/140 itr 99/271: lr: 8.53129e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.7488 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9309
[92m03-13 04:02:26[0m Epoch 35/140 itr 199/271: lr: 8.50203e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.2486 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9135
[92m03-13 04:06:21[0m Epoch 36/140 itr 99/271: lr: 8.45175e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.9224 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9118
[92m03-13 04:08:36[0m Epoch 36/140 itr 199/271: lr: 8.42186e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.8745 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9408
[92m03-13 04:12:31[0m Epoch 37/140 itr 99/271: lr: 8.37051e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.9388 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9063
[92m03-13 04:14:46[0m Epoch 37/140 itr 199/271: lr: 8.34e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.1690 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9121
[92m03-13 04:18:42[0m Epoch 38/140 itr 99/271: lr: 8.28762e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.3067 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9242
[92m03-13 04:20:57[0m Epoch 38/140 itr 199/271: lr: 8.2565e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.9923 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9423
[92m03-13 04:24:52[0m Epoch 39/140 itr 99/271: lr: 8.2031e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.1573 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9265
[92m03-13 04:27:06[0m Epoch 39/140 itr 199/271: lr: 8.1714e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.8097 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9245
[92m03-13 04:31:01[0m Epoch 40/140 itr 99/271: lr: 8.11701e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.6432 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9122
[92m03-13 04:33:16[0m Epoch 40/140 itr 199/271: lr: 8.08473e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.0096 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9358
[92m03-13 04:34:58[0m Write snapshot into /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_2/model_dump/snapshot_40.pth.tar
[92m03-13 04:37:20[0m Epoch 41/140 itr 99/271: lr: 8.02938e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.2565 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.8945
[92m03-13 04:39:35[0m Epoch 41/140 itr 199/271: lr: 7.99655e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.8495 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9204
[92m03-13 04:43:31[0m Epoch 42/140 itr 99/271: lr: 7.94027e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.7056 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9208
[92m03-13 04:45:46[0m Epoch 42/140 itr 199/271: lr: 7.9069e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.3586 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9212
[92m03-13 04:49:42[0m Epoch 43/140 itr 99/271: lr: 7.84971e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.4120 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9304
[92m03-13 04:51:57[0m Epoch 43/140 itr 199/271: lr: 7.81581e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.4037 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9307
[92m03-13 04:55:54[0m Epoch 44/140 itr 99/271: lr: 7.75775e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.1615 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9149
[92m03-13 04:58:09[0m Epoch 44/140 itr 199/271: lr: 7.72335e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 6.4583 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9216
[92m03-13 05:02:04[0m Epoch 45/140 itr 99/271: lr: 7.66444e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.3962 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9202
[92m03-13 05:04:19[0m Epoch 45/140 itr 199/271: lr: 7.62955e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.1435 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9301
[92m03-13 05:08:14[0m Epoch 46/140 itr 99/271: lr: 7.56983e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 7.2120 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9199
[92m03-13 05:10:29[0m Epoch 46/140 itr 199/271: lr: 7.53446e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.7402 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9486
[92m03-13 05:14:25[0m Epoch 47/140 itr 99/271: lr: 7.47395e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.6135 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9172
[92m03-13 05:16:40[0m Epoch 47/140 itr 199/271: lr: 7.43813e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 6.5716 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9345
[92m03-13 05:20:36[0m Epoch 48/140 itr 99/271: lr: 7.37686e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.3095 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9214
[92m03-13 05:22:51[0m Epoch 48/140 itr 199/271: lr: 7.3406e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.5299 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9149
[92m03-13 05:26:46[0m Epoch 49/140 itr 99/271: lr: 7.27861e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.2761 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9167
[92m03-13 05:29:02[0m Epoch 49/140 itr 199/271: lr: 7.24193e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.6668 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9426
[92m03-13 05:32:58[0m Epoch 50/140 itr 99/271: lr: 7.17924e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.2300 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9303
[92m03-13 05:35:13[0m Epoch 50/140 itr 199/271: lr: 7.14217e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.5614 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9206
[92m03-13 05:36:55[0m Write snapshot into /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_2/model_dump/snapshot_50.pth.tar
[92m03-13 05:39:18[0m Epoch 51/140 itr 99/271: lr: 7.07881e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.3175 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9036
[92m03-13 05:41:32[0m Epoch 51/140 itr 199/271: lr: 7.04136e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.0269 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9199
[92m03-13 05:45:29[0m Epoch 52/140 itr 99/271: lr: 6.97737e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.8188 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9255
[92m03-13 05:47:44[0m Epoch 52/140 itr 199/271: lr: 6.93955e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.7257 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9360
[92m03-13 05:51:41[0m Epoch 53/140 itr 99/271: lr: 6.87496e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.9594 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9458
[92m03-13 05:53:56[0m Epoch 53/140 itr 199/271: lr: 6.8368e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.8600 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9259
[92m03-13 05:57:52[0m Epoch 54/140 itr 99/271: lr: 6.77164e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.0063 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9146
[92m03-13 06:00:07[0m Epoch 54/140 itr 199/271: lr: 6.73315e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.6327 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9194
[92m03-13 06:04:04[0m Epoch 55/140 itr 99/271: lr: 6.66746e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 6.2284 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9196
[92m03-13 06:06:20[0m Epoch 55/140 itr 199/271: lr: 6.62867e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.5137 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9376
[92m03-13 06:10:18[0m Epoch 56/140 itr 99/271: lr: 6.56247e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.4800 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9418
[92m03-13 06:12:33[0m Epoch 56/140 itr 199/271: lr: 6.52339e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.3530 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9123
[92m03-13 06:16:28[0m Epoch 57/140 itr 99/271: lr: 6.45673e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.6339 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9443
[92m03-13 06:18:44[0m Epoch 57/140 itr 199/271: lr: 6.41738e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.2970 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9379
[92m03-13 06:22:41[0m Epoch 58/140 itr 99/271: lr: 6.35028e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.9392 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9184
[92m03-13 06:24:56[0m Epoch 58/140 itr 199/271: lr: 6.31069e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.5926 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9237
[92m03-13 06:28:53[0m Epoch 59/140 itr 99/271: lr: 6.24318e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.6110 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9220
[92m03-13 06:31:08[0m Epoch 59/140 itr 199/271: lr: 6.20336e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.0454 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9224
[92m03-13 06:35:03[0m Epoch 60/140 itr 99/271: lr: 6.13549e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.6298 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9165
[92m03-13 06:37:19[0m Epoch 60/140 itr 199/271: lr: 6.09546e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.3466 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9213
[92m03-13 06:39:01[0m Write snapshot into /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_2/model_dump/snapshot_60.pth.tar
[92m03-13 06:41:23[0m Epoch 61/140 itr 99/271: lr: 6.02725e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.1396 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9157
[92m03-13 06:43:38[0m Epoch 61/140 itr 199/271: lr: 5.98704e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.4958 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9379
[92m03-13 06:47:35[0m Epoch 62/140 itr 99/271: lr: 5.91853e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.5721 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9248
[92m03-13 06:49:50[0m Epoch 62/140 itr 199/271: lr: 5.87815e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.4573 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9208
[92m03-13 06:53:47[0m Epoch 63/140 itr 99/271: lr: 5.80937e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.8905 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.8957
[92m03-13 06:56:03[0m Epoch 63/140 itr 199/271: lr: 5.76884e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.6654 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9295
[92m03-13 07:00:00[0m Epoch 64/140 itr 99/271: lr: 5.69983e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.3449 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9230
[92m03-13 07:02:15[0m Epoch 64/140 itr 199/271: lr: 5.65917e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 7.0344 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9269
[92m03-13 07:06:11[0m Epoch 65/140 itr 99/271: lr: 5.58997e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.2895 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9359
[92m03-13 07:08:27[0m Epoch 65/140 itr 199/271: lr: 5.5492e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.2061 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9129
[92m03-13 07:12:22[0m Epoch 66/140 itr 99/271: lr: 5.47983e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.4292 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9189
[92m03-13 07:14:38[0m Epoch 66/140 itr 199/271: lr: 5.43899e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.0691 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9412
[92m03-13 07:18:34[0m Epoch 67/140 itr 99/271: lr: 5.36948e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.3503 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9283
[92m03-13 07:20:49[0m Epoch 67/140 itr 199/271: lr: 5.32857e-05 speed: 1.37(1.34s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.4808 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9126
[92m03-13 07:24:46[0m Epoch 68/140 itr 99/271: lr: 5.25898e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.2933 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9086
[92m03-13 07:27:02[0m Epoch 68/140 itr 199/271: lr: 5.21802e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.1130 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9255
[92m03-13 07:30:58[0m Epoch 69/140 itr 99/271: lr: 5.14836e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.0092 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9391
[92m03-13 07:33:14[0m Epoch 69/140 itr 199/271: lr: 5.10738e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.4495 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9350
[92m03-13 07:37:11[0m Epoch 70/140 itr 99/271: lr: 5.0377e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.6665 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9481
[92m03-13 07:39:27[0m Epoch 70/140 itr 199/271: lr: 4.99672e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.2135 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9209
[92m03-13 07:41:09[0m Write snapshot into /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_2/model_dump/snapshot_70.pth.tar
[92m03-13 07:43:31[0m Epoch 71/140 itr 99/271: lr: 4.92705e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.1298 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9303
[92m03-13 07:45:47[0m Epoch 71/140 itr 199/271: lr: 4.88608e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.0409 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9275
[92m03-13 07:49:45[0m Epoch 72/140 itr 99/271: lr: 4.81645e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.5489 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9208
[92m03-13 07:52:00[0m Epoch 72/140 itr 199/271: lr: 4.77552e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.1483 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9169
[92m03-13 07:55:58[0m Epoch 73/140 itr 99/271: lr: 4.70598e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.9853 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9084
[92m03-13 07:58:13[0m Epoch 73/140 itr 199/271: lr: 4.6651e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.0035 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9176
[92m03-13 08:02:11[0m Epoch 74/140 itr 99/271: lr: 4.59567e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.1064 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9233
[92m03-13 08:04:26[0m Epoch 74/140 itr 199/271: lr: 4.55487e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.9272 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9137
[92m03-13 08:08:24[0m Epoch 75/140 itr 99/271: lr: 4.48559e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 1.6262 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9400
[92m03-13 08:10:40[0m Epoch 75/140 itr 199/271: lr: 4.44489e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.2850 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9259
[92m03-13 08:14:37[0m Epoch 76/140 itr 99/271: lr: 4.3758e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.8031 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9050
[92m03-13 08:16:53[0m Epoch 76/140 itr 199/271: lr: 4.33522e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.7592 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9369
[92m03-13 08:20:50[0m Epoch 77/140 itr 99/271: lr: 4.26634e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.3350 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9387
[92m03-13 08:23:05[0m Epoch 77/140 itr 199/271: lr: 4.2259e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.0071 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9284
[92m03-13 08:27:02[0m Epoch 78/140 itr 99/271: lr: 4.15727e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.5518 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9132
[92m03-13 08:29:18[0m Epoch 78/140 itr 199/271: lr: 4.11699e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.3243 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9089
[92m03-13 08:33:16[0m Epoch 79/140 itr 99/271: lr: 4.04865e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.3264 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9226
[92m03-13 08:35:32[0m Epoch 79/140 itr 199/271: lr: 4.00855e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.3222 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9244
[92m03-13 08:39:29[0m Epoch 80/140 itr 99/271: lr: 3.94053e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.8353 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9280
[92m03-13 08:41:44[0m Epoch 80/140 itr 199/271: lr: 3.90062e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.8924 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9315
[92m03-13 08:43:27[0m Write snapshot into /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_2/model_dump/snapshot_80.pth.tar
[92m03-13 08:45:49[0m Epoch 81/140 itr 99/271: lr: 3.83296e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.6558 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9372
[92m03-13 08:48:04[0m Epoch 81/140 itr 199/271: lr: 3.79328e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.8955 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9201
[92m03-13 08:52:03[0m Epoch 82/140 itr 99/271: lr: 3.72601e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 6.2955 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9055
[92m03-13 08:54:19[0m Epoch 82/140 itr 199/271: lr: 3.68656e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.4387 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9126
[92m03-13 08:58:17[0m Epoch 83/140 itr 99/271: lr: 3.61971e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.6081 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9276
[92m03-13 09:00:32[0m Epoch 83/140 itr 199/271: lr: 3.58052e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.8358 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9343
[92m03-13 09:04:30[0m Epoch 84/140 itr 99/271: lr: 3.51413e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.8647 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9348
[92m03-13 09:06:46[0m Epoch 84/140 itr 199/271: lr: 3.47522e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.8196 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9409
[92m03-13 09:10:43[0m Epoch 85/140 itr 99/271: lr: 3.40932e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.1837 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9384
[92m03-13 09:13:00[0m Epoch 85/140 itr 199/271: lr: 3.3707e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.2885 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9261
[92m03-13 09:16:58[0m Epoch 86/140 itr 99/271: lr: 3.30532e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.8878 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9248
[92m03-13 09:19:13[0m Epoch 86/140 itr 199/271: lr: 3.26703e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.0743 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9216
[92m03-13 09:23:10[0m Epoch 87/140 itr 99/271: lr: 3.2022e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.6792 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9477
[92m03-13 09:25:26[0m Epoch 87/140 itr 199/271: lr: 3.16424e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.4057 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9265
[92m03-13 09:29:25[0m Epoch 88/140 itr 99/271: lr: 3.1e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.8805 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9295
[92m03-13 09:31:41[0m Epoch 88/140 itr 199/271: lr: 3.0624e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.8588 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9186
[92m03-13 09:35:39[0m Epoch 89/140 itr 99/271: lr: 2.99878e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.8892 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9213
[92m03-13 09:37:55[0m Epoch 89/140 itr 199/271: lr: 2.96155e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.8574 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9173
[92m03-13 09:41:52[0m Epoch 90/140 itr 99/271: lr: 2.89858e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.0612 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9332
[92m03-13 09:44:09[0m Epoch 90/140 itr 199/271: lr: 2.86174e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.3706 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9120
[92m03-13 09:45:51[0m Write snapshot into /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_2/model_dump/snapshot_90.pth.tar
[92m03-13 09:48:15[0m Epoch 91/140 itr 99/271: lr: 2.79946e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.1244 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9294
[92m03-13 09:50:30[0m Epoch 91/140 itr 199/271: lr: 2.76303e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.5862 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9149
[92m03-13 09:54:27[0m Epoch 92/140 itr 99/271: lr: 2.70146e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.2840 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9381
[92m03-13 09:56:43[0m Epoch 92/140 itr 199/271: lr: 2.66546e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.1440 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9358
[92m03-13 10:00:42[0m Epoch 93/140 itr 99/271: lr: 2.60464e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.1748 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9242
[92m03-13 10:02:58[0m Epoch 93/140 itr 199/271: lr: 2.56909e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.5286 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9139
[92m03-13 10:06:55[0m Epoch 94/140 itr 99/271: lr: 2.50904e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.5600 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9282
[92m03-13 10:09:11[0m Epoch 94/140 itr 199/271: lr: 2.47395e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.9393 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9346
[92m03-13 10:13:09[0m Epoch 95/140 itr 99/271: lr: 2.41471e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.4354 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9170
[92m03-13 10:15:25[0m Epoch 95/140 itr 199/271: lr: 2.3801e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.8389 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9147
[92m03-13 10:19:23[0m Epoch 96/140 itr 99/271: lr: 2.32169e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.1889 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9325
[92m03-13 10:21:39[0m Epoch 96/140 itr 199/271: lr: 2.28759e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.9786 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9306
[92m03-13 10:25:36[0m Epoch 97/140 itr 99/271: lr: 2.23004e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.4014 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9459
[92m03-13 10:27:53[0m Epoch 97/140 itr 199/271: lr: 2.19645e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 6.0909 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9089
[92m03-13 10:31:49[0m Epoch 98/140 itr 99/271: lr: 2.1398e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.7331 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9270
[92m03-13 10:34:05[0m Epoch 98/140 itr 199/271: lr: 2.10674e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.9481 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9025
[92m03-13 10:38:04[0m Epoch 99/140 itr 99/271: lr: 2.05101e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.8828 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9246
[92m03-13 10:40:20[0m Epoch 99/140 itr 199/271: lr: 2.01851e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.9141 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9239
[92m03-13 10:44:18[0m Epoch 100/140 itr 99/271: lr: 1.96373e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.1292 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9170
[92m03-13 10:46:34[0m Epoch 100/140 itr 199/271: lr: 1.93179e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.2105 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9222
[92m03-13 10:48:17[0m Write snapshot into /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_2/model_dump/snapshot_100.pth.tar
[92m03-13 10:50:40[0m Epoch 101/140 itr 99/271: lr: 1.87798e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.0898 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9265
[92m03-13 10:52:56[0m Epoch 101/140 itr 199/271: lr: 1.84662e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.4428 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9117
[92m03-13 10:56:54[0m Epoch 102/140 itr 99/271: lr: 1.79382e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.2414 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9282
[92m03-13 10:59:10[0m Epoch 102/140 itr 199/271: lr: 1.76306e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.7265 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9094
[92m03-13 11:03:08[0m Epoch 103/140 itr 99/271: lr: 1.71129e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.4484 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9388
[92m03-13 11:05:24[0m Epoch 103/140 itr 199/271: lr: 1.68114e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.3857 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9340
[92m03-13 11:09:21[0m Epoch 104/140 itr 99/271: lr: 1.63043e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.6625 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9381
[92m03-13 11:11:37[0m Epoch 104/140 itr 199/271: lr: 1.60091e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.0567 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9066
[92m03-13 11:15:35[0m Epoch 105/140 itr 99/271: lr: 1.55127e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.2978 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9444
[92m03-13 11:17:51[0m Epoch 105/140 itr 199/271: lr: 1.5224e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.6172 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9353
[92m03-13 11:21:48[0m Epoch 106/140 itr 99/271: lr: 1.47387e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.6500 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9057
[92m03-13 11:24:04[0m Epoch 106/140 itr 199/271: lr: 1.44565e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.2470 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9275
[92m03-13 11:28:02[0m Epoch 107/140 itr 99/271: lr: 1.39825e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 2.8045 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9235
[92m03-13 11:30:18[0m Epoch 107/140 itr 199/271: lr: 1.3707e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 4.1047 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9154
[92m03-13 11:34:16[0m Epoch 108/140 itr 99/271: lr: 1.32445e-05 speed: 1.38(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 1.8992 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9213
[92m03-13 11:36:32[0m Epoch 108/140 itr 199/271: lr: 1.29759e-05 speed: 1.37(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.6573 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9173
[92m03-13 11:40:30[0m Epoch 109/140 itr 99/271: lr: 1.25252e-05 speed: 1.38(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 3.6682 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9159
[92m03-13 11:42:47[0m Epoch 109/140 itr 199/271: lr: 1.22636e-05 speed: 1.38(1.35s r0.02)s/itr 0.10h/epoch loss_joint_proj: 5.7168 loss_loss_dsr_c: 2.0794 loss_loss_dsr_mc: 0.9404
