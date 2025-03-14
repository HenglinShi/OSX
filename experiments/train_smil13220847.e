PYTHONPATH has been set to
/software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages 
Please do not modify PYTHONPATH while using this module. 
 
/home/x_hensh/.local/lib/python3.10/site-packages/mmcv/__init__.py:20: UserWarning: On January 1, 2023, MMCV will release v2.0.0, in which it will remove components related to the training process and add a data transformation module. In addition, it will rename the package names mmcv to mmcv-lite and mmcv-full to mmcv. See https://github.com/open-mmlab/mmcv/blob/master/docs/en/compatibility.md for more details.
  warnings.warn(
/home/x_hensh/.local/lib/python3.10/site-packages/timm/models/layers/__init__.py:48: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
[92m03-14 17:44:53[0m Creating dataset...
[92m03-14 17:44:53[0m Creating graph and optimizer...
[92m03-14 17:45:03[0m Load checkpoint from ../pretrained_models/osx_l.pth.tar
[92m03-14 17:45:03[0m set lr to 0.0001
[92m03-14 17:45:03[0m set debug to False
[92m03-14 17:45:03[0m set continue_train to True
[92m03-14 17:45:03[0m set device to cuda
[92m03-14 17:45:03[0m set gpu_ids to ['0']
[92m03-14 17:45:03[0m set exp_name to output/train_kpt1dsr1_p3drender_fixed/
[92m03-14 17:45:03[0m set num_thread to 16
[92m03-14 17:45:03[0m set train_batch_size to 4
[92m03-14 17:45:03[0m set encoder_setting to osx_l
[92m03-14 17:45:03[0m set decoder_setting to normal
[92m03-14 17:45:03[0m set end_epoch to 140
[92m03-14 17:45:03[0m set pretrained_model_path to ../pretrained_models/osx_l.pth.tar
[92m03-14 17:45:03[0m set agora_benchmark to False
[92m03-14 17:45:03[0m set ubody_benchmark to False
[92m03-14 17:45:03[0m set ima_benchmark to True
[92m03-14 17:45:03[0m set model_type to smil_h
[92m03-14 17:45:03[0m set output_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_fixed/
[92m03-14 17:45:03[0m set model_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_fixed/model_dump
[92m03-14 17:45:03[0m set vis_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_fixed/vis
[92m03-14 17:45:03[0m set log_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_fixed/log
[92m03-14 17:45:03[0m set code_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_fixed/code
[92m03-14 17:45:03[0m set result_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_fixed/result
[92m03-14 17:45:03[0m set encoder_config_file to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../main/transformer_utils/configs/osx/encoder/body_encoder_large.py
[92m03-14 17:45:03[0m set encoder_pretrained_model_path to ../pretrained_models/osx_vit_l.pth
[92m03-14 17:45:03[0m set feat_dim to 1024
[92m03-14 17:45:03[0m set trainset_3d to []
[92m03-14 17:45:03[0m set trainset_2d to ['IMA']
[92m03-14 17:45:03[0m set testset to IMA
/proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../common/utils/transforms.py:80: UserWarning: Using torch.cross without specifying the dim arg is deprecated.
Please either pass the dim explicitly or simply use torch.linalg.cross.
The default value of dim will change to agree with that of linalg.cross in a future release. (Triggered internally at /opt/conda/conda-bld/pytorch_1712608935911/work/aten/src/ATen/native/Cross.cpp:62.)
  b3 = torch.cross(b1, b2)
/software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages/torch/functional.py:512: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /opt/conda/conda-bld/pytorch_1712608935911/work/aten/src/ATen/native/TensorShape.cpp:3587.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
[92m03-14 17:46:22[0m Epoch 0/140 itr 99/1083: lr: 9.99999e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.0782 loss_loss_dsr_c: 1.9403 loss_loss_dsr_mc: 0.9187
[92m03-14 17:47:34[0m Epoch 0/140 itr 199/1083: lr: 9.99996e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 8.1382 loss_loss_dsr_c: 1.8297 loss_loss_dsr_mc: 0.9281
[92m03-14 17:48:46[0m Epoch 0/140 itr 299/1083: lr: 9.9999e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.6763 loss_loss_dsr_c: 1.9060 loss_loss_dsr_mc: 0.9163
[92m03-14 17:49:58[0m Epoch 0/140 itr 399/1083: lr: 9.99983e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 7.7093 loss_loss_dsr_c: 1.9092 loss_loss_dsr_mc: 0.9384
[92m03-14 17:51:09[0m Epoch 0/140 itr 499/1083: lr: 9.99973e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 11.5153 loss_loss_dsr_c: 1.9259 loss_loss_dsr_mc: 0.9602
[92m03-14 17:52:21[0m Epoch 0/140 itr 599/1083: lr: 9.99962e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 7.1576 loss_loss_dsr_c: 1.8937 loss_loss_dsr_mc: 0.9161
[92m03-14 17:53:33[0m Epoch 0/140 itr 699/1083: lr: 9.99948e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 6.5516 loss_loss_dsr_c: 1.9208 loss_loss_dsr_mc: 0.9483
[92m03-14 17:54:45[0m Epoch 0/140 itr 799/1083: lr: 9.99932e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 4.6149 loss_loss_dsr_c: 1.9153 loss_loss_dsr_mc: 0.9406
[92m03-14 17:55:57[0m Epoch 0/140 itr 899/1083: lr: 9.99914e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.9777 loss_loss_dsr_c: 1.8882 loss_loss_dsr_mc: 0.9437
[92m03-14 17:57:08[0m Epoch 0/140 itr 999/1083: lr: 9.99894e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 8.8706 loss_loss_dsr_c: 1.9128 loss_loss_dsr_mc: 0.9541
[92m03-14 17:58:15[0m Write snapshot into /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_fixed/model_dump/snapshot_0.pth.tar
[92m03-14 17:59:29[0m Epoch 1/140 itr 99/1083: lr: 9.99852e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 7.6156 loss_loss_dsr_c: 1.9422 loss_loss_dsr_mc: 0.9374
[92m03-14 18:00:41[0m Epoch 1/140 itr 199/1083: lr: 9.99825e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 4.6236 loss_loss_dsr_c: 1.8960 loss_loss_dsr_mc: 0.9634
[92m03-14 18:01:53[0m Epoch 1/140 itr 299/1083: lr: 9.99797e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.2193 loss_loss_dsr_c: 1.9053 loss_loss_dsr_mc: 0.9251
[92m03-14 18:03:05[0m Epoch 1/140 itr 399/1083: lr: 9.99767e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 11.0975 loss_loss_dsr_c: 1.8499 loss_loss_dsr_mc: 0.9551
[92m03-14 18:04:17[0m Epoch 1/140 itr 499/1083: lr: 9.99734e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 8.1614 loss_loss_dsr_c: 1.9062 loss_loss_dsr_mc: 0.9139
[92m03-14 18:05:29[0m Epoch 1/140 itr 599/1083: lr: 9.99699e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 4.0090 loss_loss_dsr_c: 1.8230 loss_loss_dsr_mc: 0.9475
[92m03-14 18:06:41[0m Epoch 1/140 itr 699/1083: lr: 9.99663e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 6.9444 loss_loss_dsr_c: 1.9142 loss_loss_dsr_mc: 0.9009
[92m03-14 18:07:52[0m Epoch 1/140 itr 799/1083: lr: 9.99624e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 9.6922 loss_loss_dsr_c: 1.9365 loss_loss_dsr_mc: 0.9475
[92m03-14 18:09:05[0m Epoch 1/140 itr 899/1083: lr: 9.99583e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.3526 loss_loss_dsr_c: 1.8652 loss_loss_dsr_mc: 0.9144
[92m03-14 18:10:17[0m Epoch 1/140 itr 999/1083: lr: 9.99539e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.4689 loss_loss_dsr_c: 1.8752 loss_loss_dsr_mc: 0.8803
[92m03-14 18:12:30[0m Epoch 2/140 itr 99/1083: lr: 9.99455e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.0537 loss_loss_dsr_c: 1.9001 loss_loss_dsr_mc: 0.9295
[92m03-14 18:13:41[0m Epoch 2/140 itr 199/1083: lr: 9.99406e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 7.5531 loss_loss_dsr_c: 1.8824 loss_loss_dsr_mc: 0.9366
[92m03-14 18:14:53[0m Epoch 2/140 itr 299/1083: lr: 9.99355e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 7.3325 loss_loss_dsr_c: 1.9079 loss_loss_dsr_mc: 0.9387
[92m03-14 18:16:05[0m Epoch 2/140 itr 399/1083: lr: 9.99302e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.9247 loss_loss_dsr_c: 1.8731 loss_loss_dsr_mc: 0.9145
[92m03-14 18:17:17[0m Epoch 2/140 itr 499/1083: lr: 9.99246e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 7.0903 loss_loss_dsr_c: 1.8892 loss_loss_dsr_mc: 0.9352
[92m03-14 18:18:29[0m Epoch 2/140 itr 599/1083: lr: 9.99188e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 7.6746 loss_loss_dsr_c: 1.8706 loss_loss_dsr_mc: 0.9509
[92m03-14 18:19:41[0m Epoch 2/140 itr 699/1083: lr: 9.99129e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 10.1481 loss_loss_dsr_c: 1.9415 loss_loss_dsr_mc: 0.9373
[92m03-14 18:20:53[0m Epoch 2/140 itr 799/1083: lr: 9.99067e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 8.9425 loss_loss_dsr_c: 1.9674 loss_loss_dsr_mc: 0.9350
[92m03-14 18:22:04[0m Epoch 2/140 itr 899/1083: lr: 9.99003e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 17.7887 loss_loss_dsr_c: 1.9283 loss_loss_dsr_mc: 0.9480
[92m03-14 18:23:16[0m Epoch 2/140 itr 999/1083: lr: 9.98937e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.2089 loss_loss_dsr_c: 1.8386 loss_loss_dsr_mc: 0.9001
[92m03-14 18:25:29[0m Epoch 3/140 itr 99/1083: lr: 9.98811e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 4.0774 loss_loss_dsr_c: 1.9490 loss_loss_dsr_mc: 0.9533
[92m03-14 18:26:41[0m Epoch 3/140 itr 199/1083: lr: 9.98739e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 4.9954 loss_loss_dsr_c: 1.8309 loss_loss_dsr_mc: 0.9287
[92m03-14 18:27:53[0m Epoch 3/140 itr 299/1083: lr: 9.98664e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 8.3145 loss_loss_dsr_c: 1.9520 loss_loss_dsr_mc: 0.9354
[92m03-14 18:29:05[0m Epoch 3/140 itr 399/1083: lr: 9.98588e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.9549 loss_loss_dsr_c: 1.8370 loss_loss_dsr_mc: 0.9386
[92m03-14 18:30:17[0m Epoch 3/140 itr 499/1083: lr: 9.9851e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 4.8069 loss_loss_dsr_c: 1.9350 loss_loss_dsr_mc: 0.9032
[92m03-14 18:31:29[0m Epoch 3/140 itr 599/1083: lr: 9.98429e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 7.6566 loss_loss_dsr_c: 1.8605 loss_loss_dsr_mc: 0.9302
[92m03-14 18:32:41[0m Epoch 3/140 itr 699/1083: lr: 9.98346e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.1222 loss_loss_dsr_c: 1.8646 loss_loss_dsr_mc: 0.9424
[92m03-14 18:33:53[0m Epoch 3/140 itr 799/1083: lr: 9.98262e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 6.6058 loss_loss_dsr_c: 1.8768 loss_loss_dsr_mc: 0.9467
[92m03-14 18:35:04[0m Epoch 3/140 itr 899/1083: lr: 9.98175e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 7.1887 loss_loss_dsr_c: 1.8703 loss_loss_dsr_mc: 0.9617
[92m03-14 18:36:16[0m Epoch 3/140 itr 999/1083: lr: 9.98086e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.9877 loss_loss_dsr_c: 1.8575 loss_loss_dsr_mc: 0.8872
[92m03-14 18:38:29[0m Epoch 4/140 itr 99/1083: lr: 9.97918e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 9.0155 loss_loss_dsr_c: 1.9452 loss_loss_dsr_mc: 0.9131
[92m03-14 18:39:40[0m Epoch 4/140 itr 199/1083: lr: 9.97823e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.5400 loss_loss_dsr_c: 1.8472 loss_loss_dsr_mc: 0.9405
[92m03-14 18:40:52[0m Epoch 4/140 itr 299/1083: lr: 9.97726e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 7.7087 loss_loss_dsr_c: 1.8654 loss_loss_dsr_mc: 0.8912
[92m03-14 18:42:04[0m Epoch 4/140 itr 399/1083: lr: 9.97627e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 4.4252 loss_loss_dsr_c: 1.8884 loss_loss_dsr_mc: 0.8746
[92m03-14 18:43:16[0m Epoch 4/140 itr 499/1083: lr: 9.97525e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.0770 loss_loss_dsr_c: 1.9097 loss_loss_dsr_mc: 0.9074
[92m03-14 18:44:28[0m Epoch 4/140 itr 599/1083: lr: 9.97422e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 4.0917 loss_loss_dsr_c: 1.8922 loss_loss_dsr_mc: 0.9587
[92m03-14 18:45:40[0m Epoch 4/140 itr 699/1083: lr: 9.97316e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.0546 loss_loss_dsr_c: 1.7976 loss_loss_dsr_mc: 0.9010
[92m03-14 18:46:52[0m Epoch 4/140 itr 799/1083: lr: 9.97208e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 10.7688 loss_loss_dsr_c: 1.9134 loss_loss_dsr_mc: 0.9294
[92m03-14 18:48:03[0m Epoch 4/140 itr 899/1083: lr: 9.97099e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 7.9755 loss_loss_dsr_c: 1.8875 loss_loss_dsr_mc: 0.9584
[92m03-14 18:49:15[0m Epoch 4/140 itr 999/1083: lr: 9.96987e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 4.1469 loss_loss_dsr_c: 1.9269 loss_loss_dsr_mc: 0.9063
[92m03-14 18:51:28[0m Epoch 5/140 itr 99/1083: lr: 9.96777e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 4.8115 loss_loss_dsr_c: 1.9115 loss_loss_dsr_mc: 0.9322
[92m03-14 18:52:40[0m Epoch 5/140 itr 199/1083: lr: 9.9666e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.4313 loss_loss_dsr_c: 1.8105 loss_loss_dsr_mc: 0.9429
[92m03-14 18:53:52[0m Epoch 5/140 itr 299/1083: lr: 9.9654e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.9875 loss_loss_dsr_c: 1.8615 loss_loss_dsr_mc: 0.9283
[92m03-14 18:55:04[0m Epoch 5/140 itr 399/1083: lr: 9.96417e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 9.6213 loss_loss_dsr_c: 1.8296 loss_loss_dsr_mc: 0.9403
[92m03-14 18:56:16[0m Epoch 5/140 itr 499/1083: lr: 9.96293e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 6.5929 loss_loss_dsr_c: 1.8279 loss_loss_dsr_mc: 0.9380
[92m03-14 18:57:27[0m Epoch 5/140 itr 599/1083: lr: 9.96167e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 8.9494 loss_loss_dsr_c: 1.9320 loss_loss_dsr_mc: 0.9009
[92m03-14 18:58:39[0m Epoch 5/140 itr 699/1083: lr: 9.96038e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.2695 loss_loss_dsr_c: 1.9170 loss_loss_dsr_mc: 0.9008
[92m03-14 18:59:51[0m Epoch 5/140 itr 799/1083: lr: 9.95908e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 6.2620 loss_loss_dsr_c: 1.8847 loss_loss_dsr_mc: 0.9335
[92m03-14 19:01:03[0m Epoch 5/140 itr 899/1083: lr: 9.95775e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 6.3899 loss_loss_dsr_c: 1.8709 loss_loss_dsr_mc: 0.9074
[92m03-14 19:02:15[0m Epoch 5/140 itr 999/1083: lr: 9.9564e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 6.6009 loss_loss_dsr_c: 1.9098 loss_loss_dsr_mc: 0.9161
[92m03-14 19:04:28[0m Epoch 6/140 itr 99/1083: lr: 9.9539e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.1942 loss_loss_dsr_c: 1.9675 loss_loss_dsr_mc: 0.9253
[92m03-14 19:05:40[0m Epoch 6/140 itr 199/1083: lr: 9.95249e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 6.4222 loss_loss_dsr_c: 1.9480 loss_loss_dsr_mc: 0.9411
[92m03-14 19:06:52[0m Epoch 6/140 itr 299/1083: lr: 9.95106e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.1098 loss_loss_dsr_c: 1.8385 loss_loss_dsr_mc: 0.9204
[92m03-14 19:08:04[0m Epoch 6/140 itr 399/1083: lr: 9.94961e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 12.2562 loss_loss_dsr_c: 1.9453 loss_loss_dsr_mc: 0.9229
[92m03-14 19:09:15[0m Epoch 6/140 itr 499/1083: lr: 9.94814e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 8.8971 loss_loss_dsr_c: 1.9416 loss_loss_dsr_mc: 0.9209
[92m03-14 19:10:27[0m Epoch 6/140 itr 599/1083: lr: 9.94665e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.9274 loss_loss_dsr_c: 1.8799 loss_loss_dsr_mc: 0.9358
[92m03-14 19:11:39[0m Epoch 6/140 itr 699/1083: lr: 9.94514e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.4161 loss_loss_dsr_c: 1.8244 loss_loss_dsr_mc: 0.9095
[92m03-14 19:12:51[0m Epoch 6/140 itr 799/1083: lr: 9.94361e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.3330 loss_loss_dsr_c: 1.8840 loss_loss_dsr_mc: 0.9555
[92m03-14 19:14:03[0m Epoch 6/140 itr 899/1083: lr: 9.94205e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.8890 loss_loss_dsr_c: 1.8520 loss_loss_dsr_mc: 0.9255
[92m03-14 19:15:15[0m Epoch 6/140 itr 999/1083: lr: 9.94048e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 8.4276 loss_loss_dsr_c: 1.8807 loss_loss_dsr_mc: 0.9410
[92m03-14 19:17:28[0m Epoch 7/140 itr 99/1083: lr: 9.93756e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.6644 loss_loss_dsr_c: 1.8186 loss_loss_dsr_mc: 0.9290
[92m03-14 19:18:40[0m Epoch 7/140 itr 199/1083: lr: 9.93592e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 8.2734 loss_loss_dsr_c: 1.8552 loss_loss_dsr_mc: 0.9623
[92m03-14 19:19:52[0m Epoch 7/140 itr 299/1083: lr: 9.93427e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 2.9757 loss_loss_dsr_c: 1.8264 loss_loss_dsr_mc: 0.8970
[92m03-14 19:21:04[0m Epoch 7/140 itr 399/1083: lr: 9.93259e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 2.8894 loss_loss_dsr_c: 1.9496 loss_loss_dsr_mc: 0.9558
[92m03-14 19:22:16[0m Epoch 7/140 itr 499/1083: lr: 9.93089e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 4.2117 loss_loss_dsr_c: 1.9000 loss_loss_dsr_mc: 0.9272
[92m03-14 19:23:28[0m Epoch 7/140 itr 599/1083: lr: 9.92917e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.3064 loss_loss_dsr_c: 1.8436 loss_loss_dsr_mc: 0.9289
[92m03-14 19:24:40[0m Epoch 7/140 itr 699/1083: lr: 9.92743e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 6.0833 loss_loss_dsr_c: 1.9301 loss_loss_dsr_mc: 0.9094
[92m03-14 19:25:52[0m Epoch 7/140 itr 799/1083: lr: 9.92567e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 12.0436 loss_loss_dsr_c: 1.9446 loss_loss_dsr_mc: 0.8895
[92m03-14 19:27:04[0m Epoch 7/140 itr 899/1083: lr: 9.92389e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.7755 loss_loss_dsr_c: 1.8902 loss_loss_dsr_mc: 0.9077
[92m03-14 19:28:15[0m Epoch 7/140 itr 999/1083: lr: 9.92209e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.6141 loss_loss_dsr_c: 1.8729 loss_loss_dsr_mc: 0.9385
[92m03-14 19:30:29[0m Epoch 8/140 itr 99/1083: lr: 9.91876e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.7319 loss_loss_dsr_c: 1.8786 loss_loss_dsr_mc: 0.9350
[92m03-14 19:31:41[0m Epoch 8/140 itr 199/1083: lr: 9.9169e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.5125 loss_loss_dsr_c: 1.8826 loss_loss_dsr_mc: 0.9472
[92m03-14 19:32:52[0m Epoch 8/140 itr 299/1083: lr: 9.91501e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 2.5608 loss_loss_dsr_c: 1.8804 loss_loss_dsr_mc: 0.9283
[92m03-14 19:34:04[0m Epoch 8/140 itr 399/1083: lr: 9.91311e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.3091 loss_loss_dsr_c: 1.9173 loss_loss_dsr_mc: 0.9535
[92m03-14 19:35:16[0m Epoch 8/140 itr 499/1083: lr: 9.91119e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 10.2745 loss_loss_dsr_c: 1.9260 loss_loss_dsr_mc: 0.9358
[92m03-14 19:36:28[0m Epoch 8/140 itr 599/1083: lr: 9.90924e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 7.3361 loss_loss_dsr_c: 1.9116 loss_loss_dsr_mc: 0.9196
[92m03-14 19:37:40[0m Epoch 8/140 itr 699/1083: lr: 9.90728e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 6.6502 loss_loss_dsr_c: 1.8686 loss_loss_dsr_mc: 0.9529
[92m03-14 19:38:52[0m Epoch 8/140 itr 799/1083: lr: 9.90529e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 7.1300 loss_loss_dsr_c: 1.9146 loss_loss_dsr_mc: 0.8995
[92m03-14 19:40:04[0m Epoch 8/140 itr 899/1083: lr: 9.90328e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 10.1147 loss_loss_dsr_c: 1.8725 loss_loss_dsr_mc: 0.9442
[92m03-14 19:41:16[0m Epoch 8/140 itr 999/1083: lr: 9.90126e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 4.6891 loss_loss_dsr_c: 1.9047 loss_loss_dsr_mc: 0.9259
[92m03-14 19:43:28[0m Epoch 9/140 itr 99/1083: lr: 9.89751e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.7176 loss_loss_dsr_c: 1.9298 loss_loss_dsr_mc: 0.9485
[92m03-14 19:44:40[0m Epoch 9/140 itr 199/1083: lr: 9.89543e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.3534 loss_loss_dsr_c: 1.9025 loss_loss_dsr_mc: 0.8954
[92m03-14 19:45:52[0m Epoch 9/140 itr 299/1083: lr: 9.89332e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 2.1436 loss_loss_dsr_c: 1.9250 loss_loss_dsr_mc: 0.9236
[92m03-14 19:47:04[0m Epoch 9/140 itr 399/1083: lr: 9.89119e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 12.4032 loss_loss_dsr_c: 1.9615 loss_loss_dsr_mc: 0.9064
[92m03-14 19:48:16[0m Epoch 9/140 itr 499/1083: lr: 9.88904e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 6.7878 loss_loss_dsr_c: 1.8290 loss_loss_dsr_mc: 0.9118
[92m03-14 19:49:28[0m Epoch 9/140 itr 599/1083: lr: 9.88687e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 9.2063 loss_loss_dsr_c: 1.8873 loss_loss_dsr_mc: 0.9124
[92m03-14 19:50:40[0m Epoch 9/140 itr 699/1083: lr: 9.88468e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.4902 loss_loss_dsr_c: 1.8850 loss_loss_dsr_mc: 0.9427
[92m03-14 19:51:51[0m Epoch 9/140 itr 799/1083: lr: 9.88247e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 8.0214 loss_loss_dsr_c: 1.9216 loss_loss_dsr_mc: 0.9099
[92m03-14 19:53:03[0m Epoch 9/140 itr 899/1083: lr: 9.88024e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 2.9290 loss_loss_dsr_c: 1.8502 loss_loss_dsr_mc: 0.9223
[92m03-14 19:54:15[0m Epoch 9/140 itr 999/1083: lr: 9.87798e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 7.1486 loss_loss_dsr_c: 1.8976 loss_loss_dsr_mc: 0.9321
[92m03-14 19:56:28[0m Epoch 10/140 itr 99/1083: lr: 9.87383e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.4098 loss_loss_dsr_c: 1.9173 loss_loss_dsr_mc: 0.8895
[92m03-14 19:57:40[0m Epoch 10/140 itr 199/1083: lr: 9.87152e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 2.2472 loss_loss_dsr_c: 1.8226 loss_loss_dsr_mc: 0.8839
[92m03-14 19:58:52[0m Epoch 10/140 itr 299/1083: lr: 9.86919e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.3951 loss_loss_dsr_c: 1.9336 loss_loss_dsr_mc: 0.9071
[92m03-14 20:00:04[0m Epoch 10/140 itr 399/1083: lr: 9.86683e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.8018 loss_loss_dsr_c: 1.9007 loss_loss_dsr_mc: 0.9329
[92m03-14 20:01:16[0m Epoch 10/140 itr 499/1083: lr: 9.86446e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 2.8883 loss_loss_dsr_c: 1.8785 loss_loss_dsr_mc: 0.9308
[92m03-14 20:02:27[0m Epoch 10/140 itr 599/1083: lr: 9.86207e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 8.4208 loss_loss_dsr_c: 1.8985 loss_loss_dsr_mc: 0.9036
[92m03-14 20:03:39[0m Epoch 10/140 itr 699/1083: lr: 9.85965e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 4.5791 loss_loss_dsr_c: 1.8995 loss_loss_dsr_mc: 0.9295
[92m03-14 20:04:51[0m Epoch 10/140 itr 799/1083: lr: 9.85722e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.9381 loss_loss_dsr_c: 1.9256 loss_loss_dsr_mc: 0.9322
[92m03-14 20:06:02[0m Epoch 10/140 itr 899/1083: lr: 9.85476e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 4.4520 loss_loss_dsr_c: 1.9119 loss_loss_dsr_mc: 0.9394
[92m03-14 20:07:14[0m Epoch 10/140 itr 999/1083: lr: 9.85228e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 2.1724 loss_loss_dsr_c: 1.9156 loss_loss_dsr_mc: 0.9072
[92m03-14 20:08:21[0m Write snapshot into /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_fixed/model_dump/snapshot_10.pth.tar
[92m03-14 20:09:36[0m Epoch 11/140 itr 99/1083: lr: 9.84772e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.4320 loss_loss_dsr_c: 1.9211 loss_loss_dsr_mc: 0.8935
[92m03-14 20:10:47[0m Epoch 11/140 itr 199/1083: lr: 9.84519e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 4.4195 loss_loss_dsr_c: 2.0063 loss_loss_dsr_mc: 0.9471
[92m03-14 20:11:59[0m Epoch 11/140 itr 299/1083: lr: 9.84263e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 2.9039 loss_loss_dsr_c: 1.8966 loss_loss_dsr_mc: 0.8990
[92m03-14 20:13:11[0m Epoch 11/140 itr 399/1083: lr: 9.84006e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 4.8989 loss_loss_dsr_c: 1.8617 loss_loss_dsr_mc: 0.9277
[92m03-14 20:14:23[0m Epoch 11/140 itr 499/1083: lr: 9.83746e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.9340 loss_loss_dsr_c: 1.9036 loss_loss_dsr_mc: 0.9311
[92m03-14 20:15:34[0m Epoch 11/140 itr 599/1083: lr: 9.83485e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 6.6793 loss_loss_dsr_c: 1.8713 loss_loss_dsr_mc: 0.9418
[92m03-14 20:16:46[0m Epoch 11/140 itr 699/1083: lr: 9.83221e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.5990 loss_loss_dsr_c: 1.8729 loss_loss_dsr_mc: 0.9329
[92m03-14 20:17:58[0m Epoch 11/140 itr 799/1083: lr: 9.82955e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 9.9234 loss_loss_dsr_c: 1.9203 loss_loss_dsr_mc: 0.9188
[92m03-14 20:19:10[0m Epoch 11/140 itr 899/1083: lr: 9.82687e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.2999 loss_loss_dsr_c: 1.9017 loss_loss_dsr_mc: 0.9308
[92m03-14 20:20:22[0m Epoch 11/140 itr 999/1083: lr: 9.82417e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.3204 loss_loss_dsr_c: 1.8767 loss_loss_dsr_mc: 0.9362
[92m03-14 20:22:34[0m Epoch 12/140 itr 99/1083: lr: 9.81921e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 2.5999 loss_loss_dsr_c: 1.9050 loss_loss_dsr_mc: 0.9258
[92m03-14 20:23:46[0m Epoch 12/140 itr 199/1083: lr: 9.81645e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.3779 loss_loss_dsr_c: 1.8977 loss_loss_dsr_mc: 0.9086
[92m03-14 20:24:58[0m Epoch 12/140 itr 299/1083: lr: 9.81367e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.9996 loss_loss_dsr_c: 1.9182 loss_loss_dsr_mc: 0.8884
[92m03-14 20:26:10[0m Epoch 12/140 itr 399/1083: lr: 9.81088e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 8.2496 loss_loss_dsr_c: 1.9342 loss_loss_dsr_mc: 0.8958
[92m03-14 20:27:22[0m Epoch 12/140 itr 499/1083: lr: 9.80806e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 9.0117 loss_loss_dsr_c: 1.9158 loss_loss_dsr_mc: 0.9039
[92m03-14 20:28:34[0m Epoch 12/140 itr 599/1083: lr: 9.80522e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.1072 loss_loss_dsr_c: 1.9189 loss_loss_dsr_mc: 0.8863
[92m03-14 20:29:46[0m Epoch 12/140 itr 699/1083: lr: 9.80236e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 4.2995 loss_loss_dsr_c: 1.9051 loss_loss_dsr_mc: 0.9089
[92m03-14 20:30:58[0m Epoch 12/140 itr 799/1083: lr: 9.79948e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.3548 loss_loss_dsr_c: 1.9192 loss_loss_dsr_mc: 0.9270
[92m03-14 20:32:09[0m Epoch 12/140 itr 899/1083: lr: 9.79658e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.1691 loss_loss_dsr_c: 1.9120 loss_loss_dsr_mc: 0.9428
[92m03-14 20:33:21[0m Epoch 12/140 itr 999/1083: lr: 9.79366e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 7.9443 loss_loss_dsr_c: 1.9816 loss_loss_dsr_mc: 0.9390
[92m03-14 20:35:34[0m Epoch 13/140 itr 99/1083: lr: 9.78829e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 4.8853 loss_loss_dsr_c: 1.8937 loss_loss_dsr_mc: 0.9491
[92m03-14 20:36:47[0m Epoch 13/140 itr 199/1083: lr: 9.78531e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 4.6406 loss_loss_dsr_c: 1.8771 loss_loss_dsr_mc: 0.9366
[92m03-14 20:37:59[0m Epoch 13/140 itr 299/1083: lr: 9.78232e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 7.3407 loss_loss_dsr_c: 1.9400 loss_loss_dsr_mc: 0.9494
[92m03-14 20:39:11[0m Epoch 13/140 itr 399/1083: lr: 9.7793e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 10.3805 loss_loss_dsr_c: 1.9291 loss_loss_dsr_mc: 0.9408
[92m03-14 20:40:23[0m Epoch 13/140 itr 499/1083: lr: 9.77626e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.9981 loss_loss_dsr_c: 1.8197 loss_loss_dsr_mc: 0.9207
[92m03-14 20:41:35[0m Epoch 13/140 itr 599/1083: lr: 9.7732e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 16.2717 loss_loss_dsr_c: 1.9724 loss_loss_dsr_mc: 0.9566
[92m03-14 20:42:48[0m Epoch 13/140 itr 699/1083: lr: 9.77012e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 17.7470 loss_loss_dsr_c: 2.0091 loss_loss_dsr_mc: 0.9454
[92m03-14 20:44:01[0m Epoch 13/140 itr 799/1083: lr: 9.76702e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 17.5826 loss_loss_dsr_c: 1.9807 loss_loss_dsr_mc: 0.9561
[92m03-14 20:45:14[0m Epoch 13/140 itr 899/1083: lr: 9.7639e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 8.3946 loss_loss_dsr_c: 1.9475 loss_loss_dsr_mc: 0.9581
[92m03-14 20:46:26[0m Epoch 13/140 itr 999/1083: lr: 9.76076e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 6.0776 loss_loss_dsr_c: 1.8803 loss_loss_dsr_mc: 0.9390
[92m03-14 20:48:39[0m Epoch 14/140 itr 99/1083: lr: 9.755e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.3793 loss_loss_dsr_c: 1.8416 loss_loss_dsr_mc: 0.8982
[92m03-14 20:49:51[0m Epoch 14/140 itr 199/1083: lr: 9.7518e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 2.8908 loss_loss_dsr_c: 1.8500 loss_loss_dsr_mc: 0.9276
[92m03-14 20:51:03[0m Epoch 14/140 itr 299/1083: lr: 9.74858e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 12.2143 loss_loss_dsr_c: 1.9474 loss_loss_dsr_mc: 0.9424
[92m03-14 20:52:15[0m Epoch 14/140 itr 399/1083: lr: 9.74535e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 7.2278 loss_loss_dsr_c: 1.9013 loss_loss_dsr_mc: 0.9579
[92m03-14 20:53:27[0m Epoch 14/140 itr 499/1083: lr: 9.74209e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 6.7871 loss_loss_dsr_c: 1.9370 loss_loss_dsr_mc: 0.9148
[92m03-14 20:54:38[0m Epoch 14/140 itr 599/1083: lr: 9.73881e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 6.1167 loss_loss_dsr_c: 1.9217 loss_loss_dsr_mc: 0.9019
[92m03-14 20:55:50[0m Epoch 14/140 itr 699/1083: lr: 9.73551e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.7700 loss_loss_dsr_c: 1.8377 loss_loss_dsr_mc: 0.8951
[92m03-14 20:57:02[0m Epoch 14/140 itr 799/1083: lr: 9.73219e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.3444 loss_loss_dsr_c: 1.9168 loss_loss_dsr_mc: 0.9311
[92m03-14 20:58:14[0m Epoch 14/140 itr 899/1083: lr: 9.72886e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 4.4060 loss_loss_dsr_c: 1.9328 loss_loss_dsr_mc: 0.9098
[92m03-14 20:59:26[0m Epoch 14/140 itr 999/1083: lr: 9.7255e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 8.6096 loss_loss_dsr_c: 1.9409 loss_loss_dsr_mc: 0.8963
[92m03-14 21:01:38[0m Epoch 15/140 itr 99/1083: lr: 9.71934e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 4.1348 loss_loss_dsr_c: 1.8177 loss_loss_dsr_mc: 0.9096
[92m03-14 21:02:51[0m Epoch 15/140 itr 199/1083: lr: 9.71592e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.6546 loss_loss_dsr_c: 1.9170 loss_loss_dsr_mc: 0.9344
[92m03-14 21:04:03[0m Epoch 15/140 itr 299/1083: lr: 9.71249e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 4.1413 loss_loss_dsr_c: 1.8750 loss_loss_dsr_mc: 0.9423
[92m03-14 21:05:15[0m Epoch 15/140 itr 399/1083: lr: 9.70903e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.7480 loss_loss_dsr_c: 1.8812 loss_loss_dsr_mc: 0.9295
[92m03-14 21:06:27[0m Epoch 15/140 itr 499/1083: lr: 9.70556e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 11.1948 loss_loss_dsr_c: 1.9283 loss_loss_dsr_mc: 0.9480
[92m03-14 21:07:38[0m Epoch 15/140 itr 599/1083: lr: 9.70206e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 2.8809 loss_loss_dsr_c: 1.9045 loss_loss_dsr_mc: 0.9656
[92m03-14 21:08:50[0m Epoch 15/140 itr 699/1083: lr: 9.69855e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.8413 loss_loss_dsr_c: 1.9042 loss_loss_dsr_mc: 0.9285
[92m03-14 21:10:02[0m Epoch 15/140 itr 799/1083: lr: 9.69501e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.2345 loss_loss_dsr_c: 1.8624 loss_loss_dsr_mc: 0.9348
[92m03-14 21:11:14[0m Epoch 15/140 itr 899/1083: lr: 9.69146e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 9.2906 loss_loss_dsr_c: 1.8883 loss_loss_dsr_mc: 0.9474
[92m03-14 21:12:26[0m Epoch 15/140 itr 999/1083: lr: 9.68788e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 2.3353 loss_loss_dsr_c: 1.7635 loss_loss_dsr_mc: 0.9327
[92m03-14 21:14:39[0m Epoch 16/140 itr 99/1083: lr: 9.68133e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 4.1759 loss_loss_dsr_c: 1.8872 loss_loss_dsr_mc: 0.9089
[92m03-14 21:15:50[0m Epoch 16/140 itr 199/1083: lr: 9.6777e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 6.1125 loss_loss_dsr_c: 1.9608 loss_loss_dsr_mc: 0.8837
[92m03-14 21:17:02[0m Epoch 16/140 itr 299/1083: lr: 9.67405e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 2.5257 loss_loss_dsr_c: 1.8334 loss_loss_dsr_mc: 0.9201
[92m03-14 21:18:14[0m Epoch 16/140 itr 399/1083: lr: 9.67038e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.8997 loss_loss_dsr_c: 1.9112 loss_loss_dsr_mc: 0.9022
[92m03-14 21:19:26[0m Epoch 16/140 itr 499/1083: lr: 9.66669e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 2.0533 loss_loss_dsr_c: 1.8437 loss_loss_dsr_mc: 0.9195
[92m03-14 21:20:38[0m Epoch 16/140 itr 599/1083: lr: 9.66298e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 8.0586 loss_loss_dsr_c: 1.9429 loss_loss_dsr_mc: 0.9038
[92m03-14 21:21:50[0m Epoch 16/140 itr 699/1083: lr: 9.65925e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 2.7007 loss_loss_dsr_c: 1.9276 loss_loss_dsr_mc: 0.9292
[92m03-14 21:23:02[0m Epoch 16/140 itr 799/1083: lr: 9.6555e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 10.5344 loss_loss_dsr_c: 1.8887 loss_loss_dsr_mc: 0.9515
[92m03-14 21:24:14[0m Epoch 16/140 itr 899/1083: lr: 9.65173e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 4.7718 loss_loss_dsr_c: 1.8799 loss_loss_dsr_mc: 0.8956
[92m03-14 21:25:25[0m Epoch 16/140 itr 999/1083: lr: 9.64794e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 8.9608 loss_loss_dsr_c: 1.9823 loss_loss_dsr_mc: 0.9160
[92m03-14 21:27:39[0m Epoch 17/140 itr 99/1083: lr: 9.64099e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.7856 loss_loss_dsr_c: 1.7692 loss_loss_dsr_mc: 0.8836
[92m03-14 21:28:51[0m Epoch 17/140 itr 199/1083: lr: 9.63715e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 9.5290 loss_loss_dsr_c: 1.8803 loss_loss_dsr_mc: 0.9242
[92m03-14 21:30:03[0m Epoch 17/140 itr 299/1083: lr: 9.63328e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 3.3375 loss_loss_dsr_c: 1.9519 loss_loss_dsr_mc: 0.8960
[92m03-14 21:31:15[0m Epoch 17/140 itr 399/1083: lr: 9.6294e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 8.4218 loss_loss_dsr_c: 1.9136 loss_loss_dsr_mc: 0.9244
[92m03-14 21:32:27[0m Epoch 17/140 itr 499/1083: lr: 9.6255e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 6.9009 loss_loss_dsr_c: 1.8858 loss_loss_dsr_mc: 0.9326
[92m03-14 21:33:39[0m Epoch 17/140 itr 599/1083: lr: 9.62157e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 2.2911 loss_loss_dsr_c: 1.8552 loss_loss_dsr_mc: 0.9197
[92m03-14 21:34:50[0m Epoch 17/140 itr 699/1083: lr: 9.61763e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 4.5745 loss_loss_dsr_c: 1.9020 loss_loss_dsr_mc: 0.9088
[92m03-14 21:36:02[0m Epoch 17/140 itr 799/1083: lr: 9.61367e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 8.6040 loss_loss_dsr_c: 1.9101 loss_loss_dsr_mc: 0.9113
[92m03-14 21:37:14[0m Epoch 17/140 itr 899/1083: lr: 9.60969e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 5.9819 loss_loss_dsr_c: 1.9547 loss_loss_dsr_mc: 0.9096
[92m03-14 21:38:26[0m Epoch 17/140 itr 999/1083: lr: 9.60568e-05 speed: 0.72(0.72s r0.00)s/itr 0.22h/epoch loss_joint_proj: 1.8356 loss_loss_dsr_c: 1.8178 loss_loss_dsr_mc: 0.9100
