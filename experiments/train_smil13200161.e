PYTHONPATH has been set to
/software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages 
Please do not modify PYTHONPATH while using this module. 
 
/home/x_hensh/.local/lib/python3.10/site-packages/mmcv/__init__.py:20: UserWarning: On January 1, 2023, MMCV will release v2.0.0, in which it will remove components related to the training process and add a data transformation module. In addition, it will rename the package names mmcv to mmcv-lite and mmcv-full to mmcv. See https://github.com/open-mmlab/mmcv/blob/master/docs/en/compatibility.md for more details.
  warnings.warn(
/home/x_hensh/.local/lib/python3.10/site-packages/timm/models/layers/__init__.py:48: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
[92m03-13 00:05:53[0m Creating dataset...
[92m03-13 00:05:54[0m Creating graph and optimizer...
[92m03-13 00:06:08[0m Load checkpoint from ../pretrained_models/osx_l.pth.tar
[92m03-13 00:06:08[0m set lr to 0.0001
[92m03-13 00:06:08[0m set debug to False
[92m03-13 00:06:08[0m set continue_train to True
[92m03-13 00:06:08[0m set device to cuda
[92m03-13 00:06:08[0m set gpu_ids to ['0']
[92m03-13 00:06:08[0m set exp_name to output/train_kpt1dsr1_p3drender_1/
[92m03-13 00:06:08[0m set num_thread to 16
[92m03-13 00:06:08[0m set train_batch_size to 32
[92m03-13 00:06:08[0m set encoder_setting to osx_l
[92m03-13 00:06:08[0m set decoder_setting to normal
[92m03-13 00:06:08[0m set end_epoch to 140
[92m03-13 00:06:08[0m set pretrained_model_path to ../pretrained_models/osx_l.pth.tar
[92m03-13 00:06:08[0m set agora_benchmark to False
[92m03-13 00:06:08[0m set ubody_benchmark to False
[92m03-13 00:06:08[0m set ima_benchmark to True
[92m03-13 00:06:08[0m set model_type to smil_h
[92m03-13 00:06:08[0m set output_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_1/
[92m03-13 00:06:08[0m set model_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_1/model_dump
[92m03-13 00:06:08[0m set vis_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_1/vis
[92m03-13 00:06:08[0m set log_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_1/log
[92m03-13 00:06:08[0m set code_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_1/code
[92m03-13 00:06:08[0m set result_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_1/result
[92m03-13 00:06:08[0m set encoder_config_file to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../main/transformer_utils/configs/osx/encoder/body_encoder_large.py
[92m03-13 00:06:08[0m set encoder_pretrained_model_path to ../pretrained_models/osx_vit_l.pth
[92m03-13 00:06:08[0m set feat_dim to 1024
[92m03-13 00:06:08[0m set trainset_3d to []
[92m03-13 00:06:08[0m set trainset_2d to ['IMA']
[92m03-13 00:06:08[0m set testset to IMA
/proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../common/utils/transforms.py:80: UserWarning: Using torch.cross without specifying the dim arg is deprecated.
Please either pass the dim explicitly or simply use torch.linalg.cross.
The default value of dim will change to agree with that of linalg.cross in a future release. (Triggered internally at /opt/conda/conda-bld/pytorch_1712608935911/work/aten/src/ATen/native/Cross.cpp:62.)
  b3 = torch.cross(b1, b2)
/software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages/torch/functional.py:512: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /opt/conda/conda-bld/pytorch_1712608935911/work/aten/src/ATen/native/TensorShape.cpp:3587.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
Traceback (most recent call last):
  File "/proj/berzelius-2024-331/users/x_hensh/git/OSX/experiments/../main/train.py", line 139, in <module>
    main()
  File "/proj/berzelius-2024-331/users/x_hensh/git/OSX/experiments/../main/train.py", line 104, in main
    loss = trainer.model(inputs, targets, meta_info, 'train')
  File "/software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages/torch/nn/parallel/data_parallel.py", line 183, in forward
    return self.module(*inputs[0], **module_kwargs[0])
  File "/software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/proj/berzelius-2024-331/users/x_hensh/git/OSX/main/OSX.py", line 473, in forward
    silhouette, joint_proj = self.camera_screen(
  File "/proj/berzelius-2024-331/users/x_hensh/git/OSX/main/render_p3d.py", line 119, in __call__
    silhouette = self.silhouette_renderer(meshes_world=torch_mesh.clone(),
  File "/software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/x_hensh/.local/lib/python3.10/site-packages/pytorch3d/renderer/mesh/renderer.py", line 64, in forward
    images = self.shader(fragments, meshes_world, **kwargs)
  File "/software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/x_hensh/.local/lib/python3.10/site-packages/pytorch3d/renderer/mesh/shader.py", line 302, in forward
    colors = torch.ones_like(fragments.bary_coords)
RuntimeError: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

terminate called after throwing an instance of 'c10::Error'
  what():  CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Exception raised from c10_cuda_check_implementation at /opt/conda/conda-bld/pytorch_1712608935911/work/c10/cuda/CUDAException.cpp:43 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x57 (0x1491caa71897 in /software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages/torch/lib/libc10.so)
frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::string const&) + 0x64 (0x1491caa21b25 in /software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages/torch/lib/libc10.so)
frame #2: c10::cuda::c10_cuda_check_implementation(int, char const*, char const*, int, bool) + 0x118 (0x1491cab4b718 in /software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages/torch/lib/libc10_cuda.so)
frame #3: <unknown function> + 0x1d8d6 (0x1491cab168d6 in /software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages/torch/lib/libc10_cuda.so)
frame #4: <unknown function> + 0x1f5e3 (0x1491cab185e3 in /software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages/torch/lib/libc10_cuda.so)
frame #5: <unknown function> + 0x1f922 (0x1491cab18922 in /software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages/torch/lib/libc10_cuda.so)
frame #6: <unknown function> + 0x5a5860 (0x14921923e860 in /software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages/torch/lib/libtorch_python.so)
frame #7: <unknown function> + 0x6a36f (0x1491caa5636f in /software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages/torch/lib/libc10.so)
frame #8: c10::TensorImpl::~TensorImpl() + 0x21b (0x1491caa4f1cb in /software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages/torch/lib/libc10.so)
frame #9: c10::TensorImpl::~TensorImpl() + 0x9 (0x1491caa4f379 in /software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages/torch/lib/libc10.so)
frame #10: <unknown function> + 0x850f48 (0x1492194e9f48 in /software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages/torch/lib/libtorch_python.so)
frame #11: THPVariable_subclass_dealloc(_object*) + 0x2f6 (0x1492194ea2c6 in /software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages/torch/lib/libtorch_python.so)
<omitting python frames>
frame #27: __libc_start_main + 0xe5 (0x1492221fe7e5 in /lib64/libc.so.6)

/var/lib/slurm/slurmd/job13200161/slurm_script: line 25: 2113130 Aborted                 (core dumped) python ../main/train.py --devices gpu:0 --lr 1e-4 --exp_name output/train_kpt1dsr1_p3drender_1/ --end_epoch 140 --pretrained_model_path ../pretrained_models/osx_l.pth.tar --ima_benchmark --train_batch_size 32 --continue --decoder_setting normal --model_type smil_h --continue
