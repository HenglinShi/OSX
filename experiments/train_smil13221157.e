PYTHONPATH has been set to
/software/sse/manual/PyTorch/2.3.0/python-3.10/envs/pytorch_2.3.0/lib/python3.10/site-packages 
Please do not modify PYTHONPATH while using this module. 
 
/home/x_hensh/.local/lib/python3.10/site-packages/mmcv/__init__.py:20: UserWarning: On January 1, 2023, MMCV will release v2.0.0, in which it will remove components related to the training process and add a data transformation module. In addition, it will rename the package names mmcv to mmcv-lite and mmcv-full to mmcv. See https://github.com/open-mmlab/mmcv/blob/master/docs/en/compatibility.md for more details.
  warnings.warn(
/home/x_hensh/.local/lib/python3.10/site-packages/timm/models/layers/__init__.py:48: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
[92m03-14 21:09:44[0m Creating dataset...
[92m03-14 21:09:45[0m Creating graph and optimizer...
[92m03-14 21:09:59[0m Load checkpoint from ../pretrained_models/osx_l.pth.tar
[92m03-14 21:09:59[0m set lr to 0.0001
[92m03-14 21:09:59[0m set debug to False
[92m03-14 21:09:59[0m set continue_train to True
[92m03-14 21:09:59[0m set device to cuda
[92m03-14 21:09:59[0m set gpu_ids to ['0']
[92m03-14 21:09:59[0m set exp_name to output/train_kpt1dsr1_p3drender_fixed2/
[92m03-14 21:09:59[0m set num_thread to 16
[92m03-14 21:09:59[0m set train_batch_size to 8
[92m03-14 21:09:59[0m set encoder_setting to osx_l
[92m03-14 21:09:59[0m set decoder_setting to normal
[92m03-14 21:09:59[0m set end_epoch to 140
[92m03-14 21:09:59[0m set pretrained_model_path to ../pretrained_models/osx_l.pth.tar
[92m03-14 21:09:59[0m set agora_benchmark to False
[92m03-14 21:09:59[0m set ubody_benchmark to False
[92m03-14 21:09:59[0m set ima_benchmark to True
[92m03-14 21:09:59[0m set model_type to smil_h
[92m03-14 21:09:59[0m set output_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_fixed2/
[92m03-14 21:09:59[0m set model_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_fixed2/model_dump
[92m03-14 21:09:59[0m set vis_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_fixed2/vis
[92m03-14 21:09:59[0m set log_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_fixed2/log
[92m03-14 21:09:59[0m set code_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_fixed2/code
[92m03-14 21:09:59[0m set result_dir to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../output/train_kpt1dsr1_p3drender_fixed2/result
[92m03-14 21:09:59[0m set encoder_config_file to /proj/berzelius-2024-331/users/x_hensh/git/OSX/main/../main/transformer_utils/configs/osx/encoder/body_encoder_large.py
[92m03-14 21:09:59[0m set encoder_pretrained_model_path to ../pretrained_models/osx_vit_l.pth
[92m03-14 21:09:59[0m set feat_dim to 1024
[92m03-14 21:09:59[0m set trainset_3d to []
[92m03-14 21:09:59[0m set trainset_2d to ['IMA']
[92m03-14 21:09:59[0m set testset to IMA
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
  File "/proj/berzelius-2024-331/users/x_hensh/git/OSX/main/OSX.py", line 474, in forward
    silhouette, joint_proj = self.camera_screen(
  File "/proj/berzelius-2024-331/users/x_hensh/git/OSX/main/render_p3d.py", line 149, in __call__
    rendered = self.renderer(meshes_world=torch_mesh.clone(),
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
  File "/home/x_hensh/.local/lib/python3.10/site-packages/pytorch3d/renderer/mesh/shader.py", line 132, in forward
    colors = phong_shading(
  File "/home/x_hensh/.local/lib/python3.10/site-packages/pytorch3d/renderer/mesh/shading.py", line 121, in phong_shading
    colors, _ = _phong_shading_with_pixels(
  File "/home/x_hensh/.local/lib/python3.10/site-packages/pytorch3d/renderer/mesh/shading.py", line 93, in _phong_shading_with_pixels
    ambient, diffuse, specular = _apply_lighting(
  File "/home/x_hensh/.local/lib/python3.10/site-packages/pytorch3d/renderer/mesh/shading.py", line 43, in _apply_lighting
    specular_color = materials.specular_color * light_specular
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.96 GiB. GPU 
