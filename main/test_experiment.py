'''
This module evaluates one experiment with weight trained at different epoches
By default it will evaluate model trained at [0, 20, 40, 70] epoches



'''


import os
import pdb
import sys
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, "../main/transformer_utils")
from config import cfg
import cv2
from common.base import Demoer
from common.utils.preprocessing import load_img
from common.utils.vis import render_mesh, save_obj, vis_keypoints
import matplotlib.pyplot as plt
import platform
if not platform.system() == 'Windows':
    os.environ["PYOPENGL_PLATFORM"] = "egl"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', type=str, default='cpu', dest='devices')
    parser.add_argument('--img_path', type=str, default='input.png')
    #parser.add_argument('--output_folder', type=str, default='output')
    parser.add_argument('--encoder_setting', type=str, default='osx_l', choices=['osx_b', 'osx_l'])
    parser.add_argument('--decoder_setting', type=str, default='normal', choices=['normal', 'wo_face_decoder', 'wo_decoder'])
    parser.add_argument('--pretrained_model_path', type=str, default='../pretrained_models/osx_l.pth.tar')
    parser.add_argument('--evaluation_epoches', type=list, default=[0, 20, 40, 70])
    parser.add_argument('--model_type', type=str, default='smil_h')

    args = parser.parse_args()

    if not args.devices:
        assert 0, "please set proper devices"

    if args.devices[:3] == 'gpu':
        args.device = 'cuda'
        args.gpu_ids = [i for i in args.devices[4:].split(',')]

    elif args.devices[:3] == 'cpu':
        args.device = 'cpu'
        args.gpu_ids = None
    else:
        raise NotImplementedError()

    return args

args = parse_args()
cudnn.benchmark = True

model_type = args.model_type

if model_type == 'smpl_h':
    from common.utils.human_models import smpl_h as smpl
elif model_type == 'smpl_x':
    from common.utils.human_models import smpl_x as smpl
elif model_type == 'smil_h':
    from common.utils.human_models import smil_h as smpl
else:
    raise NotImplementedError()


eval_dir = f'{args.pretrained_model_path}/vis_eval'
os.makedirs(eval_dir, exist_ok=True)
evaluation_epoches = args.evaluation_epoches

demoer = None
cfg = None
transform = transforms.ToTensor()
fig, ax = plt.subplots(1,len(evaluation_epoches))
for i, epoch in enumerate(evaluation_epoches):
    model_path = f'{args.pretrained_model_path}/model_dump/snapshot_{epoch}.pth.tar'

    assert osp.exists(model_path), 'Cannot find model at ' + model_path
    #print('Load checkpoint from {}'.format(model_path))

    del demoer
    del cfg
    from config import cfg
    # load model
    cfg.set_args(args.device, args.gpu_ids)
    cfg.set_additional_args(
        encoder_setting=args.encoder_setting,
        decoder_setting=args.decoder_setting,
        pretrained_model_path=model_path,
        model_type=model_type)



    demoer = Demoer()
    demoer._make_model(smpl)
    demoer.model.eval()

    # prepare input image

    original_img = load_img(args.img_path)
    original_img_height, original_img_width = original_img.shape[:2]

    image_name = args.img_path.split('/')[-1][:-4]
    video_name = image_name[1:7]
    mask_path = os.path.join('../dataset/IMA/mask/render/', 'video_'+video_name, image_name + '.jpg')
    original_mask = load_img(mask_path)

    vis_mesh = original_mask.copy()
    vis_kpts = original_mask.copy()

    img = transform(original_img.astype(np.float32))/255
    img = img.cuda()[None,:,:,:]
    inputs = {'img': img}
    targets = {}
    meta_info = {}

    with torch.no_grad():
        out = demoer.model(inputs, targets, meta_info, 'test')

    mesh = out['smplx_mesh_cam'][0]
    save_obj(mesh.detach().cpu().numpy(), smpl.face, f'{eval_dir}/mesh_{epoch}.obj')

   
    focal = [
        cfg.focal[0] / cfg.input_body_shape[1] * original_img_width,
        cfg.focal[1] / cfg.input_body_shape[0] * original_img_height
        ]
    princpt = [
        cfg.princpt[0] / cfg.input_body_shape[1] * original_img_width, # / cfg.input_body_shape[1],
        cfg.princpt[1] / cfg.input_body_shape[0] * original_img_height# / cfg.input_body_shape[0]
        ]

    vis_mesh = render_mesh(vis_mesh, mesh.cpu().numpy(), smpl.face, {'focal': focal, 'princpt': princpt})

    joint_proj = out['smplx_joint_proj'].detach().cpu().numpy()[0]

    joint_proj[:, 0] = joint_proj[:, 0] / cfg.input_body_shape[1] * original_img_width
    joint_proj[:, 1] = joint_proj[:, 1] / cfg.input_body_shape[0] * original_img_height


    vis_mesh = vis_keypoints(vis_mesh, joint_proj)
    #vis_kpts = vis_keypoints(vis_kpts, joint_proj)



    cv2.imwrite(f'{eval_dir}/render_at_epoch{epoch}.jpg', vis_mesh[:, :, ::-1])
    #cv2.imwrite(f'{eval_dir}/kpts_epoch{epoch}.jpg', vis_kpts[:, :, ::-1])


    #pdb.set_trace()
    ax[i].imshow(vis_mesh[:, :, ::-1].astype(np.uint8))
    #ax[1, i].imshow(vis_kpts[:, :, ::-1].astype(np.uint8))

plt.show()

