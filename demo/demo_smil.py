import os
import sys
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
from config import cfg
import cv2
import pdb

os.environ["PYOPENGL_PLATFORM"] = "egl"
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids', default='0')
    parser.add_argument('--img_path', type=str, default='input.png')
    parser.add_argument('--output_folder', type=str, default='output')
    parser.add_argument('--encoder_setting', type=str, default='osx_l', choices=['osx_b', 'osx_l'])
    parser.add_argument('--decoder_setting', type=str, default='normal', choices=['normal', 'wo_face_decoder', 'wo_decoder'])
    parser.add_argument('--pretrained_model_path', type=str, default='../pretrained_models/osx_l.pth.tar')
    parser.add_argument('--model_type', type=str, default='smil_h')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    return args

args = parse_args()
cfg.set_args(args.gpu_ids)
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

# load model
cfg.set_additional_args(
    encoder_setting=args.encoder_setting, 
    decoder_setting=args.decoder_setting, 
    pretrained_model_path=args.pretrained_model_path,
    model_type=model_type)
import sys
sys.path.insert(0, "../main/transformer_utils")
from common.base import Demoer
from common.utils.preprocessing import load_img, process_bbox, generate_patch_image
from common.utils.vis import render_mesh, save_obj, vis_keypoints
from common.utils.human_models import smpl_x
from common.utils.human_models import smpl_h
from main.render_p3d import base_renderer
demoer = Demoer()
demoer._make_model(smpl)


model_path = args.pretrained_model_path
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))

demoer.model.eval()

# prepare input image
transform = transforms.ToTensor()
original_img = load_img(args.img_path)
original_img_height, original_img_width = original_img.shape[:2]
os.makedirs(args.output_folder, exist_ok=True)

# detect human bbox with yolov5s
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
with torch.no_grad():
    results = detector(original_img)
person_results = results.xyxy[0][results.xyxy[0][:, 5] == 0]
class_ids, confidences, boxes = [], [], []
for detection in person_results:
    x1, y1, x2, y2, confidence, class_id = detection.tolist()
    class_ids.append(class_id)
    confidences.append(confidence)
    boxes.append([x1, y1, x2 - x1, y2 - y1])
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
vis_mesh = original_img.copy()
vis_kpts = original_img.copy()


if len(indices) == 0:
    indices = [0]
    boxes = [[0,0,original_img_width,original_img_height]]


for num, indice in enumerate(indices):
    bbox = boxes[indice]  # x,y,h,w
    bbox = process_bbox(bbox, original_img_width, original_img_height)
    img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
    vis_mesh2 = img.copy()
    vis_kpts2 = img.copy()
    img = transform(img.astype(np.float32))/255
    img = img.cuda()[None,:,:,:]
    inputs = {'img': img}
    targets = {}
    meta_info = {}

    # mesh recovery
    with torch.no_grad():
        out = demoer.model(inputs, targets, meta_info, 'test')

    mesh = out['smplx_mesh_cam'][0]
    points = out['outpu'].joints[0, smpl.joint_idx, :]
    

    save_obj(mesh.detach().cpu().numpy(), smpl.face, os.path.join(args.output_folder, f'person_{num}.obj')) 

    # render mesh
    focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
    princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
    vis_mesh = render_mesh(vis_mesh, mesh.detach().cpu().numpy(), smpl.face, {'focal': focal, 'princpt': princpt})

    #get_2d_pts
    joint_proj = out['smplx_joint_proj'].detach().cpu().numpy()[0]
    joint_proj[:, 0] = joint_proj[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
    joint_proj[:, 1] = joint_proj[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
    #pdb.set_trace()
    joint_proj = np.concatenate((joint_proj, np.ones_like(joint_proj[:, :1])), 1)
    joint_proj = np.dot(bb2img_trans, joint_proj.transpose(1, 0)).transpose(1, 0)
    vis_kpts = vis_keypoints(vis_kpts, joint_proj)




    


    pdb.set_trace()
    render = base_renderer(size=cfg.input_img_shape,
                           focal=focal,
                           principal_point=princpt,
                           device='cuda',
                           colorRender=True,)

    sil, p2d = render(vertices=mesh.unsqueeze(0), faces=torch.from_numpy(smpl.face.astype('int')).unsqueeze(0), points=points.unsqueeze(0))
    silhouette = sil[..., 3].unsqueeze(1)  # [N,1,256,256]
    point2d = p2d  # [N,17,2]
    #return silhouette, point2d
    #img = rgb * valid_mask + img * (1-valid_mask)
    vis_kpts = vis_keypoints(vis_kpts, joint_proj)








    focal = [
        cfg.focal[0] / cfg.input_body_shape[1] * cfg.input_img_shape[1], 
        cfg.focal[1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
             ]
    #focal = [cfg.focal[0] / cfg.input_body_shape[1], cfg.focal[1] / cfg.input_body_shape[0]]
    princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * cfg.input_img_shape[1], # / cfg.input_body_shape[1], 
               cfg.princpt[1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]# / cfg.input_body_shape[0]
               ]
    vis_mesh2 = render_mesh(vis_mesh2, mesh, smpl.face, {'focal': focal, 'princpt': princpt})

    cv2.imwrite(os.path.join(args.output_folder, f'render{indice}.jpg'), vis_mesh2[:, :, ::-1])

    joint_proj = out['smplx_joint_proj'].detach().cpu().numpy()[0]
    joint_proj[:, 0] = joint_proj[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
    joint_proj[:, 1] = joint_proj[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
    joint_proj = np.concatenate((joint_proj, np.ones_like(joint_proj[:, :1])), 1)
    #joint_proj = np.dot(bb2img_trans, joint_proj.transpose(1, 0)).transpose(1, 0)
    vis_kpts2 = vis_keypoints(vis_kpts2, joint_proj)
    cv2.imwrite(os.path.join(args.output_folder, f'kpts{indice}.jpg'), vis_kpts2[:, :, ::-1])

# save rendered image
cv2.imwrite(os.path.join(args.output_folder, f'render.jpg'), vis_mesh[:, :, ::-1])
cv2.imwrite(os.path.join(args.output_folder, f'kpts.jpg'), vis_kpts[:, :, ::-1])