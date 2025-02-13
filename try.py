import torch
import os
import cv2
import numpy as np

import torchvision.transforms as transforms
import sys
sys.path.insert(0, "main")

from common.utils.preprocessing import load_img, process_bbox, generate_patch_image
from config import cfg
import pdb


img_path = 'demo/input.png'
original_img = load_img(img_path)
original_img_height, original_img_width = original_img.shape[:2]

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

persons_deteted = []
for num, indice in enumerate(indices):
    bbox = boxes[indice]  # x,y,h,w
    bbox = process_bbox(bbox, original_img_width, original_img_height)
    img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
    persons_deteted.append(img)


sys.path.insert(0, "main/transformer_utils")

from common.base import Demoer
from common.utils.human_models import smpl_x
from common.utils.human_models import smpl_h
from common.utils.human_models import smil_h


transform = transforms.ToTensor()
cfg.set_additional_args(
    encoder_setting='osx_l',
    decoder_setting='wo_face_decoder', 
    pretrained_model_path='pretrained_models/osx_l.pth.tar')

demoerx = Demoer()
demoerx._make_model(smpl_x)
demoerx.model.eval()

demoerh = Demoer()
demoerh._make_model(smpl_h)
demoerh.model.eval()

demoeri = Demoer()
demoeri._make_model(smil_h)
demoeri.model.eval()



# Now we tartet on the second image.

image = persons_deteted[0]
img = transform(img.astype(np.float32))/255
img = img.cuda()[None,:,:,:]
inputs = {'img': img}



with torch.no_grad():
    outx = demoerx.model(inputs, {}, {}, 'test')
    outh = demoerh.model(inputs, {}, {}, 'test')
    outi = demoeri.model(inputs, {}, {}, 'test')

meshx = outx['smplx_mesh_cam'].detach().cpu().numpy()
meshx = meshx[0]

meshh = outh['smplx_mesh_cam'].detach().cpu().numpy()
meshh = meshh[0]

meshi = outi['smplx_mesh_cam'].detach().cpu().numpy()
meshi = meshi[0]


#print (outx.keys())
#pdb.set_trace()

import sys
import numpy as np
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer

from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.models.smpl import SMPLLayer
from aitviewer.configuration import CONFIG as C
C.smplx_models = "common/utils/human_model_files"

#print (outx.keys())
template_smplx = SMPLLayer(model_type="smplx", gender="male", device=C.device)#, name="SMPLX")
template_smplh = SMPLLayer(model_type="smplh", gender="male", device=C.device)#, name="SMPLH")
template_smilh = SMPLLayer(model_type="smilh", gender="male", device=C.device)#, name="SMILH")


v = Viewer()
print (outx['smplx_body_pose'] - outh['smplx_body_pose'])
print (outh['smplx_body_pose'] - outi['smplx_body_pose'])
#print (outi['smplx_body_pose'])

v.scene.add(SMPLSequence(outx['smplx_body_pose'], 
                         template_smplx,
                         #poses_root=outx['smplx_root_pose'],
                         betas=outx['smplx_shape'],
                         #trans=outx['cam_trans'],
                         poses_left_hand=outx['smplx_lhand_pose'],
                         poses_right_hand=outx['smplx_rhand_pose']))

v.scene.add(SMPLSequence(outh['smplx_body_pose'], 
                         template_smplh, 
                         #poses_root=outh['smplx_root_pose'],
                         betas=outh['smplx_shape'],
                         #trans=outx['cam_trans'],
                         poses_left_hand=outh['smplx_lhand_pose'],
                         poses_right_hand=outh['smplx_rhand_pose'],
                         position=np.array((-1.0, 0.0, -0.0))))

v.scene.add(SMPLSequence(outi['smplx_body_pose'], 
                         template_smilh, 
                         #poses_root=outi['smplx_root_pose'],
                         betas=outi['smplx_shape'],
                         #trans=outx['cam_trans'],
                         poses_left_hand=outi['smplx_lhand_pose'],
                         poses_right_hand=outi['smplx_rhand_pose'],
                         position=np.array((-2.0, 0.0, -0.0))))
v.run()












