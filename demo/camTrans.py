
import torch
import os
import cv2
import numpy as np

import torchvision.transforms as transforms
import sys
sys.path.insert(0, "../main")
sys.path.insert(0, "../")

from common.utils.preprocessing import load_img, process_bbox, generate_patch_image
from config import cfg



img_path = 'input.png'
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


sys.path.insert(0, "../main/transformer_utils")

from common.base import Demoer
from common.utils.human_models import smpl_x
from common.utils.human_models import smpl_h
from common.utils.human_models import smil_h

transform = transforms.ToTensor()
cfg.set_additional_args(
    encoder_setting='osx_l',
    decoder_setting='wo_face_decoder', 
    pretrained_model_path='../pretrained_models/osx_l.pth.tar')

demoerx = Demoer()
demoerx._make_model(smpl_x)
demoerx.model.eval()

demoerh = Demoer()
demoerh._make_model(smpl_h)
demoerh.model.eval()

demoeri = Demoer()
demoeri._make_model(smil_h)
demoeri.model.eval()

outx = []
outh = []
outi = []

for person_detected in persons_deteted:
    img = transform(person_detected.astype(np.float32))/255
    img = img.cuda()[None,:,:,:]
    inputs = {'img': img}

    with torch.no_grad():
        outx.append(demoerx.model(inputs, {}, {}, 'test'))
        outh.append(demoerh.model(inputs, {}, {}, 'test'))
        outi.append(demoeri.model(inputs, {}, {}, 'test'))



import sys
import numpy as np
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer

from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.models.smpl import SMPLLayer
from aitviewer.configuration import CONFIG as C
C.smplx_models = "../common/utils/human_model_files"

#print (outx.keys())
template_smplx = SMPLLayer(model_type="smplx", gender="male", device=C.device)#, name="SMPLX")



v = Viewer()

for out in outx:
    v.scene.add(SMPLSequence(out['smplx_body_pose'], 
                         template_smplx,
                         #poses_root=outx['smplx_root_pose'],
                         betas=out['smplx_shape'],
                         #trans=out['cam_trans'],
                         poses_left_hand=out['smplx_lhand_pose'],
                         poses_right_hand=out['smplx_rhand_pose']))


v.run()