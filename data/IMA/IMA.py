import os
import os.path as osp
from glob import glob
import numpy as np
import pandas as pd

from common.nets.loss import CoordLoss
from config import cfg
import copy
import json
import pickle
import cv2
import torch
import pdb
from pycocotools.coco import COCO
from common.utils.smplx.smplx.joint_names import SMPLH_JOINT_NAMES, JOINT_NAMES
#from common.utils.human_models import smpl_x

if cfg.model_type == 'smpl_h':
    from common.utils.human_models import smpl_h as smpl
elif cfg.model_type == 'smpl_x':
    from common.utils.human_models import smpl_x as smpl
elif cfg.model_type == 'smil_h':
    from common.utils.human_models import smil_h as smpl
else:
    raise NotImplementedError()


from common.utils.preprocessing import load_img, sanitize_bbox, process_bbox, augmentation, get_augmentation_config, \
    process_db_coord, augmentation2, generate_patch_image, \
    process_human_model_output, load_ply, load_obj
from common.utils.vis import render_mesh, save_obj, vis_keypoints
from common.utils.transforms import rigid_align

class IMA(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.data_path = osp.join(cfg.data_dir, 'IMA', 'data')
        self.annot_path = osp.join(cfg.data_dir, 'IMA', 'annotations')
        self.img_path = osp.join(cfg.data_dir, 'IMA', 'images')
        self.mask_gt_root = '/proj/berzelius-2024-331/users/x_hensh/git/OSX/dataset/IMA/mask/raw_masks'

        #self.resolution = (2160, 3840)  # height, width. one of (720, 1280) and (2160, 3840)
        self.resolution = (0, 0)  # height, width. one of (720, 1280) and (2160, 3840)
        self.test_set = 'test' if cfg.agora_benchmark else 'val'  # val, test

        self.coord_loss = CoordLoss()


        self.keypoints_posemodel = [
            'left_hip', 
            'left_eye', 
            'left_knee', 
            'left_elbow', 
            'neck', 
            'right_elbow', 
            'right_knee', 
            'left_ear',
            'right_shoulder', 
            'right_ear', 
            'left_wrist', 
            'left_shoulder', 
            'left_ankle', 
            'right_eye', 
            'right_wrist', 
            'right_hip', 
            'right_ankle', 
            'nose', 
        ]

        self.keypoint_idx_posemodel = [SMPLH_JOINT_NAMES[i] for i in smpl.joint_idx]#SMPLH_JOINT_NAMES[smpl.joint_idx]
        self.keypoint_idx_posemodel = [self.keypoint_idx_posemodel.index(i) for i in self.keypoints_posemodel]

        






        self.POSE_PAIRS = [
            [1,2], 
            [1,5], 
            [2,3], 
            [3,4], 
            [5,6], 
            [6,7],
            [1,8], 
            [8,9], 
            [9,10], 
            [1,11], 
            [11,12], 
            [12,13],
            [1,0],
            [0,14], 
            [14,16], 
            [0,15], 
            [15,17],
            [2,17], 
            [5,16] ]

        # AGORA joint set
        self.joint_set = {
            'joint_num': 127,
            'joints_name': \
                ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3',
                 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow',
                 'R_Elbow', 'L_Wrist', 'R_Wrist',  # body
                 'Jaw', 'L_Eye_SMPLH', 'R_Eye_SMPLH',  # SMPLH
                 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1',
                 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3',
                 # fingers
                 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1',
                 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3',
                 # fingers
                 'Nose', 'R_Eye', 'L_Eye', 'R_Ear', 'L_Ear',  # face in body
                 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel',  # feet
                 'L_Thumb_4', 'L_Index_4', 'L_Middle_4', 'L_Ring_4', 'L_Pinky_4',  # finger tips
                 'R_Thumb_4', 'R_Index_4', 'R_Middle_4', 'R_Ring_4', 'R_Pinky_4',  # finger tips
                 *['Face_' + str(i) for i in range(5, 56)]  # face
                 ),
                 
            'flip_pairs': \
                ((0, 15), (1, 13), (2, 6), (3, 5), (7, 9), (8, 11), (10, 14), (12, 16),  # body
                 )
        }












        self.joint_set['joint_part'] = {
            'body': list(range(self.joint_set['joints_name'].index('Pelvis'),
                               self.joint_set['joints_name'].index('R_Eye_SMPLH') + 1)) + list(
                range(self.joint_set['joints_name'].index('Nose'), self.joint_set['joints_name'].index('R_Heel') + 1)),
            'lhand': list(range(self.joint_set['joints_name'].index('L_Index_1'),
                                self.joint_set['joints_name'].index('L_Thumb_3') + 1)) + list(
                range(self.joint_set['joints_name'].index('L_Thumb_4'),
                      self.joint_set['joints_name'].index('L_Pinky_4') + 1)),
            'rhand': list(range(self.joint_set['joints_name'].index('R_Index_1'),
                                self.joint_set['joints_name'].index('R_Thumb_3') + 1)) + list(
                range(self.joint_set['joints_name'].index('R_Thumb_4'),
                      self.joint_set['joints_name'].index('R_Pinky_4') + 1)),
            'face': list(range(self.joint_set['joints_name'].index('Face_5'),
                               self.joint_set['joints_name'].index('Face_55') + 1))}
        self.joint_set['root_joint_idx'] = self.joint_set['joints_name'].index('Pelvis')
        self.joint_set['lwrist_idx'] = self.joint_set['joints_name'].index('L_Wrist')
        self.joint_set['rwrist_idx'] = self.joint_set['joints_name'].index('R_Wrist')
        self.joint_set['neck_idx'] = self.joint_set['joints_name'].index('Neck')

        self.datalist = self.load_data()

    def load_data(self):
        # load frame size

        video_sizes_df = pd.read_excel('/proj/berzelius-2024-331/users/x_hensh/git/OSX/dataset/IMA/annotations/annotations_size.xlsx')
        video_sizes_dict = {}
        n_row, _ = video_sizes_df.shape
        for row in range(n_row):
            #source = video_sizes_df['Unnamed: 7'][row]
            #pdb.set_trace()
            if pd.isna(video_sizes_df['Unnamed: 7'][row]) is True:
                height = -1
                width = -1
            elif video_sizes_df['Unnamed: 7'][row] == 'resize' :
                height = video_sizes_df.height_annotated[row]
                width = video_sizes_df.width_annotated[row]
                pass
            elif video_sizes_df['Unnamed: 7'][row] == 'original' or video_sizes_df['Unnamed: 7'][row] == 'original/resize': 
                height = video_sizes_df.height_real[row]
                width = video_sizes_df.width_real[row]
                pass
            #elif video_sizes_df['Unnamed: 7'][row] == 'original/resize': 
            #    height = video_sizes_df.height_real[row]
            #    width = video_sizes_df.width_real[row]
            #    pass
            elif video_sizes_df['Unnamed: 7'][row] == 'Neither': 
                height = video_sizes_df.height_real[row]
                width = video_sizes_df.width_real[row]
                pass
            else:
                raise NotImplementedError()
            video_sizes_dict[video_sizes_df.video_name[row]] = [width, height]


        #







        datalist = []
        if self.data_split == 'train' or (self.data_split == 'test' and self.test_set == 'val'):
            if self.data_split == 'train':
                db = COCO(osp.join(self.annot_path, 'annotations_train.json'))
            else:
                db = COCO(osp.join(self.annot_path, 'annotations_valid.json'))

            for aid in db.anns.keys():
                ann = db.anns[aid]
                image_id = ann['image_id']
                img = db.loadImgs(image_id)[0]

                imgname = f"1{img['video'][-6:]}{img['frame']:06d}.jpg"
                img_path = osp.join(self.img_path, imgname)
                mask_gt_path = osp.join(self.mask_gt_root, img['video'], imgname)


                # get the desired image shape
                img_shape = video_sizes_dict[img['video']]
                #pdb.set_trace()
                #img['image_shape']

                
                bbox = ann['bbox'] if 'box' in ann.keys() else [0, 0, img_shape[0], img_shape[1]]
                bbox = process_bbox(bbox, img_shape[0], img_shape[1])
                if bbox is None:
                    continue

                keypoints = np.array(ann['keypoints'])

                data_dict = {'img_path': img_path,
                             'mask_gt_path': mask_gt_path,
                             'img_shape': img_shape,
                             'bbox': bbox,
                             #'lhand_bbox': lhand_bbox,
                             #'rhand_bbox': rhand_bbox,
                             #'face_bbox': face_bbox,
                             'joints_2d': keypoints,
                             #'joints_3d_path': joints_3d_path,
                             #'verts_path': verts_path,
                             #'smplx_param_path': smplx_param_path,
                             'ann_id': str(aid)}
                datalist.append(data_dict)




        elif self.data_split == 'test' and self.test_set == 'test':
            raise NotImplementedError()
            with open(osp.join(self.data_path, 'AGORA_test_bbox.json')) as f:
                bboxs = json.load(f)

            for filename in bboxs.keys():
                if self.resolution == (720, 1280):
                    img_path = osp.join(self.data_path, 'test', filename)
                    img_shape = self.resolution
                    person_num = len(bboxs[filename])
                    for pid in range(person_num):
                        # change bbox from (2160,3840) to target resoution
                        bbox = np.array(bboxs[filename][pid]['bbox']).reshape(2, 2)
                        bbox[:, 0] = bbox[:, 0] / 3840 * 1280
                        bbox[:, 1] = bbox[:, 1] / 2160 * 720
                        bbox = bbox.reshape(4)
                        bbox = process_bbox(bbox, img_shape[1], img_shape[0])
                        if bbox is None:
                            continue
                        datalist.append({'img_path': img_path, 'img_shape': img_shape, 'bbox': bbox, 'person_idx': pid})

                elif self.resolution == (2160,
                                         3840):  # use cropped and resized images. loading 4K images in pytorch dataloader takes too much time...
                    person_num = len(bboxs[filename])
                    for pid in range(person_num):
                        img_path = osp.join(self.data_path, '3840x2160', 'test_crop',
                                            filename[:-4] + '_pid_' + str(pid) + '.png')
                        json_path = osp.join(self.data_path, '3840x2160', 'test_crop',
                                             filename[:-4] + '_pid_' + str(pid) + '.json')
                        if not osp.isfile(json_path):
                            continue
                        with open(json_path) as f:
                            crop_resize_info = json.load(f)
                            img2bb_trans_from_orig = np.array(crop_resize_info['img2bb_trans'], dtype=np.float32)
                            resized_height, resized_width = crop_resize_info['resized_height'], crop_resize_info[
                                'resized_width']
                        img_shape = (resized_height, resized_width)
                        bbox = np.array([0, 0, resized_width, resized_height], dtype=np.float32)

                        #keypoints = ann['keypoints']



                        datalist.append({'img_path': img_path,
                                         'img_shape': img_shape,
                                         'img2bb_trans_from_orig': img2bb_trans_from_orig,
                                         'bbox': bbox,
                                         'person_idx': pid})
        else:
            raise NotImplementedError()

        return datalist

    def process_hand_face_bbox(self, bbox, do_flip, img_shape, img2bb_trans):
        if bbox is None:
            bbox = np.array([0, 0, 1, 1], dtype=np.float32).reshape(2, 2)  # dummy value
            bbox_valid = float(False)  # dummy value
        else:
            # reshape to top-left (x,y) and bottom-right (x,y)
            bbox = bbox.reshape(2, 2)

            # flip augmentation
            if do_flip:
                bbox[:, 0] = img_shape[1] - bbox[:, 0] - 1
                bbox[0, 0], bbox[1, 0] = bbox[1, 0].copy(), bbox[0, 0].copy()  # xmin <-> xmax swap

            # make four points of the bbox
            bbox = bbox.reshape(4).tolist()
            xmin, ymin, xmax, ymax = bbox
            bbox = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32).reshape(4, 2)

            # affine transformation (crop, rotation, scale)
            bbox_xy1 = np.concatenate((bbox, np.ones_like(bbox[:, :1])), 1)
            bbox = np.dot(img2bb_trans, bbox_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
            bbox[:, 0] = bbox[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            bbox[:, 1] = bbox[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

            # make box a rectangle without rotation
            xmin = np.min(bbox[:, 0]);
            xmax = np.max(bbox[:, 0]);
            ymin = np.min(bbox[:, 1]);
            ymax = np.max(bbox[:, 1]);
            bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

            bbox_valid = float(True)
            bbox = bbox.reshape(2, 2)

        return bbox, bbox_valid

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):

        inputs = {}



        # Load data metadata
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']
        mask_gt_path = data['mask_gt_path']

        # Load data and gt
        joint_2d_orignal = data['joints_2d']
        joint_2d_orignal[np.isnan(joint_2d_orignal)] = 0
              
        img_orignal = load_img(img_path)
        mask_original = cv2.imread(mask_gt_path)
        mask_original[mask_original<125] = 0
        mask_original[mask_original>125] = 255
        #
     
        h, w, c = img_orignal.shape

        # Reshape the image according to the label
        if img_shape[0] != w or img_shape[1] != h:
            img_resized = cv2.resize(img_orignal, img_shape, interpolation = cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask_original, img_shape, interpolation = cv2.INTER_NEAREST)
        else:
            img_resized = img_orignal
            mask_resized = mask_original
        

        scale, rot, color_scale, do_flip = get_augmentation_config(self.data_split)

        img_aug, trans_img, inv_trans_img = generate_patch_image(img_resized, bbox, scale, rot, do_flip, cfg.input_img_shape)     
        img_aug = np.clip(img_aug * color_scale[None, None, :], 0, 255)

        #img2bb_trans, bb2img_trans
        mask_aug, trans_aug, inv_trans_aug = generate_patch_image(mask_resized, bbox, scale, rot, do_flip, cfg.input_img_shape, flags=cv2.INTER_NEAREST)     
        #
        #mask_aug = np.clip(mask_aug * color_scale[None, None, :], 0, 255)
        

        # input_body_shape [256, 192]
        # input_img_shape [512, 384]
        # Augmentation
        # affine transform
  
        #img, img_kpt_aug, img2bb_trans, bb2img_trans, rot, do_flip = augmentation2(img, img_kpt, bbox, self.data_split)
        img_scaled = self.transform(img_aug.astype(np.float32)) / 255.

        inputs['img'] = img_scaled #{'img': img_scaled},


        mask_scaled = self.transform(mask_aug[...,2].astype(np.float32)) / 255.
        #pdb.set_trace()


        #img_kpt = cv2.resize(img_kpt, (cfg.input_body_shape[1],cfg.input_body_shape[0]), interpolation = cv2.INTER_LINEAR)
        #img_kpt_aug = cv2.resize(img_kpt_aug, (cfg.input_body_shape[1],cfg.input_body_shape[0]), interpolation = cv2.INTER_LINEAR)



        if self.data_split == 'train':
            if do_flip:
                joint_2d_orignal[:, 0] = img_shape[0] - 1 - joint_2d_orignal[:, 0]

                for pair in self.joint_set['flip_pairs']:
                    joint_2d_orignal[pair[0], :], joint_2d_orignal[pair[1], :] = joint_2d_orignal[pair[1], :].copy(), joint_2d_orignal[pair[0], :].copy()

            # img -> bboxs & augmentation
            joint_valid = np.ones_like(joint_2d_orignal[:, :1])
            joint_img_xy1 = np.concatenate((joint_2d_orignal[:, :2], np.ones_like(joint_2d_orignal[:, :1])), 1)

            joint_2d_aug = joint_2d_orignal.copy()
            joint_2d_aug[:, :2] = np.dot(trans_img, joint_img_xy1.transpose(1, 0)).transpose(1, 0) #img - > input shape

            


            if cfg.debug:
                kpt_image_original = img_orignal.copy()
                if img_shape[0] != w or img_shape[1] != h:
                    kpt_image_resize = cv2.resize(kpt_image_original, img_shape, interpolation = cv2.INTER_LINEAR)
                else:
                    kpt_image_resize = kpt_image_original

                # For debugging
                # KPT on original image
                for i in range(joint_2d_orignal.shape[0]):
                    if ((joint_2d_orignal[i, 0] > 0) *  (joint_2d_orignal[i, 0] < img_shape[0]) *
                        (joint_2d_orignal[i, 1] > 0) *  (joint_2d_orignal[i, 1] < img_shape[1])):
                        cv2.circle(
                            kpt_image_resize,  
                            (joint_2d_orignal[i,0].astype(np.int32), joint_2d_orignal[i,1].astype(np.int32)), 
                            radius=5, color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)    
                            
                #cv2.imwrite(f'kpt_on_resized.jpg', kpt_image_resize[:, :, ::-1])

                kpt_image_aug, trans_img, inv_trans_img = generate_patch_image(kpt_image_resize, bbox, scale, rot, do_flip, cfg.input_img_shape)     
                kpt_image_aug = np.clip(kpt_image_aug * color_scale[None, None, :], 0, 255)

                # KPT on augmented image
                for i in range(joint_2d_aug.shape[0]):
                    if ((joint_2d_aug[i, 0] > 0) * (joint_2d_aug[i, 0] < cfg.input_img_shape[1]) *
                        (joint_2d_aug[i, 1] > 0) * (joint_2d_aug[i, 1] < cfg.input_img_shape[0])):
                                    
                        cv2.circle(
                            kpt_image_aug, 
                            (joint_2d_aug[i,0].astype(np.int32), joint_2d_aug[i,1].astype(np.int32)), 
                            radius=2, color=(255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)

                #cv2.imwrite(f'kpt_on_augmented.jpg', kpt_image_aug[:, :, ::-1])

                #pdb.set_trace()

                #inputs['kpt_image_resize'] = kpt_image_resize
                inputs['kpt_image_aug'] = kpt_image_aug
                
                # Render augmented keypoints on the augmented image.
                #img2 = np.ascontiguousarray((kpt_image_aug.detach().numpy().copy()*255).transpose([1,2,0]) , dtype=np.uint8)
            

            joint_trunc_aug = joint_valid * \
                ((joint_2d_aug[:, 0] >= 0) * (joint_2d_aug[:, 0] < cfg.input_img_shape[0]) *
                 (joint_2d_aug[:, 1] >= 0) * (joint_2d_aug[:, 1] < cfg.input_img_shape[1])# * 
                 ).reshape(-1, 1).astype(np.float32)


            joint_2d_hm = joint_2d_aug.copy()
            joint_2d_hm[:, 0] = joint_2d_hm[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2] # # input shape to mesh shape
            joint_2d_hm[:, 1] = joint_2d_hm[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1] # input shape to mesh shape



            joint_trunc_hm = joint_valid * \
                ((joint_2d_hm[:, 0] >= 0) * (joint_2d_hm[:, 0] < cfg.output_hm_shape[2]) *
                 (joint_2d_hm[:, 1] >= 0) * (joint_2d_hm[:, 1] < cfg.output_hm_shape[1])# * 
                 ).reshape(-1, 1).astype(np.float32)
            

            joint_2d_gt = joint_2d_aug.copy()
            joint_2d_gt[:, 0] = joint_2d_gt[:, 0] / cfg.input_img_shape[1] * cfg.input_body_shape[1] # # input shape to mesh shape
            joint_2d_gt[:, 1] = joint_2d_gt[:, 1] / cfg.input_img_shape[0] * cfg.input_body_shape[0] # input shape to mesh shape



            joint_trunc_gt = joint_valid * \
                ((joint_2d_gt[:, 0] >= 0) * (joint_2d_gt[:, 0] < cfg.input_body_shape[1]) *
                 (joint_2d_gt[:, 1] >= 0) * (joint_2d_gt[:, 1] < cfg.input_body_shape[0])# * 
                 ).reshape(-1, 1).astype(np.float32)

            
            #joint_2d_hm[np.isnan(joint_2d_hm)] = 0
            




            targets = {'joint_img': joint_2d_gt, 
                       'mask_gt': mask_scaled,
                       'smplx_joint_img': joint_2d_hm,
                       #'joint_cam': joint_cam, '
                       #'smplx_joint_cam': joint_cam, 'smplx_pose': smplx_pose, 'smplx_shape': smplx_shape,
                       #'smplx_expr': smplx_expr, 'lhand_bbox_center': lhand_bbox_center,
                       #'lhand_bbox_size': lhand_bbox_size, 'rhand_bbox_center': rhand_bbox_center,
                       #'rhand_bbox_size': rhand_bbox_size, 'face_bbox_center': face_bbox_center,
                       #'face_bbox_size': face_bbox_size
                       }
            meta_info = {
                'joint_idx': np.array(self.keypoint_idx_posemodel),
                'joint_valid': joint_valid, 
                'joint_trunc': joint_trunc_gt,
                         #'smplx_joint_valid': np.zeros_like(joint_valid),
                         #'smplx_joint_trunc': np.zeros_like(joint_trunc), 'smplx_pose_valid': smplx_pose_valid,
                         #'smplx_shape_valid': float(smplx_shape_valid), 'smplx_expr_valid': float(smplx_expr_valid),
                         #'is_3D': float(True), 'lhand_bbox_valid': lhand_bbox_valid,
                         #'rhand_bbox_valid': rhand_bbox_valid, 'face_bbox_valid': face_bbox_valid
                         }


            return inputs, targets, meta_info
        

            # hand and face bbox transform
            #lhand_bbox, rhand_bbox, face_bbox = data['lhand_bbox'], data['rhand_bbox'], data['face_bbox']
            #lhand_bbox, lhand_bbox_valid = self.process_hand_face_bbox(lhand_bbox, do_flip, img_shape, img2bb_trans)
            #rhand_bbox, rhand_bbox_valid = self.process_hand_face_bbox(rhand_bbox, do_flip, img_shape, img2bb_trans)
            #face_bbox, face_bbox_valid = self.process_hand_face_bbox(face_bbox, do_flip, img_shape, img2bb_trans)
            #if do_flip:
            #    lhand_bbox, rhand_bbox = rhand_bbox, lhand_bbox
            #    lhand_bbox_valid, rhand_bbox_valid = rhand_bbox_valid, lhand_bbox_valid
            #lhand_bbox_center = (lhand_bbox[0] + lhand_bbox[1]) / 2.;
            #rhand_bbox_center = (rhand_bbox[0] + rhand_bbox[1]) / 2.;
            #face_bbox_center = (face_bbox[0] + face_bbox[1]) / 2.
            #lhand_bbox_size = lhand_bbox[1] - lhand_bbox[0];
            #rhand_bbox_size = rhand_bbox[1] - rhand_bbox[0];
            #face_bbox_size = face_bbox[1] - face_bbox[0];

            """
            # for debug
            _img = img.numpy().transpose(1,2,0)[:,:,::-1].copy() * 255
            if lhand_bbox_valid:
                _tmp = lhand_bbox.copy().reshape(2,2)
                _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
                _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
                cv2.rectangle(_img, (int(_tmp[0,0]), int(_tmp[0,1])), (int(_tmp[1,0]), int(_tmp[1,1])), (255,0,0), 3)
                cv2.imwrite('agora_' + str(idx) + '_lhand.jpg', _img)
            if rhand_bbox_valid:
                _tmp = rhand_bbox.copy().reshape(2,2)
                _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
                _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
                cv2.rectangle(_img, (int(_tmp[0,0]), int(_tmp[0,1])), (int(_tmp[1,0]), int(_tmp[1,1])), (255,0,0), 3)
                cv2.imwrite('agora_' + str(idx) + '_rhand.jpg', _img)
            if face_bbox_valid:
                _tmp = face_bbox.copy().reshape(2,2)
                _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
                _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
                cv2.rectangle(_img, (int(_tmp[0,0]), int(_tmp[0,1])), (int(_tmp[1,0]), int(_tmp[1,1])), (255,0,0), 3)
                cv2.imwrite('agora_' + str(idx) + '_face.jpg', _img)
            #cv2.imwrite('agora_' + str(idx) + '.jpg', _img)
            """

            # coordinates
            #joint_cam = joint_cam - joint_cam[self.joint_set['root_joint_idx'], None, :]  # root-relative
            #joint_cam[self.joint_set['joint_part']['lhand'], :] = joint_cam[self.joint_set['joint_part']['lhand'],
            #                                                      :] - joint_cam[self.joint_set['lwrist_idx'], None,
            #                                                           :]  # left hand root-relative
            #joint_cam[self.joint_set['joint_part']['rhand'], :] = joint_cam[self.joint_set['joint_part']['rhand'],
            #                                                      :] - joint_cam[self.joint_set['rwrist_idx'], None,
            #                                                           :]  # right hand root-relative
            #joint_cam[self.joint_set['joint_part']['face'], :] = joint_cam[self.joint_set['joint_part']['face'],
            #                                                     :] - joint_cam[self.joint_set['neck_idx'], None,
            #                                                          :]  # face root-relative
            #joint_img = np.concatenate((joint_img[:, :2], joint_cam[:, 2:]), 1)  # x, y, depth
            #joint_img[self.joint_set['joint_part']['body'], 2] = (joint_cam[self.joint_set['joint_part'][
            #                                                                    'body'], 2].copy() / (
            #                                                                  cfg.body_3d_size / 2) + 1) / 2. * \
            #                                                     cfg.output_hm_shape[0]  # body depth discretize
            #joint_img[self.joint_set['joint_part']['lhand'], 2] = (joint_cam[self.joint_set['joint_part'][
            #                                                                     'lhand'], 2].copy() / (
            #                                                                   cfg.hand_3d_size / 2) + 1) / 2. * \
            #                                                      cfg.output_hm_shape[0]  # left hand depth discretize
            #joint_img[self.joint_set['joint_part']['rhand'], 2] = (joint_cam[self.joint_set['joint_part'][
            #                                                                     'rhand'], 2].copy() / (
            #                                                                   cfg.hand_3d_size / 2) + 1) / 2. * \
            #                                                      cfg.output_hm_shape[0]  # right hand depth discretize
            #joint_img[self.joint_set['joint_part']['face'], 2] = (joint_cam[self.joint_set['joint_part'][
            #                                                                    'face'], 2].copy() / (
            #                                                                  cfg.face_3d_size / 2) + 1) / 2. * \
            #                                                     cfg.output_hm_shape[0]  # face depth discretize
            #joint_valid = np.ones_like(joint_img[:, :1])
            #joint_img, joint_cam, joint_valid, joint_trunc = process_db_coord(joint_img, joint_cam, joint_valid,
            #                                                                  do_flip, img_shape,
            #                                                                  self.joint_set['flip_pairs'],
            #                                                                  img2bb_trans, rot,
            #                                                                  self.joint_set['joints_name'],
            #                                                                  smpl.joints_name)
            
            

            


            
            #pdb.set_trace()



            #joint_bbox = 

            #pdb.set_trace()

            """
            # for debug
            _tmp = joint_img.copy() 
            _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
            _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
            _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
            _img = vis_keypoints(_img.copy(), _tmp)
            cv2.imwrite('agora_' + str(idx) + '.jpg', _img)
            """

            """
            # for debug
            _tmp = joint_cam.copy()[:,:2]
            _tmp[:,0] = _tmp[:,0] / (cfg.body_3d_size / 2) * cfg.input_img_shape[1] + cfg.input_img_shape[1]/2
            _tmp[:,1] = _tmp[:,1] / (cfg.body_3d_size / 2) * cfg.input_img_shape[0] + cfg.input_img_shape[0]/2
            _img = np.zeros((cfg.input_img_shape[0], cfg.input_img_shape[1], 3), dtype=np.float32)
            _img = vis_keypoints(_img.copy(), _tmp)
            cv2.imwrite('agora_' + str(idx) + '_cam.jpg', _img)
            """

            # smplx parameters
            #root_pose = np.array(smplx_param['global_orient'], dtype=np.float32).reshape(
            #    -1)  # rotation to world coordinate
            #body_pose = np.array(smplx_param['body_pose'], dtype=np.float32).reshape(-1)
            #shape = np.array(smplx_param['betas'], dtype=np.float32).reshape(-1)[:10]  # bug?
            #lhand_pose = np.array(smplx_param['left_hand_pose'], dtype=np.float32).reshape(-1)
            #rhand_pose = np.array(smplx_param['right_hand_pose'], dtype=np.float32).reshape(-1)
            #jaw_pose = np.array(smplx_param['jaw_pose'], dtype=np.float32).reshape(-1)
            #expr = np.array(smplx_param['expression'], dtype=np.float32).reshape(-1)
            #trans = np.array(smplx_param['transl'], dtype=np.float32).reshape(-1)  # translation to world coordinate
            cam_param = {'focal': cfg.focal,
                         'princpt': cfg.princpt}  # put random camera paraemter as we do not use coordinates from smplx parameters
            #smplx_param = {'root_pose': root_pose, 'body_pose': body_pose, 'shape': shape,
            #               'lhand_pose': lhand_pose, 'lhand_valid': True,
            #               'rhand_pose': rhand_pose, 'rhand_valid': True,
            #               'jaw_pose': jaw_pose, 'expr': expr, 'face_valid': True,
            #               'trans': trans}
            #_, _, _, smplx_pose, smplx_shape, smplx_expr, smplx_pose_valid, _, smplx_expr_valid, _ = process_human_model_output(
            #    smplx_param, cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx')
            #smplx_pose_valid = np.tile(smplx_pose_valid[:, None], (1, 3)).reshape(-1)
            #smplx_pose_valid[
            #:3] = 0  # global orient of the provided parameter is a rotation to world coordinate system. I want camera coordinate system.
            #smplx_shape_valid = True

            
        else:
            # load crop and resize information (for the 4K setting)
            #if self.resolution == (2160, 3840):
            #    img2bb_trans = np.dot(
            #        np.concatenate((img2bb_trans,
            #                        np.array([0, 0, 1], dtype=np.float32).reshape(1, 3))),
            #        np.concatenate((data['img2bb_trans_from_orig'],
            #                        np.array([0, 0, 1], dtype=np.float32).reshape(1, 3)))
            #    )
            #    bb2img_trans = np.linalg.inv(img2bb_trans)[:2, :]
            #    img2bb_trans = img2bb_trans[:2, :]

            pdb.set_trace()
            if self.test_set == 'val':
                # gt load
                #with open(data['verts_path']) as f:
                #    verts = np.array(json.load(f)).reshape(-1, 3)

                inputs = {'img': img_scaled}
                targets = {'smplx_mesh_cam': data['joints_2d']}
                meta_info = {'bb2img_trans': inv_trans_img,#trans_img, inv_trans_img
                             'img_path': img_path,
                             'img_shape': np.array(img_shape)}
            else:
                inputs = {'img': img_scaled}
                targets = {'smplx_mesh_cam': np.zeros((smpl.vertex_num, 3), dtype=np.float32)}  # dummy vertex
                meta_info = {'bb2img_trans': inv_trans_img,
                        'joint_idx': np.array(self.keypoint_idx_posemodel),
                             'img_path': img_path}

            return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {
            'pa_mpvpe_all': [], 
            'pa_mpvpe_hand': [], 
            'pa_mpvpe_face': [], 
            'mpvpe_all': [], 
            'mpvpe_hand': [], 
            'mpvpe_face': [],
            'mse_joint': [],
            'loss_lhand_bbox': [],
            'loss_rhand_bbox': [],
            'loss_face_bbox': [],
            'loss_joint_proj': [],
            'loss_joint_img': [],
            'loss_joint_img_face': [],
            'loss_smplx_joint_img': []
            }
        

  

        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]

            joint_image_gt = annot['joints_2d']

            #a = smpl.reduce_joint_set(joint_image_gt)


            # The ground truth is using COCO format which has 18 keypoints




            out = outs[n]

            # get output

            inpuit_img = out['img']
            joint_img = out['joint_img']
            joint_proj = out['smplx_joint_proj']
            root_pose = out['smplx_root_pose']
            body_pose = out['smplx_body_pose']# = body_pose

            lhand_pose = out['smplx_lhand_pose']# = lhand_pose
            lhand_bbox = out['lhand_bbox']# = lhand_bbox

            rhand_pose = out['smplx_rhand_pose']# = rhand_pose
            rhand_bbox = out['rhand_bbox']# = rhand_bbox

            jaw_pose = out['smplx_jaw_pose']# = jaw_pose
            face_bbox = out['face_bbox']# = face_bbox

            shape = out['smplx_shape']# = shape
            expr = out['smplx_expr']# = expr

            mesh_cam = out['smplx_mesh_cam']
            cam_trans = out['cam_trans']# = cam_trans

            bb2img_trans = out['bb2img_trans']
            
            mesh_gt = out['smplx_mesh_cam_target']
            mesh_out = out['smplx_mesh_cam']
            #3mg = out['img'].detach().cpu().numpy().transpose([1,2,0])

            # Ground truth keypoint, you only have on the original frame
            # You have to project the detection result back to the original image

            joint_proj = joint_proj[self.keypoint_idx_posemodel]
            joint_proj[:, 0] = joint_proj[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
            joint_proj[:, 1] = joint_proj[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
            
            joint_proj = np.concatenate((joint_proj, np.ones_like(joint_proj[:, :1])), 1)
            joint_proj = np.dot(bb2img_trans, joint_proj.transpose(1, 0)).transpose(1, 0)

            # render mesh

            # get
            vis = True
            if vis:
                from common.utils.vis import vis_keypoints, vis_mesh, save_obj, render_mesh, vis_keypoints_error
                vis_mesh = load_img(annot['img_path'])
                vis_mesh = cv2.resize(vis_mesh, annot['img_shape'], interpolation = cv2.INTER_LINEAR)
                vis_kpts = vis_mesh.copy()

                bbox = annot['bbox']
                focal = [
                    cfg.focal[0] / cfg.input_body_shape[1] * bbox[2],
                    cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]
                    ]

                princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0],
                        cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]




                vis_mesh = render_mesh(vis_mesh, out['smplx_mesh_cam'],
                                    smpl.face,
                                    {'focal': focal, 'princpt': princpt})

                #vis_kpts = vis_keypoints(vis_mesh, joint_proj)
                vis_kpts = vis_keypoints_error(vis_mesh, joint_proj, joint_image_gt)

                cv2.imwrite(os.path.join(cfg.vis_dir, out['img_path'].split('/')[-1].split('.')[0] + f'kpt.jpg'), vis_kpts[:, :, ::-1])
                cv2.imwrite(os.path.join(cfg.vis_dir, out['img_path'].split('/')[-1].split('.')[0] + f'render.jpg'), vis_mesh[:, :, ::-1])


            acc = 0.0
            cut = 0
            for i in range(joint_image_gt.shape[0]):
                if joint_image_gt[i, 0]>0 and joint_image_gt[i,0] < annot['img_shape'][0] and joint_image_gt[i, 1]>0 and joint_image_gt[i,1] < annot['img_shape'][1]:
                    acc = acc + np.sqrt((joint_image_gt[i, 0]-joint_proj[i, 0]) ** 2 + (joint_image_gt[i, 1] - joint_proj[i, 1]) **2)
                    cut = cut + 1
                else:
                    #raise NotImplementedError()
                    pass
            if cut == 0:
                acc = 0
            else:
                acc = acc / cut

            eval_result['loss_joint_img'].append(acc)


        return eval_result

    def print_eval_result(self, eval_result):

        print('AGORA test results are dumped at: ' + osp.join(cfg.result_dir, 'predictions'))

        if self.data_split == 'test' and self.test_set == 'test':  # do not print. just submit the results to the official evaluation server
            return

        #print('PA MPVPE (All): %.2f mm' % np.mean(eval_result['pa_mpvpe_all']))
        #print('PA MPVPE (Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_hand']))
        #print('PA MPVPE (Face): %.2f mm' % np.mean(eval_result['pa_mpvpe_face']))

        #print('MPVPE (All): %.2f mm' % np.mean(eval_result['mpvpe_all']))
        #print('MPVPE (Hands): %.2f mm' % np.mean(eval_result['mpvpe_hand']))
        #print('MPVPE (Face): %.2f mm' % np.mean(eval_result['mpvpe_face']))
        print('loss_joint_img: %.2f mm' % np.mean(eval_result['loss_joint_img']))