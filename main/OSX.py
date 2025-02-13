import torch
import torch.nn as nn
from torch.nn import functional as F
from common.nets.module import PositionNet, HandRotationNet, FaceRegressor, BoxNet, BoxSizeNet, HandRoI, FaceRoI, BodyRotationNet
from common.nets.loss import CoordLoss, ParamLoss, CELoss, SilhouetteLoss
#from common.utils.human_models import smpl_h, smpl_x
from common.utils.transforms import rot6d_to_axis_angle, restore_bbox
from config import cfg
from common.utils.vis import render_mesh, save_obj, vis_keypoints
import cv2
import math
import copy
from mmpose.models import build_posenet
from mmcv import Config
import os
import pdb
import numpy as np
from render_p3d import base_renderer
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self, encoder, body_position_net, body_rotation_net, box_net, hand_position_net, hand_roi_net, hand_decoder,
                 hand_rotation_net, face_position_net, face_roi_net, face_decoder, face_regressor, smpl):
        super(Model, self).__init__()
        # body
        self.encoder = encoder
        self.body_position_net = body_position_net
        self.body_regressor = body_rotation_net
        self.box_net = box_net

        # hand
        self.hand_roi_net = hand_roi_net
        self.hand_position_net = hand_position_net
        self.hand_decoder = hand_decoder
        self.hand_regressor = hand_rotation_net

        # face
        self.face_roi_net = face_roi_net
        self.face_position_net = face_position_net
        self.face_decoder = face_decoder
        self.face_regressor = face_regressor

        self.smpl = smpl
        self.smpl_layer = copy.deepcopy(self.smpl.layer['neutral']).cuda()

        self.coord_loss = CoordLoss()
        self.param_loss = ParamLoss()
        self.ce_loss = CELoss()
        self.mask_loss = SilhouetteLoss()

        self.body_num_joints = len(smpl.pos_joint_part['body'])
        self.hand_num_joints = len(smpl.pos_joint_part['rhand'])

        self.trainable_modules = [self.encoder, self.body_position_net, self.body_regressor,
                                  self.box_net, self.hand_position_net, self.hand_roi_net, self.hand_regressor,
                                  self.face_regressor, self.face_roi_net, self.face_position_net]
        self.special_trainable_modules = [self.hand_decoder, self.face_decoder]


        # Definining the render

        focal = [
            cfg.focal[0] / cfg.input_body_shape[1] * cfg.input_body_shape[1], 
            cfg.focal[1] / cfg.input_body_shape[0] * cfg.input_body_shape[0]
        ]

        princpt = [
            cfg.princpt[0] / cfg.input_body_shape[1] * cfg.input_body_shape[1], # / cfg.input_body_shape[1], 
            cfg.princpt[1] / cfg.input_body_shape[0] * cfg.input_body_shape[0]# / cfg.input_body_shape[0]
        ]



        #pdb.set_trace()

        self.camera_screen = base_renderer(
            size=cfg.input_body_shape,
            focal=focal,
            principal_point=princpt,
            device='cuda',
            colorRender=True)














    def get_camera_trans(self, cam_param):
        # camera translation
        if cfg.model_type == 'smil_h':
            t_xy = cam_param[:, :2] + torch.Tensor([0.1, -1]).cuda()
            #print (t_xy)
        else: 
            t_xy = cam_param[:, :2]
        gamma = torch.sigmoid(cam_param[:, 2])  # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(cfg.focal[0] * cfg.focal[1] * cfg.camera_3d_size * cfg.camera_3d_size / (
                cfg.input_body_shape[0] * cfg.input_body_shape[1]))]).cuda().view(-1)
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:, None]), 1)
        if cfg.model_type == 'smil_h':
            return cam_trans / 3
        else:
            return cam_trans

    def get_coord(self, root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr, cam_trans, mode):
        batch_size = root_pose.shape[0]
        zero_pose = torch.zeros((1, 3)).float().cuda().repeat(batch_size, 1)  # eye poses
        output = self.smpl_layer(betas=shape, body_pose=body_pose, global_orient=root_pose, right_hand_pose=rhand_pose,
                                  left_hand_pose=lhand_pose, jaw_pose=jaw_pose, leye_pose=zero_pose,
                                  reye_pose=zero_pose, expression=expr)
        # camera-centered 3D coordinate
        mesh_cam = output.vertices
        if mode == 'test' and cfg.testset == 'AGORA':  # use 144 joints for AGORA evaluation
            joint_cam = output.joints
        else:
            joint_cam = output.joints[:, self.smpl.joint_idx, :] # get the joint regressed by the mesh

        if mode == 'train' and len(cfg.trainset_3d) == 1 and cfg.trainset_3d[0] == 'AGORA' and len(
                cfg.trainset_2d) == 0:  # prevent gradients from backpropagating to SMPLX parameter regression module
            x = (joint_cam[:, :, 0].detach() + cam_trans[:, None, 0]) / (
                    joint_cam[:, :, 2].detach() + cam_trans[:, None, 2] + 1e-4) * cfg.focal[0] + cfg.princpt[0]
            y = (joint_cam[:, :, 1].detach() + cam_trans[:, None, 1]) / (
                    joint_cam[:, :, 2].detach() + cam_trans[:, None, 2] + 1e-4) * cfg.focal[1] + cfg.princpt[1]
        else:
            x = (joint_cam[:, :, 0] + cam_trans[:, None, 0]) / (joint_cam[:, :, 2] + cam_trans[:, None, 2] + 1e-4) * \
                cfg.focal[0] + cfg.princpt[0]
            y = (joint_cam[:, :, 1] + cam_trans[:, None, 1]) / (joint_cam[:, :, 2] + cam_trans[:, None, 2] + 1e-4) * \
                cfg.focal[1] + cfg.princpt[1]
        x = x / cfg.input_body_shape[1] * cfg.output_hm_shape[2]
        y = y / cfg.input_body_shape[0] * cfg.output_hm_shape[1]
        joint_proj = torch.stack((x, y), 2)

        # root-relative 3D coordinates
        root_cam = joint_cam[:, self.smpl.root_joint_idx, None, :]
        joint_cam = joint_cam - root_cam
        mesh_cam = mesh_cam + cam_trans[:, None, :]  # for rendering

        # left hand root (left wrist)-relative 3D coordinates
        lhand_idx = self.smpl.joint_part['lhand']
        lhand_cam = joint_cam[:, lhand_idx, :]
        lwrist_cam = joint_cam[:, self.smpl.lwrist_idx, None, :]
        lhand_cam = lhand_cam - lwrist_cam
        joint_cam = torch.cat((joint_cam[:, :lhand_idx[0], :], lhand_cam, joint_cam[:, lhand_idx[-1] + 1:, :]), 1)

        # right hand root (right wrist)-relative 3D coordinates
        rhand_idx = self.smpl.joint_part['rhand']
        rhand_cam = joint_cam[:, rhand_idx, :]
        rwrist_cam = joint_cam[:, self.smpl.rwrist_idx, None, :]
        rhand_cam = rhand_cam - rwrist_cam
        joint_cam = torch.cat((joint_cam[:, :rhand_idx[0], :], rhand_cam, joint_cam[:, rhand_idx[-1] + 1:, :]), 1)

        # face root (neck)-relative 3D coordinates
        face_idx = self.smpl.joint_part['face']
        if joint_cam.shape[1] >70:
            face_cam = joint_cam[:, face_idx, :]
            neck_cam = joint_cam[:, self.smpl.neck_idx, None, :]
            face_cam = face_cam - neck_cam
            joint_cam = torch.cat((joint_cam[:, :face_idx[0], :], face_cam, joint_cam[:, face_idx[-1] + 1:, :]), 1)

        return joint_proj, joint_cam, mesh_cam

    def get_coord_hshi(self, root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr, cam_trans, mode):
        batch_size = root_pose.shape[0]
        zero_pose = torch.zeros((1, 3)).float().cuda().repeat(batch_size, 1)  # eye poses
        output = self.smpl_layer(betas=shape, body_pose=body_pose, global_orient=root_pose, right_hand_pose=rhand_pose,
                                  left_hand_pose=lhand_pose, jaw_pose=jaw_pose, leye_pose=zero_pose,
                                  reye_pose=zero_pose, expression=expr)
        # camera-centered 3D coordinate
        mesh_cam = output.vertices
        if mode == 'test' and cfg.testset == 'AGORA':  # use 144 joints for AGORA evaluation
            joint_cam = output.joints
        else:
            joint_cam = output.joints[:, self.smpl.joint_idx, :]

        joint_proj = joint_cam + cam_trans[:, None, :]

        # root-relative 3D coordinates
        joint_cam_tr = joint_cam + cam_trans[:, None, :]
        root_cam = joint_cam[:, self.smpl.root_joint_idx, None, :]  #??????? why taking this root cam away?
        joint_cam = joint_cam - root_cam
        mesh_cam = mesh_cam + cam_trans[:, None, :]  # for rendering MOVING THE MESH ACCORDING TO THE TRANSLATION

        # left hand root (left wrist)-relative 3D coordinates
        lhand_idx = self.smpl.joint_part['lhand']
        lhand_cam = joint_cam[:, lhand_idx, :]
        lwrist_cam = joint_cam[:, self.smpl.lwrist_idx, None, :]
        lhand_cam = lhand_cam - lwrist_cam
        joint_cam = torch.cat((joint_cam[:, :lhand_idx[0], :], lhand_cam, joint_cam[:, lhand_idx[-1] + 1:, :]), 1)

        # right hand root (right wrist)-relative 3D coordinates
        rhand_idx = self.smpl.joint_part['rhand']
        rhand_cam = joint_cam[:, rhand_idx, :]
        rwrist_cam = joint_cam[:, self.smpl.rwrist_idx, None, :]
        rhand_cam = rhand_cam - rwrist_cam
        joint_cam = torch.cat((joint_cam[:, :rhand_idx[0], :], rhand_cam, joint_cam[:, rhand_idx[-1] + 1:, :]), 1)

        # face root (neck)-relative 3D coordinates
        face_idx = self.smpl.joint_part['face']
        if joint_cam.shape[1] >70:
            face_cam = joint_cam[:, face_idx, :]
            neck_cam = joint_cam[:, self.smpl.neck_idx, None, :]
            face_cam = face_cam - neck_cam
            joint_cam = torch.cat((joint_cam[:, :face_idx[0], :], face_cam, joint_cam[:, face_idx[-1] + 1:, :]), 1)

        return joint_proj, joint_cam, mesh_cam, output, joint_cam_tr

    def generate_mesh_gt(self, targets, mode):
        if 'smplx_mesh_cam' in targets:
            return targets['smplx_mesh_cam']
        nums = [3, 63, 45, 45, 3]
        accu = []
        temp = 0
        for num in nums:
            temp += num
            accu.append(temp)
        pose = targets['smplx_pose']
        root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose = \
            pose[:, :accu[0]], pose[:, accu[0]:accu[1]], pose[:, accu[1]:accu[2]], pose[:, accu[2]:accu[3]], pose[:,accu[3]:accu[4]]
        shape = targets['smplx_shape']
        expr = targets['smplx_expr']
        cam_trans = targets['smplx_cam_trans']

        # final output
        joint_proj, joint_cam, mesh_cam = self.get_coord(root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape,
                                                         expr, cam_trans, mode)

        return mesh_cam

    def norm2heatmap(self, input, hm_shape):
        assert input.shape[-1] in [2, 3, 4]
        if input.shape[-1] == 2:
            x, y = input[..., 0], input[..., 1]
            x = x * hm_shape[2]
            y = y * hm_shape[1]
            output = torch.stack((x, y), dim=-1)
        elif input.shape[-1] == 3:
            x, y, z = input[..., 0], input[..., 1], input[..., 2]
            x = x * hm_shape[2]
            y = y * hm_shape[1]
            z = z * hm_shape[0]
            output = torch.stack((x, y, z), dim=-1)
        elif input.shape[-1] == 4:
            x, y, w, h = input[..., 0], input[..., 1], input[..., 2], input[..., 3]
            x = x * hm_shape[2]
            y = y * hm_shape[1]
            w = w * hm_shape[2]
            h = h * hm_shape[1]
            output = torch.stack((x, y, w, h), dim=-1)
        return output

    def heatmap2norm(self, input, hm_shape):
        assert input.shape[-1] in [2, 3, 4]
        if input.shape[-1] == 2:
            x, y = input[..., 0], input[..., 1]
            x = x / hm_shape[2]
            y = y / hm_shape[1]
            output = torch.stack((x, y), dim=-1)
        elif input.shape[-1] == 3:
            x, y, z = input[..., 0], input[..., 1], input[..., 2]
            x = x / hm_shape[2]
            y = y / hm_shape[1]
            z = z / hm_shape[0]
            output = torch.stack((x, y, z), dim=-1)
        elif input.shape[-1] == 4:
            x, y, w, h = input[..., 0], input[..., 1], input[..., 2], input[..., 3]
            x = x / hm_shape[2]
            y = y / hm_shape[1]
            w = w / hm_shape[2]
            h = h / hm_shape[1]
            output = torch.stack((x, y, w, h), dim=-1)

        return output

    def bbox_split(self, bbox):
        # bbox:[bs, 3, 3]
        lhand_bbox_center, rhand_bbox_center, face_bbox_center = \
            bbox[:, 0, :2], bbox[:, 1, :2], bbox[:, 2, :2]
        return lhand_bbox_center, rhand_bbox_center, face_bbox_center

    def forward(self, inputs, targets, meta_info, mode):
        
        body_img = F.interpolate(inputs['img'], cfg.input_body_shape)
        batch_size = body_img.shape[0]

        # 1. Encoder
        img_feat, task_tokens = self.encoder(body_img)  # task_token:[bs, N, c]
        shape_token, cam_token, expr_token, jaw_pose_token, hand_token, body_pose_token = \
            task_tokens[:, 0], task_tokens[:, 1], task_tokens[:, 2], task_tokens[:, 3], task_tokens[:, 4:6], task_tokens[:, 6:]

        # 2. Body Regressor
        body_joint_hm, body_joint_img = self.body_position_net(img_feat)
        root_pose, body_pose, shape, cam_param, = self.body_regressor(body_pose_token, shape_token, cam_token, body_joint_img.detach())
        root_pose = rot6d_to_axis_angle(root_pose)
        body_pose = rot6d_to_axis_angle(body_pose.reshape(-1, 6)).reshape(body_pose.shape[0], -1)  # (N, J_R*3)
        cam_trans = self.get_camera_trans(cam_param)

        # 3. Hand and Face BBox Estimation
        lhand_bbox_center, lhand_bbox_size, rhand_bbox_center, rhand_bbox_size, face_bbox_center, face_bbox_size = self.box_net(img_feat, body_joint_hm.detach())
        lhand_bbox = restore_bbox(lhand_bbox_center, lhand_bbox_size, cfg.input_hand_shape[1] / cfg.input_hand_shape[0], 2.0).detach()  # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space
        rhand_bbox = restore_bbox(rhand_bbox_center, rhand_bbox_size, cfg.input_hand_shape[1] / cfg.input_hand_shape[0], 2.0).detach()  # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space
        face_bbox = restore_bbox(face_bbox_center, face_bbox_size, cfg.input_face_shape[1] / cfg.input_face_shape[0], 1.5).detach()  # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space

        # 4. Differentiable Feature-level Hand/Face Crop-Upsample
        # hand_feat: list, [bsx2, c, cfg.output_hand_hm_shape[1]*scale, cfg.output_hand_hm_shape[2]*scale]
        hand_feats = self.hand_roi_net(img_feat, lhand_bbox, rhand_bbox)  # list, hand_feat: flipped left hand + right hand
        # face_feat: list, [bs, c, cfg.output_face_hm_shape[1]*scale, cfg.output_face_hm_shape[2]*scale]
        face_feats = self.face_roi_net(img_feat, face_bbox)

        # 4. keypoint-guided deformable decoder
        # hand keypoint-guided deformable decoder
        _, hand_joint_img, hand_img_feat_joints = self.hand_position_net(hand_feats[-2])  # (2N, J_P, 3) in (hand_hm_shape[2], hand_hm_shape[1], hand_hm_shape[0]) space
        # [-2]: scale=2, because the roi size = (hand_hm_shape*scale//2)
        hand_coord_init = self.heatmap2norm(hand_joint_img, cfg.output_hand_hm_shape)
        hand_img_feat_joints = self.hand_decoder(hand_feats, coord_init=hand_coord_init.detach(), query_init=hand_img_feat_joints)
        # hand regression head
        hand_pose = self.hand_regressor(hand_img_feat_joints, hand_joint_img.detach())
        hand_pose = rot6d_to_axis_angle(hand_pose.reshape(-1, 6)).reshape(hand_img_feat_joints.shape[0], -1)  # (2N, J_R*3)
        # restore flipped left hand joint coordinates
        batch_size = hand_joint_img.shape[0] // 2
        lhand_joint_img = hand_joint_img[:batch_size, :, :]
        lhand_joint_img = torch.cat(
            (cfg.output_hand_hm_shape[2] - 1 - lhand_joint_img[:, :, 0:1], lhand_joint_img[:, :, 1:]), 2)
        rhand_joint_img = hand_joint_img[batch_size:, :, :]
        # restore flipped left hand joint rotations
        batch_size = hand_pose.shape[0] // 2
        lhand_pose = hand_pose[:batch_size, :].reshape(-1, len(self.smpl.orig_joint_part['lhand']), 3)
        lhand_pose = torch.cat((lhand_pose[:, :, 0:1], -lhand_pose[:, :, 1:3]), 2).view(batch_size, -1)
        rhand_pose = hand_pose[batch_size:, :]

        # face keypoint-guided deformable decoder
        _, face_joint_img, face_img_feat_joints = self.face_position_net(face_feats[-2])  # (N, J_P, 3) in (face_hm_shape[2], face_hm_shape[1], face_hm_shape[0]) space
        face_coord_init = self.heatmap2norm(face_joint_img, cfg.output_face_hm_shape)
        face_img_feat_joints = self.face_decoder(face_feats, coord_init=face_coord_init.detach(), query_init=face_img_feat_joints)
        # face regression head
        expr, jaw_pose = self.face_regressor(face_img_feat_joints, face_joint_img.detach(), face_feats[-1])
        jaw_pose = rot6d_to_axis_angle(jaw_pose)

        # final output
        joint_proj, joint_cam, mesh_cam, regoutput, joint_cam_tr = self.get_coord_hshi(root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr, cam_trans, mode)
        #pdb.set_trace()
        #joint_proj, joint_cam, mesh_cam, regoutput = self.get_coord_hshi(root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr, cam_trans, mode)
        pose = torch.cat((root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose), 1)
        joint_img = torch.cat((body_joint_img, lhand_joint_img, rhand_joint_img), 1)

        #pdb.set_trace()
        #print (mesh_cam.shape)
        #print (torch.from_numpy(self.smpl.face.astype('int')).unsqueeze(0).repeat(batch_size,1,1).shape)
        #print (joint_proj.shape)
        #pdb.set_trace()
        silhouette, joint_proj = self.camera_screen(mesh_cam, torch.from_numpy(self.smpl.face.astype('int')).unsqueeze(0).repeat(batch_size,1,1), joint_proj)
        


            
        if mode == 'test' and 'smplx_pose' in targets:
            mesh_pseudo_gt = self.generate_mesh_gt(targets, mode)

        if mode == 'train':

            joint_proj = joint_proj[:, meta_info['joint_idx'][0,:], :]
            
            if cfg.debug:
                from common.utils.vis import vis_keypoints, vis_mesh, save_obj, render_mesh, vis_keypoints_error

                vis_ind = -1
                #pdb.set_trace()
                #cv2.imwrite(f'debug_org_pkt.jpg', np.ascontiguousarray(inputs['kpt_image_aug'][vis_ind, ...].detach().cpu().numpy()[:, :, ::-1], dtype=np.uint8))
                
                cv2.imwrite(f'debug_org_kpt_aug.jpg', 
                            np.ascontiguousarray(inputs['kpt_image_aug'][vis_ind, ...].detach().cpu().numpy()[:, :, ::-1], dtype=np.uint8))
                
                cv2.imwrite(f'debug_mask_gt.jpg', 
                            np.ascontiguousarray(targets['mask_gt'][vis_ind, ...].permute([1,2,0]).detach().cpu().numpy()[:, :, ::-1] * 255, dtype=np.uint8))

                kpt_gt = targets['joint_img'][vis_ind, ...].cpu().numpy().copy()


                kpt_pred_debug = joint_proj[vis_ind,...].detach().cpu().numpy()

                # 
                img_input = np.ascontiguousarray(body_img[vis_ind, ...].permute([1,2,0]).detach().cpu().numpy() * 255, dtype=np.uint8)
                img_input = vis_keypoints(img_input, kpt_gt)
                cv2.imwrite(f'debug_kpt_gt.jpg', img_input[:, :, ::-1])


                # See kpt render result

                img_input = np.ascontiguousarray(body_img[vis_ind, ...].permute([1,2,0]).detach().cpu().numpy() * 255, dtype=np.uint8)
                img_input = vis_keypoints(img_input, kpt_pred_debug)
                cv2.imwrite(f'debug_kpt_pred.jpg', img_input[:, :, ::-1])
                    

                    #
                img_input = np.ascontiguousarray(body_img[vis_ind, ...].permute([1,2,0]).detach().cpu().numpy() * 255, dtype=np.uint8)
                img_input = vis_keypoints_error(img_input, kpt_pred_debug, kpt_gt)
                cv2.imwrite(f'debug_kpt_error.jpg', img_input[:, :, ::-1])

                    
                # See mesh render result and predicted kpts
                rendered = torch.multiply(body_img[vis_ind,...].permute([1,2,0]), 1 - silhouette[vis_ind, :, :, 3].unsqueeze(-1)) + silhouette[vis_ind, :, :, 3].unsqueeze(-1)
                img_input = vis_keypoints(np.ascontiguousarray(rendered.detach().cpu().numpy() * 255, dtype=np.uint8), kpt_gt)
                cv2.imwrite(f'debug_kpt_gt_rendered.jpg', img_input[:, :, ::-1])


                rendered = torch.multiply(body_img[vis_ind,...].permute([1,2,0]), 1 - silhouette[vis_ind, :, :, 3].unsqueeze(-1)) + silhouette[vis_ind, :, :, 3].unsqueeze(-1)
                img_input = vis_keypoints(np.ascontiguousarray(rendered.detach().cpu().numpy() * 255, dtype=np.uint8), kpt_pred_debug)
                cv2.imwrite(f'debug_kpt_pred_rendered.jpg', img_input[:, :, ::-1])


            


            #img_input = render_mesh(img_input, mesh_cam[-1, ...].cpu().detach().numpy(), self.smpl.face, {'focal': focal, 'princpt': princpt})
            img_input = np.ascontiguousarray(body_img[vis_ind, ...].permute([1,2,0]).detach().cpu().numpy() * 255, dtype=np.uint8)
            img_input = render_mesh(img_input, mesh_cam[vis_ind, ...].cpu().detach().numpy(), self.smpl.face, {'focal': cfg.focal, 'princpt': cfg.princpt})
            #pdb.set_trace()
            for i in range(kpt_pred_debug.shape[0]):
                if ((kpt_pred_debug[i, 0] >= 0) * (kpt_pred_debug[i, 0] < cfg.input_body_shape[1]) *
                    (kpt_pred_debug[i, 1] >= 0) * (kpt_pred_debug[i, 1] < cfg.input_body_shape[0])):
                    cv2.circle(img_input, 
                               (int(kpt_pred_debug[i, 0]), int(kpt_pred_debug[i, 1])),  radius=1, color=(0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)

            cv2.imwrite(f'debug_kpt_pred_old_render.jpg', img_input[:, :, ::-1])

            # loss functions
            loss = {}
            loss['joint_proj'] = self.coord_loss(joint_proj, 
                                                 targets['joint_img'], 
                                                 meta_info['joint_trunc'])
            mask_gt = F.interpolate(targets['mask_gt'], cfg.input_body_shape, mode='nearest')
            #pdb.set_trace()
            loss['mask'] = self.mask_loss(silhouette[...,3], mask_gt.squeeze(1))

            #print (loss['joint_proj'])
            #loss['joint_img'] = self.coord_loss(joint_img, 
            #                                    self.smpl.reduce_joint_set(targets['joint_img']),
            #                                    self.smpl.reduce_joint_set(meta_info['joint_trunc']), 
            #                                    meta_info['is_3D'])
            
            #loss['joint_img_face'] = self.coord_loss(face_joint_img, 
            #                                         targets['joint_img'][:, smpl_x.joint_part['face']],
            #                                         meta_info['joint_trunc'][:, smpl_x.joint_part['face']], 
            #                                         meta_info['is_3D'])
            
            #loss['smplx_joint_img'] = self.coord_loss(joint_img, 
            #                                          smpl_x.reduce_joint_set(targets['smplx_joint_img']),
            #                                          smpl_x.reduce_joint_set(meta_info['smplx_joint_trunc']))
            return loss
        else:
            # change hand output joint_img according to hand bbox
            for part_name, bbox in (('lhand', lhand_bbox), ('rhand', rhand_bbox)):
                joint_img[:, self.smpl.pos_joint_part[part_name], 0] *= (
                        ((bbox[:, None, 2] - bbox[:, None, 0]) / cfg.input_body_shape[1] * cfg.output_hm_shape[2]) /
                        cfg.output_hand_hm_shape[2])
                joint_img[:, self.smpl.pos_joint_part[part_name], 0] += (
                        bbox[:, None, 0] / cfg.input_body_shape[1] * cfg.output_hm_shape[2])
                joint_img[:, self.smpl.pos_joint_part[part_name], 1] *= (
                        ((bbox[:, None, 3] - bbox[:, None, 1]) / cfg.input_body_shape[0] * cfg.output_hm_shape[1]) /
                        cfg.output_hand_hm_shape[1])
                joint_img[:, self.smpl.pos_joint_part[part_name], 1] += (
                        bbox[:, None, 1] / cfg.input_body_shape[0] * cfg.output_hm_shape[1])

            # change input_body_shape to input_img_shape
            for bbox in (lhand_bbox, rhand_bbox, face_bbox):
                bbox[:, 0] *= cfg.input_img_shape[1] / cfg.input_body_shape[1]
                bbox[:, 1] *= cfg.input_img_shape[0] / cfg.input_body_shape[0]
                bbox[:, 2] *= cfg.input_img_shape[1] / cfg.input_body_shape[1]
                bbox[:, 3] *= cfg.input_img_shape[0] / cfg.input_body_shape[0]


            out = {}
            out['img'] = inputs['img']
            out['debug_img'] = body_img
            #out['debug_silhouette'] = silhouette
            #out['debug_joint_proj'] = joint_proj
            #out['debug_rendered'] = rendered
            out['joint_img'] = joint_proj#joint_img
            out['smplx_joint_proj'] = joint_proj
            out['smplx_mesh_cam'] = mesh_cam
            out['smplx_root_pose'] = root_pose
            out['smplx_body_pose'] = body_pose
            out['smplx_lhand_pose'] = lhand_pose
            out['smplx_rhand_pose'] = rhand_pose
            out['smplx_jaw_pose'] = jaw_pose
            out['smplx_shape'] = shape
            out['smplx_expr'] = expr
            out['cam_trans'] = cam_trans
            out['lhand_bbox'] = lhand_bbox
            out['rhand_bbox'] = rhand_bbox
            out['face_bbox'] = face_bbox
            out['outpu'] = regoutput
            out['joint_cam_tr'] = joint_cam_tr
            if 'smplx_pose' in targets:
                out['smplx_mesh_cam_pseudo_gt'] = mesh_pseudo_gt
            if 'smplx_mesh_cam' in targets:
                out['smplx_mesh_cam_target'] = targets['smplx_mesh_cam']
            if 'bb2img_trans' in meta_info:
                out['bb2img_trans'] = meta_info['bb2img_trans']
            return out

def init_weights(m):
    try:
        if type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight, std=0.001)
        elif type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.BatchNorm2d:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
    except AttributeError:
        pass


def get_model(smpl, mode):
    # body
    vit_cfg = Config.fromfile(cfg.encoder_config_file)
    vit = build_posenet(vit_cfg.model)
    body_position_net = PositionNet('body', feat_dim=cfg.feat_dim)
    body_rotation_net = BodyRotationNet(feat_dim=cfg.feat_dim)
    box_net = BoxNet(feat_dim=cfg.feat_dim)

    # hand
    hand_roi_net = HandRoI(feat_dim=cfg.feat_dim, upscale=cfg.upscale)
    hand_position_net = PositionNet('hand', feat_dim=cfg.feat_dim//2)
    hand_rotation_net = HandRotationNet('hand', feat_dim=256)
    decoder_cfg = Config.fromfile(os.path.join(cfg.root_dir, 'main/transformer_utils/configs/osx/decoder/hand_decoder.py'))
    hand_decoder = build_posenet(decoder_cfg.model)

    # face
    face_roi_net = FaceRoI(feat_dim=cfg.feat_dim, upscale=cfg.upscale)
    face_position_net = PositionNet('face', feat_dim=cfg.feat_dim//2)
    face_regressor = FaceRegressor(feat_dim=cfg.feat_dim, joint_feat_dim=256)
    decoder_cfg = Config.fromfile(os.path.join(cfg.root_dir, 'main/transformer_utils/configs/osx/decoder/face_decoder.py'))
    face_decoder = build_posenet(decoder_cfg.model)

    if mode == 'train':
        body_position_net.apply(init_weights)
        body_rotation_net.apply(init_weights)
        box_net.apply(init_weights)
        encoder_pretrained_model_path = torch.load(cfg.encoder_pretrained_model_path)['state_dict']
        vit.load_state_dict(encoder_pretrained_model_path, strict=False)
        print(f"Initialize backbone from {cfg.encoder_pretrained_model_path}")

        # hand
        hand_position_net.apply(init_weights)
        hand_roi_net.apply(init_weights)
        hand_rotation_net.apply(init_weights)
        hand_decoder.apply(init_weights)

        # face
        face_position_net.apply(init_weights)
        face_roi_net.apply(init_weights)
        face_decoder.apply(init_weights)
        face_regressor.apply(init_weights)

    encoder = vit.backbone
    model = Model(encoder, body_position_net, body_rotation_net, box_net, hand_position_net, hand_roi_net, hand_decoder, hand_rotation_net,
                  face_position_net, face_roi_net, face_decoder, face_regressor, smpl)
    return model
