import numpy as np
import torch
import os.path as osp
from config import cfg
from common.utils.smplx import smplx
import pickle
import pdb
from smplx.joint_names import SMPLH_JOINT_NAMES, JOINT_NAMES
class SMPLX(object):
    def __init__(self):
        self.layer_arg = {'create_global_orient': False, 'create_body_pose': False, 'create_left_hand_pose': False, 'create_right_hand_pose': False, 'create_jaw_pose': False, 'create_leye_pose': False, 'create_reye_pose': False, 'create_betas': False, 'create_expression': False, 'create_transl': False}
        self.layer = {'neutral': smplx.create(cfg.human_model_path, 'smplx', gender='NEUTRAL', use_pca=False, use_face_contour=True, **self.layer_arg),
                        'male': smplx.create(cfg.human_model_path, 'smplx', gender='MALE', use_pca=False, use_face_contour=True, **self.layer_arg),
                        'female': smplx.create(cfg.human_model_path, 'smplx', gender='FEMALE', use_pca=False, use_face_contour=True, **self.layer_arg)
                        }
        self.vertex_num = 10475
        self.face = self.layer['neutral'].faces
        self.shape_param_dim = 10
        self.expr_code_dim = 10
        with open(osp.join(cfg.human_model_path, 'smplx', 'SMPLX_to_J14.pkl'), 'rb') as f:
            self.j14_regressor = pickle.load(f, encoding='latin1')
        with open(osp.join(cfg.human_model_path, 'smplx', 'MANO_SMPLX_vertex_ids.pkl'), 'rb') as f:
            self.hand_vertex_idx = pickle.load(f, encoding='latin1')
        self.face_vertex_idx = np.load(osp.join(cfg.human_model_path, 'smplx', 'SMPL-X__FLAME_vertex_ids.npy'))
        self.J_regressor = self.layer['neutral'].J_regressor.numpy()
        self.J_regressor_idx = {'pelvis': 0, 'lwrist': 20, 'rwrist': 21, 'neck': 12}
        self.orig_hand_regressor = self.make_hand_regressor()
        #self.orig_hand_regressor = {'left': self.layer.J_regressor.numpy()[[20,37,38,39,25,26,27,28,29,30,34,35,36,31,32,33],:], 'right': self.layer.J_regressor.numpy()[[21,52,53,54,40,41,42,43,44,45,49,50,51,46,47,48],:]}

        # original SMPLX joint set
        self.orig_joint_num = 53 # 22 (body joints) + 30 (hand joints) + 1 (face jaw joint)
        self.orig_joints_name = \
        ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', # body joints
        'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', # left hand joints
        'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', # right hand joints
        'Jaw' # face jaw joint
        )
        self.orig_flip_pairs = \
        ( (1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), # body joints
        (22,37), (23,38), (24,39), (25,40), (26,41), (27,42), (28,43), (29,44), (30,45), (31,46), (32,47), (33,48), (34,49), (35,50), (36,51) # hand joints
        )
        self.orig_root_joint_idx = self.orig_joints_name.index('Pelvis')
        self.orig_joint_part = \
        {'body': range(self.orig_joints_name.index('Pelvis'), self.orig_joints_name.index('R_Wrist')+1),
        'lhand': range(self.orig_joints_name.index('L_Index_1'), self.orig_joints_name.index('L_Thumb_3')+1),
        'rhand': range(self.orig_joints_name.index('R_Index_1'), self.orig_joints_name.index('R_Thumb_3')+1),
        'face': range(self.orig_joints_name.index('Jaw'), self.orig_joints_name.index('Jaw')+1)}

        # changed SMPLX joint set for the supervision
        self.joint_num = 137 # 25 (body joints) + 40 (hand joints) + 72 (face keypoints)
        self.joints_name = \
        ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Neck', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Ear', 'R_Ear', 'L_Eye', 'R_Eye', 'Nose',# body joints
         'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand joints
         'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4', # right hand joints
         *['Face_' + str(i) for i in range(1,73)] # face keypoints (too many keypoints... omit real names. have same name of keypoints defined in FLAME class)
         )
        self.root_joint_idx = self.joints_name.index('Pelvis')
        self.lwrist_idx = self.joints_name.index('L_Wrist')
        self.rwrist_idx = self.joints_name.index('R_Wrist')
        self.neck_idx = self.joints_name.index('Neck')
        self.flip_pairs = \
        ( (1,2), (3,4), (5,6), (8,9), (10,11), (12,13), (14,17), (15,18), (16,19), (20,21), (22,23), # body joints
        (25,45), (26,46), (27,47), (28,48), (29,49), (30,50), (31,51), (32,52), (33,53), (34,54), (35,55), (36,56), (37,57), (38,58), (39,59), (40,60), (41,61), (42,62), (43,63), (44,64), # hand joints
        (67,68), # face eyeballs
        (69,78), (70,77), (71,76), (72,75), (73,74), # face eyebrow
        (83,87), (84,86), # face below nose
        (88,97), (89,96), (90,95), (91,94), (92,99), (93,98), # face eyes
        (100,106), (101,105), (102,104), (107,111), (108,110), # face mouth
        (112,116), (113,115), (117,119), # face lip
        (120,136), (121,135), (122,134), (123,133), (124,132), (125,131), (126,130), (127,129) # face contours
        )
        self.joint_idx = \
        (0,1,2,4,5,7,8,12,16,17,18,19,20,21,60,61,62,63,64,65,59,58,57,56,55, # body joints
        37,38,39,66,25,
        26,27,67,28,29,
        30,68,34,35,36,
        69,31,32,33,70, # left hand joints
        52,53,54,71,40,41,42,72,43,44,45,73,49,50,51,74,46,47,48,75, # right hand joints
        22,15, # jaw, head
        57,56, # eyeballs
        76,77,78,79,80,81,82,83,84,85, # eyebrow
        86,87,88,89, # nose
        90,91,92,93,94, # below nose
        95,96,97,98,99,100,101,102,103,104,105,106, # eyes
        107, # right mouth
        108,109,110,111,112, # upper mouth
        113, # left mouth
        114,115,116,117,118, # lower mouth
        119, # right lip
        120,121,122, # upper lip
        123, # left lip
        124,125,126, # lower lip
        127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143 # face contour
        )

        



        self.joint_part = \
        {'body': range(self.joints_name.index('Pelvis'), self.joints_name.index('Nose')+1),
        'lhand': range(self.joints_name.index('L_Thumb_1'), self.joints_name.index('L_Pinky_4')+1),
        'rhand': range(self.joints_name.index('R_Thumb_1'), self.joints_name.index('R_Pinky_4')+1),
        'hand': range(self.joints_name.index('L_Thumb_1'), self.joints_name.index('R_Pinky_4')+1),
        'face': range(self.joints_name.index('Face_1'), self.joints_name.index('Face_72')+1)}
        
        # changed SMPLX joint set for PositionNet prediction
        self.pos_joint_num = 65 # 25 (body joints) + 40 (hand joints)
        self.pos_joints_name = \
        ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Neck', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Ear', 'R_Ear', 'L_Eye', 'R_Eye', 'Nose', # body joints
         'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand joints
         'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4', # right hand joints
         )
        self.pos_joint_part = \
        {'body': range(self.pos_joints_name.index('Pelvis'), self.pos_joints_name.index('Nose')+1),
        'lhand': range(self.pos_joints_name.index('L_Thumb_1'), self.pos_joints_name.index('L_Pinky_4')+1),
        'rhand': range(self.pos_joints_name.index('R_Thumb_1'), self.pos_joints_name.index('R_Pinky_4')+1),
        'hand': range(self.pos_joints_name.index('L_Thumb_1'), self.pos_joints_name.index('R_Pinky_4')+1)}
        self.pos_joint_part['L_MCP'] = [self.pos_joints_name.index('L_Index_1') - len(self.pos_joint_part['body']),
                                        self.pos_joints_name.index('L_Middle_1') - len(self.pos_joint_part['body']),
                                        self.pos_joints_name.index('L_Ring_1') - len(self.pos_joint_part['body']),
                                        self.pos_joints_name.index('L_Pinky_1') - len(self.pos_joint_part['body'])]
        self.pos_joint_part['R_MCP'] = [self.pos_joints_name.index('R_Index_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand']),
                                        self.pos_joints_name.index('R_Middle_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand']),
                                        self.pos_joints_name.index('R_Ring_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand']),
                                        self.pos_joints_name.index('R_Pinky_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand'])]
    
    def make_hand_regressor(self):
        regressor = self.layer['neutral'].J_regressor.numpy()
        lhand_regressor = np.concatenate((regressor[[20,37,38,39],:],                   #left_wrist,left_thumb1, left_thumb2, left_thumb3
                                            np.eye(self.vertex_num)[5361,None],         #lthumb
                                                regressor[[25,26,27],:],                #left_index1, left_index2, left_index3, 
                                                np.eye(self.vertex_num)[4933,None],     #lindex
                                                regressor[[28,29,30],:],                #left_middle1, left_middle2, left_middle3
                                                np.eye(self.vertex_num)[5058,None],     #lmiddle
                                                regressor[[34,35,36],:],                #left_ring1, left_ring2, left_ring3
                                                np.eye(self.vertex_num)[5169,None],     #lring
                                                regressor[[31,32,33],:],                #left_pinky1, left_pinky2, left_pinky3
                                                np.eye(self.vertex_num)[5286,None]))    #lpinky
        
        rhand_regressor = np.concatenate((regressor[[21,52,53,54],:],                   #right_ wrist, right_thumb1, right_thumb2, right_thumb3
                                            np.eye(self.vertex_num)[8079,None],         #rthumb
                                                regressor[[40,41,42],:],                #right_index1, right_index2, right_index3
                                                np.eye(self.vertex_num)[7669,None],     #rindex
                                                regressor[[43,44,45],:],                #right_middle1, right_middle2, right_middle3
                                                np.eye(self.vertex_num)[7794,None],     #rmiddle
                                                regressor[[49,50,51],:],                #right_ring1, right_ring2, righint_ring3
                                                np.eye(self.vertex_num)[7905,None],     #rring
                                                regressor[[46,47,48],:],                #right_pinky1, right_pinky2, right_pinky3
                                                np.eye(self.vertex_num)[8022,None]))    #rpinky
        hand_regressor = {'left': lhand_regressor, 'right': rhand_regressor}
        return hand_regressor

        
    def reduce_joint_set(self, joint):
        new_joint = []
        for name in self.pos_joints_name:
            idx = self.joints_name.index(name)
            new_joint.append(joint[:,idx,:])
        new_joint = torch.stack(new_joint,1)
        return new_joint

class SMPL(object):
    def __init__(self):
        self.layer_arg = {'create_body_pose': False, 'create_betas': False, 'create_global_orient': False, 'create_transl': False}
        self.layer = {'neutral': smplx.create(cfg.human_model_path, 'smpl', gender='NEUTRAL', **self.layer_arg), 'male': smplx.create(cfg.human_model_path, 'smpl', gender='MALE', **self.layer_arg), 'female': smplx.create(cfg.human_model_path, 'smpl', gender='FEMALE', **self.layer_arg)}
        self.vertex_num = 6890
        self.face = self.layer['neutral'].faces
        self.shape_param_dim = 10
        self.vposer_code_dim = 32

        # original SMPL joint set
        self.orig_joint_num = 24
        self.orig_joints_name = ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 
                                 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 
                                 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 
                                 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 
                                 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand')
        self.orig_flip_pairs = ( (1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), (22,23) )
        self.orig_root_joint_idx = self.orig_joints_name.index('Pelvis')
        self.orig_joint_regressor = self.layer['neutral'].J_regressor.numpy().astype(np.float32)
        
        self.joint_num = self.orig_joint_num
        self.joints_name = self.orig_joints_name
        self.flip_pairs = self.orig_flip_pairs
        self.root_joint_idx = self.orig_root_joint_idx
        self.joint_regressor = self.orig_joint_regressor




class SMPLH(object):
    def __init__(self):
        self.layer_arg = {
            'create_global_orient': False, 
            'create_body_pose': False, 
            'create_left_hand_pose': False, 
            'create_right_hand_pose': False, 
            'create_jaw_pose': False, 
            'create_leye_pose': False, 
            'create_reye_pose': False, 
            'create_betas': False, 
            'create_expression': False, 
            'create_transl': False
            }
        
        self.layer = {'neutral': smplx.create(cfg.human_model_path, 'smplh', gender='NEUTRAL', use_pca=False, use_face_contour=True, **self.layer_arg),
                        'male': smplx.create(cfg.human_model_path, 'smplh', gender='MALE', use_pca=False, use_face_contour=True, **self.layer_arg),
                        'female': smplx.create(cfg.human_model_path, 'smplh', gender='FEMALE', use_pca=False, use_face_contour=True, **self.layer_arg)
                        }
        #pdb.set_trace()
        self.vertex_num = 6890
        self.face = self.layer['neutral'].faces
        self.shape_param_dim = 10
        #self.expr_code_dim = 10
        

        self.J_regressor = self.layer['neutral'].J_regressor.numpy()
        
        self.J_regressor_idx = {
            'pelvis': 0, 
            'lwrist': 20, 
            'rwrist': 21, 
            'neck': 12
            }
        
        self.orig_hand_regressor = self.make_hand_regressor()

        with open(osp.join(cfg.human_model_path, 'smplx', 'SMPLX_to_J14.pkl'), 'rb') as f:
            self.j14_regressor = pickle.load(f, encoding='latin1')
        with open(osp.join(cfg.human_model_path, 'smplx', 'MANO_SMPLX_vertex_ids.pkl'), 'rb') as f:
            self.hand_vertex_idx = pickle.load(f, encoding='latin1')
        self.face_vertex_idx = np.load(osp.join(cfg.human_model_path, 'smplx', 'SMPL-X__FLAME_vertex_ids.npy'))

        #self.orig_hand_regressor = {'left': self.layer.J_regressor.numpy()[[20,37,38,39,25,26,27,28,29,30,34,35,36,31,32,33],:], 'right': self.layer.J_regressor.numpy()[[21,52,53,54,40,41,42,43,44,45,49,50,51,46,47,48],:]}

        # original SMPLX joint set
        self.orig_joint_num = 52 # 22 (body joints) + 30 (hand joints)
        self.orig_joints_name = \
        ('Pelvis', 
         'L_Hip', 
         'R_Hip', 
         'Spine_1',
         'L_Knee', 
         'R_Knee', 
         'Spine_2',
         'L_Ankle',
         'R_Ankle', 
         'Spine_3', 
         'L_Foot', 
         'R_Foot', 
         'Neck', 
         'L_Collar', 
         'R_Collar', 
         'Head', 
         'L_Shoulder', 
         'R_Shoulder', 
         'L_Elbow', 'R_Elbow', 
         'L_Wrist', 'R_Wrist', # body joints
        'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 
        'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', # left hand joints
        'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', # right hand joints
        )


        self.orig_flip_pairs = \
        ( (1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), # body joints
        (22,37), (23,38), (24,39), (25,40), (26,41), (27,42), (28,43), (29,44), (30,45), (31,46), (32,47), (33,48), (34,49), (35,50), (36,51) # hand joints
        )
        self.orig_root_joint_idx = self.orig_joints_name.index('Pelvis')
        self.orig_joint_part = \
        {'body': range(self.orig_joints_name.index('Pelvis'), self.orig_joints_name.index('R_Wrist')+1),
        'lhand': range(self.orig_joints_name.index('L_Index_1'), self.orig_joints_name.index('L_Thumb_3')+1),
        'rhand': range(self.orig_joints_name.index('R_Index_1'), self.orig_joints_name.index('R_Thumb_3')+1),
        }

        # changed SMPLX joint set for the supervision
        self.joint_num = 137 # 25 (body joints) + 40 (hand joints) + 72 (face keypoints)
        self.joints_name = \
        ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 
         'L_Ankle', 'R_Ankle', 'Neck', 'L_Shoulder', 'R_Shoulder', 
         'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Big_toe', 
         'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 
         'L_Ear', 'R_Ear', 'L_Eye', 'R_Eye', 'Nose',# body joints
         'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand joints
         'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4', # right hand joints
         *['Face_' + str(i) for i in range(1,73)] # face keypoints (too many keypoints... omit real names. have same name of keypoints defined in FLAME class)
         )
        self.root_joint_idx = self.joints_name.index('Pelvis')
        self.lwrist_idx = self.joints_name.index('L_Wrist')
        self.rwrist_idx = self.joints_name.index('R_Wrist')
        self.neck_idx = self.joints_name.index('Neck')
        self.flip_pairs = \
        ( (1,2), (3,4), (5,6), (8,9), (10,11), (12,13), (14,17), (15,18), (16,19), (20,21), (22,23), # body joints
        (25,45), (26,46), (27,47), (28,48), (29,49), (30,50), (31,51), (32,52), (33,53), (34,54), (35,55), (36,56), (37,57), (38,58), (39,59), (40,60), (41,61), (42,62), (43,63), (44,64), # hand joints
        (67,68), # face eyeballs
        (69,78), (70,77), (71,76), (72,75), (73,74), # face eyebrow
        (83,87), (84,86), # face below nose
        (88,97), (89,96), (90,95), (91,94), (92,99), (93,98), # face eyes
        (100,106), (101,105), (102,104), (107,111), (108,110), # face mouth
        (112,116), (113,115), (117,119), # face lip
        (120,136), (121,135), (122,134), (123,133), (124,132), (125,131), (126,130), (127,129) # face contours
        )

        joint_idx_tmp = \
        (0,1,2,4,5,7,8,12,16,17,18,19,20,21,60,61,62,63,64,65,59,58,57,56,55, # body joints
        37,38,39,66,25,26,27,67,28,29,30,68,34,35,36,69,31,32,33,70, # left hand joints
        52,53,54,71,40,41,42,72,43,44,45,73,49,50,51,74,46,47,48,75, # right hand joints
        22,15, # jaw, head
        57,56, # eyeballs
        76,77,78,79,80,81,82,83,84,85, # eyebrow
        86,87,88,89, # nose
        90,91,92,93,94, # below nose
        95,96,97,98,99,100,101,102,103,104,105,106, # eyes
        107, # right mouth
        108,109,110,111,112, # upper mouth
        113, # left mouth
        114,115,116,117,118, # lower mouth
        119, # right lip
        120,121,122, # upper lip
        123, # left lip
        124,125,126, # lower lip
        127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143 # face contour
        )

        self.joint_idx = [SMPLH_JOINT_NAMES.index(JOINT_NAMES[i]) for i in joint_idx_tmp  if JOINT_NAMES[i] in SMPLH_JOINT_NAMES]
        #                        if JOINT_NAMES[i] in SMPLH_JOINT_NAMES else None for i in joint_idx_tmp]
        #self.joint_idx = [i ]


        self.joint_part = \
        {'body': range(self.joints_name.index('Pelvis'), self.joints_name.index('Nose')+1),
        'lhand': range(self.joints_name.index('L_Thumb_1'), self.joints_name.index('L_Pinky_4')+1),
        'rhand': range(self.joints_name.index('R_Thumb_1'), self.joints_name.index('R_Pinky_4')+1),
        'hand': range(self.joints_name.index('L_Thumb_1'), self.joints_name.index('R_Pinky_4')+1),
        'face': range(self.joints_name.index('Face_1'), self.joints_name.index('Face_72')+1)}
        
        # changed SMPLX joint set for PositionNet prediction
        self.pos_joint_num = 65 # 25 (body joints) + 40 (hand joints)
        self.pos_joints_name = \
        ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Neck', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Ear', 'R_Ear', 'L_Eye', 'R_Eye', 'Nose', # body joints
         'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand joints
         'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4', # right hand joints
         )
        self.pos_joint_part = \
        {'body': range(self.pos_joints_name.index('Pelvis'), self.pos_joints_name.index('Nose')+1),
        'lhand': range(self.pos_joints_name.index('L_Thumb_1'), self.pos_joints_name.index('L_Pinky_4')+1),
        'rhand': range(self.pos_joints_name.index('R_Thumb_1'), self.pos_joints_name.index('R_Pinky_4')+1),
        'hand': range(self.pos_joints_name.index('L_Thumb_1'), self.pos_joints_name.index('R_Pinky_4')+1)}
        self.pos_joint_part['L_MCP'] = [self.pos_joints_name.index('L_Index_1') - len(self.pos_joint_part['body']),
                                        self.pos_joints_name.index('L_Middle_1') - len(self.pos_joint_part['body']),
                                        self.pos_joints_name.index('L_Ring_1') - len(self.pos_joint_part['body']),
                                        self.pos_joints_name.index('L_Pinky_1') - len(self.pos_joint_part['body'])]
        self.pos_joint_part['R_MCP'] = [self.pos_joints_name.index('R_Index_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand']),
                                        self.pos_joints_name.index('R_Middle_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand']),
                                        self.pos_joints_name.index('R_Ring_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand']),
                                        self.pos_joints_name.index('R_Pinky_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand'])]
    
    def make_hand_regressor(self):
        regressor = self.layer['neutral'].J_regressor.numpy()
        lhand_regressor = np.concatenate((regressor[[20,34,35,36],:],                   #left_wrist,left_thumb1, left_thumb2, left_thumb3
                                            np.eye(self.vertex_num)[2746,None],         #lthumb
                                                regressor[[22,23,24],:],                #left_index1, left_index2, left_index3, 
                                                np.eye(self.vertex_num)[2319,None],     #lindex
                                                regressor[[25,26,37],:],                #left_middle1, left_middle2, left_middle3
                                                np.eye(self.vertex_num)[2445,None],     #lmiddle
                                                regressor[[31,32,33],:],                #left_ring1, left_ring2, left_ring3
                                                np.eye(self.vertex_num)[2556,None],     #lring
                                                regressor[[28,29,30],:],                #left_pinky1, left_pinky2, left_pinky3
                                                np.eye(self.vertex_num)[2673,None]))    #lpinky
        
        rhand_regressor = np.concatenate((regressor[[21,49,50,51],:],                   #right_ wrist, right_thumb1, right_thumb2, right_thumb3
                                            np.eye(self.vertex_num)[6191,None],         #rthumb
                                                regressor[[37,38,39],:],                #right_index1, right_index2, right_index3
                                                np.eye(self.vertex_num)[5782,None],     #rindex
                                                regressor[[40,41,42],:],                #right_middle1, right_middle2, right_middle3
                                                np.eye(self.vertex_num)[5905,None],     #rmiddle
                                                regressor[[46,47,48],:],                #right_ring1, right_ring2, righint_ring3
                                                np.eye(self.vertex_num)[6016,None],     #rring
                                                regressor[[43,44,45],:],                #right_pinky1, right_pinky2, right_pinky3
                                                np.eye(self.vertex_num)[6133,None]))    #rpinky
        hand_regressor = {'left': lhand_regressor, 'right': rhand_regressor}
        return hand_regressor

        
    def reduce_joint_set(self, joint):
        new_joint = []
        for name in self.pos_joints_name:
            idx = self.joints_name.index(name)
            new_joint.append(joint[:,idx,:])
        new_joint = torch.stack(new_joint,1)
        return new_joint




class SMILH(object):
    def __init__(self):
        self.layer_arg = {
            'create_global_orient': False, 
            'create_body_pose': False, 
            'create_left_hand_pose': False, 
            'create_right_hand_pose': False, 
            'create_jaw_pose': False, 
            'create_leye_pose': False, 
            'create_reye_pose': False, 
            'create_betas': False, 
            'create_expression': False, 
            'create_transl': False
            }
        
        self.layer = {'neutral': smplx.create(cfg.human_model_path, 'smilh', gender='NEUTRAL', use_pca=False, use_face_contour=True, **self.layer_arg),
                        'male': smplx.create(cfg.human_model_path, 'smilh', gender='MALE', use_pca=False, use_face_contour=True, **self.layer_arg),
                        'female': smplx.create(cfg.human_model_path, 'smilh', gender='FEMALE', use_pca=False, use_face_contour=True, **self.layer_arg)
                        }
        #pdb.set_trace()
        self.vertex_num = 6890
        self.face = self.layer['neutral'].faces
        self.shape_param_dim = 10
        #self.expr_code_dim = 10
        

        self.J_regressor = self.layer['neutral'].J_regressor.numpy()
        
        self.J_regressor_idx = {
            'pelvis': 0, 
            'lwrist': 20, 
            'rwrist': 21, 
            'neck': 12
            }
        
        self.orig_hand_regressor = self.make_hand_regressor()

        with open(osp.join(cfg.human_model_path, 'smplx', 'SMPLX_to_J14.pkl'), 'rb') as f:
            self.j14_regressor = pickle.load(f, encoding='latin1')
        with open(osp.join(cfg.human_model_path, 'smplx', 'MANO_SMPLX_vertex_ids.pkl'), 'rb') as f:
            self.hand_vertex_idx = pickle.load(f, encoding='latin1')
        self.face_vertex_idx = np.load(osp.join(cfg.human_model_path, 'smplx', 'SMPL-X__FLAME_vertex_ids.npy'))

        #self.orig_hand_regressor = {'left': self.layer.J_regressor.numpy()[[20,37,38,39,25,26,27,28,29,30,34,35,36,31,32,33],:], 'right': self.layer.J_regressor.numpy()[[21,52,53,54,40,41,42,43,44,45,49,50,51,46,47,48],:]}

        # original SMPLX joint set
        self.orig_joint_num = 52 # 22 (body joints) + 30 (hand joints)
        self.orig_joints_name = \
        ('Pelvis', 
         'L_Hip', 
         'R_Hip', 
         'Spine_1',
         'L_Knee', 
         'R_Knee', 
         'Spine_2',
         'L_Ankle',
         'R_Ankle', 
         'Spine_3', 
         'L_Foot', 
         'R_Foot', 
         'Neck', 
         'L_Collar', 
         'R_Collar', 
         'Head', 
         'L_Shoulder', 
         'R_Shoulder', 
         'L_Elbow', 'R_Elbow', 
         'L_Wrist', 'R_Wrist', # body joints
        'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 
        'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', # left hand joints
        'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', # right hand joints
        )


        self.orig_flip_pairs = \
        ( (1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), # body joints
        (22,37), (23,38), (24,39), (25,40), (26,41), (27,42), (28,43), (29,44), (30,45), (31,46), (32,47), (33,48), (34,49), (35,50), (36,51) # hand joints
        )
        self.orig_root_joint_idx = self.orig_joints_name.index('Pelvis')
        self.orig_joint_part = \
        {'body': range(self.orig_joints_name.index('Pelvis'), self.orig_joints_name.index('R_Wrist')+1),
        'lhand': range(self.orig_joints_name.index('L_Index_1'), self.orig_joints_name.index('L_Thumb_3')+1),
        'rhand': range(self.orig_joints_name.index('R_Index_1'), self.orig_joints_name.index('R_Thumb_3')+1),
        }

        # changed SMPLX joint set for the supervision
        self.joint_num = 137 # 25 (body joints) + 40 (hand joints) + 72 (face keypoints)
        self.joints_name = \
        ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 
         'L_Ankle', 'R_Ankle', 'Neck', 'L_Shoulder', 'R_Shoulder', 
         'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Big_toe', 
         'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 
         'L_Ear', 'R_Ear', 'L_Eye', 'R_Eye', 'Nose',# body joints
         'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand joints
         'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4', # right hand joints
         *['Face_' + str(i) for i in range(1,73)] # face keypoints (too many keypoints... omit real names. have same name of keypoints defined in FLAME class)
         )
        self.root_joint_idx = self.joints_name.index('Pelvis')
        self.lwrist_idx = self.joints_name.index('L_Wrist')
        self.rwrist_idx = self.joints_name.index('R_Wrist')
        self.neck_idx = self.joints_name.index('Neck')
        self.flip_pairs = \
        ( (1,2), (3,4), (5,6), (8,9), (10,11), (12,13), (14,17), (15,18), (16,19), (20,21), (22,23), # body joints
        (25,45), (26,46), (27,47), (28,48), (29,49), (30,50), (31,51), (32,52), (33,53), (34,54), (35,55), (36,56), (37,57), (38,58), (39,59), (40,60), (41,61), (42,62), (43,63), (44,64), # hand joints
        (67,68), # face eyeballs
        (69,78), (70,77), (71,76), (72,75), (73,74), # face eyebrow
        (83,87), (84,86), # face below nose
        (88,97), (89,96), (90,95), (91,94), (92,99), (93,98), # face eyes
        (100,106), (101,105), (102,104), (107,111), (108,110), # face mouth
        (112,116), (113,115), (117,119), # face lip
        (120,136), (121,135), (122,134), (123,133), (124,132), (125,131), (126,130), (127,129) # face contours
        )

        joint_idx_tmp = \
        (0,1,2,4,5,7,8,12,16,17,18,19,20,21,60,61,62,63,64,65,59,58,57,56,55, # body joints
        37,38,39,66,25,26,27,67,28,29,30,68,34,35,36,69,31,32,33,70, # left hand joints
        52,53,54,71,40,41,42,72,43,44,45,73,49,50,51,74,46,47,48,75, # right hand joints
        22,15, # jaw, head
        57,56, # eyeballs
        76,77,78,79,80,81,82,83,84,85, # eyebrow
        86,87,88,89, # nose
        90,91,92,93,94, # below nose
        95,96,97,98,99,100,101,102,103,104,105,106, # eyes
        107, # right mouth
        108,109,110,111,112, # upper mouth
        113, # left mouth
        114,115,116,117,118, # lower mouth
        119, # right lip
        120,121,122, # upper lip
        123, # left lip
        124,125,126, # lower lip
        127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143 # face contour
        )

        self.joint_idx = [SMPLH_JOINT_NAMES.index(JOINT_NAMES[i]) for i in joint_idx_tmp  if JOINT_NAMES[i] in SMPLH_JOINT_NAMES]
        #                        if JOINT_NAMES[i] in SMPLH_JOINT_NAMES else None for i in joint_idx_tmp]
        #self.joint_idx = [i ]


        self.joint_part = \
        {'body': range(self.joints_name.index('Pelvis'), self.joints_name.index('Nose')+1),
        'lhand': range(self.joints_name.index('L_Thumb_1'), self.joints_name.index('L_Pinky_4')+1),
        'rhand': range(self.joints_name.index('R_Thumb_1'), self.joints_name.index('R_Pinky_4')+1),
        'hand': range(self.joints_name.index('L_Thumb_1'), self.joints_name.index('R_Pinky_4')+1),
        'face': range(self.joints_name.index('Face_1'), self.joints_name.index('Face_72')+1)}
        
        # changed SMPLX joint set for PositionNet prediction
        self.pos_joint_num = 65 # 25 (body joints) + 40 (hand joints)
        self.pos_joints_name = \
        ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Neck', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Ear', 'R_Ear', 'L_Eye', 'R_Eye', 'Nose', # body joints
         'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand joints
         'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4', # right hand joints
         )
        self.pos_joint_part = \
        {'body': range(self.pos_joints_name.index('Pelvis'), self.pos_joints_name.index('Nose')+1),
        'lhand': range(self.pos_joints_name.index('L_Thumb_1'), self.pos_joints_name.index('L_Pinky_4')+1),
        'rhand': range(self.pos_joints_name.index('R_Thumb_1'), self.pos_joints_name.index('R_Pinky_4')+1),
        'hand': range(self.pos_joints_name.index('L_Thumb_1'), self.pos_joints_name.index('R_Pinky_4')+1)}
        self.pos_joint_part['L_MCP'] = [self.pos_joints_name.index('L_Index_1') - len(self.pos_joint_part['body']),
                                        self.pos_joints_name.index('L_Middle_1') - len(self.pos_joint_part['body']),
                                        self.pos_joints_name.index('L_Ring_1') - len(self.pos_joint_part['body']),
                                        self.pos_joints_name.index('L_Pinky_1') - len(self.pos_joint_part['body'])]
        self.pos_joint_part['R_MCP'] = [self.pos_joints_name.index('R_Index_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand']),
                                        self.pos_joints_name.index('R_Middle_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand']),
                                        self.pos_joints_name.index('R_Ring_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand']),
                                        self.pos_joints_name.index('R_Pinky_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand'])]
    
    def make_hand_regressor(self):
        regressor = self.layer['neutral'].J_regressor.numpy()
        lhand_regressor = np.concatenate((regressor[[20,34,35,36],:],                   #left_wrist,left_thumb1, left_thumb2, left_thumb3
                                            np.eye(self.vertex_num)[2746,None],         #lthumb
                                                regressor[[22,23,24],:],                #left_index1, left_index2, left_index3, 
                                                np.eye(self.vertex_num)[2319,None],     #lindex
                                                regressor[[25,26,37],:],                #left_middle1, left_middle2, left_middle3
                                                np.eye(self.vertex_num)[2445,None],     #lmiddle
                                                regressor[[31,32,33],:],                #left_ring1, left_ring2, left_ring3
                                                np.eye(self.vertex_num)[2556,None],     #lring
                                                regressor[[28,29,30],:],                #left_pinky1, left_pinky2, left_pinky3
                                                np.eye(self.vertex_num)[2673,None]))    #lpinky
        
        rhand_regressor = np.concatenate((regressor[[21,49,50,51],:],                   #right_ wrist, right_thumb1, right_thumb2, right_thumb3
                                            np.eye(self.vertex_num)[6191,None],         #rthumb
                                                regressor[[37,38,39],:],                #right_index1, right_index2, right_index3
                                                np.eye(self.vertex_num)[5782,None],     #rindex
                                                regressor[[40,41,42],:],                #right_middle1, right_middle2, right_middle3
                                                np.eye(self.vertex_num)[5905,None],     #rmiddle
                                                regressor[[46,47,48],:],                #right_ring1, right_ring2, righint_ring3
                                                np.eye(self.vertex_num)[6016,None],     #rring
                                                regressor[[43,44,45],:],                #right_pinky1, right_pinky2, right_pinky3
                                                np.eye(self.vertex_num)[6133,None]))    #rpinky
        hand_regressor = {'left': lhand_regressor, 'right': rhand_regressor}
        return hand_regressor

        
    def reduce_joint_set(self, joint):
        new_joint = []
        for name in self.pos_joints_name:
            idx = self.joints_name.index(name)
            new_joint.append(joint[:,idx,:])
        new_joint = torch.stack(new_joint,1)
        return new_joint

smpl_x = SMPLX()
smpl_h = SMPLH()
smil_h = SMILH()
smpl = SMPL()
