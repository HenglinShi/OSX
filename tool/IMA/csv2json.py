import json
import pandas as pd
import os.path as osp

import numpy as np
import pdb
# Missing id and image id


annotations_csv = pd.read_csv('../../dataset/IMA/pose_estimates_youtube_dataset.csv')
video_names = set(annotations_csv.video)


video_names_train = ['video_000135', 'video_000099', 'video_000297', 'video_000088', 'video_000268', 
                     'video_000295', 'video_000296', 'video_000369', 'video_000282', 'video_000347', 
                     'video_000191', 'video_000399', 'video_000011', 'video_000398', 'video_000068', 
                     'video_000385', 'video_000021', 'video_000001', 'video_000267', 'video_000290', 
                     'video_000005', 'video_000052', 'video_000073', 'video_000389', 'video_000173', 
                     'video_000352', 'video_000283', 'video_000285', 'video_000110', 'video_000346', 
                     'video_000127', 'video_000031', 'video_000344', 'video_000169', 'video_000292', 
                     'video_000339', 'video_000349', 'video_000190', 'video_000409', 'video_000364', 
                     'video_000379', 'video_000047', 'video_000338', 'video_000112', 'video_000236', 
                     'video_000204', 'video_000269', 'video_000287', 'video_000141', 'video_000291', 
                     'video_000123', 'video_000360', 'video_000227', 'video_000244', 'video_000358', 
                     'video_000353', 'video_000288', 'video_000340', 'video_000004']



video_names_test = ['video_000106', 'video_000284', 'video_000345', 'video_000070', 'video_000079', 
                    'video_000107', 'video_000122', 'video_000121', 'video_000241', 'video_000059', 
                    'video_000000', 'video_000394', 'video_000405', 'video_000341', 'video_000111', 
                    'video_000186', 'video_000179', 'video_000412', 'video_000090', 'video_000086', 
                    'video_000286', 'video_000172', 'video_000077', 'video_000276', 'video_000396', 
                    'video_000348']


assert (set(video_names_test + video_names_train) == video_names)

del video_names

#video_names = list(video_names)
#import random
#random.shuffle(video_names)
#num_videos = len(video_names)
#cut = int(0.7*num_videos)
#video_names_train = video_names[:cut]
#video_names_test = video_names[cut:]



body_parts = [
    'LHip',     'LEye',     'LKnee',     'LElbow',     'Neck',
    'RElbow',     'RKnee',     'LEar',     'RShoulder',     'REar',
    'LWrist',     'LShoulder',     'LAnkle',     'REye',     'RWrist',
    'RHip',     'RAnkle',     'Nose'
]


annotations = dict()
annotations['annotations'] = list()
annotations['images'] = list()

global_id = 0

for video_name in video_names_train:
    annotations_video = annotations_csv[annotations_csv.video == video_name].reset_index(drop=True)
    frame_names = set(annotations_video.frame)

    for frame_name in frame_names:
        annotations_frame = annotations_video[annotations_video.frame==frame_name].reset_index(drop=True)

        imgname = f"1{video_name[-6:]}{frame_name:06d}.jpg"
        img_path = osp.join('../../dataset/IMA/images', imgname)

        if not osp.exists(img_path): continue
        #pdb.set_trace()

        annotations['annotations'].append({
            'image_id': global_id,
            'id': global_id,
            'keypoints': {}
        })

        annotations['images'].append({
            'id': global_id,
            'video': video_name,
            'frame': frame_name,
            'image_shape': [int(annotations_frame.pixel_x[0]), int(annotations_frame.pixel_y[0])]
        })

        n_row, n_col = annotations_frame.shape
        body_parts = set(annotations_frame.bp) if body_parts is None else body_parts

        keypoint_buf = list()

        for body_part in body_parts:
            keypoint_buf.append([annotations_frame[annotations_frame.bp==body_part].reset_index().x[0],
                                 annotations_frame[annotations_frame.bp==body_part].reset_index().y[0]])

        #annotations['annotations'][-1] = list()
        #for i_row in range(n_row):
        #    annotations['annotations'][-1]['keypoints'][annotations_frame.bp.iloc[i_row]] = [annotations_frame.x[i_row], annotations_frame.y[i_row]]

        annotations['annotations'][-1]['keypoints'] = keypoint_buf# np.array(keypoint_buf)
        global_id = global_id + 1
        #print (annotations_frame.shape)




with open('../../dataset/IMA/annotations/annotations_train.json', 'w') as f:
    json.dump(annotations, f)




del annotations

annotations = dict()
annotations['annotations'] = list()
annotations['images'] = list()



global_id = 0

for video_name in video_names_test:
    annotations_video = annotations_csv[annotations_csv.video == video_name].reset_index(drop=True)
    frame_names = set(annotations_video.frame)

    for frame_name in frame_names:
        annotations_frame = annotations_video[annotations_video.frame==frame_name].reset_index(drop=True)

        imgname = f"1{video_name[-6:]}{frame_name:06d}.jpg"
        img_path = osp.join('../../dataset/IMA/images', imgname)

        if not osp.exists(img_path): continue

        #pdb.set_trace()

        annotations['annotations'].append({
            'image_id': global_id,
            'id': global_id,
            'keypoints': {}
        })

        annotations['images'].append({
            'id': global_id,
            'video': video_name,
            'frame': frame_name,
            'image_shape': [int(annotations_frame.pixel_x[0]), int(annotations_frame.pixel_y[0])]
        })

        n_row, n_col = annotations_frame.shape
        body_parts = set(annotations_frame.bp) if body_parts is None else body_parts

        keypoint_buf = list()

        for body_part in body_parts:
            keypoint_buf.append([annotations_frame[annotations_frame.bp==body_part].reset_index().x[0],
                                 annotations_frame[annotations_frame.bp==body_part].reset_index().y[0]])

        #annotations['annotations'][-1] = list()
        #for i_row in range(n_row):
        #    annotations['annotations'][-1]['keypoints'][annotations_frame.bp.iloc[i_row]] = [annotations_frame.x[i_row], annotations_frame.y[i_row]]

        annotations['annotations'][-1]['keypoints'] = keypoint_buf# np.array(keypoint_buf)
        global_id = global_id + 1
        #print (annotations_frame.shape)




with open('../../dataset/IMA/annotations/annotations_valid.json', 'w') as f:
    json.dump(annotations, f)