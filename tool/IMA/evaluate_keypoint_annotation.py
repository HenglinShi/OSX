import pandas as pd
import os
import shutil
import os.path as osp
import sys
import pdb
sys.path.insert(0, "/home/x_hensh/.local/lib/python3.10/site-packages")
sys.path.insert(0, "/proj/berzelius-2024-331/users/x_hensh/git/OSX/")
sys.path.insert(0, "/proj/berzelius-2024-331/users/x_hensh/git/OSX/main")
from common.utils.vis import vis_keypoints
import cv2
import numpy as np

annotations_csv = pd.read_csv('/proj/berzelius-2024-331/users/x_hensh/git/OSX/dataset/IMA/pose_estimates_youtube_dataset.csv')
videos_dir =            '/proj/berzelius-2024-331/users/x_hensh/git/OSX/dataset/IMA/images'
renders_dir =           '/proj/berzelius-2024-331/users/x_hensh/git/OSX/dataset/IMA/renders_video_separated'
renders_dir_original =  osp.join(renders_dir, 'original')
renders_dir_resize =    osp.join(renders_dir, 'resized')    #'/proj/berzelius-2024-331/users/x_hensh/git/OSX/dataset/IMA/renders/resized'

os.makedirs(renders_dir, exist_ok=True)


body_parts = [
    'LHip',     'LEye',     'LKnee',     'LElbow',     'Neck',
    'RElbow',     'RKnee',     'LEar',     'RShoulder',     'REar',
    'LWrist',     'LShoulder',     'LAnkle',     'REye',     'RWrist',
    'RHip',     'RAnkle',     'Nose'
]


os.makedirs(renders_dir_original, exist_ok=True)
os.makedirs(renders_dir_resize, exist_ok=True)

video_names = list(set(annotations_csv.video))
video_names.sort()


#pdb.set_trace()
for video_name in video_names:
    annotations_video = annotations_csv[annotations_csv.video == video_name].reset_index(drop=True)
    frame_names = set(annotations_video.frame)
    
    os.makedirs(osp.join(renders_dir_original, video_name), exist_ok=True)
    os.makedirs(osp.join(renders_dir_resize, video_name), exist_ok=True)

    for frame_name in frame_names:
        annotations_frame = annotations_video[annotations_video.frame==frame_name].reset_index(drop=True)

        imgname = f"1{video_name[-6:]}{frame_name:06d}.jpg"
        img_raw_path = osp.join(videos_dir, imgname)

        if not osp.exists(img_raw_path): continue

        pixel_x, pixel_y = int(annotations_frame.pixel_x[0]), int(annotations_frame.pixel_y[0])

        n_row, n_col = annotations_frame.shape
        body_parts = set(annotations_frame.bp) if body_parts is None else body_parts

        keypoint_buf = list()
        for body_part in body_parts:
            keypoint_buf.append([annotations_frame[annotations_frame.bp==body_part].reset_index().x[0],
                                 annotations_frame[annotations_frame.bp==body_part].reset_index().y[0]])

        keypoint_buf = np.array(keypoint_buf)
        keypoint_buf[np.isnan(keypoint_buf)] = 0

        frame_original = cv2.imread(img_raw_path)
        render_original = vis_keypoints(frame_original, keypoint_buf)

        render_orignal_path = osp.join(renders_dir_original, video_name, imgname)
        print ('Render Original ' + render_orignal_path)
        cv2.imwrite(render_orignal_path, render_original)

        frame_resized = cv2.resize(frame_original, (pixel_x, pixel_y), interpolation = cv2.INTER_CUBIC)
        render_resized = vis_keypoints(frame_resized, keypoint_buf)

        render_resized_path = osp.join(renders_dir_resize, video_name, imgname)
        print ('Render Resized ' + render_resized_path)
        cv2.imwrite(render_resized_path, render_resized)