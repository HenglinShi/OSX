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
videos_dir = '/proj/berzelius-2024-331/users/x_hensh/git/OSX/dataset/IMA/images'
#renders_dir =  '/proj/berzelius-2024-331/users/x_hensh/git/OSX/dataset/IMA/renders'
#renders_dir_original = '/proj/berzelius-2024-331/users/x_hensh/git/OSX/dataset/IMA/renders/original'
#renders_dir_resize = '/proj/berzelius-2024-331/users/x_hensh/git/OSX/dataset/IMA/renders/resized'

video_names = list(set(annotations_csv.video))
video_names.sort()
widths_annotated = []
heights_annotated = []

widths_real = []
heights_real = []

i = 0
video_names_valid = []
sizes_equal = []
for video_name in video_names:
    annotations_video = annotations_csv[annotations_csv.video == video_name].reset_index(drop=True)
    frame_names = set(annotations_video.frame)

    width_annotated, height_annotated = -1, -1
    width_real, height_real = -1, -1
    size_equal = False

    for frame_name in frame_names:
        annotations_frame = annotations_video[annotations_video.frame==frame_name].reset_index(drop=True)
        width_annotated, height_annotated = int(annotations_frame.pixel_x[0]), int(annotations_frame.pixel_y[0])
        
        imgname = f"1{video_name[-6:]}{frame_name:06d}.jpg"
        img_raw_path = osp.join(videos_dir, imgname)

        if not osp.exists(img_raw_path): continue
        i = i + 1
        print (i)

        
        frame_original = cv2.imread(img_raw_path)
        height_real, width_real, _ = frame_original.shape

        
        if ((width_real == width_annotated) and (height_real == height_annotated)):
            size_equal = True
        else:
            size_equal = False
            #print (video_name + '\t' + 
            #    'width_annotated: ' + str(width_annotated) + ',' + '\t' + 'height_annotated: ' + str(height_annotated) + ';' + '\t' + 
            #    'width_real: ' + str(width_real) + ',' + '\t' + 'height_real: ' + str(height_real)) 
        break
        
    video_names_valid.append(video_name)
    widths_annotated.append(width_annotated)
    heights_annotated.append(height_annotated)
    widths_real.append(width_real)
    heights_real.append(height_real)
    sizes_equal.append(size_equal)


annotations_size = {
    'video_name': video_names_valid,
    'width_annotated': widths_annotated,
    'height_annotated': heights_annotated,
    'width_real': widths_real,
    'height_real': heights_real,
    'size_equal': sizes_equal
}

annotations_size_df = pd.DataFrame(annotations_size)

annotations_size_df.to_excel('/proj/berzelius-2024-331/users/x_hensh/git/OSX/dataset/IMA/annotations_size.xlsx')