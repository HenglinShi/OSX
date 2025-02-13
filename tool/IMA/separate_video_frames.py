import pandas as pd
import os
import cv2
import shutil

annotations_csv = pd.read_csv('/proj/berzelius-2024-331/users/x_hensh/git/OSX/dataset/IMA/pose_estimates_youtube_dataset.csv')
destination_dir = '/proj/berzelius-2024-331/users/x_hensh/data/Youtube-Infant-Body-Parsing/frames_video_separated'
source_dir = '/proj/berzelius-2024-331/users/x_hensh/git/OSX/dataset/IMA/images'

os.mkdir(destination_dir)

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


video_names = video_names_train + video_names_test

frame_names_all = os.listdir(source_dir)

for video_name in video_names:
    os.mkdir(os.path.join(destination_dir, video_name))
    #print ('processing ' + video_name)
    frame_names_video = [frame_name for frame_name in frame_names_all if frame_name.startswith('1'+video_name[6:])]
    frame_names_video.sort()
    for frame_name in frame_names_video:
        print ('processing ' + video_name + " and frame " + frame_name)
        #annotation = annotations_csv[
        #    (annotations_csv.video==video_name) &
        #    (annotations_csv.frame==int(frame_name[7:].split('.')[0]))
        #].reset_index()
        #width_target, height_target = int(annotation.pixel_x[0]), int(annotation.pixel_y[0])

        image_path = os.path.join(source_dir, frame_name)
        #frame = cv2.imread(image_path)

        #height_src, width_src, _ = frame.shape

        shutil.copyfile(image_path, os.path.join(destination_dir, video_name, frame_name))
        #if ((width_src == width_target) and (height_src == height_target)):
        #    shutil.copyfile(image_path, os.path.join(destination_dir, video_name, frame_name))
        #else:
        #    frame = cv2.resize(frame, (width_target, height_target), interpolation=cv2.INTER_CUBIC)
        #    cv2.imwrite(os.path.join(destination_dir, video_name, frame_name), frame)
