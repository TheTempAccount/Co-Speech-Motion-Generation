import os
import json
import numpy as np
from glob import glob
import cv2 as cv
from moviepy.editor import *

def get_gts(clip):    
    keypoints_files = sorted(glob(os.path.join(clip, 'keypoints_new/person_1')+'/*.json'))
    
    upper_body_points = list(np.arange(0, 25))
    poses = [] 
    confs = []
    neck_to_nose_len = []
    mean_position = []
    for kp_file in keypoints_files:
        kp_load = json.load(open(kp_file, 'r'))['people'][0]
        posepts = kp_load['pose_keypoints_2d']
        lhandpts = kp_load['hand_left_keypoints_2d']
        rhandpts = kp_load['hand_right_keypoints_2d']
        facepts = kp_load['face_keypoints_2d']

        neck = np.array(posepts).reshape(-1,3)[1]
        nose = np.array(posepts).reshape(-1,3)[0]
        x_offset = abs(neck[0]-nose[0])
        y_offset = abs(neck[1]-nose[1])
        neck_to_nose_len.append(y_offset)
        mean_position.append([neck[0],neck[1]])

        keypoints=np.array(posepts+lhandpts+rhandpts+facepts).reshape(-1,3)[:,:2]

        upper_body = keypoints[upper_body_points, :]
        hand_points = keypoints[25:, :]
        keypoints = np.vstack([upper_body, hand_points])

        poses.append(keypoints)

    if len(neck_to_nose_len) > 0:
        scale_factor = np.mean(neck_to_nose_len)
    else:
        raise ValueError(clip)
    mean_position = np.mean(np.array(mean_position), axis=0)
    
    unlocalized_poses = np.array(poses).copy()
    localized_poses = []
    for i in range(len(poses)):
        keypoints = poses[i]
        neck = keypoints[1].copy()

        keypoints[:, 0] = (keypoints[:, 0] - neck[0]) / scale_factor
        keypoints[:, 1] = (keypoints[:, 1] - neck[1]) / scale_factor
        localized_poses.append(keypoints.reshape(-1))
        
    localized_poses=np.array(localized_poses)
    return unlocalized_poses, localized_poses, (scale_factor, mean_position)

def load_json(fn):
    annotations = json.load(open(fn))['people'][0]

    posepts=annotations['pose_keypoints_2d']
    hand_left_pts=annotations['hand_left_keypoints_2d']
    hand_right_pts=annotations['hand_right_keypoints_2d']
    face_pts = annotations['face_keypoints_2d']

    all_keypoints = posepts + hand_left_pts + hand_right_pts + face_pts

    all_keypoints = np.array(all_keypoints).reshape(-1, 3)

    return all_keypoints

def make_video(image_dir, vid_file, fps=25):
    image_files = sorted(glob(image_dir+'/*.jpg'))

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    im = cv.imread(image_files[0])
    height, width = im.shape[0], im.shape[1]
    videowriter = cv.VideoWriter(vid_file, fourcc, fps, (width, height))
    for im_name in image_files:
        frame = cv.imread(im_name)
        videowriter.write(frame)
    videowriter.release()

def add_audio(video_pth, audio_pth):
    new_vid_pth = os.path.splitext(video_pth)[0]+'_aud.mp4'
    video = VideoFileClip(video_pth)
    audio = AudioFileClip(audio_pth)
    video = video.set_audio(audio)
    video.write_videofile(new_vid_pth)
    os.remove(video_pth)