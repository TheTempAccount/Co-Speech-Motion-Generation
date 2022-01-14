import json
import numpy as np

def load_json(fn):
    annotations = json.load(open(fn))['people'][0]

    posepts=annotations['pose_keypoints_2d']
    hand_left_pts=annotations['hand_left_keypoints_2d']
    hand_right_pts=annotations['hand_right_keypoints_2d']
    face_pts = annotations['face_keypoints_2d']

    all_keypoints = posepts + hand_left_pts + hand_right_pts + face_pts

    all_keypoints = np.array(all_keypoints).reshape(-1, 3)

    return all_keypoints