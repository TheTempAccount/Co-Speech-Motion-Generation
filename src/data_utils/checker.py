from __future__ import absolute_import

import numpy as np
from .consts import checker_stats


def hand_angle(keypoints, confs):
    
    rwrist=4
    lwrist=7
    assert keypoints.shape[0] == 50 and keypoints.shape[1] == 108
    assert confs.shape[0] == 50 and confs.shape[1] == 108
    
    keypoints = keypoints.reshape(50, -1, 2)
    confs = confs.reshape(50, -1, 2)
    
    lhand = keypoints[:, lwrist, :]
    rhand = keypoints[:, rwrist, :]
    
    tan = (np.abs(rhand-lhand))[:, 1] / (np.abs(rhand-lhand))[:, 0] 
    angles = np.arctan(tan) 
    
    
    
    his_angles = angles[15:25]
    fut_angles = angles[40:50]
    mean_his_angles = np.mean(his_angles)
    mean_fut_angles = np.mean(fut_angles)
    
    label = abs(mean_fut_angles - mean_his_angles) > 0.2
    return mean_his_angles, mean_fut_angles, label

def hand_offset(keypoints, confs):
    
    rwrist=4
    lwrist=7
    assert keypoints.shape[0] == 50 and keypoints.shape[1] == 108
    assert confs.shape[0] == 50 and confs.shape[1] == 108
    
    keypoints = keypoints.reshape(50, -1, 2)
    confs = confs.reshape(50, -1, 2)
    
    lhand = keypoints[:, lwrist, :]
    rhand = keypoints[:, rwrist, :]

    y_offset = (np.abs(rhand-lhand))[:, 1] 
    x_offset = (np.abs(rhand-lhand))[:, 0] 
    
    mean_his_y_offset = np.mean(y_offset[15:25])
    mean_his_x_offset = np.mean(x_offset[15:25])
    mean_fut_y_offset = np.mean(y_offset[40:50])
    mean_fut_x_offset = np.mean(x_offset[40:50])
    
    label = (abs(mean_his_y_offset-mean_fut_y_offset) > 1.0) or ((abs(mean_his_x_offset-mean_fut_x_offset)) > 1.0)
    return mean_his_y_offset, mean_fut_y_offset, mean_his_x_offset, mean_fut_x_offset, label

def hand_position(keypoints, confs):
    
    rwrist=4
    lwrist=7
    assert keypoints.shape[0] == 50 and keypoints.shape[1] == 108
    assert confs.shape[0] == 50 and confs.shape[1] == 108
    
    keypoints = keypoints.reshape(50, -1, 2)
    
    lhand = keypoints[:, lwrist, :]
    rhand = keypoints[:, rwrist, :]

    his_l_pos = lhand[15:25]
    his_r_pos = rhand[15:25]
    fut_l_pos = lhand[40:50]
    fut_r_pos = rhand[40:50]
    
    mean_his_l_pos = np.mean(his_l_pos, axis=0)
    mean_his_r_pos = np.mean(his_r_pos, axis=0)
    mean_fut_l_pos = np.mean(fut_l_pos, axis=0)
    mean_fut_r_pos = np.mean(fut_r_pos, axis=0)
    
    
    mean_l_x_offset = abs(mean_fut_l_pos[0] - mean_his_l_pos[0])
    mean_l_y_offset = abs(mean_fut_l_pos[1] - mean_his_l_pos[1])
    mean_r_x_offset = abs(mean_fut_r_pos[0] - mean_his_r_pos[0])
    mean_r_y_offset = abs(mean_fut_r_pos[1] - mean_his_r_pos[1])
    
    label = (mean_l_x_offset > 1) or (mean_l_y_offset > 1) or (mean_r_x_offset > 1) or (mean_r_y_offset > 1)
    return mean_l_x_offset, mean_l_y_offset, mean_r_x_offset, mean_r_y_offset, label


def finger_distance(keypoints, confs, thres=0.9):
    
    assert keypoints.shape[0] == 50 and keypoints.shape[1] == 108
    assert confs.shape[0] == 50 and confs.shape[1] == 108
    
    keypoints = keypoints.reshape(50, -1, 2)
    confs = confs.reshape(50, -1, 2)
    lhand = keypoints[:, 12:12+21, :]
    rhand = keypoints[:, 12+21:12+21+21, :]
    
    hand_dis = np.sqrt(np.sum(np.square(lhand - rhand), axis=-1))
    
    
    
    his_hand_dis = hand_dis[15:25, ...]
    fut_hand_dis = hand_dis[40:50, ...]
    mean_his_dis = np.mean(his_hand_dis)
    mean_fut_dis = np.mean(fut_hand_dis)
    
    label = (abs(mean_fut_dis-mean_his_dis)) > thres 
    

    return abs(mean_fut_dis-mean_his_dis), label, (1-label)

def finger_angle(keypoints, confs, thres=0.3):
    
    assert keypoints.shape[0] == 50 and keypoints.shape[1] == 108
    assert confs.shape[0] == 50 and confs.shape[1] == 108
    
    keypoints = keypoints.reshape(50, -1, 2)
    confs = confs.reshape(50, -1, 2)
    
    lhand = keypoints[:, 12:21+12, :]
    rhand = keypoints[:, 21+12:12+21+21, :]
    
    
    tan = (np.abs(rhand-lhand))[:, :, 1] / ((np.abs(rhand-lhand))[:, :, 0]+1e-9) 
    angles = np.arctan(tan) 
    angles[np.where(np.isnan(angles))] = 1
    
    angles = np.mean(angles, axis=1)
    
    his_angles = angles[15:25]
    fut_angles = angles[40:50]
    mean_his_angles = np.mean(his_angles)
    mean_fut_angles = np.mean(fut_angles)
    
    positive = abs(mean_fut_angles - mean_his_angles) > thres
    negative = abs(mean_fut_angles - mean_his_angles) < 0.2

    return abs(mean_fut_angles - mean_his_angles), positive, negative

def finger_offset(keypoints, confs, thres_list=[1.5, 1.5]):
    
    assert keypoints.shape[0] == 50 and keypoints.shape[1] == 108
    assert confs.shape[0] == 50 and confs.shape[1] == 108
    
    keypoints = keypoints.reshape(50, -1, 2)
    confs = confs.reshape(50, -1, 2)
    
    lhand = keypoints[:, 12:12+21, :]
    rhand = keypoints[:, 12+21:12+21+21, :]

    y_offset = (np.abs(rhand-lhand))[:, :, 1] 
    x_offset = (np.abs(rhand-lhand))[:, :, 0] 
    y_offset = np.mean(y_offset, axis=1)
    x_offset = np.mean(x_offset, axis=1)
    
    mean_his_y_offset = np.mean(y_offset[15:25])
    mean_his_x_offset = np.mean(x_offset[15:25])
    mean_fut_y_offset = np.mean(y_offset[40:50])
    mean_fut_x_offset = np.mean(x_offset[40:50])
    
    positive = (abs(mean_his_y_offset-mean_fut_y_offset) > thres_list[0]) or ((abs(mean_his_x_offset-mean_fut_x_offset)) > thres_list[1])

    negative = (abs(mean_his_y_offset-mean_fut_y_offset) < 1.0) and ((abs(mean_his_x_offset-mean_fut_x_offset)) < 1.0)
    return abs(mean_his_y_offset-mean_fut_y_offset), abs(mean_his_x_offset-mean_fut_x_offset), positive, negative

def finger_position(keypoints, confs, thres_list=[1.5, 1.5, 1.5, 1.5]):
    
    assert keypoints.shape[0] == 50 and keypoints.shape[1] == 108
    assert confs.shape[0] == 50 and confs.shape[1] == 108
    
    keypoints = keypoints.reshape(50, -1, 2)
    confs = confs.reshape(50, -1, 2)
    
    lhand = keypoints[:, 12:12+21, :]
    rhand = keypoints[:, 12+21:12+21+21, :]
    lhand = np.mean(lhand, axis=1)
    rhand = np.mean(rhand, axis=1)
    

    his_l_pos = lhand[15:25, :]
    his_r_pos = rhand[15:25, :]
    fut_l_pos = lhand[40:50, :]
    fut_r_pos = rhand[40:50, :]
    
    mean_his_l_pos = np.mean(his_l_pos, axis=0)
    mean_his_r_pos = np.mean(his_r_pos, axis=0)
    mean_fut_l_pos = np.mean(fut_l_pos, axis=0)
    mean_fut_r_pos = np.mean(fut_r_pos, axis=0)
    
    
    mean_l_x_offset = abs(mean_fut_l_pos[0] - mean_his_l_pos[0])
    mean_l_y_offset = abs(mean_fut_l_pos[1] - mean_his_l_pos[1])
    mean_r_x_offset = abs(mean_fut_r_pos[0] - mean_his_r_pos[0])
    mean_r_y_offset = abs(mean_fut_r_pos[1] - mean_his_r_pos[1])
    
    positive = (mean_l_x_offset > thres_list[0]) or (mean_l_y_offset > thres_list[1]) or (mean_r_x_offset > thres_list[2]) or (mean_r_y_offset > thres_list[3])

    negative = (mean_l_x_offset < 1.0) and (mean_l_y_offset < 1.0) and (mean_r_x_offset < 1.0) and (mean_r_y_offset < 1.0)
    return mean_l_x_offset, mean_l_y_offset, mean_r_x_offset, mean_r_y_offset, positive, negative

def finger_velocity(keypoints, confs, thres_list=[0.08, 0.08, 0.08, 0.08], floor_thres_list=[0.06, 0.06, 0.06, 0.06], ceil_thres_list=[0.15,0.15,0.15, 0.15]):
    
    assert keypoints.shape[0] == 50 and keypoints.shape[1] == 108
    assert confs.shape[0] == 50 and confs.shape[1] == 108
    
    keypoints = keypoints.reshape(50, -1, 2)
    keypoints = keypoints[:, 12:, :]
    velocity = keypoints[1:,:,:] - keypoints[:-1,:,:] 
    velocity = velocity[25:35, ...]
    velocity = np.abs(velocity)
    
    
    l_x_v = velocity[:, :21, 0]
    l_y_v = velocity[:, :21, 1]
    r_x_v = velocity[:, 21:, 0]
    r_y_v = velocity[:, 21:, 1]
    
    mean_l_x_v = np.mean(l_x_v)
    mean_l_y_v = np.mean(l_y_v)
    mean_r_x_v = np.mean(r_x_v)
    mean_r_y_v = np.mean(r_y_v)
    
    if mean_l_x_v > 1:
        mean_l_x_v = 0
    if mean_l_y_v > 1:
        mean_l_y_v = 0
    if mean_r_x_v > 1:
        mean_r_x_v = 0    
    if mean_r_y_v > 1:
        mean_r_y_v = 0
    
    soft_label = 0
    hard_label = None
    
    soft_label = (mean_l_x_v > thres_list[0]) or (mean_l_y_v > thres_list[1]) or (mean_r_x_v > thres_list[2]) or (mean_r_y_v > thres_list[3])
    
    
    if (mean_l_x_v < floor_thres_list[0]) and (mean_l_y_v < floor_thres_list[1]) and (mean_r_x_v < floor_thres_list[2]) and (mean_r_y_v < floor_thres_list[3]):
        hard_label = 0 
    
    
    
    
    
    return mean_l_x_v, mean_l_y_v, mean_r_x_v, mean_r_y_v, soft_label, hard_label


def auto_label(bat, bat_confs, speaker):
    '''
    bat: [50, 108]
    bat_confs: [50, 108]
    '''
    
    positive = 0
    negative = 0
    keypoints, confs = bat, bat_confs
        






    if speaker in checker_stats:
        res=finger_distance(keypoints, confs, checker_stats[speaker]['finger_distance'])
        positive += res[-2]
        negative += res[-1]
        res=finger_angle(keypoints, confs, checker_stats[speaker]['finger_angle'])
        positive += res[-2]
        negative += res[-1]
        res=finger_offset(keypoints, confs, checker_stats[speaker]['finger_offset'])
        positive += res[-2]
        negative += res[-1]
        res=finger_position(keypoints, confs, checker_stats[speaker]['finger_position'])
        positive += res[-2]
        negative += res[-1]
        res=finger_velocity(keypoints, confs, checker_stats[speaker]['finger_velocity'][0], checker_stats[speaker]['finger_velocity'][1], checker_stats[speaker]['finger_velocity'][2])
        soft_label,hard_label = res[-2], res[-1]

        positive += soft_label
        negative += (1-soft_label)
    else:
        res=finger_distance(keypoints, confs)
        positive += res[-2]
        negative += res[-1]
        res=finger_angle(keypoints, confs)
        positive += res[-2]
        negative += res[-1]
        res=finger_offset(keypoints, confs)
        positive += res[-2]
        negative += res[-1]
        res=finger_position(keypoints, confs)
        positive += res[-2]
        negative += res[-1]
        res=finger_velocity(keypoints, confs)
        soft_label,hard_label = res[-2], res[-1]
        positive += soft_label
        negative += (1-soft_label)

    if speaker in checker_stats:
        positive_thres = checker_stats[speaker]['positive_thres']
        negative_thres = checker_stats[speaker]['negative_thres']
    else:
        positive_thres = 3
        negative_thres = 2
   
    #print(speaker)
    #print(positive_thres, negative_thres)
    if positive > positive_thres: 
        finnal_label = 1
    elif negative > negative_thres: 
        finnal_label = 0
    else:
        finnal_label = -1

    return finnal_label
