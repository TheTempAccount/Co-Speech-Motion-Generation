from __future__ import absolute_import

import math
import os
import random
import sys

import time
import argparse
import numpy as np
from .consts import checker_stats


def hand_angle(keypoints, confs):
    #根据手部的夹角进行判断
    rwrist=4
    lwrist=7
    assert keypoints.shape[0] == 50 and keypoints.shape[1] == 108
    assert confs.shape[0] == 50 and confs.shape[1] == 108
    
    keypoints = keypoints.reshape(50, -1, 2)
    confs = confs.reshape(50, -1, 2)
    
    lhand = keypoints[:, lwrist, :]#seq_len是50,每个seq中取出wrist的位置
    rhand = keypoints[:, rwrist, :]
    #计算每一帧中两只手之间的夹角
    tan = (np.abs(rhand-lhand))[:, 1] / (np.abs(rhand-lhand))[:, 0] #(rhand-lhand, 50, 1)
    angles = np.arctan(tan) #(50)    
    
    #区分出是在his还是在fut里面发生了状态的转换的
    #只看末尾的十帧?
    his_angles = angles[15:25]#(25, 21)
    fut_angles = angles[40:50]#(25, 21)
    mean_his_angles = np.mean(his_angles)
    mean_fut_angles = np.mean(fut_angles)
    
    label = abs(mean_fut_angles - mean_his_angles) > 0.2
    return mean_his_angles, mean_fut_angles, label

def hand_offset(keypoints, confs):
    #根据两手之间的距离变化判断，实际上就是两手之间的距离，但是分成了x,y两个维度上的距离
    rwrist=4
    lwrist=7
    assert keypoints.shape[0] == 50 and keypoints.shape[1] == 108
    assert confs.shape[0] == 50 and confs.shape[1] == 108
    
    keypoints = keypoints.reshape(50, -1, 2)
    confs = confs.reshape(50, -1, 2)
    
    lhand = keypoints[:, lwrist, :]#seq_len是50,每个seq中取出wrist的位置
    rhand = keypoints[:, rwrist, :]

    y_offset = (np.abs(rhand-lhand))[:, 1] #(50)#y坐标上的距离 
    x_offset = (np.abs(rhand-lhand))[:, 0] #(50)
    
    mean_his_y_offset = np.mean(y_offset[15:25])
    mean_his_x_offset = np.mean(x_offset[15:25])
    mean_fut_y_offset = np.mean(y_offset[40:50])
    mean_fut_x_offset = np.mean(x_offset[40:50])
    
    label = (abs(mean_his_y_offset-mean_fut_y_offset) > 1.0) or ((abs(mean_his_x_offset-mean_fut_x_offset)) > 1.0)
    return mean_his_y_offset, mean_fut_y_offset, mean_his_x_offset, mean_fut_x_offset, label

def hand_position(keypoints, confs):
    #根据手部的位置前后差异，例如，演讲者可能会同时抬起双手
    rwrist=4
    lwrist=7
    assert keypoints.shape[0] == 50 and keypoints.shape[1] == 108
    assert confs.shape[0] == 50 and confs.shape[1] == 108
    
    keypoints = keypoints.reshape(50, -1, 2)
    
    lhand = keypoints[:, lwrist, :]#seq_len是50,每个seq中取出wrist的位置
    rhand = keypoints[:, rwrist, :]

    his_l_pos = lhand[15:25]#(50, 2)
    his_r_pos = rhand[15:25]
    fut_l_pos = lhand[40:50]
    fut_r_pos = rhand[40:50]
    
    mean_his_l_pos = np.mean(his_l_pos, axis=0)#(2)
    mean_his_r_pos = np.mean(his_r_pos, axis=0)
    mean_fut_l_pos = np.mean(fut_l_pos, axis=0)
    mean_fut_r_pos = np.mean(fut_r_pos, axis=0)
    
    #在两个方向上和初始时候的位置上的差异
    mean_l_x_offset = abs(mean_fut_l_pos[0] - mean_his_l_pos[0])
    mean_l_y_offset = abs(mean_fut_l_pos[1] - mean_his_l_pos[1])
    mean_r_x_offset = abs(mean_fut_r_pos[0] - mean_his_r_pos[0])
    mean_r_y_offset = abs(mean_fut_r_pos[1] - mean_his_r_pos[1])
    
    label = (mean_l_x_offset > 1) or (mean_l_y_offset > 1) or (mean_r_x_offset > 1) or (mean_r_y_offset > 1)
    return mean_l_x_offset, mean_l_y_offset, mean_r_x_offset, mean_r_y_offset, label


def finger_distance(keypoints, confs, thres=0.9):
    #根据两只手之间的距离来判断是否发生了姿态的变换
    assert keypoints.shape[0] == 50 and keypoints.shape[1] == 108
    assert confs.shape[0] == 50 and confs.shape[1] == 108
    
    keypoints = keypoints.reshape(50, -1, 2)
    confs = confs.reshape(50, -1, 2)
    lhand = keypoints[:, 12:12+21, :]
    rhand = keypoints[:, 12+21:12+21+21, :]
    
    hand_dis = np.sqrt(np.sum(np.square(lhand - rhand), axis=-1))
    
    #区分出是在his还是在fut里面发生了状态的转换的
    #只看末尾的十帧?
    his_hand_dis = hand_dis[15:25, ...]#(25, 21)
    fut_hand_dis = hand_dis[40:50, ...]#(25, 21)
    mean_his_dis = np.mean(his_hand_dis)
    mean_fut_dis = np.mean(fut_hand_dis)
    
    label = (abs(mean_fut_dis-mean_his_dis)) > thres #相对位移,scale_factor大约在100左右，1.0也就是100左右，双手已经打开的状况下继续向外移动就会导致这样的
    
#     return mean_his_dis, mean_fut_dis, label
    return abs(mean_fut_dis-mean_his_dis), label, (1-label)

def finger_angle(keypoints, confs, thres=0.3):
    #根据手部的夹角进行判断
    assert keypoints.shape[0] == 50 and keypoints.shape[1] == 108
    assert confs.shape[0] == 50 and confs.shape[1] == 108
    
    keypoints = keypoints.reshape(50, -1, 2)
    confs = confs.reshape(50, -1, 2)
    
    lhand = keypoints[:, 12:21+12, :]#seq_len是50,每个seq中取出wrist的位置
    rhand = keypoints[:, 21+12:12+21+21, :]#(50, 21, 2)
    
    #计算每一帧中两只手之间的夹角
    tan = (np.abs(rhand-lhand))[:, :, 1] / ((np.abs(rhand-lhand))[:, :, 0]+1e-9) #(rhand-lhand, 50, 1)
    angles = np.arctan(tan) #(50, 21)
    angles[np.where(np.isnan(angles))] = 1#0/0应该是1？
    
    angles = np.mean(angles, axis=1)#(50)
    
    his_angles = angles[15:25]
    fut_angles = angles[40:50]
    mean_his_angles = np.mean(his_angles)
    mean_fut_angles = np.mean(fut_angles)
    
    positive = abs(mean_fut_angles - mean_his_angles) > thres#原0.3
    negative = abs(mean_fut_angles - mean_his_angles) < 0.2
#     return mean_his_angles, mean_fut_angles, label
    return abs(mean_fut_angles - mean_his_angles), positive, negative

def finger_offset(keypoints, confs, thres_list=[1.5, 1.5]):
    #根据手部横纵方向上的偏移进行判，这个可能和distance有重叠，例如演讲者打开了双手，两只手在水平方向上的距离会增加断，即两只手在水平方向和竖直方向上是否发生了偏移
    assert keypoints.shape[0] == 50 and keypoints.shape[1] == 108
    assert confs.shape[0] == 50 and confs.shape[1] == 108
    
    keypoints = keypoints.reshape(50, -1, 2)
    confs = confs.reshape(50, -1, 2)
    
    lhand = keypoints[:, 12:12+21, :]#
    rhand = keypoints[:, 12+21:12+21+21, :]

    y_offset = (np.abs(rhand-lhand))[:, :, 1] #(50) 
    x_offset = (np.abs(rhand-lhand))[:, :, 0] #(50, 21)
    y_offset = np.mean(y_offset, axis=1)
    x_offset = np.mean(x_offset, axis=1)
    
    mean_his_y_offset = np.mean(y_offset[15:25])
    mean_his_x_offset = np.mean(x_offset[15:25])
    mean_fut_y_offset = np.mean(y_offset[40:50])
    mean_fut_x_offset = np.mean(x_offset[40:50])
    
    positive = (abs(mean_his_y_offset-mean_fut_y_offset) > thres_list[0]) or ((abs(mean_his_x_offset-mean_fut_x_offset)) > thres_list[1])#原1.5
#     return mean_his_y_offset, mean_fut_y_offset, mean_his_x_offset, mean_fut_x_offset, label
    negative = (abs(mean_his_y_offset-mean_fut_y_offset) < 1.0) and ((abs(mean_his_x_offset-mean_fut_x_offset)) < 1.0)
    return abs(mean_his_y_offset-mean_fut_y_offset), abs(mean_his_x_offset-mean_fut_x_offset), positive, negative

def finger_position(keypoints, confs, thres_list=[1.5, 1.5, 1.5, 1.5]):
    #根据手部的位置差异，例如，演讲者可能会同时抬起双手
    assert keypoints.shape[0] == 50 and keypoints.shape[1] == 108
    assert confs.shape[0] == 50 and confs.shape[1] == 108
    
    keypoints = keypoints.reshape(50, -1, 2)
    confs = confs.reshape(50, -1, 2)
    
    lhand = keypoints[:, 12:12+21, :]#seq_len是50,每个seq中取出wrist的位置
    rhand = keypoints[:, 12+21:12+21+21, :]#(50, 21, 2)
    lhand = np.mean(lhand, axis=1)
    rhand = np.mean(rhand, axis=1)
    
#     lhand_confs = confs[:, lwrist, :]
    his_l_pos = lhand[15:25, :]
    his_r_pos = rhand[15:25, :]
    fut_l_pos = lhand[40:50, :]
    fut_r_pos = rhand[40:50, :]
    
    mean_his_l_pos = np.mean(his_l_pos, axis=0)#(2)
    mean_his_r_pos = np.mean(his_r_pos, axis=0)
    mean_fut_l_pos = np.mean(fut_l_pos, axis=0)
    mean_fut_r_pos = np.mean(fut_r_pos, axis=0)
    
    #在两个方向上和初始时候的位置上的差异
    mean_l_x_offset = abs(mean_fut_l_pos[0] - mean_his_l_pos[0])
    mean_l_y_offset = abs(mean_fut_l_pos[1] - mean_his_l_pos[1])
    mean_r_x_offset = abs(mean_fut_r_pos[0] - mean_his_r_pos[0])
    mean_r_y_offset = abs(mean_fut_r_pos[1] - mean_his_r_pos[1])
    
    positive = (mean_l_x_offset > thres_list[0]) or (mean_l_y_offset > thres_list[1]) or (mean_r_x_offset > thres_list[2]) or (mean_r_y_offset > thres_list[3])

    negative = (mean_l_x_offset < 1.0) and (mean_l_y_offset < 1.0) and (mean_r_x_offset < 1.0) and (mean_r_y_offset < 1.0)
    return mean_l_x_offset, mean_l_y_offset, mean_r_x_offset, mean_r_y_offset, positive, negative

def finger_velocity(keypoints, confs, thres_list=[0.08, 0.08, 0.08, 0.08], floor_thres_list=[0.06, 0.06, 0.06, 0.06], ceil_thres_list=[0.15,0.15,0.15, 0.15]):
    #通过future的起始阶段的velocity来判断，floor和ceil作为hard judge
    assert keypoints.shape[0] == 50 and keypoints.shape[1] == 108
    assert confs.shape[0] == 50 and confs.shape[1] == 108
    
    keypoints = keypoints.reshape(50, -1, 2)#(50, 54, 2)
    keypoints = keypoints[:, 12:, :]#(50, 42, 2)，只关注手部的加速度
    velocity = keypoints[1:,:,:] - keypoints[:-1,:,:] #(49, 54, 2)
    velocity = velocity[25:35, ...]#(10, 54, 2)#只看future的起始steps中的加速度
    velocity = np.abs(velocity)#变化的绝对值
    
    #两只手，每只手有两个维度
    l_x_v = velocity[:, :21, 0]#(10, 21)
    l_y_v = velocity[:, :21, 1]
    r_x_v = velocity[:, 21:, 0]
    r_y_v = velocity[:, 21:, 1]
    
    mean_l_x_v = np.mean(l_x_v)#左手在前十帧里面的平均加速度，每个点的加速度都平均进来了
    mean_l_y_v = np.mean(l_y_v)
    mean_r_x_v = np.mean(r_x_v)
    mean_r_y_v = np.mean(r_y_v)
    
    if mean_l_x_v > 1:#太大了认为是检测点错误
        mean_l_x_v = 0
    if mean_l_y_v > 1:#太大了认为是检测点错误
        mean_l_y_v = 0
    if mean_r_x_v > 1:#太大了认为是检测点错误
        mean_r_x_v = 0    
    if mean_r_y_v > 1:#太大了认为是检测点错误
        mean_r_y_v = 0
    
    soft_label = 0
    hard_label = None
    
    soft_label = (mean_l_x_v > thres_list[0]) or (mean_l_y_v > thres_list[1]) or (mean_r_x_v > thres_list[2]) or (mean_r_y_v > thres_list[3])
    #hard_label
    #全都小于hard_thres，则认为是zero
    if (mean_l_x_v < floor_thres_list[0]) and (mean_l_y_v < floor_thres_list[1]) and (mean_r_x_v < floor_thres_list[2]) and (mean_r_y_v < floor_thres_list[3]):
        hard_label = 0 
    
    #有一个大于hard_ceil，则认为是trans
    # if (mean_l_x_v > ceil_thres_list[0]) or (mean_l_y_v > ceil_thres_list[1]) or (mean_r_x_v > ceil_thres_list[2]) or (mean_r_y_v > ceil_thres_list[3]):
    #     hard_label = 1 
    
    return mean_l_x_v, mean_l_y_v, mean_r_x_v, mean_r_y_v, soft_label, hard_label


def auto_label(bat, bat_confs, speaker):
    #对一个sample进行label, bat: (50, 108)
    #TODO: batch的形式
    positive = 0
    negative = 0
    keypoints, confs = bat, bat_confs
        #首先根据两手之间的距离进行判断
#         res=hand_angle(keypoints, confs)
#         label += res[-1]
#         res=hand_offset(keypoints, confs)
#         label += res[-1]
#         res=hand_position(keypoints, confs)
#         label += res[-1]
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
#         label += soft_label
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

    if positive > positive_thres: #全部是1, 则认为发生了较大的状态变化
        finnal_label = 1
    elif negative > negative_thres: #有三个以上的指标认为
        finnal_label = 0
    else:
        finnal_label = -1
#     if hard_label is not None:
#         finnal_label = hard_label

    return finnal_label