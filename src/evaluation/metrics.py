'''
Warning: metrics are only for reference only, may have limited significance
'''
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from data_utils.checker import auto_label

def data_driven_baselines(gt_kps):
    '''
    gt_kps: T, D
    '''
    gt_velocity = np.abs(gt_kps[1:] - gt_kps[:-1])
    
    mean= np.mean(gt_velocity, axis=0)[np.newaxis] #(1, D)
    mean = np.mean(np.abs(gt_velocity-mean))
    last_step = gt_kps[1] - gt_kps[0]
    last_step = last_step[np.newaxis] #(1, D)
    last_step = np.mean(np.abs(gt_velocity-last_step))
    return last_step, mean

def Batch_LVD(gt_kps, pr_kps):
    # length = np.minimum(gt_kps.shape[0], pr_kps.shape[1])
    length = gt_kps.shape[0]-10
    gt_kps=gt_kps[25:length]
    pr_kps = pr_kps[:, 25:length, :]
    gt_velocity = np.abs(gt_kps[1:] - gt_kps[:-1])[np.newaxis, ...]
    pr_velocity = np.abs(pr_kps[:, 1:, :] - pr_kps[:, :-1, :])
    
    v_diff = np.mean(np.abs(pr_velocity - gt_velocity))
    return v_diff

def LVD(gt_kps, pr_kps):
    gt_kps = gt_kps.squeeze()
    pr_kps = pr_kps.squeeze()
    if len(pr_kps.shape) == 3:
        return Batch_LVD(gt_kps, pr_kps)
    # length = np.minimum(gt_kps.shape[0], pr_kps.shape[0])
    length = gt_kps.shape[0]-10
    gt_kps = gt_kps[25:length]
    pr_kps = pr_kps[25:length] #(T, D)
    if pr_kps.shape[0] < gt_kps.shape[0]:
        pr_kps = np.pad(pr_kps, [[0, int(gt_kps.shape[0]-pr_kps.shape[0])], [0, 0]], mode='constant')
    
    gt_velocity = np.abs(gt_kps[1:] - gt_kps[:-1])
    pr_velocity = np.abs(pr_kps[1:] - pr_kps[:-1])
    
    return np.mean(np.abs(pr_velocity-gt_velocity))

def diversity(kps):
    '''
    kps: bs, seq, dim
    '''
    dis_list = []
    #the distance between each pair
    for i in range(kps.shape[0]):
        for j in range(i+1, kps.shape[0]):
            seq_i = kps[i]
            seq_j = kps[j]
            
            dis = np.mean(np.abs(seq_i - seq_j))
            dis_list.append(dis)
    return np.mean(dis_list)

def peak_velocity(pose_sequence, order=2):
    '''
    pose_sequence: (B, seq, pose_dim).
    '''
    speed = pose_sequence[:,1:, :] - pose_sequence[:, :-1, :] #(B, seq_len-1, pose_dim)
    if order == 2:#velocity
        speed = speed[:, 1:, :] - speed[:, :-1, :]
    
    speed = np.abs(speed)
    speed = np.mean(speed, axis=-1) #(B, seq_len)

    return speed

def velocity_consistency(query, target, length=10):
    '''
    query: (B, seq)
    target: (B, seq)
    we find the frame idx of the peak velocity in gt sequences and pred sequences, and calculate the difference
    '''
    query = np.argsort(query, axis=1)[:, ::-1][:, :length]#(B, 10), frame idx of the top #length velocity
    target = np.argsort(target, axis=1)[:, ::-1][:, :length]

    query = query.reshape(query.shape[0], -1, 1)#(B, 10, 1)
    target = target.reshape(target.shape[0], 1, -1)#(B, 1, 10)

    diff = query - target #(B, 10, 10)，
    diff = np.abs(diff) # 
    diff = np.min(diff, axis=-1) #(B, seq) 
    return diff.reshape(-1)

def mode_transition_seq(pose_sequence, speaker):
    '''
    pose_sequence: (B, seq, pose_dim) 
    ret: binary sequence indicating posture transitions
    '''
    conf = np.zeros_like(pose_sequence)
    num_seconds = (pose_sequence.shape[1]-25) // 25
    res = np.zeros([pose_sequence.shape[0], num_seconds])

    for i in range(pose_sequence.shape[0]):
        bat = pose_sequence[i]
        for j in range(num_seconds):
            label = auto_label(bat[j*25:j*25+50], conf[0,j*25:j*25+50], speaker)
            res[i][j] = (label==1)

    return res.astype(int)

def mode_transition_consistency(query, target):
    '''
    query: (B, seq)
    target: (B, seq)
    we find the posture transitions in gt sequences and pred sequences, and calculate the difference
    '''
    # print(query.shape, target.shape)
    co_appear = query & target#(B, seq)
    co_appear_count = np.sum(co_appear, axis=1)#(B,)
    query_count = np.maximum(np.sum(query, axis=1), 1) #(B,)
    target_count = np.maximum(np.sum(target, axis=1), 1)#(B)
    # print(co_appear_count, target_count)
    precision = co_appear_count / query_count
    recall = co_appear_count / target_count
    assert (recall <= 1).all(), (co_appear_count.shape, target_count.shape)

    accuracy = np.mean(query == target.astype(int))

    return precision, recall, accuracy