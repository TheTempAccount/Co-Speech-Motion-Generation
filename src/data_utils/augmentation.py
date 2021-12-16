import numpy as np

def _get_limbs(keypoints):
    '''
    计算每一条边
    keypoints: (T, J, D)(50, 54, 2), 12+21+21
    '''
    T, J, D = keypoints.shape
    assert T == 50 and J == 12+21+21 and D == 2
    limbs = np.zeros((T, 51, D))
    #与neck直接相连的边
    limbs[:, 0] = (keypoints[:, 0] - keypoints[:, 1])#(neck-nose)
    limbs[:, 1] = (keypoints[:, 2] - keypoints[:, 1])
    limbs[:, 2] = (keypoints[:, 5] - keypoints[:, 1])

    #头部的边
    #与nose相连的边
    limbs[:, 3] = (keypoints[:, 8] - keypoints[:, 0])
    limbs[:, 4] = (keypoints[:, 9] - keypoints[:, 0])
    #与两边眼睛相连的边
    limbs[:, 5] = (keypoints[:, 10] - keypoints[:, 8])
    limbs[:, 6] = (keypoints[:, 11] - keypoints[:, 9])

    #右边身体边
    #rshoulder
    limbs[:, 7] = (keypoints[:, 3] - keypoints[:, 2])
    #relbow
    limbs[:, 8] = (keypoints[:, 4] - keypoints[:, 3])
    #rwrist
    # limbs[:, 9] = (keypoints[:, 33] - keypoints[:, 4])
    #右手， 每根手指和手部起始点的边
    limbs[:, 9] = (keypoints[:, 34] - keypoints[:, 33])
    limbs[:, 10] = (keypoints[:, 38] - keypoints[:, 33])
    limbs[:, 11] = (keypoints[:, 42] - keypoints[:, 33])
    limbs[:, 12] = (keypoints[:, 46] - keypoints[:, 33])
    limbs[:, 13] = (keypoints[:, 50] - keypoints[:, 33])
    #大拇指
    limbs[:, 14] = (keypoints[:, 35] - keypoints[:, 34])
    limbs[:, 15] = (keypoints[:, 36] - keypoints[:, 35])
    limbs[:, 16] = (keypoints[:, 37] - keypoints[:, 36])
    #食指
    limbs[:, 17] = (keypoints[:, 39] - keypoints[:, 38])
    limbs[:, 18] = (keypoints[:, 40] - keypoints[:, 39])
    limbs[:, 19] = (keypoints[:, 41] - keypoints[:, 40])
    #中指
    limbs[:, 20] = (keypoints[:, 43] - keypoints[:, 42])
    limbs[:, 21] = (keypoints[:, 44] - keypoints[:, 43])
    limbs[:, 22] = (keypoints[:, 45] - keypoints[:, 44])
    #无名指
    limbs[:, 23] = (keypoints[:, 47] - keypoints[:, 46])
    limbs[:, 24] = (keypoints[:, 48] - keypoints[:, 47])
    limbs[:, 25] = (keypoints[:, 49] - keypoints[:, 48])
    #小拇指
    limbs[:, 26] = (keypoints[:, 51] - keypoints[:, 50])
    limbs[:, 27] = (keypoints[:, 52] - keypoints[:, 51])
    limbs[:, 28] = (keypoints[:, 53] - keypoints[:, 52])

    #左边身体边
    #rshoulder
    limbs[:, 29] = (keypoints[:, 6] - keypoints[:, 5])
    #relbow
    limbs[:, 30] = (keypoints[:, 7] - keypoints[:, 6])
    #rwrist
    # limbs[:, 32] = (keypoints[:, 12] - keypoints[:, 7])
    #右手， 每根手指和手部起始点的边
    limbs[:, 31] = (keypoints[:, 13] - keypoints[:, 12])
    limbs[:, 32] = (keypoints[:, 17] - keypoints[:, 12])
    limbs[:, 33] = (keypoints[:, 21] - keypoints[:, 12])
    limbs[:, 34] = (keypoints[:, 25] - keypoints[:, 12])
    limbs[:, 35] = (keypoints[:, 29] - keypoints[:, 12])
    #大拇指
    limbs[:, 36] = (keypoints[:, 14] - keypoints[:, 13])
    limbs[:, 37] = (keypoints[:, 15] - keypoints[:, 14])
    limbs[:, 38] = (keypoints[:, 16] - keypoints[:, 15])
    #食指
    limbs[:, 39] = (keypoints[:, 18] - keypoints[:, 17])
    limbs[:, 40] = (keypoints[:, 19] - keypoints[:, 18])
    limbs[:, 41] = (keypoints[:, 20] - keypoints[:, 19])
    #中指
    limbs[:, 42] = (keypoints[:, 22] - keypoints[:, 21])
    limbs[:, 43] = (keypoints[:, 23] - keypoints[:, 22])
    limbs[:, 44] = (keypoints[:, 24] - keypoints[:, 23])
    #无名指
    limbs[:, 45] = (keypoints[:, 26] - keypoints[:, 25])
    limbs[:, 46] = (keypoints[:, 27] - keypoints[:, 26])
    limbs[:, 47] = (keypoints[:, 28] - keypoints[:, 27])
    #小拇指
    limbs[:, 48] = (keypoints[:, 30] - keypoints[:, 29])
    limbs[:, 49] = (keypoints[:, 31] - keypoints[:, 30])
    limbs[:, 50] = (keypoints[:, 32] - keypoints[:, 31])

    return limbs



def _scale_limbs():
    pass

def LimbScaling(keypoints, global_scale=None, local_scales=None):
    '''
    keypoints: (T, J*D)(50, 108)
    global_scale: 对关键点整体的缩放，不一定需要？
    local_scale: 对每一条边的缩放系数，12个关键点总共13条边，手部21个关键点20条边
    '''
    #每一条边关联到的点，从neck开始逆时针
    if global_scale is  None:
        global_scale = 1
    if local_scales is None:#
        local_scales = 0.75 + np.random.random([12])*(1.25-0.75)

    limb_dependents = [
        #从neck关联到的点
        [0, 8, 9, 10, 11], #(neck-nose)，关联到头部的点
        [2, 3, 4] + list(np.arange(33, 54)), #(neck-rshoulder)，关联到右手
        [5, 6, 7] + list(np.arange(12, 33)), #neck-lshoulder), 关联到左手
        #逆时针递归

        #头部点
        #nose
        [8, 10], #(nose-reye)
        [9, 11], #(nose-leye)
        #leye
        [10], 
        #reye
        [11],

        #右边身体点
        #rshoulder
        [3, 4] + list(np.arange(33, 54)),
        #relbow
        [4] + list(np.arange(33, 54)),
        #rwrist
        # list(np.arange(33, 54)),
        #右手 33
        # list(np.arange(34, 54)), #手指上的20个关键点都会和这个点关联
        [34, 25, 36, 37],
        [38, 39, 40, 41],
        [42, 43, 44, 45],
        [46, 47, 48, 49],
        [50, 51, 52, 53],
        #大拇指 34
        [35, 36, 37],
        [36, 37],
        [37],
        #食指 38
        [39, 40, 41],
        [40, 41],
        [41],
        #中指 42
        [43, 44, 45],
        [44, 45],
        [45],
        #无名指 46
        [47, 48, 49],
        [48, 49],
        [49],
        #小拇指 50
        [51, 52, 53],
        [52, 53],
        [53],

        #左边身体点
        #lshoulder
        [6, 7] + list(np.arange(12, 33)),
        #lelbow
        [7] + list(np.arange(12, 33)),
        #lwrist
        # list(np.arange(12, 33)),
        #左手 12
        # list(np.arange(13, 33)), #手指上的20个关键点都会和这个点关联
        [13, 14, 15, 16],
        [17, 18, 19, 20],
        [21, 22, 23, 24],
        [25, 26, 27, 28],
        [29, 30, 31, 32],
        #大拇指 13
        [14, 15, 16],
        [15, 16],
        [16],
        #食指 17
        [18, 19, 20],
        [19, 20],
        [20],
        #中指 21
        [22, 23, 24],
        [23, 24],
        [24],
        #无名指 25
        [26, 27, 28],
        [27, 28],
        [28],
        #小拇指 29
        [30, 31, 32],
        [31, 32],
        [32],
    ]

    #获取在scale之前的边的长度
    T, JD = keypoints.shape
    assert T == 50 and JD == 108
    keypoints = keypoints.reshape(T, -1, 2) #(T, J, D)
    limbs = _get_limbs(keypoints) #(T, 53, D) #53条边在scale之前的长度

    #进行缩放
    scaled_limbs = limbs.copy()

    #global scale，整体放大或缩小
    scaled_limbs = scaled_limbs * global_scale

    #local scale, 对每一条边进行放缩， 对称的边使用相同的放缩系数
    #local_scale: (3+2+3+4), 分别对应了neck周围的边，头部的边，手臂的边，手掌的边，每一层级使用相同的放缩系数
    #与neck直接相连的边
    scaled_limbs[:, 0] *= local_scales[0]
    #两个肩膀应该用相同的缩放系数
    scaled_limbs[:, 1] *= local_scales[1]
    scaled_limbs[:, 2] *= local_scales[1]

    #头部的边
    #与nose相连的边
    scaled_limbs[:, 3] *= local_scales[3]
    scaled_limbs[:, 4] *= local_scales[3]
    #与两边眼睛相连的边
    scaled_limbs[:, 5] *= local_scales[4]
    scaled_limbs[:, 6] *= local_scales[4]

    #右边身体边
    #rshoulder
    scaled_limbs[:, 7] *= local_scales[5]
    #relbow
    scaled_limbs[:, 8] *= local_scales[6]
    #rwrist
    # scaled_limbs[:, 9] *= local_scales[7]
    #右手， 每根手指和手部起始点的边
    scaled_limbs[:, 9] *= local_scales[8]
    scaled_limbs[:, 10] *= local_scales[8]
    scaled_limbs[:, 11] *= local_scales[8]
    scaled_limbs[:, 12] *= local_scales[8]
    scaled_limbs[:, 13] *= local_scales[8]
    #大拇指
    scaled_limbs[:, 14] *= local_scales[9]
    scaled_limbs[:, 15] *= local_scales[10]
    scaled_limbs[:, 16] *= local_scales[11]
    #食指
    scaled_limbs[:, 17] *= local_scales[9]
    scaled_limbs[:, 18] *= local_scales[10]
    scaled_limbs[:, 19] *= local_scales[11]
    #中指
    scaled_limbs[:, 20] *= local_scales[9]
    scaled_limbs[:, 21] *= local_scales[10]
    scaled_limbs[:, 22] *= local_scales[11]
    #无名指
    scaled_limbs[:, 23] *= local_scales[9]
    scaled_limbs[:, 24] *= local_scales[10]
    scaled_limbs[:, 25] *= local_scales[11]
    #小拇指
    scaled_limbs[:, 26] *= local_scales[9]
    scaled_limbs[:, 27] *= local_scales[10]
    scaled_limbs[:, 28] *= local_scales[11]

    #左边身体边
    #rshoulder
    scaled_limbs[:, 29] *= local_scales[5]
    #relbow
    scaled_limbs[:, 30] *= local_scales[6]
    #rwrist
    # scaled_limbs[:, 32] *= local_scales[7]
    #右手， 每根手指和手部起始点的边
    scaled_limbs[:, 31] *= local_scales[8]
    scaled_limbs[:, 32] *= local_scales[8]
    scaled_limbs[:, 33] *= local_scales[8]
    scaled_limbs[:, 34] *= local_scales[8]
    scaled_limbs[:, 35] *= local_scales[8]
    #大拇指
    scaled_limbs[:, 36] *= local_scales[9]
    scaled_limbs[:, 37] *= local_scales[10]
    scaled_limbs[:, 38] *= local_scales[11]
    #食指
    scaled_limbs[:, 39] *= local_scales[9]
    scaled_limbs[:, 40] *= local_scales[10]
    scaled_limbs[:, 41] *= local_scales[11]
    #中指
    scaled_limbs[:, 42] *= local_scales[9]
    scaled_limbs[:, 43] *= local_scales[10]
    scaled_limbs[:, 44] *= local_scales[11]
    #无名指
    scaled_limbs[:, 45] *= local_scales[9]
    scaled_limbs[:, 46] *= local_scales[10]
    scaled_limbs[:, 47] *= local_scales[11]
    #小拇指
    scaled_limbs[:, 48] *= local_scales[9]
    scaled_limbs[:, 49] *= local_scales[10]
    scaled_limbs[:, 50] *= local_scales[11]

    #计算缩放前后每条边向量的变化
    delta = scaled_limbs - limbs

    scaled_keypoints = keypoints.copy()
    #递归修改关键点的位置
    assert len(limb_dependents) == limbs.shape[1], print(len(limb_dependents), limbs.shape[1])
    for i in range(len(limb_dependents)):
        scaled_keypoints[:, limb_dependents[i]] += delta[:, i:i+1, :]
    
    return scaled_keypoints.reshape(T, JD)
    
