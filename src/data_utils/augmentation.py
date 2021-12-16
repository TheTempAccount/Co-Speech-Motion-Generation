import numpy as np

def _get_limbs(keypoints):
    '''
    keypoints: (T, J, D)(50, 54, 2), 12+21+21
    '''
    T, J, D = keypoints.shape
    assert T == 50 and J == 12+21+21 and D == 2
    limbs = np.zeros((T, 51, D))
    
    limbs[:, 0] = (keypoints[:, 0] - keypoints[:, 1])
    limbs[:, 1] = (keypoints[:, 2] - keypoints[:, 1])
    limbs[:, 2] = (keypoints[:, 5] - keypoints[:, 1])

    
    
    limbs[:, 3] = (keypoints[:, 8] - keypoints[:, 0])
    limbs[:, 4] = (keypoints[:, 9] - keypoints[:, 0])
    
    limbs[:, 5] = (keypoints[:, 10] - keypoints[:, 8])
    limbs[:, 6] = (keypoints[:, 11] - keypoints[:, 9])

    
    
    limbs[:, 7] = (keypoints[:, 3] - keypoints[:, 2])
    
    limbs[:, 8] = (keypoints[:, 4] - keypoints[:, 3])
    
    
    
    limbs[:, 9] = (keypoints[:, 34] - keypoints[:, 33])
    limbs[:, 10] = (keypoints[:, 38] - keypoints[:, 33])
    limbs[:, 11] = (keypoints[:, 42] - keypoints[:, 33])
    limbs[:, 12] = (keypoints[:, 46] - keypoints[:, 33])
    limbs[:, 13] = (keypoints[:, 50] - keypoints[:, 33])
    
    limbs[:, 14] = (keypoints[:, 35] - keypoints[:, 34])
    limbs[:, 15] = (keypoints[:, 36] - keypoints[:, 35])
    limbs[:, 16] = (keypoints[:, 37] - keypoints[:, 36])
    
    limbs[:, 17] = (keypoints[:, 39] - keypoints[:, 38])
    limbs[:, 18] = (keypoints[:, 40] - keypoints[:, 39])
    limbs[:, 19] = (keypoints[:, 41] - keypoints[:, 40])
    
    limbs[:, 20] = (keypoints[:, 43] - keypoints[:, 42])
    limbs[:, 21] = (keypoints[:, 44] - keypoints[:, 43])
    limbs[:, 22] = (keypoints[:, 45] - keypoints[:, 44])
    
    limbs[:, 23] = (keypoints[:, 47] - keypoints[:, 46])
    limbs[:, 24] = (keypoints[:, 48] - keypoints[:, 47])
    limbs[:, 25] = (keypoints[:, 49] - keypoints[:, 48])
    
    limbs[:, 26] = (keypoints[:, 51] - keypoints[:, 50])
    limbs[:, 27] = (keypoints[:, 52] - keypoints[:, 51])
    limbs[:, 28] = (keypoints[:, 53] - keypoints[:, 52])

    
    
    limbs[:, 29] = (keypoints[:, 6] - keypoints[:, 5])
    
    limbs[:, 30] = (keypoints[:, 7] - keypoints[:, 6])
    
    
    
    limbs[:, 31] = (keypoints[:, 13] - keypoints[:, 12])
    limbs[:, 32] = (keypoints[:, 17] - keypoints[:, 12])
    limbs[:, 33] = (keypoints[:, 21] - keypoints[:, 12])
    limbs[:, 34] = (keypoints[:, 25] - keypoints[:, 12])
    limbs[:, 35] = (keypoints[:, 29] - keypoints[:, 12])
    
    limbs[:, 36] = (keypoints[:, 14] - keypoints[:, 13])
    limbs[:, 37] = (keypoints[:, 15] - keypoints[:, 14])
    limbs[:, 38] = (keypoints[:, 16] - keypoints[:, 15])
    
    limbs[:, 39] = (keypoints[:, 18] - keypoints[:, 17])
    limbs[:, 40] = (keypoints[:, 19] - keypoints[:, 18])
    limbs[:, 41] = (keypoints[:, 20] - keypoints[:, 19])
    
    limbs[:, 42] = (keypoints[:, 22] - keypoints[:, 21])
    limbs[:, 43] = (keypoints[:, 23] - keypoints[:, 22])
    limbs[:, 44] = (keypoints[:, 24] - keypoints[:, 23])
    
    limbs[:, 45] = (keypoints[:, 26] - keypoints[:, 25])
    limbs[:, 46] = (keypoints[:, 27] - keypoints[:, 26])
    limbs[:, 47] = (keypoints[:, 28] - keypoints[:, 27])
    
    limbs[:, 48] = (keypoints[:, 30] - keypoints[:, 29])
    limbs[:, 49] = (keypoints[:, 31] - keypoints[:, 30])
    limbs[:, 50] = (keypoints[:, 32] - keypoints[:, 31])

    return limbs



def _scale_limbs():
    pass

def LimbScaling(keypoints, global_scale=None, local_scales=None):
    '''
    keypoints: (T, J*D)(50, 108)
    global_scale: 
    local_scale: 对每一条边的缩放系数，12个关键点总共13条边，手部21个关键点20条边
    '''
    
    if global_scale is  None:
        global_scale = 1
    if local_scales is None:
        local_scales = 0.75 + np.random.random([12])*(1.25-0.75)

    limb_dependents = [
        
        [0, 8, 9, 10, 11], 
        [2, 3, 4] + list(np.arange(33, 54)), 
        [5, 6, 7] + list(np.arange(12, 33)), 
        

        
        
        [8, 10], 
        [9, 11], 
        
        [10], 
        
        [11],

        
        
        [3, 4] + list(np.arange(33, 54)),
        
        [4] + list(np.arange(33, 54)),
        
        
        
        
        [34, 25, 36, 37],
        [38, 39, 40, 41],
        [42, 43, 44, 45],
        [46, 47, 48, 49],
        [50, 51, 52, 53],
        
        [35, 36, 37],
        [36, 37],
        [37],
        
        [39, 40, 41],
        [40, 41],
        [41],
        
        [43, 44, 45],
        [44, 45],
        [45],
        
        [47, 48, 49],
        [48, 49],
        [49],
        
        [51, 52, 53],
        [52, 53],
        [53],

        
        
        [6, 7] + list(np.arange(12, 33)),
        
        [7] + list(np.arange(12, 33)),
        
        
        
        
        [13, 14, 15, 16],
        [17, 18, 19, 20],
        [21, 22, 23, 24],
        [25, 26, 27, 28],
        [29, 30, 31, 32],
        
        [14, 15, 16],
        [15, 16],
        [16],
        
        [18, 19, 20],
        [19, 20],
        [20],
        
        [22, 23, 24],
        [23, 24],
        [24],
        
        [26, 27, 28],
        [27, 28],
        [28],
        
        [30, 31, 32],
        [31, 32],
        [32],
    ]

    
    T, JD = keypoints.shape
    assert T == 50 and JD == 108
    keypoints = keypoints.reshape(T, -1, 2) 
    limbs = _get_limbs(keypoints) 

    
    scaled_limbs = limbs.copy()

    
    scaled_limbs = scaled_limbs * global_scale

    
    
    
    scaled_limbs[:, 0] *= local_scales[0]
    
    scaled_limbs[:, 1] *= local_scales[1]
    scaled_limbs[:, 2] *= local_scales[1]

    
    
    scaled_limbs[:, 3] *= local_scales[3]
    scaled_limbs[:, 4] *= local_scales[3]
    
    scaled_limbs[:, 5] *= local_scales[4]
    scaled_limbs[:, 6] *= local_scales[4]

    
    
    scaled_limbs[:, 7] *= local_scales[5]
    
    scaled_limbs[:, 8] *= local_scales[6]
    
    
    
    scaled_limbs[:, 9] *= local_scales[8]
    scaled_limbs[:, 10] *= local_scales[8]
    scaled_limbs[:, 11] *= local_scales[8]
    scaled_limbs[:, 12] *= local_scales[8]
    scaled_limbs[:, 13] *= local_scales[8]
    
    scaled_limbs[:, 14] *= local_scales[9]
    scaled_limbs[:, 15] *= local_scales[10]
    scaled_limbs[:, 16] *= local_scales[11]
    
    scaled_limbs[:, 17] *= local_scales[9]
    scaled_limbs[:, 18] *= local_scales[10]
    scaled_limbs[:, 19] *= local_scales[11]
    
    scaled_limbs[:, 20] *= local_scales[9]
    scaled_limbs[:, 21] *= local_scales[10]
    scaled_limbs[:, 22] *= local_scales[11]
    
    scaled_limbs[:, 23] *= local_scales[9]
    scaled_limbs[:, 24] *= local_scales[10]
    scaled_limbs[:, 25] *= local_scales[11]
    
    scaled_limbs[:, 26] *= local_scales[9]
    scaled_limbs[:, 27] *= local_scales[10]
    scaled_limbs[:, 28] *= local_scales[11]

    
    
    scaled_limbs[:, 29] *= local_scales[5]
    
    scaled_limbs[:, 30] *= local_scales[6]
    
    
    
    scaled_limbs[:, 31] *= local_scales[8]
    scaled_limbs[:, 32] *= local_scales[8]
    scaled_limbs[:, 33] *= local_scales[8]
    scaled_limbs[:, 34] *= local_scales[8]
    scaled_limbs[:, 35] *= local_scales[8]
    
    scaled_limbs[:, 36] *= local_scales[9]
    scaled_limbs[:, 37] *= local_scales[10]
    scaled_limbs[:, 38] *= local_scales[11]
    
    scaled_limbs[:, 39] *= local_scales[9]
    scaled_limbs[:, 40] *= local_scales[10]
    scaled_limbs[:, 41] *= local_scales[11]
    
    scaled_limbs[:, 42] *= local_scales[9]
    scaled_limbs[:, 43] *= local_scales[10]
    scaled_limbs[:, 44] *= local_scales[11]
    
    scaled_limbs[:, 45] *= local_scales[9]
    scaled_limbs[:, 46] *= local_scales[10]
    scaled_limbs[:, 47] *= local_scales[11]
    
    scaled_limbs[:, 48] *= local_scales[9]
    scaled_limbs[:, 49] *= local_scales[10]
    scaled_limbs[:, 50] *= local_scales[11]

    
    delta = scaled_limbs - limbs

    scaled_keypoints = keypoints.copy()
    
    assert len(limb_dependents) == limbs.shape[1], print(len(limb_dependents), limbs.shape[1])
    for i in range(len(limb_dependents)):
        scaled_keypoints[:, limb_dependents[i]] += delta[:, i:i+1, :]
    
    return scaled_keypoints.reshape(T, JD)
    
