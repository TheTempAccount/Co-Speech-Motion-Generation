'''
给出body graph的定义。我们肢体部分使用12个点，手掌使用21个点，总共54个点。分成三个部分：左右手、左右手臂、躯干
逐步添加graph的定义：
body graph -> 相似部位共享权重 -> 类邻接矩阵

actually this is for my FYP
'''
import numpy as np

num_joints = 54

#编号0是手掌的起始点，设置手掌和手臂的共同点的时候，应该以手腕为共同点，并且跳过手掌起始点
hand_edge_list = np.array([
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20]
    ])

left_hand_edge_list = hand_edge_list + 12 #和2,3,4关联的
right_hand_edge_list = hand_edge_list + 12 + 21 

left_arm_edge_list = np.array([
    [1, 2, 3, 4]
])

right_arm_edge_list = np.array([
    [1, 5, 6, 7]
])

body_edge_list = np.array([
    [1, 0],
    [0, 8, 10],
    [0, 9, 11]
])

hand_jump_edge_list = np.hstack([np.arange(12, 21+12).reshape(-1, 1),np.arange(12+21, 12+21+21).reshape(-1, 1)])
arm_jump_edge_list = np.hstack([np.array([2, 3, 4]).reshape(-1, 1), np.array([5, 6, 7]).reshape(-1, 1)])

def construct_edges(edge_list):
    '''
    从关联矩阵中构建邻接矩阵
    '''
    A = np.zeros([num_joints, num_joints])
    for edge in edge_list:
        for i in range(len(edge)-1):
            p1 = edge[i]
            p2 = edge[i+1]
            #跳过手掌的起始点
            if p1 == 12:
                p1 = 7
            elif p1 == 12+21:
                p1 = 4
            
            A[p1, p2] = 1
            A[p2, p1] = 1

    return A

left_hand_graph = construct_edges(left_hand_edge_list)
right_hand_graph = construct_edges(right_hand_edge_list)
left_arm_graph = construct_edges(left_arm_edge_list)
right_arm_graph = construct_edges(right_arm_edge_list)
body_graph = construct_edges(body_edge_list)

self_graph = np.eye(num_joints, num_joints)
hand_graph = left_hand_graph + right_hand_graph
arm_graph = left_arm_graph + right_arm_graph
whole_graph = ((hand_graph + arm_graph + body_graph) != 0).astype(int)
complete_graph = np.ones_like(whole_graph) - self_graph

hand_class_graph = construct_edges(hand_jump_edge_list)
arm_class_graph = construct_edges(arm_jump_edge_list)

if __name__ == "__main__":
    # print(np.stack([self_graph, hand_graph, arm_graph, body_graph, hand_class_graph, arm_class_graph]).shape)
    # print(arm_jump_edge_list)
    print(np.stack([self_graph, complete_graph]))