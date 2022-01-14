from tqdm import tqdm
from scipy.optimize import curve_fit

import os
from glob import glob
import numpy as np
import json

import os
import imageio
import cv2 as cv
import random
import warnings

from functools import partial
from multiprocessing import Pool


def cvt25(our_poses):
    seq_len = our_poses.shape[0]
    gt_poses = np.zeros([seq_len, 135*2])
    stats = (100, np.array([694.51276396, 319.63906205]))
    length = min(our_poses.shape[0], gt_poses.shape[0])
    our_poses = our_poses[:length, :]
    gt_poses = gt_poses[:length, :]
    gt_poses = gt_poses.reshape(gt_poses.shape[0], -1, 2)
    our_poses = our_poses.reshape(our_poses.shape[0], -1, 2)
    
    if our_poses.shape[1] == 49:
        gt_poses[:, [1, 2, 3, 4, 5, 6, 7], :] = our_poses[:, 0:7, :]
        gt_poses[:, 25:25+21+21, :] = our_poses[:, 7:, :]
    else:
        gt_poses[:, [0,1, 2, 3, 4, 5, 6, 7,15,16,17,18], :] = our_poses[:, 0:12, :]
        gt_poses[:, 25:25+21+21, :] = our_poses[:, 12:, :]
    
    gt_poses[:, :, 0] = gt_poses[:, :, 0] * stats[0] + stats[1][0]
    gt_poses[:, :, 1] = gt_poses[:, :, 1] * stats[0] + stats[1][1]
    
    dummy_prob = np.ones([gt_poses.shape[0], gt_poses.shape[1], 1])
    
    gt_poses = np.concatenate([gt_poses, dummy_prob], axis = -1)
    
    return gt_poses


def smooth(res):
    smoothed = []
    for i in range(len(res)):
        finnal_res = res[i]
        window = [finnal_res[7], finnal_res[8], finnal_res[9], finnal_res[10], finnal_res[11], finnal_res[12]]
        w_size = 7
        for i in range(10, len(finnal_res)-3):
            window.append(finnal_res[i+3])
            if len(window)>w_size:
                window = window[1:]
            if (i % 25) in [22, 23, 24, 0, 1, 2, 3]:
                finnal_res[i] = np.mean(window, axis = 0)
                
        smoothed.append(finnal_res)
            
    return np.array(smoothed)

def make_img(keypoint_npy, edge_lists, h, w, nc, basic_points_only, remove_face_labels, random_drop_prob):
    person_keypoints = np.asarray(keypoint_npy).reshape(-1, 135, 3)[0]
    # Separate out the keypoint array to different parts.
    pose_pts = person_keypoints[:25]
    face_pts = person_keypoints[-68: ]
    hand_pts_l = person_keypoints[(25): (25 + 21)]
    hand_pts_r = person_keypoints[25+21:25+21+21]
    all_pts = [pose_pts, face_pts, hand_pts_l, hand_pts_r]
    all_pts = [extract_valid_keypoints(pts, edge_lists)
                for pts in all_pts]

    pose_img = connect_pose_keypoints(all_pts, edge_lists,
                                        (h, w, nc),
                                        basic_points_only,
                                        remove_face_labels,
                                        random_drop_prob)
    pose_img = pose_img.astype(np.uint8)
    
    return pose_img

def draw_openpose_npy(resize_h, resize_w, crop_h, crop_w, original_h,
                      original_w, is_flipped, cfgdata, keypoints_npy, remove_face_labels=False):
    r"""Connect the OpenPose keypoints to edges and draw the pose map.

    Args:
        resize_h (int): Height the input image was resized to.
        resize_w (int): Width the input image was resized to.
        crop_h (int): Height the input image was cropped.
        crop_w (int): Width the input image was cropped.
        original_h (int): Original height of the input image.
        original_w (int): Original width of the input image.
        is_flipped (bool): Is the input image flipped.
        cfgdata (obj): Data configuration.
        keypoints_npy (dict): OpenPose keypoint dict.

    Returns:
        (list of HxWxC numpy array): Drawn label map.
    """
    
    basic_points_only = False
    

    
    random_drop_prob = 0

    h, w, nc = crop_h, crop_w, 3

    
    edge_lists = define_edge_lists(basic_points_only)

    outputs = []
    draw_img = partial(make_img, edge_lists=edge_lists, h=h, w=w, nc=nc, basic_points_only=basic_points_only, remove_face_labels=remove_face_labels, random_drop_prob=random_drop_prob)

    # pbar = tqdm(total = len(keypoints_npy), ncols=50)
    num_process = int(os.cpu_count()/2)
    with tqdm(total = len(keypoints_npy), ncols=50) as pbar:
        with Pool(processes=num_process) as pool:
            for v in pool.map(draw_img, keypoints_npy):
                assert v.shape[0] == h and v.shape[1] == w, v.shape
                outputs.append(v)
                pbar.update()
    # for keypoint_npy in tqdm(keypoints_npy, ncols=50):
    #     person_keypoints = np.asarray(keypoint_npy).reshape(-1, 135, 3)[0]
        
    #     pose_pts = person_keypoints[:25]
    #     face_pts = person_keypoints[-68:]
    #     hand_pts_l = person_keypoints[(25): (25 + 21)]
    #     hand_pts_r = person_keypoints[25+21:25+21+21]
    #     all_pts = [pose_pts, face_pts, hand_pts_l, hand_pts_r]
        
    #     all_pts = [extract_valid_keypoints(pts, edge_lists)
    #                for pts in all_pts]

        
    #     pose_img = connect_pose_keypoints(all_pts, edge_lists,
    #                                       (h, w, nc),
    #                                       basic_points_only,
    #                                       remove_face_labels,
    #                                       random_drop_prob)





    #     pose_img = pose_img.astype(np.uint8)
    #     outputs.append(pose_img)
    return outputs


def extract_valid_keypoints(pts, edge_lists):
    r"""Use only the valid keypoints by looking at the detection confidences.
    If the confidences for all keypoints in an edge are above threshold,
    keep the keypoints. Otherwise, their coordinates will be set to zero.

    Args:
        pts (Px3 numpy array): Keypoint xy coordinates + confidence.
        edge_lists (nested list of ints):  List of keypoint indices for edges.

    Returns:
        (Px2 numpy array): Output keypoints.
    """
    pose_edge_list, _, hand_edge_list, _, face_list = edge_lists
    p = pts.shape[0]
    thre = 0.1 if p == 68 else 0.01
    output = np.zeros((p, 2))

    if p == 68:  
        for edge_list in face_list:
            for edge in edge_list:
                if (pts[edge, 2] > thre).all():
                    output[edge, :] = pts[edge, :2]
    elif p == 21:  
        for edge in hand_edge_list:
            if (pts[edge, 2] > thre).all():
                output[edge, :] = pts[edge, :2]
    else:  
        valid = (pts[:, 2] > thre)
        output[valid, :] = pts[valid, :2]

    return output


def linear(x, a, b):
    r"""Linear fitting function."""
    return a * x + b


def func(x, a, b, c):
    r"""Quadratic fitting function."""
    return a * x**2 + b * x + c


def interp_points(x, y):
    r"""Given the start and end points, interpolate to get a curve/line.

    Args:
        x (1D array): x coordinates of the points to interpolate.
        y (1D array): y coordinates of the points to interpolate.

    Returns:
        (dict):
          - curve_x (1D array): x coordinates of the interpolated points.
          - curve_y (1D array): y coordinates of the interpolated points.
    """
    if abs(x[:-1] - x[1:]).max() < abs(y[:-1] - y[1:]).max():
        curve_y, curve_x = interp_points(y, x)
        if curve_y is None:
            return None, None
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if len(x) < 3:
                    popt, _ = curve_fit(linear, x, y)
                else:
                    popt, _ = curve_fit(func, x, y)
                    if abs(popt[0]) > 1:
                        return None, None
            except Exception:
                return None, None
        if x[0] > x[-1]:
            x = list(reversed(x))
            y = list(reversed(y))
        curve_x = np.linspace(x[0], x[-1], int(np.round(x[-1]-x[0])))
        if len(x) < 3:
            curve_y = linear(curve_x, *popt)
        else:
            curve_y = func(curve_x, *popt)
    return curve_x.astype(int), curve_y.astype(int)


def connect_pose_keypoints(pts, edge_lists, size, basic_points_only,
                           remove_face_labels, random_drop_prob):
    r"""Draw edges by connecting the keypoints onto the label map.

    Args:
        pts (Px3 numpy array): Keypoint xy coordinates + confidence.
        edge_lists (nested list of ints):  List of keypoint indices for edges.
        size (tuple of int): Output size.
        basic_points_only (bool): Whether to use only the basic keypoints.
        remove_face_labels (bool): Whether to remove face labels.
        random_drop_prob (float): Probability to randomly drop keypoints.

    Returns:
        (HxWxC numpy array): Output label map.
    """
    pose_pts, face_pts, hand_pts_l, hand_pts_r = pts
    h, w, c = size
    body_edges = np.ones((h, w, c), np.uint8)
    
    
    use_one_hot = False
    c = 3

    pose_edge_list, pose_color_list, hand_edge_list, hand_color_list, \
        face_list = edge_lists

    
    h = int(pose_pts[:, 1].max() - pose_pts[:, 1].min())
    bw = max(1, h // 150)  
    body_edges = draw_edges(body_edges, pose_pts, [pose_edge_list], bw,
                            use_one_hot, random_drop_prob,
                            colors=pose_color_list, draw_end_points=True)

    if not basic_points_only:
        
        bw = max(2, h // 450)
        for i, hand_pts in enumerate([hand_pts_l, hand_pts_r]):
            if use_one_hot:
                k = 24 + i
                body_edges[:, :, k] = draw_edges(body_edges[:, :, k], hand_pts,
                                                 [hand_edge_list],
                                                 bw, False, random_drop_prob,
                                                 colors=[255] * len(hand_pts))
            else:
                body_edges = draw_edges(body_edges, hand_pts, [hand_edge_list],
                                        bw, False, random_drop_prob,
                                        colors=hand_color_list)
        
        if not remove_face_labels:
            if use_one_hot:
                k = 26
                body_edges[:, :, k] = draw_edges(body_edges[:, :, k], face_pts,
                                                 face_list, bw, False,
                                                 random_drop_prob)
            else:
                body_edges = draw_edges(body_edges, face_pts, face_list, bw,
                                        False, random_drop_prob)
    return body_edges


def draw_edges(canvas, keypoints, edges_list, bw, use_one_hot,
               random_drop_prob=0, edge_len=2, colors=None,
               draw_end_points=False):
    r"""Draw all the edges in the edge list on the canvas.

    Args:
        canvas (HxWxK numpy array): Canvas to draw.
        keypoints (Px2 numpy array): Keypoints.
        edge_list (nested list of ints):  List of keypoint indices for edges.
        bw (int): Stroke width.
        use_one_hot (bool): Use one-hot encoding or not.
        random_drop_prob (float): Probability to randomly drop keypoints.
        edge_len (int): Number of keypoints in an edge.
        colors (tuple of int): Color to draw.
        draw_end_points (bool): Whether to draw end points for edges.

    Returns:
        (HxWxK numpy array): Output.
    """
    k = 0
    for edge_list in edges_list:
        for i, edge in enumerate(edge_list):
            for j in range(0, max(1, len(edge) - 1), edge_len - 1):
                if random.random() > random_drop_prob:
                    sub_edge = edge[j:j + edge_len]
                    x, y = keypoints[sub_edge, 0], keypoints[sub_edge, 1]
                    if 0 not in x:  
                        curve_x, curve_y = interp_points(x, y)
                        if use_one_hot:
                            
                            
                            draw_edge(canvas[:, :, k], curve_x, curve_y,
                                      bw=bw, color=255,
                                      draw_end_points=draw_end_points)
                        else:

                            color = colors[i] if colors is not None \
                                else (255, 255, 255)
                            draw_edge(canvas, curve_x, curve_y,
                                      bw=bw, color=color,
                                      draw_end_points=draw_end_points)
                k += 1
    return canvas


def define_edge_lists(basic_points_only):
    r"""Define the list of keypoints that should be connected to form the edges.

    Args:
        basic_points_only (bool): Whether to use only the basic keypoints.
    """
    
    pose_edge_list = [
        [17, 15], [15, 0], [0, 16], [16, 18],  
        [0, 1],                        
        [1, 2], [2, 3], [3, 4],                
        [1, 5], [5, 6], [6, 7],                
        
        
    ]
    pose_color_list = [
        [153, 0, 153], [153, 0, 102], [102, 0, 153], [51, 0, 153],
        [153, 0, 51],
        [153, 51, 0], [153, 102, 0], [153, 153, 0],
        [102, 153, 0], [51, 153, 0], [0, 153, 0],
        
        
        
        
        
        
    ]

    if not basic_points_only:
        pose_edge_list += [
            
            
        ]
        pose_color_list += [
            
            
        ]

    
    hand_edge_list = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20]
    ]
    hand_color_list = [
        [204, 0, 0], [163, 204, 0], [0, 204, 82], [0, 82, 204], [163, 0, 204]
    ]

    
    face_list = [
        
        
        
        
        
        
        
    ]

    return pose_edge_list, pose_color_list, hand_edge_list, hand_color_list, \
        face_list


def draw_edge(im, x, y, bw=1, color=(255, 255, 255), draw_end_points=False):
    r"""Set colors given a list of x and y coordinates for the edge.

    Args:
        im (HxWxC numpy array): Canvas to draw.
        x (1D numpy array): x coordinates of the edge.
        y (1D numpy array): y coordinates of the edge.
        bw (int): Width of the stroke.
        color (list or tuple of int): Color to draw.
        draw_end_points (bool): Whether to draw end points of the edge.
    """
    if x is not None and x.size:
        h, w = im.shape[0], im.shape[1]
        
        for i in range(-bw, bw):
            for j in range(-bw, bw):
                yy = np.maximum(0, np.minimum(h - 1, y + i))
                xx = np.maximum(0, np.minimum(w - 1, x + j))
                set_color(im, yy, xx, color)

        
        if draw_end_points:
            for i in range(-bw * 2, bw * 2):
                for j in range(-bw * 2, bw * 2):
                    if (i ** 2) + (j ** 2) < (4 * bw ** 2):
                        yy = np.maximum(0, np.minimum(h - 1, np.array(
                            [y[0], y[-1]]) + i))
                        xx = np.maximum(0, np.minimum(w - 1, np.array(
                            [x[0], x[-1]]) + j))
                        set_color(im, yy, xx, color)


def set_color(im, yy, xx, color):
    r"""Set pixels of the image to the given color.

    Args:
        im (HxWxC numpy array): Canvas to draw.
        xx (1D numpy array): x coordinates of the pixels.
        yy (1D numpy array): y coordinates of the pixels.
        color (list or tuple of int): Color to draw.
    """
    if type(color) != list and type(color) != tuple:
        color = [color] * 3
    if len(im.shape) == 3 and im.shape[2] == 3:
        if (im[yy, xx] == 0).all():
            im[yy, xx, 0], im[yy, xx, 1], im[yy, xx, 2] = \
                color[0], color[1], color[2]
        else:
            for c in range(3):
                im[yy, xx, c] = ((im[yy, xx, c].astype(float)
                                  + color[c]) / 2).astype(np.uint8)
    else:
        im[yy, xx] = color[0]


def save_images(imgs, save_root):
    os.makedirs(save_root, exist_ok=True)

    for i in range(len(imgs)):
        save_name = os.path.join(save_root, '%06d.jpg' % (i))
        h, w = imgs[i].shape[0], imgs[i].shape[1]
        imageio.imwrite(save_name, cv.resize(imgs[i], (512, int(h*512 / w))))