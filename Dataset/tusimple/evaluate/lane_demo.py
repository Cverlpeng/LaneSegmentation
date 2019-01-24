# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 13:57:59 2018

@author: user
"""

import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from lane import LaneEval
import pdb
#%matplotlib inline


def load_json(path):
    '''打开json文件，按行取数据'''
    per_line = [json.loads(line) for line in open(path,"r").readlines()]
    
    return per_line

def get_xycoord(lanes,y_samples):
    
    lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in lanes]
    return lanes_vis
    

def show_orig_image(path):
    
    img = plt.imread(path)
    plt.imshow(img)
    plt.show()
    return img
def show_gt_image(gt_lanes,y_samples,img):
    gt_lanes_vis = get_xycoord(gt_lanes,y_samples)
    img_vis = img.copy()
    
    for lane in gt_lanes_vis:
        for pt in lane:
            cv2.circle(img_vis, pt, radius=5, color=(0, 255, 0))
    
    plt.imshow(img_vis)
    plt.show()
def show_gt_pred_image(pred_lanes,gt_lanes,y_samples,img):
    
    gt_lanes_vis = get_xycoord(gt_lanes,y_samples)
    pred_lanes_vis = get_xycoord(pred_lanes,y_samples)
    img_vis = img.copy()
    
    for lane in gt_lanes_vis:
        cv2.polylines(img_vis, np.int32([lane]), isClosed=False, color=(0,255,0), thickness=5)
    for lane in pred_lanes_vis:
        cv2.polylines(img_vis, np.int32([lane]), isClosed=False, color=(0,0,255), thickness=2)
    
    plt.imshow(img_vis)
    plt.show()

def mean_eval(eval_ret_sum):
    
    acc = 0
    fp = 0
    fn = 0
    num = len(eval_ret_sum)
    #pdb.set_trace()
    for per_eval in eval_ret_sum:
        acc += per_eval[0]
        fp += per_eval[1]
        fn += per_eval[2]
    
    return acc/num, fp/num, fn/num
        

def main():
    gt_path = 'gt.json'
    pred_path = 'pred.json'
    json_gt = load_json(gt_path)
    json_pred = load_json(pred_path)

    eval_ret_sum = []
    for num in range(len(json_pred)):
        pred, gt = json_pred[num], json_gt[num]
        pred_lanes = pred['lanes']
        #run_time = pred['run_time']
        gt_lanes = gt['lanes']
        y_samples = gt['h_samples']
        raw_file = pred['raw_file']
        img = show_orig_image(raw_file)
        #show_gt_image(gt_lanes,y_samples,img)
        show_gt_pred_image(pred_lanes,gt_lanes,y_samples,img)
        np.random.shuffle(pred_lanes)
        eval_ret = LaneEval.bench(pred_lanes, gt_lanes, y_samples)
        print('图片%s的测试ACC:%f---FP:%f---FN:%f'%(raw_file,eval_ret[0],eval_ret[1],eval_ret[2]))
        eval_ret_sum.append(eval_ret)
    
    mean_ret = mean_eval(eval_ret_sum)
    print('平均ACC:%f---FP:%f---FN:%f'%(mean_ret[0],mean_ret[1],mean_ret[2]))


main()



    
    
    