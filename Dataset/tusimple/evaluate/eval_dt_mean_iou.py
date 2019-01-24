#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 18:00:47 2019

@author: user
"""

'''
evaluate dt result
    1.set threshold ,if predictions map pixle value > threshold , predictions map value = 1
    2.use tf.metrics.mean_iou()

'''
#import tensorflow as tf
import numpy as np
import cv2
import argparse
import os 
from functools import reduce
import pdb

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, help='The image path or the src image save dir')
    parser.add_argument('--pred_dir', type=str, help='The model weights path')
    

    return parser.parse_args()


def computeIOU(gt,pred):
    intersection = np.logical_and(gt, pred)
    union = np.logical_or(gt, pred)
    iou = np.sum(intersection) / np.sum(union)
    return iou

    
def main(gt_dir,pred_dir):
    #pdb.set_trace()
    img_name = os.listdir(gt_dir)
    iou_list = []
    for name in img_name:
        pred_path = os.path.join(pred_dir,name)
        gt_path = os.path.join(gt_dir,name)
        pred = cv2.imread(pred_path,cv2.IMREAD_GRAYSCALE)
        ret, thresh = cv2.threshold(pred,70,255,0)
        
        gt = cv2.imread(gt_path,cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt,(pred.shape[1],pred.shape[0]),interpolation=cv2.INTER_NEAREST)
        
        iou = computeIOU(gt,thresh)
        iou_list.append(iou)
    mean_iou = reduce(lambda x,y: x+y, iou_list)/len(iou_list)
    print('------mean iou:{}--------'.format(mean_iou))
    
    

if __name__ == '__main__':
    # init args
    args = init_args()
    main(args.gt_dir,args.pred_dir)
    
