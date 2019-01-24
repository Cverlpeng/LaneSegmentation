# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 13:44:22 2019

@author: lpeng
"""
import cv2
import numpy as np
import argparse
import os
import sys
import heapq
import pdb

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_dir', type=str, help='The image dir path')
    parser.add_argument('--freespace_dir', type=str, help='The freespace image dir path')
    parser.add_argument('--save_dir', type=str, help='The save result path')
    parser.add_argument('--lane_dir', type=str, help='The freespace image dir path')
    parser.add_argument('--vehicle_dir', type=str, help='The vehicle image dir path')
    return parser.parse_args()

def get_img_list(img_dir):
    
    img_name = []
    for file_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir,file_name)
        if os.path.splitext(img_path)[1] == '.bmp':
            img_name.append(file_name.split('.')[0])
    return img_name
    

def find_largest_contour(freespace):
        
    # get binary image
    ret, thresh = cv2.threshold(freespace,127,255,0)
    
    #find all contours
    image ,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    mask_contour = np.zeros(freespace.shape)
    
    #find largest index of contours
    contours_size = [c.size for c in contours]
    max_index = contours_size.index(max(contours_size))
    #find max two index of contours
    #max_index = map(contours_size.index, heapq.nlargest(2, contours_size))
           
    cv2.drawContours(freespace,contours,max_index,(255,0,255),cv2.FILLED)
    cv2.drawContours(mask_contour,contours,max_index,(255,0,0),1)    
    
    return freespace, mask_contour
            
def main(img_root_dir, freespace_dir, lane_dir,vehicle_dir, save_root_dir):
    
    
 
    dir_list = ['front_28','front_425','front_120','rear','side_right_front',
                           'side_right_rear','side_left_front','side_left_rear']
    #dir_list = ['rear']
    
    for per_dir in dir_list:
        
        img_dir = img_root_dir+per_dir+'/'        
        img_name = get_img_list(img_dir)
        save_dir = save_root_dir+'result/'

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_dir = save_dir+per_dir+'/'
        if not os.path.exists(save_dir):    
                os.mkdir(save_dir)
                
        for index,name in enumerate(img_name):
            
            img_path = img_dir + name + '.bmp'
            freespace_path = freespace_dir+per_dir+'/'+name+'._prob.png'
            solid_lane_path = lane_dir+per_dir+'/'+name+ '_1_avg.png'
            dash_lane_path = lane_dir+per_dir+'/'+name+ '_2_avg.png' 
            douSolid_lane_path = lane_dir+per_dir+'/'+name+ '_3_avg.png'
            vehicle_path = vehicle_dir+per_dir+'/'+name+'.txt'
            
            img = cv2.imread(img_path)
    
            freespace = cv2.imread(freespace_path, cv2.IMREAD_GRAYSCALE)
            solid_lane = cv2.imread(solid_lane_path, cv2.IMREAD_GRAYSCALE)
            dash_lane = cv2.imread(dash_lane_path, cv2.IMREAD_GRAYSCALE)
            douSolid_lane = cv2.imread(douSolid_lane_path, cv2.IMREAD_GRAYSCALE)
            
            
            if  per_dir == 'front_120':
                
                resized_freespace = cv2.resize(freespace, (img.shape[1], img.shape[0]), cv2.INTER_NEAREST)
                fill_free , mask_contour = find_largest_contour(resized_freespace[0:520,0:img.shape[1]])
                resized_solid_lane = cv2.resize(solid_lane, (1020, 528), cv2.INTER_NEAREST)
                resized_dash_lane = cv2.resize(dash_lane, (1020, 528), cv2.INTER_NEAREST)
                resized_douSolid_lane = cv2.resize(douSolid_lane, (1020, 528), cv2.INTER_NEAREST)
                mask_free = np.zeros(img.shape)
                mask_roi = np.zeros([528,1020,3])
                mask_orig = np.zeros(img.shape)
                
                #Remove the driving area from false detection
                mask_free[0:520,0:img.shape[1],1] = fill_free[0:520,0:img.shape[1]]*0.2
                mask_free[0:520,0:img.shape[1],2] = mask_contour[0:520,0:img.shape[1]]
                
                
                mask_roi[:,:,0] = resized_solid_lane*0.5
                mask_roi[:,:,2] = resized_dash_lane*0.6
                mask_roi[:,:,0] += resized_douSolid_lane*0.3
                
                mask_orig[0:528, 300:1320] = mask_roi
               
                masked_img = mask_orig + img + mask_free[0:]
                
                print('Visualizing image: {} from camera: {}'.format(name, per_dir))

            else:      
                fill_free , mask_contour = find_largest_contour(freespace)
                mask = np.zeros(img.shape)
                #Rear camera, only need to visualize lane lines and BBOX
                if not per_dir == 'rear':
                    resized_freespace = cv2.resize(fill_free, (img.shape[1], img.shape[0]), cv2.INTER_NEAREST)
                    resized_mask_contour = cv2.resize(mask_contour, (img.shape[1], img.shape[0]), cv2.INTER_NEAREST)
                    mask[:,:,1] = resized_freespace*0.1
                    mask[:,:,2] = resized_mask_contour
                
                resized_solid_lane = cv2.resize(solid_lane, (img.shape[1], img.shape[0]), cv2.INTER_NEAREST)
                resized_dash_lane = cv2.resize(dash_lane, (img.shape[1], img.shape[0]), cv2.INTER_NEAREST)
                resized_douSolid_lane = cv2.resize(douSolid_lane, (img.shape[1], img.shape[0]), cv2.INTER_NEAREST)
                
                
                mask[:,:,0] = resized_solid_lane*0.5
                mask[:,:,2] += resized_dash_lane*0.6
                mask[:,:,2] += resized_douSolid_lane*0.3
                
                
                masked_img = img + mask
                print('Visualizing image: {} from camera: {}'.format(name, per_dir))
                
            
            
            #visualize vehicle bbox
            data = open(vehicle_path, 'r')
            #content = data.readline()
            lines = [l.strip() for l in data.readlines()]
    
            for index,line in enumerate(lines):
                #skip blank lines and the first line
                if index == 0 or line == '':  
                    continue
                else:
                    xmin = int(line.split(" ")[3])
                    ymin = int(line.split(" ")[4])
                    w = int(line.split(" ")[5])
                    h = int(line.split(" ")[6])
                    xmax = xmin + w
                    ymax = ymin + h
                    cv2.rectangle(masked_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,255,255), 2)
                    
            
            masked_img_path = save_dir+'/'+name+'_vis.png'
            if not cv2.imwrite(masked_img_path, masked_img):
                print("failed to write {}".format(masked_img_path))
                sys.exit(1)
        
        print('Camera: {} visualization completed'.format(per_dir))
            
    


if __name__ == '__main__':
    # init args
    args = init_args()
    
    main(args.img_dir,args.freespace_dir,args.lane_dir, args.vehicle_dir, args.save_dir)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    