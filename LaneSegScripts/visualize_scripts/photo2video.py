# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 18:38:30 2019

@author: user
"""


import cv2
import argparse
import os
import pdb
 
def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_dir', type=str, help='The image dir path')

    return parser.parse_args()
 

    

def main(img_dir):
    fps = 24  
    
    dir_list = ['front_28','front_425','front_120','rear','side_right_front',
                           'side_right_rear','side_left_front','side_left_rear']
                           
    #dir_list = ['front_rear']                           
    for index,per_dir in enumerate(dir_list):
        
        img_path = img_dir+per_dir+'/'
        img_name = os.listdir(img_path)
        img_per = cv2.imread(img_path+img_name[0])
        size = (img_per.shape[1],img_per.shape[0])
        save_dir = './'+per_dir+'_'+'fps'+str(fps)+'.avi'
        split_name = img_name[0].split('_')
        
        videowriter = cv2.VideoWriter(save_dir,cv2.VideoWriter_fourcc('M','J','P','G'),fps,size)
        
        sorted_name = [int(name.split('_')[2]) for name in img_name ]
        sorted_name.sort()
        
        for i in sorted_name:
            save_name = split_name[0]+'_'+split_name[1]+'_'+str(i)+'_'+split_name[3]
            img = cv2.imread(img_path+save_name)
            
            videowriter.write(img)
        
        print('Camera: {} Image sequence to video completion ....'.format(per_dir))


if __name__ == '__main__':
    # init args
    args = init_args()
    #main(args.img_dir, args.freespace_dir, args.lane_dir, args.vehicle_dir)
    main(args.img_dir)
    