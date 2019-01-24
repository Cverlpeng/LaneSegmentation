# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 16:02:36 2018

@author: user
"""

import json
from sklearn.linear_model import LinearRegression
import pdb
import numpy as np
import matplotlib.pyplot as plt



def fit_lane_vis(pred_x,pred_y,gt_x,gt_y):
    '''
    画出原始坐标点和预测坐标点的曲线    
    
    '''
    plt.plot(pred_x, pred_y, '*',label='original values')
    plt.plot(gt_x, gt_y, 'r',label='polyfit values')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.legend(loc=4)#指定legend的位置,读者可以自己help它的用法
    plt.title('polyfitting')
    plt.show()
    
def scale(gt_w,gt_h,pre_w,pre_h,pred_x,pred_y):
    '''
    将网络输出256*512的特征图坐标，放大到780*1280    
    '''
    new_x = np.array(pred_x)*(gt_w/pre_w)
    new_y = np.array(pred_y)*(gt_h/pre_h)
    return new_x,new_y

def fit_lane(pred_x,pred_y,gt_y):
    '''
    使用线性回归拟合车道线    
    '''
    lr = LinearRegression()
    lr.fit(pred_y.reshape(-1,1), pred_x)
    array_gt_y = np.array(gt_y).reshape(-1,1)
    gt_x = lr.predict(array_gt_y)
    #fit_lane_vis(pred_x,pred_y,gt_x,array_gt_y)
    #曲线拟合后预测的x坐标，小于0的都赋值为-2
    ret_x = []
    for index,fit_x in enumerate(gt_x):
        if index<10 or fit_x<0:
            fit_x = -2
            ret_x.append(fit_x)
        elif fit_x>1280:
            fit_x = -2
            ret_x.append(int(fit_x))
        else:
            ret_x.append(int(fit_x))
        
    return ret_x
    
def solv_equations(coeff,gt_y):
#==============================================================================
# # 解2次方程
#      k,m,n= coeff[0],coeff[1],coeff[2]
#      gt_x = []
#      for y in gt_y:
#          n -= y
#          D = m**2-4*k*n
#          if D<0:
#              print("方程无实数解")
#          elif D==0:
#              gt_x.append(-1*m/(2*k))
#              print("方程有两个相同的实根")
#          else:
#              x1 = -1*m/(2*k)+D**(1/2)/(2*k)
#              x2 = -1*m/(2*k)-D**(1/2)/(2*k)
#              if 0<x1<1280:
#                  gt_x.append(x1)
#              elif 0<x2<1280:
#                  gt_x.append(x2)
#              elif x1>1280 or x2>1280:
#                  if x1>x2:
#                      gt_x.append(x2)
#                  else:
#                      gt_x.append(x1)
#              elif x1<0 or x2<0:
#                  if x1>x2:
#                      gt_x.append(x1)
#                  else:
#                      gt_x.append(x2)
#          n = coeff[2]
#              
# 
#==============================================================================
#==============================================================================

    a,b= coeff[0],coeff[1]
    gt_x = []
    for y in gt_y:
        y -= b
        x = y/a
        gt_x.append(x)

    return gt_x
    
    
def solv_linear_equa(pred_x,pred_y,gt_y):
    '''
    图像中的车道线坐标满足以下关系：
    k/(y-h) + b(y-h) + c = x
    
    h为车道线消失点y坐标，k,b,c为所求系数。    

    '''
    pred_x = np.array(pred_x)
    pred_y = np.array(pred_y)
    gt_y = np.array(gt_y)
    vanish_y = 202
    lines = []

    for y in pred_y:
        per_line = [1/(y-vanish_y),(y-vanish_y),1]
        lines.append(per_line)
    lines = np.array(lines)
    coeff = np.linalg.lstsq(lines,pred_x)
    
    new_lines = []
    for y in gt_y:
        per_line = [1/(y-vanish_y),(y-vanish_y),1]
        new_lines.append(per_line)
    new_lines = np.array(new_lines)
    gt_x = np.dot(new_lines,coeff[0])
    #fit_lane_vis(pred_x,pred_y,gt_x,gt_y)
    ret_x = []
    for index,fit_x in enumerate(gt_x):
        if fit_x<0:
            fit_x = -2
            ret_x.append(fit_x)
        elif fit_x>1280:
            fit_x = -2
            ret_x.append(int(fit_x))
        else:
            ret_x.append(int(fit_x))
    return ret_x
    

    
    
    
        
    
def lane_fit(pred_x,pred_y,gt_y):
    """
    车道线多项式拟合
    :param lane_pts:
    :return:
    """
    if not isinstance(gt_y, np.ndarray):
        gt_y = np.array(gt_y, np.float32)
    f1 = np.polyfit(pred_y, pred_x, 3)
    p1 = np.poly1d(f1)
    gt_x = p1(gt_y)
    #gt_x = solv_equations(f1,gt_y)
    #fit_lane_vis(pred_x,pred_y,gt_x,gt_y)
    ret_x = []
    for index,fit_x in enumerate(gt_x):
        if fit_x<0:
            fit_x = -2
            ret_x.append(fit_x)
        elif fit_x>1280:
            fit_x = -2
            ret_x.append(int(fit_x))
        else:
            ret_x.append(int(fit_x))
    return ret_x


def load_json(path):
    '''打开json文件，按行取数据'''
    per_line = [json.loads(line) for line in open(path,"r").readlines()]
    
    return per_line

def get_h_samples(gt_path):
    '''从gt_json文件中拿到y坐标，一遍后面预测拟合后的x坐标'''
    per_line = load_json(gt_path)
    for line in per_line:
        h_samples = line["h_samples"]
        break
    
    return h_samples
    
def writeJsonFile(xy_coord,path):
    '''将每张Test_image的车道线信息字典，按行存入json文件中'''
    xy_coord= json.dumps(xy_coord)
    with open(path+".json","a+") as f:
        f.writelines(xy_coord+"\n")



    
def get_xy_coord(pred_xy_coord,gt_path,gt_w,gt_h,pred_w,pred_h,save_pred_file_path):
    '''
    1.先将pred_json文件中的X,Y坐标逐车道取出。
    2.在还原到780*1280大小
    3.拟合车道线，并预测指定Y坐标的X坐标值
    4.写入json文件
    5.返回处理test_image路径    
    '''
    return_file_path = []
    for key in pred_xy_coord:
        lane_x = key["lane_x"]
        lane_y = key["lane_y"]
        gt_y = get_h_samples(gt_path)#拿到统一的y坐标
        file_path = key["raw_file"]
        return_file_path.append(file_path)#将每张测试图片路径记录到list中
        pred_dict = {"raw_file":file_path}
        gt_x = []    
        #按照车道线进行缩放和拟合
        for num in range(len(lane_x)):
            new_x,new_y = scale(gt_w,gt_h,pred_w,pred_h,lane_x[num],lane_y[num])
            gt_x.append(lane_fit(new_x,new_y,gt_y))
            #gt_x.append(fit_lane(new_x,new_y,gt_y))
            #gt_x.append(solv_linear_equa(new_x,new_y,gt_y))
        pred_dict["lanes"] = gt_x 
        pred_dict["h_samples"] = gt_y  
        writeJsonFile(pred_dict,save_pred_file_path+'pred')#写入json文件
    return return_file_path

def file_path_split(pred_image_path):
    new_path = []
    for image_path in pred_image_path:
        image_name = image_path.split('/')[-4:]
        new_path.append(image_name[0]+"/"+image_name[1]+"/"+image_name[2]+"/"+image_name[3])
    return new_path
 
def get_gtJsonFile(pred_image_path,gt_path,save_pred_file_path):  
    new_path = file_path_split(pred_image_path)
    per_line = load_json(gt_path)
    
    for per_image in new_path:
        for per_file in per_line:
            if per_file["raw_file"] == per_image:
                writeJsonFile(per_file,save_pred_file_path+'gt')

                
    
    


def main():
    gt_w,gt_h,pred_w,pred_h = 1280,720,512,256
    save_pred_file_path = "/home/user/LaneSegmention/lanenet-lane/data/tusimple_test_image/ret/"
    test_jsonFilePath = "/home/user/LaneSegmention/lanenet-lane/data/tusimple_test_image/ret/test_image_xycoord.json"
    gt_jsonFilePath = "/home/user/dataset/tusimple/test_set/test_labels.json"
    per_line = load_json(test_jsonFilePath)
    pred_image_path = get_xy_coord(per_line,gt_jsonFilePath,gt_w,gt_h,pred_w,pred_h,save_pred_file_path)
    get_gtJsonFile(pred_image_path,gt_jsonFilePath,save_pred_file_path)
    
main()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    