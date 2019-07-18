#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 19:20:55 2019

@author: vaeahc
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def cal_Dark_Channel(im, width = 15):
    #参数按照论文中设置,无导向滤波和soft matting
    im_dark = np.min(im, axis = 2)
    border = int((width - 1) / 2)
    im_dark_1 = cv2.copyMakeBorder(im_dark, border, border, border, border, cv2.BORDER_DEFAULT)
    res = np.zeros(np.shape(im_dark))
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            
            res[i][j] = np.min(im_dark_1[i: i + width, j: j + width])
            
    return res
#计算A参数, im为暗通道图像, img为原图
def cal_Light_A(im, img):
    
    s_dict = {}
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            s_dict[(i, j)] = im[i][j]
        
    s_dict = sorted(s_dict.items(), key = lambda x: x[1])   
    
    A = np.zeros((3, ))
    num = int(im.shape[0] * im.shape[1] * 0.001)
    
    for i in range(len(s_dict) - 1, len(s_dict) - num - 1, -1):
        
        X_Y = s_dict[i][0]
        A = np.maximum(A, img[X_Y[0], X_Y[1], :])

    return A
        
def harz_Rec(A, img, t, t0 = 0.1):
    
    img_o = np.zeros(np.shape(img))
    
    img_o[:, :, 0] = (img[:, :, 0] - A[0]) / (np.maximum(t, t0)) + A[0]
    img_o[:, :, 1] = (img[:, :, 1] - A[1]) / (np.maximum(t, t0)) + A[1]
    img_o[:, :, 2] = (img[:, :, 2] - A[2]) / (np.maximum(t, t0)) + A[2]

    return img_o

def cal_trans(A, img, w = 0.95):
    
    dark = cal_Dark_Channel(img / A)
    t = np.maximum(1 - w * dark, 0)
    
    return t

#guided image filtering
def Guided_filtering(t, img_gray, width, sigma = 0.0001):
    
    mean_I = np.zeros(np.shape(img_gray))
    cv2.boxFilter(img_gray, -1, (width, width), mean_I, (-1, -1), True, cv2.BORDER_DEFAULT)
    mean_t = np.zeros(np.shape(t))
    cv2.boxFilter(t, -1, (width, width), mean_t, (-1, -1), True, cv2.BORDER_DEFAULT)
    corr_I = np.zeros(np.shape(img_gray))
    cv2.boxFilter(img_gray * img_gray, -1, (width, width), corr_I, (-1, -1), True, cv2.BORDER_DEFAULT)
    corr_IT = np.zeros(np.shape(t))
    cv2.boxFilter(img_gray * t, -1, (width, width), corr_IT, (-1, -1), True, cv2.BORDER_DEFAULT)
    
    var_I = corr_I - mean_I * mean_I
    cov_IT = corr_IT - mean_I * mean_t

    a = cov_IT / (var_I + sigma)    
    b = mean_t - a * mean_I
    
    mean_a = np.zeros(np.shape(a))
    mean_b = np.zeros(np.shape(b))
    cv2.boxFilter(a, -1, (width, width), mean_a, (-1, -1), True, cv2.BORDER_DEFAULT)
    cv2.boxFilter(b, -1, (width, width), mean_b, (-1, -1), True, cv2.BORDER_DEFAULT)
    
    return mean_a * img_gray + mean_b

if __name__ == '__main__':
    
    img = cv2.imread('IMG_2170.BMP') / 255
    img_gray = cv2.imread('IMG_2170.BMP', 0) / 255
    plt.imshow(img[:, :, ::-1])
    plt.show()
    im_dark = cal_Dark_Channel(img)
    plt.imshow(im_dark, 'gray')
    plt.show()
    A = cal_Light_A(im_dark, img)
    trans = cal_trans(A, img)
    plt.imshow(trans, 'gray')
    plt.show()
    trans = Guided_filtering(trans, img_gray, 41)
    plt.imshow(trans, 'gray')
    plt.show()
    result = harz_Rec(A, img, trans)    
    plt.imshow(result, 'gray')
    plt.show()
    