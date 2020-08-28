#!/usr/bin/python
# -*- coding: UTF-8 -*-
from random import randint
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import skimage

def get_noise_img(imagesize):
    img = np.zeros((imagesize,imagesize),dtype=np.uint8)
    for i in range(randint(3,8)):
        x=random.randint(0,imagesize)
        y=random.randint(0,imagesize)
        cv2.circle(img,(x,y),randint(1,5),(255,255,255),-1)
    return img
def get_noise_img_array(imagesize,img_num):
    img = np.zeros((imagesize,imagesize,img_num),dtype=np.uint8)
    for i in range(img_num):
        img[:,:,i] = get_noise_img(imagesize)
    return img
def get_target_img():
    image_range = 1852/2.0
    imagesize = [640,640]
    img_num = 5
    image = np.zeros((imagesize[0],imagesize[1],img_num),dtype=np.uint8)
    angle = random.randint(0,3600)/10.0
    volocity = random.randint(0,350)/10.0
    v_x = np.cos(angle*np.pi/180)*volocity
    v_y = np.sin(angle*np.pi/180)*volocity
    # print(angle, v_x, v_y)
    x=random.randint(0,640)
    y=random.randint(0,640)
    radius = random.randint(2,8)
    for num in range(5):
        img = np.zeros((imagesize[0],imagesize[1]),dtype=np.uint8)
        img_x = np.int(x+v_x*num)
        img_y = np.int(y+v_y*num)
        if max(img_x,img_y)<imagesize[0] and min(img_x,img_y)>=0:
            cv2.circle(img,(img_x,img_y),randint(1,5),(255,255,255),-1)
        image[:,:,num] = img
    return image
def get_target_img_array(imagesize,img_num):
    image = np.zeros((imagesize,imagesize,img_num),dtype=np.uint8)
    for i in range(random.randint(2,6)):
        image = image + get_target_img()
        # print(i)
    return image
def img2point(img):
    nz =np.transpose(np.nonzero(img))
    if nz.shape[1]==2:
        n = np.zeros((nz.shape[0],3))
        n[:,0:2] = nz
        return list(n)
    else:
        return list(nz)

def normalize_pointset(pointset,num = 1024):
    pointset = list(pointset)
    if len(pointset)<num:
        for i in range(np.int16(num/len(pointset))+1):
            pointset += pointset
    pointset = pointset[:num]
    return np.array(pointset)

def get_train_data(data_num,img_num = 5):
    data_set = [np.zeros((data_num,1024,3))]*img_num
    label_set = []
    for i in range(data_num):
        train_image = get_target_img_array(640,img_num)
        noise_image = get_noise_img_array(640,img_num)
        data = []
        for j in range(img_num):
            img = train_image[:,:,j]
            noise = noise_image[:,:,j]
            img_nz = img2point(img)
            noise_nz = img2point(noise)
            if j ==0:
                label = normalize_pointset([1]*len(img_nz)+[0]*len(noise_nz))
            nz = normalize_pointset(img_nz+noise_nz)
            data.append(nz)
            data_set[j][i,:,:] = data[j]

        label_set.append(label)
    for j in range(img_num):
           data_set[j] = np.array(data_set[j])
    label_set = np.array(label_set)
    return data_set,label_set

# data_set,label_set = get_train_data(10)
# print(np.array(data_set[0]).shape,label_set.shape)