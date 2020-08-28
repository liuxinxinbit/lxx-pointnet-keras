# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 15:39:40 2017

@author: Gary
"""

import numpy as np
import os
import tensorflow as tf
from keras import optimizers
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Flatten, Reshape, Dropout,concatenate
from keras.layers import Convolution1D, MaxPooling1D, BatchNormalization
from keras.layers import Lambda
from keras.utils import np_utils
import h5py
from keras.utils.vis_utils import plot_model

def mat_mul(A, B):
    return tf.matmul(A, B)

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    print(data.shape,label.shape)
    return (data, label)

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data

def model_branch(num_points=1024):
  input_points = Input(shape=(num_points, 3))
  x = Convolution1D(64, 1, activation='relu',
                  input_shape=(num_points, 3))(input_points)#none 2048 64

  x = BatchNormalization()(x)

  x = Convolution1D(128, 1, activation='relu')(x)#none 2048 128
  x = BatchNormalization()(x)

  x = Convolution1D(1024, 1, activation='relu')(x) #none 2048 1024
  x = BatchNormalization()(x)

  x = MaxPooling1D(pool_size=num_points)(x) #none 1 1024

  x = Dense(512, activation='relu')(x)
  x = BatchNormalization()(x)#none 1 512

  x = Dense(256, activation='relu')(x)#none 1 256
  x = BatchNormalization()(x)

  x = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x) #none 1 9

  input_T = Reshape((3, 3))(x) #none 3 3
  # forward net
  g = Lambda(mat_mul, arguments={'B': input_T})(input_points) #none 2048 3

  g = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
  g = BatchNormalization()(g)
  g = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
  g = BatchNormalization()(g)

  # feature transform net
  f = Convolution1D(64, 1, activation='relu')(g)
  f = BatchNormalization()(f)
  f = Convolution1D(128, 1, activation='relu')(f)
  f = BatchNormalization()(f)
  f = Convolution1D(1024, 1, activation='relu')(f)
  f = BatchNormalization()(f) # none 2048 1024
  f = MaxPooling1D(pool_size=num_points)(f)# none 1 1024
  f = Dense(512, activation='relu')(f)# none 1 512
  f = BatchNormalization()(f)
  f = Dense(256, activation='relu')(f)# none 1 256
  f = BatchNormalization()(f)
  f = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)# none 1 4096
  
  feature_T = Reshape((64, 64))(f)# none 64 64

  # forward net
  g = Lambda(mat_mul, arguments={'B': feature_T})(g)  #none 2048 64
  g = Convolution1D(64, 1, activation='relu')(g)   #none 2048 64
  g = BatchNormalization()(g)
  g = Convolution1D(128, 1, activation='relu')(g)   #none 2048 128
  g = BatchNormalization()(g)
  g = Convolution1D(1024, 1, activation='relu')(g)  #none 2048 1024
  g = BatchNormalization()(g)

  # global_feature
  global_feature = MaxPooling1D(pool_size=num_points)(g)   #none 1 1024

  global_feature = Model(inputs=input_points, outputs=global_feature)
  return global_feature
# number of points in each sample
num_points = 1024

# number of categories
k = 40

# define optimizer
adam = optimizers.Adam(lr=0.001, decay=0.7)

# ------------------------------------ Pointnet Architecture
# input_Transformation_net
branch_1 = model_branch(num_points)
branch_2 = model_branch(num_points)
branch_3 = model_branch(num_points)
branch_4 = model_branch(num_points)
branch_5 = model_branch(num_points)

global_feature = concatenate([branch_1.global_feature, branch_2.global_feature,branch_3.global_feature,branch_4.global_feature,branch_5.global_feature])
# point_net_cls
c = Dense(512, activation='relu')(global_feature)   #none 1 512
c = BatchNormalization()(c)
c = Dropout(rate=0.7)(c)
c = Dense(256, activation='relu')(c)    #none 1 256
c = BatchNormalization()(c)
c = Dropout(rate=0.7)(c)
c = Dense(k, activation='softmax')(c)    #none 1 40
prediction = Flatten()(c)
# --------------------------------------------------end of pointnet

# print the model summary
model = Model(inputs=[branch_1.input_points, branch_2.input_points,branch_3.input_points,branch_4.input_points,branch_5.input_points], outputs=prediction)
print(model.summary())
# # load train points and labels
# path = os.path.dirname(os.path.realpath(__file__))
# train_path = os.path.join(path, "PrepData")
# filenames = [d for d in os.listdir(train_path)]
# # print("train_path  ",train_path)
# # print("filenames   ",filenames)
# train_points = None
# train_labels = None
# for d in filenames:
#     cur_points, cur_labels = load_h5(os.path.join(train_path, d))
#     cur_points = cur_points.reshape(1, -1, 3)
#     cur_labels = cur_labels.reshape(1, -1)
#     if train_labels is None or train_points is None:
#         train_labels = cur_labels
#         train_points = cur_points
#     else:
#         train_labels = np.hstack((train_labels, cur_labels))
#         train_points = np.hstack((train_points, cur_points))
# train_points_r = train_points.reshape(-1, num_points, 3)
# train_labels_r = train_labels.reshape(-1, 1)

# # load test points and labels
# test_path = os.path.join(path, "PrepData_test")
# filenames = [d for d in os.listdir(test_path)]
# # print("test_path  ",test_path)
# # print("filenames  ",filenames)
# test_points = None
# test_labels = None
# for d in filenames:
#     cur_points, cur_labels = load_h5(os.path.join(test_path, d))
#     cur_points = cur_points.reshape(1, -1, 3)
#     cur_labels = cur_labels.reshape(1, -1)
#     if test_labels is None or test_points is None:
#         test_labels = cur_labels
#         test_points = cur_points
#     else:
#         test_labels = np.hstack((test_labels, cur_labels))
#         test_points = np.hstack((test_points, cur_points))
# test_points_r = test_points.reshape(-1, num_points, 3)
# test_labels_r = test_labels.reshape(-1, 1)


# # label to categorical
# Y_train = np_utils.to_categorical(train_labels_r, k)
# Y_test = np_utils.to_categorical(test_labels_r, k)

# # compile classification model
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # Fit model on training data
# for i in range(1,50):
#     #model.fit(train_points_r, Y_train, batch_size=32, epochs=1, shuffle=True, verbose=1)
#     # rotate and jitter the points
#     train_points_rotate = rotate_point_cloud(train_points_r)
#     train_points_jitter = jitter_point_cloud(train_points_rotate)
#     model.fit(train_points_jitter, Y_train, batch_size=4, epochs=1, shuffle=True, verbose=1)
#     s = "Current epoch is:" + str(i)
#     print(s)
#     if i % 5 == 0:
#         score = model.evaluate(test_points_r, Y_test, verbose=1)
#         print('Test loss: ', score[0])
#         print('Test accuracy: ', score[1])

# # score the model
# score = model.evaluate(test_points_r, Y_test, verbose=1)
# print('Test loss: ', score[0])
# print('Test accuracy: ', score[1])
