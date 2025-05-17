"""
U-Net: Convolutional Networks for Biomedical Image Segmentation
Authors: Olaf Ronneberger, Philipp Fischer, Thomas Broxauthor
https://github.com/decouples/Unet/tree/master
https://arxiv.org/abs/1505.04597
"""


import os
import numpy as np
from elasticdeform import tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, BatchNormalization, Activation
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K

def unet_2d(input_size, n_class=1,):
    inputs = Input(input_size)
	# 网络结构定义，数据处理的时候已经转化为灰度图了
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    #print ("conv1 shape:",conv1.shape)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    #print ("conv1 shape:",conv1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #print ("pool1 shape:",pool1.shape)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    #print ("conv2 shape:",conv2.shape)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    #print ("conv2 shape:",conv2.shape)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #print ("pool2 shape:",pool2.shape)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    #print ("conv3 shape:",conv3.shape)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    #print ("conv3 shape:",conv3.shape)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #print ("pool3 shape:",pool3.shape)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    # merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
    merge6 = Concatenate(axis=3)([drop4, up6])
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    # merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    # merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
    merge8 = Concatenate(axis=3)([conv2,up8])
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    # merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
    merge9 = Concatenate(axis=3)([conv1,up9])
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(n_class, 1, activation = 'sigmoid')(conv9)
	# [batch, 512, 512, 1]

    model = Model(inputs = inputs, outputs = conv10)
    #model.compile(optimizer = Adam(learning_rate = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy', iou, dice_coef])
    return model

def conv_block(x, filter_size, size, dropout, batch_norm=False):
    conv = Conv2D(size, (filter_size, filter_size), padding="same")(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)
    conv = Activation("relu")(conv)

    conv = Conv2D(size, (filter_size, filter_size), padding="same")(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)
    conv = Activation("relu")(conv)

    if dropout > 0:
        conv = Dropout(dropout)(conv)

    return conv
def UNet(input_shape, n_class=1, dropout_rate=0.5, batch_norm=True):
    # network structure
    FILTER_NUM = 64 # number of filters for the first layer
    FILTER_SIZE = 3 # filter size for the first layer
    U_SAMP_SIZE =2 # Size of upsampling filters

    inputs = Input(input_shape, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = MaxPooling2D(pool_size=(2, 2))(conv_128)
    # DownRes 2
    conv_64 = conv_block(pool_64, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = MaxPooling2D(pool_size=(2, 2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_32, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = MaxPooling2D(pool_size=(2, 2))(conv_32)
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = MaxPooling2D(pool_size=(2, 2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 16 * FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    up_16 = UpSampling2D(size=(U_SAMP_SIZE, U_SAMP_SIZE), data_format="channel_last")(conv_8)
    up_16 = Concatenate([conv_16, up_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)

    # UpRes 7
    up_32 = UpSampling2D(size=(U_SAMP_SIZE, U_SAMP_SIZE), data_format="channel_last")(conv_16)
    up_32 = Concatenate([conv_32, up_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)

    # UpRes 8
    up_64 = UpSampling2D(size=(U_SAMP_SIZE, U_SAMP_SIZE), data_format="channel_last")(conv_32)
    up_64 = Concatenate([conv_64, up_64], axis=3)
    up_conv_64 = conv_block(up_64, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)

    # UpRes 9
    up_128 = UpSampling2D(size=(U_SAMP_SIZE, U_SAMP_SIZE), data_format="channel_last")(conv_64)
    up_128 = Concatenate([conv_128, up_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # 1*1 convolutional layers
    conv_final = Conv2D(n_class, (1,1))(up_conv_128)
    conv_final = BatchNormalization(axis=3)(conv_final)
    conv_final = Activation("sigmoid")(conv_final) #Change to softmax for multichannel

    model = Model(inputs=inputs, outputs=conv_final, name="UNet")
    return model