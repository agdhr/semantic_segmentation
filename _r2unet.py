"""
Automated Design of Deep Learning Methods for Biomedical Image Segmentation
Fabian Isensee, Paul F. Jäger, Simon A. A. Kohl, Jens Petersen, Klaus H. Maier-Hein
https://arxiv.org/abs/1904.08128

Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation
Md Zahangir Alom, Mahmudul Hasan, Chris Yakopcic, Tarek M. Taha, Vijayan K. Asari
https://arxiv.org/abs/1802.06955
https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf

Refcode: https://github.com/lixiaolei1982/Keras-Implementation-of-U-Net-R2U-Net-Attention-U-Net-Attention-R2U-Net.-/tree/master
"""
from pyexpat import features

from keras.src.layers import layer
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input, \
    add, multiply, concatenate, Dropout, Activation, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.backend as K

def rec_res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=(3, 3), stride=[1, 1],
                  padding='same', data_format='channels_first'):
    index_map = {'channels_first': 1, 'channels_last': 3}  # Using -1 for the last dimension
    #input_n_filters = input_layer.shape[index_map[data_format]]

    if data_format == 'channels_first':
        input_n_filters = input_layer.shape[index_map[data_format]]
    else:
        input_n_filters = input_layer.shape[index_map[data_format]]
    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, (1, 1), strides=stride, padding=padding, data_format=data_format)(
            input_layer)
    else:
        skip_layer = input_layer
    layer = skip_layer
    for j in range(2):
        for i in range(2):
            if i == 0:
                layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                    layer)
                if batch_normalization:
                    layer1 = BatchNormalization()(layer1)
                layer1 = Activation('relu')(layer1)
            layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                add([layer1, layer]))
            if batch_normalization:
                layer1 = BatchNormalization()(layer1)
            layer1 = Activation('relu')(layer1)
        layer = layer1

    out_layer = add([layer, skip_layer])
    return out_layer


def up_and_concate(down_layer, layer, data_format='channels_first'):
    index_map = {'channels_first': 1, 'channels_last': 3}  # Using -1 for the last dimension
    # input_n_filters = input_layer.shape[index_map[data_format]]
    if data_format == 'channels_first':
        in_channel = down_layer.shape[index_map[data_format]]
    else:
        in_channel = down_layer.shape[index_map[data_format]]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling2D((2, 2), data_format=data_format)(down_layer)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])

    return concate

def r2unet_2d(img_w, img_h, n_label, data_format='channels_first'):
    inputs = Input((1, img_w, img_h))
    x = inputs
    depth = 4
    features = 64
    skips = []
    for i in range(depth):
        x = rec_res_block(x, features, data_format=data_format)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)

        features = features * 2

    x = rec_res_block(x, features, data_format=data_format)

    for i in reversed(range(depth)):
        features = features // 2
        x = up_and_concate(x, skips[i], data_format=data_format)
        x = rec_res_block(x, features, data_format=data_format)

    conv6 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x)
    conv7 = Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv7)
    #model.compile(optimizer=Adam(lr=1e-6), loss=[dice_coef_loss], metrics=['accuracy', dice_coef])
    return model