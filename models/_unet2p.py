"""
UNet++
author: Peng Fei Jia
https://github.com/AlphaJia/keras_unet_plus_plus
https://arxiv.org/abs/1807.10165
"""

from tensorflow.keras import backend as K
from tensorflow.keras.layers import (concatenate, Input, Conv2D,
                                     Conv2DTranspose, Activation, AvgPool2D,
                                     BatchNormalization)
from tensorflow.keras.models import Model

dropout_rate = 0.5

def conv_batchnorm_relu_block(input_tensor, nb_filter, kernel_size=3,
                             activation='relu', padding='same'):
    """
    A block that contains a convolution layer, batch normalization, and relu activation.
    conv_bathnorm_relu_block(input_tensor, num_filters, kernel_size=3, activation='relu', padding='same')
    ----------
    Olaf Ronneberger, Philipp Fischer, and Thomas Brox. (2015). U-net: Convolutional
    networks for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.
    """
    x = Conv2D(nb_filter, (kernel_size, kernel_size), padding=padding)(input_tensor)
    x = BatchNormalization(axis=2)(x)
    x = Activation(activation)(x)
    return x

def unet2p_2d(input_shape, n_labels, using_deep_supervision=False):
    """
    The U-net++ model
    model_build_func(input_shape, n_labels, using_deep_supervision=False)
    ----------
    Olaf Ronneberger, Philipp Fischer, and Thomas Brox. (2015). U-net: Convolutional
    networks for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.
    """
    # Number of filters
    nb_filters = [32, 64, 128, 256, 512]

    # Set image data format to channels first
    global bn_axis

    K.set_image_data_format('channels_last')
    bn_axis = -1

    # Input
    inputs = Input(shape = input_shape, name='input_image')

    conv1_1 = conv_batchnorm_relu_block(inputs, nb_filter = nb_filters[0])
    pool1 = AvgPool2D(pool_size=(2, 2), strides=(2,2), name = 'pool1')(conv1_1)

    conv2_1 = conv_batchnorm_relu_block(pool1, nb_filter=nb_filters[1])
    pool2 = AvgPool2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    up1_2 = Conv2DTranspose(nb_filters[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = conv_batchnorm_relu_block(conv1_2, nb_filter=nb_filters[0])

    conv3_1 = conv_batchnorm_relu_block(pool2, nb_filter=nb_filters[2])
    pool3 = AvgPool2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filters[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = conv_batchnorm_relu_block(conv2_2, nb_filter=nb_filters[1])

    up1_3 = Conv2DTranspose(nb_filters[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
    conv1_3 = conv_batchnorm_relu_block(conv1_3, nb_filter=nb_filters[0])

    conv4_1 = conv_batchnorm_relu_block(pool3, nb_filter=nb_filters[3])
    pool4 = AvgPool2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filters[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = conv_batchnorm_relu_block(conv3_2, nb_filter=nb_filters[2])

    up2_3 = Conv2DTranspose(nb_filters[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = conv_batchnorm_relu_block(conv2_3, nb_filter=nb_filters[1])

    up1_4 = Conv2DTranspose(nb_filters[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = conv_batchnorm_relu_block(conv1_4, nb_filter=nb_filters[0])

    conv5_1 = conv_batchnorm_relu_block(pool4, nb_filter=nb_filters[4])

    up4_2 = Conv2DTranspose(nb_filters[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = conv_batchnorm_relu_block(conv4_2, nb_filter=nb_filters[3])

    up3_3 = Conv2DTranspose(nb_filters[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = conv_batchnorm_relu_block(conv3_3, nb_filter=nb_filters[2])

    up2_4 = Conv2DTranspose(nb_filters[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = conv_batchnorm_relu_block(conv2_4, nb_filter=nb_filters[1])

    up1_5 = Conv2DTranspose(nb_filters[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 = conv_batchnorm_relu_block(conv1_5, nb_filter=nb_filters[0])

    nestnet_output_1 = Conv2D(n_labels, (1, 1), activation='sigmoid', name='output_1', padding='same')(conv1_2)
    nestnet_output_2 = Conv2D(n_labels, (1, 1), activation='sigmoid', name='output_2', padding='same')(conv1_3)
    nestnet_output_3 = Conv2D(n_labels, (1, 1), activation='sigmoid', name='output_3', padding='same')(conv1_4)
    nestnet_output_4 = Conv2D(n_labels, (1, 1), activation='sigmoid', name='output_4', padding='same')(conv1_5)

    if using_deep_supervision:
        model = Model(inputs=inputs, outputs=[nestnet_output_1, nestnet_output_2, nestnet_output_3, nestnet_output_4])
    else:
        model = Model(inputs=inputs, outputs=nestnet_output_4)

    return model