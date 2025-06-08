"""
ResUNet-a: A deep learning framework for semantic segmentation of remotely sensed data
Authors: Foivos I. Diakogiannis, François Waldner, Peter Caccetta, Chen Wu
https://doi.org/10.1016/j.isprsjprs.2020.01.013

https://github.com/nikhilroxtomar/Deep-Residual-Unet/blob/master/Deep%20Residual%20UNet.ipynb
"""
from tensorflow import keras

def bn_act(x, act=True):
    x = keras.layers.BatchNormalization()(x)
    if act == True:
        x = keras.layers.Activation("relu")(x)
    return x

def conv_block(x, filters, kernel_size=(3,3), strides=1, padding="same"):
    conv = bn_act(x)
    conv = keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(conv)
    return conv

def stem(x, filters, kernel_size=(3,3), strides=1, padding="same"):
    conv = keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)

    shortcut = keras.layers.Conv2D(filters, kernel_size=(1,1), strides=strides, padding=padding)(x)
    shortcut = bn_act(shortcut, act=False)

    output = keras.layers.Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=(3,3), strides=1, padding="same"):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)

    shortcut = keras.layers.Conv2D(filters, kernel_size=(1,1), strides=strides, padding=padding)(x)
    shortcut = bn_act(shortcut, act=False)

    output = keras.layers.Add()([shortcut, res])
    return output

def upsample_concat_block(x, skip):
    u = keras.layers.UpSampling2D((2,2))(x)
    c = keras.layers.Concatenate()([u, skip])
    return c

"""ResUNet-a: A deep learning framework for semantic segmentation of remotely sensed data
Authors: Foivos I. Diakogiannis, François Waldner, Peter Caccetta, Chen Wu
https://doi.org/10.1016/j.isprsjprs.2020.01.013
"""
def resunet_a_2d(input_shape, n_labels):
    """
    The ResUNet-a model
    model_build_func(input_shape, n_labels)
    """
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input(input_shape)

    # Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)

    # Bridge
    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)

    # Decoder
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block(u1, f[4])

    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])

    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])

    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])

    outputs = keras.layers.Conv2D(n_labels, (1, 1), activation='sigmoid')(d4)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    return model