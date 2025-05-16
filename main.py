"""IMPORT LIBRARIES"""
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import (ImageDataGenerator, 
                                                  load_img,
                                                  img_to_array, 
                                                  array_to_img)
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

from metrics import iou, dice_coef

from _unet import unet_2d
from _vnet import vnet_2d

"""DATASET DIRECTORY"""
data_dir = 'd://z/master/comvis/segmentation/isbi_membrane/membrane/'
train_path = data_dir + "train"
label_path = data_dir + "train"
test_path = data_dir + "test"

output_path = "d://z/master/comvis/segmentation/isbi_membrane/membrane/"
npy_path = output_path + "npydata"
predict_path = output_path + "predict"

img_type = "tif"
out_rows = 512
out_cols = 512

dataset_name = 'isbi_membrane'
model_name = 'vnet'

"""LOAD DATASET"""
def create_train_data(train_path, label_path, out_rows, out_cols, img_type):
    i = 0
    print('-' * 30)
    print('Creating training images...')
    print('-' * 30)
    """Load images in the train path"""
    imgs = glob.glob(train_path + "/train/*." + img_type)
    """Print the number of images in the train path"""
    print('The number of train images/masks is::',len(imgs))
    imgdatas = np.ndarray((len(imgs), out_rows, out_cols, 1), dtype=np.uint8)
    imglabels = np.ndarray((len(imgs), out_rows, out_cols, 1), dtype=np.uint8)
    for imgname in imgs:
        """Define image filename and label filename"""
        midname = imgname[imgname.rindex("/") + 1:]
        """Load train images and labels from the path and change to grayscale"""
        img = load_img(train_path + "/" + midname, color_mode='grayscale')
        label = load_img(label_path + "/" + midname, color_mode='grayscale')
        """Convert images and labels to numpy arrays"""
        img = img_to_array(img)
        label = img_to_array(label)
        """Alternarives"""
        # img = cv2.imread(self.data_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
        # label = cv2.imread(self.label_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
        # img = np.array([img])
        # label = np.array([label])
        imgdatas[i] = img
        imglabels[i] = label
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, len(imgs)))
        i += 1
    print('Loading done')
    """Save images on .npy files"""
    np.save(npy_path + f'/{dataset_name}_imgs_train.npy', imgdatas)
    np.save(npy_path + f'/{dataset_name}_imgs_mask_train.npy', imglabels)
    print('Saving to .npy files done.')

def create_test_data(test_path, out_rows, out_cols, img_type):
    i = 0
    print('-' * 30)
    print('Creating test images...')
    print('-' * 30)
    imgs = glob.glob(test_path + "/test/*." + img_type)
    print('The number of test images/masks is::',len(imgs))
    imgdatas = np.ndarray((len(imgs), out_rows, out_cols, 1), dtype=np.uint8)
    for imgname in imgs:
        midname = imgname[imgname.rindex("/") + 1:]
        img = load_img(test_path + "/" + midname, color_mode='grayscale')
        img = img_to_array(img)
        # img = cv2.imread(self.test_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
        # img = np.array([img])
        imgdatas[i] = img
        i += 1
    print('loading done')
    np.save(npy_path + f'/{dataset_name}_imgs_test.npy', imgdatas)
    print('Saving to imgs_test.npy files done.')

"""Create train and test data"""
create_train_data(train_path, label_path, out_rows, out_cols, img_type)
create_test_data(test_path, out_rows, out_cols, img_type)

def load_train_data(npy_path):
        print('-' * 30)
        print('load train images...')
        print('-' * 30)
        """Load images and masks from .npy files"""
        imgs_train = np.load(npy_path + f"/{dataset_name}_imgs_train.npy")
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = np.load(npy_path + f"/{dataset_name}_imgs_mask_train.npy")
        imgs_mask_train = imgs_mask_train.astype('float32')
        """Rescale pixel images from [0, 255] to [0, 1]"""
        imgs_train /= 255
        mean = imgs_train.mean(axis=0)
        imgs_train -= mean
        imgs_mask_train /= 255
        imgs_mask_train[imgs_mask_train > 0.5] = 1
        imgs_mask_train[imgs_mask_train <= 0.5] = 0
        return imgs_train, imgs_mask_train

def load_test_data(npy_path):
        print('-' * 30)
        print('load test images...')
        print('-' * 30)
        imgs_test = np.load(npy_path + f"/{dataset_name}_imgs_test.npy")
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 255
        mean = imgs_test.mean(axis=0)
        imgs_test -= mean
        return imgs_test

imgs_train, imgs_mask_train = load_train_data(npy_path)
print(imgs_train.shape, imgs_mask_train.shape)
imgs_test = load_test_data(npy_path)
print(imgs_test.shape)

""" SEMANTIC SEGMENTATION MODEL"""
"""UNet 2D"""
#model = unet_2d((out_rows, out_cols, 1), filter_num=[64, 128, 256, 512, 1024], n_labels=1)

"""VNet 2D"""
model = vnet_2d((out_rows, out_cols, 1), num_class=1)
#model = vnet_2d()

"""Model Compilation"""
model.compile(optimizer = Adam(learning_rate = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy', iou, dice_coef])
model.summary()

save_dir = os.path.join(os.getcwd(), 'outputs')
model_filename = f'{model_name}_isbi_10epoch.keras'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_filename)
model.save(model_path)
print('saved trained model at %s' %  model_path)

model_checkpoint = ModelCheckpoint(model_path,
                                   monitor='val_accuracy',
                                   verbose=1,
                                   save_best_only=True)

print('Fitting model...')
history = model.fit(imgs_train,
                    imgs_mask_train,
                    batch_size=2,
                    epochs=10,
                    verbose=1,
                    validation_split=0.4,
                    shuffle=True,
                    callbacks=[model_checkpoint])

import matplotlib.pyplot as plt

"""Model Evaluation"""
def plot_segm_history(history, metrics=['accuracy', 'val_accuracy'], losses = ['loss', 'val_loss']):
    # summarize history for iou or accuracy
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot()
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    for metric in metrics:
        plt.plot(history.history[metric], linewidth=3)
    plt.suptitle("accuray over epochs", fontsize=12)
    plt.ylabel('accuracy', fontname="Segoe UI",fontsize=12)
    plt.xlabel('epoch', fontname="Segoe UI", fontsize=12)
    plt.legend(metrics, loc='lower right', fontsize=12)
    plt.grid(False)
    plt.savefig(f'outputs/{model_name}unet-isbi_accuracy-plot.png')
    plt.show()

    # summary history for loss
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot()
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    for loss in losses:
        plt.plot(history.history[loss], linewidth=3)
    plt.suptitle("losses over epochs", fontsize=12)
    plt.ylabel('loss', fontname="Segoe UI", fontsize=12)
    plt.xlabel('epoch', fontname="Segoe UI", fontsize=12)
    plt.legend(losses, loc='upper right', fontsize=12)
    plt.grid(False)
    plt.savefig(f'outputs/{model_name}-isbi_loss-plot_10epoch.png')
    plt.show()

plot_segm_history(history, metrics=['accuracy', 'val_accuracy'], losses = ['loss', 'val_loss'])

