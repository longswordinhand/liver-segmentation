from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from data import load_train_data, load_test_data
from keras.utils import multi_gpu_model
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries, active_contour

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

image_org_rows = 512
image_org_cols = 512

img_rows = 96
img_cols = 96

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model_num = 3
    model_list = []
    gpu_num = 3
    epoch_num = 250
    batch_size_num_init = 256
    validation_split_factor = 0.2
    for i in range(model_num):
        model = get_unet()
        # model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

        print('-'*30)
        print('Fitting model...')
        print('-'*30)

        batch_size_num = min(batch_size_num_init,int(imgs_train.shape[0]*(1-validation_split_factor)))

        parallel_model = multi_gpu_model(model, gpus=gpu_num)
        parallel_model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
        parallel_model.fit(imgs_train, imgs_mask_train, epochs=epoch_num, batch_size = batch_size_num, 
                verbose=1, shuffle=True,
                validation_split=validation_split_factor)

        imgs_mask_pred = model.predict(imgs_train, batch_size = batch_size_num, verbose=1)

        accList = np.zeros(imgs_mask_pred.shape[0])
        for j in range(imgs_mask_pred.shape[0]):
            y_true_f = imgs_mask_train[j,...].flatten()
            y_pred_f = imgs_mask_pred[j,...].flatten()
            intersection = np.sum(y_true_f * y_pred_f)
            accList[j] = (2.*intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
            

        idx = np.where(accList > np.mean(accList))
        #idx = np.array(idx).tolist()
        imgs_train = imgs_train[idx,...]
        sz = imgs_train.shape
        imgs_train = np.reshape(imgs_train,[sz[1], sz[2],sz[3],sz[4]])
        imgs_mask_train = imgs_mask_train[idx,...]
        imgs_mask_train = np.reshape(imgs_mask_train,[sz[1], sz[2],sz[3],sz[4]])

        model_list.append(model)



    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    #model.load_weights('weights.h5')
    
    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = np.zeros(imgs_test.shape)
    for i in range(model_num):
        model = model_list[i]
        imgs_mask_test = imgs_mask_test + model.predict(imgs_test, verbose=1)

    imgs_mask_test = imgs_mask_test/model_num
    np.save('imgs_mask_test.npy', imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        image = resize(image,(image_org_rows,image_org_cols))
        image = image > 0
        image = find_boundaries(image, connectivity=1, mode='thick', background=0)
        init = np.where(image > 0)
        image_org = imgs_test[image_id,...]
        snake = active_contour(gaussian(image_org, 3),
                       init, alpha=0.015, beta=10, gamma=0.001)
        image = np.dot(255*image_org,snake) + np.dot(image_org, 1-snake)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)

if __name__ == '__main__':
    train_and_predict()
