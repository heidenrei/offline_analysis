"""
    Author: Gavin Heidenreich
    Email: gheidenr@uottawa.ca
    Organization: University of Ottawa (Silasi Lab)
"""

from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

from tensorflow.keras.applications import EfficientNetB4 as Net
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import losses
from tensorflow.keras import metrics

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import math

import cv2
from PIL import Image
from PIL import ImageDraw 

np.random.seed(2021)

train_dir = '/home/silasi/offline_analysis/data/train/'
validation_dir = '/home/silasi/offline_analysis/data/valid/'
test_dir = '/home/silasi/offline_analysis/data/test/'

weights_path = '/home/silasi/offline_analysis/model/base_model.h5'
tuned_weights_path = '/home/silasi/offline_analysis/model/model.h5'
load_from_weights_path = False
batch_size = 16

width = 224
height = 224
epochs = 250

NUM_TRAIN = len(os.listdir(train_dir + '1')) + len(os.listdir(train_dir + '0'))
NUM_TEST = len(os.listdir(test_dir + 'predict'))
NUM_VAL = len(os.listdir(validation_dir + '1')) + len(os.listdir(validation_dir + '0'))

weights = None
if os.path.exists(weights_path):
    weights = weights_path

fine_tuning = False
train = False
analyze_test = False
test_on_vid = True

dropout_rate = 0.4
input_shape = (height, width, 3)

def get_callbacks():
    def scheduler(epoch, lr):
        if epoch <= 150:
            return lr
        return lr * math.exp(-0.1)

    callbacks_list = []
    callbacks_list.append(callbacks.ModelCheckpoint("model/model.h5", save_best_only=True))
    # callbacks_list.append(callbacks.LearningRateScheduler(scheduler))
    # callbacks_list.append(callbacks.EarlyStopping(monitor="val_binary_accuracy", patience=100, verbose=1, mode="max", restore_best_weights=True))
    return callbacks_list

callbacks_list = get_callbacks()


#cant really do much augmentation cause the pellet is right around the edge of the frame in some frames

def get_gens():
    train_datagen = ImageDataGenerator(
        #rotation_range=10,
        #width_shift_range=0.1,
        #height_shift_range=0.1,
        #shear_range=0.1,
        #zoom_range=0.1,
        horizontal_flip=True,
        zca_whitening=True,
        fill_mode='nearest'
        )

    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        color_mode="grayscale",
        target_size=(height, width),
        batch_size=batch_size,
        shuffle=True,
        class_mode='binary')

    test_datagen = ImageDataGenerator()#rescale=1./255)

    validation_generator = test_datagen.flow_from_directory(
        directory=validation_dir,
        color_mode="grayscale",
        target_size=(height, width),
        batch_size=batch_size,
        shuffle=True,
        class_mode='binary')

    test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=(224, 224),
        color_mode="grayscale",
        batch_size=1,
        class_mode=None,
        shuffle=False,
        seed=42)

    return train_datagen, train_generator, test_datagen, validation_generator, test_generator

def model2():
    efficient_net = Net(
        weights='imagenet',
        input_shape=(224,224,3),
        include_top=False,
        pooling='max'
    )

    model = models.Sequential()
    model.add(efficient_net)
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate, name='dropout_layer1'))
    model.add(layers.Dense(units = 64, activation = 'relu'))
    model.add(layers.Dense(units = 32, activation = 'relu'))
    model.add(layers.Dense(units = 16, activation = 'relu'))
    model.add(layers.Dense(units = 1, activation='sigmoid'))
    model.summary()

    #

    efficient_net.trainable = False
    opt = optimizers.Adam(lr=1e-4)
    if fine_tuning:
        for layer in model.layers:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
        opt = optimizers.Adam(lr=1e-5)

    if load_from_weights_path and weights:
        if not train:
            model.load_weights(tuned_weights_path)
        else:
            model.load_weights(weights)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['binary_accuracy'])
    return model

def plot2(history):
    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_x = range(len(acc))

    plt.plot(epochs_x, acc, 'r', label='Training acc')
    plt.plot(epochs_x, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs_x, loss, 'r', label='Training loss')
    plt.plot(epochs_x, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

def get_model_for_pred():
    efficient_net = Net(
        weights='imagenet',
        input_shape=(224,224,3),
        include_top=False,
        pooling='max'
    )

    model = models.Sequential()
    model.add(efficient_net)
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate, name='dropout_layer1'))
    model.add(layers.Dense(units = 64, activation = 'relu'))
    model.add(layers.Dense(units = 32, activation = 'relu'))
    model.add(layers.Dense(units = 16, activation = 'relu'))
    model.add(layers.Dense(units = 1, activation='sigmoid'))
    model.summary()
    efficient_net.trainable = False

    model.load_weights(weights_path)
    opt = optimizers.Adam(lr=1e-5)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['binary_accuracy'])
    return model

def get_pred(img):
    predict_image = np.asarray([cv2.resize(img, (224, 224))])
    # predict_image = predict_image / 255.
    result = model.predict(predict_image)[0]

    print(result)
    return result < 0.5

def predict(images):
    result = model.predict(images / 255.)
    return result

def predict_on_single_raw_image(opencv_image):
    predict_image = np.asarray([cv2.resize(opencv_image, (224, 224))])
    predict_image = predict_image / 255.
    return model.predict(predict_image)[0]

def predict_in_real_use(opencv_image):
    predict_image = np.asarray([cv2.resize(opencv_image, (224, 224))])
    predict_image = predict_image / 255.
    if model.predict(predict_image)[0] > 0.5:
        return True
    return False

def test_on_video(video_file):
    video_stream = cv2.VideoCapture(video_file)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    grab, frame = video_stream.read()
    model = get_model_for_pred()
    frame_cnt = 0
    while frame is not None:
        if frame_cnt % 1 == 0:
            img = cv2.resize(frame, (224, 224))
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.asarray([img])
            result = model.predict(img)[0][0] > 0.5
            print(result)
            print(type(result))
            cv2.putText(frame, str(result), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                            lineType=cv2.LINE_AA)

            cv2.imshow("test_on_video", frame)
            # gray = cv2.cvt
            # cap.write(frame)

            frame_cnt += 1
            grab, frame = video_stream.read()
        else:
            cv2.waitKey(80)
            # cap.write(frame)
            grab, frame = video_stream.read()
            frame_cnt += 1
        cv2.waitKey(15)

def main():
    print(f'{NUM_TRAIN=}', f'{NUM_TEST=}', f'{NUM_VAL=}')
    train_datagen, train_generator, test_datagen, validation_generator, test_generator = get_gens()

    if train:
        model = model2()
        history = model.fit(
            train_generator,
            epochs=epochs,
            callbacks=callbacks_list,
            validation_data=validation_generator,
            verbose=1,
            )

        print(history.history.keys())

        plot2(history)
    

    if analyze_test:
        model = get_model_for_pred()
        pred = model.predict_generator(test_generator)
        filenames = test_generator.filenames
        print(pred)

        res = list(zip(filenames, pred))

        for r in res:
            fn = r[0]
            res = r[1]
            print(fn, res)
            img = Image.open(test_dir + fn)
            draw = ImageDraw.Draw(img) 
            draw.text((10, 10), str(res > 0.5), (255, 0, 0)) 

            img.show()
            time.sleep(0.5)

    if test_on_vid:
        test_on_video('/home/silasi/offline_analysis/2021-03-24_(10-18-13)_00782B1A622B_24_28.avi')

if __name__ == '__main__':
    main()

